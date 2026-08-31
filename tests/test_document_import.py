from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import app.main as main_module
from app.document_parser import DocumentParser, DownloadedResource


def _resource(
    content_type: str,
    body: bytes,
    *,
    url: str = "https://public.example/report",
    disposition: str | None = None,
) -> DownloadedResource:
    return DownloadedResource(
        final_url=url,
        status_code=200,
        content_type=content_type,
        body=body,
        content_disposition=disposition,
    )


def _text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
    )
    content = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content.set_data(f"BT /F1 12 Tf 20 100 Td ({escaped}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize(
    ("resource", "expected_title", "expected_content"),
    [
        (
            _resource(
                "text/html",
                b"<title>Market Report</title><style>hidden</style><p>Demand rose.</p>",
            ),
            "Market Report",
            "Demand rose.",
        ),
        (
            _resource("application/json", b'{"supplier":"A","amount":42}'),
            "report",
            '"amount": 42',
        ),
        (
            _resource(
                "text/csv",
                b"supplier,amount\nA,42\n",
                disposition='attachment; filename="purchases.csv"',
            ),
            "purchases.csv",
            "supplier\tamount\nA\t42",
        ),
        (
            _resource("text/markdown", b"# Policy\n\nQuarterly review required."),
            "report",
            "Quarterly review required.",
        ),
    ],
)
def test_document_parser_supports_bounded_text_formats(
    resource: DownloadedResource, expected_title: str, expected_content: str
) -> None:
    parsed = DocumentParser().parse(resource)

    assert parsed.title == expected_title
    assert expected_content in parsed.content
    assert "hidden" not in parsed.content


def test_document_parser_extracts_pdf_text_and_rejects_empty_or_oversized_documents() -> None:
    parser = DocumentParser(max_chars=100, max_pdf_pages=2)
    parsed = parser.parse(
        _resource(
            "application/pdf",
            _text_pdf("Quarterly supplier policy"),
            url="https://public.example/policy.pdf",
        )
    )

    assert parsed.title == "policy.pdf"
    assert parsed.filename == "policy.pdf"
    assert parsed.content == "[Page 1]\nQuarterly supplier policy"

    with pytest.raises(ValueError, match="extracted no text"):
        parser.parse(_resource("application/pdf", _text_pdf("")))
    with pytest.raises(ValueError, match="text limit"):
        DocumentParser(max_chars=5).parse(_resource("text/plain", b"too long"))


def test_document_parser_rejects_invalid_json_and_non_utf8_text() -> None:
    parser = DocumentParser()

    with pytest.raises(ValueError, match="JSON document is invalid"):
        parser.parse(_resource("application/json", b"{invalid"))
    with pytest.raises(ValueError, match="must be UTF-8"):
        parser.parse(_resource("text/plain", b"\xff"))


class _StubFetcher:
    def __init__(self, resource: DownloadedResource) -> None:
        self.resource = resource
        self.urls: list[str] = []

    async def download(self, url: str) -> DownloadedResource:
        self.urls.append(url)
        return self.resource


def test_admin_url_import_is_idempotent_and_retrievable_with_source_url(monkeypatch) -> None:
    admin_token = "import-admin-token"
    tenant_id = f"tenant-{uuid4().hex}"
    marker = f"imported-policy-{uuid4().hex}"
    resource = _resource(
        "text/csv",
        f"policy,requirement\nsupplier,{marker} quarterly review\n".encode(),
        url="https://public.example/policy.csv",
        disposition='attachment; filename="policy.csv"',
    )
    fetcher = _StubFetcher(resource)
    monkeypatch.setattr(main_module.settings, "knowledge_admin_token", SecretStr(admin_token))
    monkeypatch.setattr(main_module.settings, "knowledge_trust_access_headers", True)
    monkeypatch.setattr(main_module, "safe_fetcher", fetcher)
    headers = {
        "X-Tenant-Id": tenant_id,
        "X-Principal-Ids": "alice",
        "X-Knowledge-Admin-Token": admin_token,
    }
    payload = {
        "url": "https://public.example/policy.csv",
        "title": "Imported supplier policy",
        "knowledge_base_id": "policy",
    }

    with TestClient(main_module.app) as client:
        first = client.post("/api/knowledge/import-url", headers=headers, json=payload)
        second = client.post("/api/knowledge/import-url", headers=headers, json=payload)
        chat = client.post("/api/chat", headers=headers, json={"query": marker})
        unauthorized = client.post(
            "/api/chat",
            headers={"X-Tenant-Id": tenant_id, "X-Principal-Ids": "bob"},
            json={"query": marker},
        )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["document_id"] == second.json()["document_id"]
    assert first.json()["content_type"] == "text/csv"
    assert first.json()["filename"] == "policy.csv"
    assert len(first.json()["content_sha256"]) == 64
    assert fetcher.urls == [payload["url"], payload["url"]]
    sources = [
        source
        for source in chat.json()["run"]["sources"]
        if source["title"] == "Imported supplier policy"
    ]
    assert len(sources) == 1
    assert sources[0]["document_id"] == first.json()["document_id"]
    assert sources[0]["url"] == payload["url"]
    assert marker in sources[0]["content_snippet"]
    assert all(
        source["title"] != "Imported supplier policy"
        for source in unauthorized.json()["run"]["sources"]
    )


def test_url_import_requires_safe_http_backend_before_downloading(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module.settings, "knowledge_admin_token", SecretStr("import-admin-token")
    )
    monkeypatch.setattr(main_module, "safe_fetcher", None)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/knowledge/import-url",
            headers={"X-Knowledge-Admin-Token": "import-admin-token"},
            json={"url": "https://public.example/report.pdf"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Safe HTTP import is disabled"
