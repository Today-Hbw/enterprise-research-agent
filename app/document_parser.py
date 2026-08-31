from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from email.message import Message
from html.parser import HTMLParser
from io import BytesIO, StringIO
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from pypdf import PdfReader
from pypdf.errors import PyPdfError


@dataclass(frozen=True)
class DownloadedResource:
    final_url: str
    status_code: int
    content_type: str
    body: bytes
    content_disposition: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    content: str
    filename: str
    content_type: str


class _DocumentHtmlParser(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.lower()
        if normalized == "title":
            self._in_title = True
        if normalized in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized == "title":
            self._in_title = False
        if normalized in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self._parts).split())


class DocumentParser:
    SUPPORTED_CONTENT_TYPES = frozenset(
        {
            "text/html",
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/csv",
            "application/json",
            "application/ld+json",
            "application/pdf",
        }
    )

    def __init__(self, *, max_chars: int = 2_000_000, max_pdf_pages: int = 100) -> None:
        if max_chars < 1:
            raise ValueError("Document parser max_chars must be positive")
        if max_pdf_pages < 1:
            raise ValueError("Document parser max_pdf_pages must be positive")
        self.max_chars = max_chars
        self.max_pdf_pages = max_pdf_pages

    def parse(self, resource: DownloadedResource) -> ParsedDocument:
        if resource.content_type not in self.SUPPORTED_CONTENT_TYPES:
            raise ValueError(f"Unsupported document content type: {resource.content_type}")
        filename = self._filename(resource)
        fallback_title = filename or urlsplit(resource.final_url).hostname or "Document"

        if resource.content_type == "text/html":
            parser = _DocumentHtmlParser()
            parser.feed(self._decode_text(resource.body))
            title = " ".join(parser.title.split()) or fallback_title
            content = parser.text
        elif resource.content_type in {"application/json", "application/ld+json"}:
            title = fallback_title
            try:
                payload = json.loads(self._decode_text(resource.body))
            except json.JSONDecodeError as exc:
                raise ValueError("Downloaded JSON document is invalid") from exc
            content = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        elif resource.content_type in {"text/csv", "application/csv"}:
            title = fallback_title
            content = self._parse_csv(resource.body)
        elif resource.content_type == "application/pdf":
            title, content = self._parse_pdf(resource.body, fallback_title)
        else:
            title = fallback_title
            content = self._decode_text(resource.body)

        normalized = content.replace("\x00", "").strip()
        if not normalized:
            raise ValueError("Document parser extracted no text")
        if len(normalized) > self.max_chars:
            raise ValueError("Parsed document exceeds configured text limit")
        return ParsedDocument(
            title=title[:500],
            content=normalized,
            filename=filename or "document",
            content_type=resource.content_type,
        )

    @staticmethod
    def _decode_text(body: bytes) -> str:
        try:
            return body.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Downloaded text document must be UTF-8 encoded") from exc

    def _parse_csv(self, body: bytes) -> str:
        text = self._decode_text(body)
        try:
            rows = csv.reader(StringIO(text, newline=""))
            return "\n".join("\t".join(cell.strip() for cell in row) for row in rows)
        except csv.Error as exc:
            raise ValueError("Downloaded CSV document is invalid") from exc

    def _parse_pdf(self, body: bytes, fallback_title: str) -> tuple[str, str]:
        try:
            reader = PdfReader(BytesIO(body), strict=False)
            if reader.is_encrypted:
                raise ValueError("Encrypted PDF documents are not supported")
            if len(reader.pages) > self.max_pdf_pages:
                raise ValueError("PDF exceeds configured page limit")
            pages = []
            extracted_chars = 0
            for index, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    extracted_chars += len(text)
                    if extracted_chars > self.max_chars:
                        raise ValueError("Parsed document exceeds configured text limit")
                    pages.append(f"[Page {index}]\n{text}")
            metadata_title = (
                str(reader.metadata.title).strip()
                if reader.metadata and reader.metadata.title
                else ""
            )
            return metadata_title or fallback_title, "\n\n".join(pages)
        except (PyPdfError, EOFError) as exc:
            raise ValueError("Downloaded PDF document is invalid") from exc

    @staticmethod
    def _filename(resource: DownloadedResource) -> str:
        candidate = ""
        if resource.content_disposition:
            message = Message()
            message["content-disposition"] = resource.content_disposition
            candidate = message.get_filename() or ""
        if not candidate:
            candidate = unquote(PurePosixPath(urlsplit(resource.final_url).path).name)
        candidate = PurePosixPath(candidate.replace("\\", "/")).name
        return "".join(character for character in candidate if character.isprintable()).strip()[
            :255
        ]
