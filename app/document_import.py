from __future__ import annotations

import asyncio
import hashlib
import json

from pydantic import AnyHttpUrl, BaseModel, Field

from app.document_parser import DocumentParser
from app.knowledge import (
    IndexedKnowledgeDocument,
    KnowledgeDocumentInput,
    KnowledgeService,
    PrincipalId,
)
from app.tools.web import SafeHttpFetcher


class KnowledgeUrlImportRequest(BaseModel):
    url: AnyHttpUrl
    title: str | None = Field(default=None, min_length=1, max_length=500)
    knowledge_base_id: str = Field(default="default", min_length=1, max_length=128)
    allowed_principal_ids: list[PrincipalId] = Field(default_factory=list, max_length=100)
    public: bool = False
    metadata: dict[str, str] = Field(default_factory=dict, max_length=20)


class ImportedKnowledgeDocument(IndexedKnowledgeDocument):
    source_url: str
    content_type: str
    filename: str
    content_sha256: str


class KnowledgeImportService:
    def __init__(
        self,
        *,
        fetcher: SafeHttpFetcher,
        parser: DocumentParser,
        knowledge_service: KnowledgeService,
    ) -> None:
        self.fetcher = fetcher
        self.parser = parser
        self.knowledge_service = knowledge_service

    async def import_url(
        self, *, tenant_id: str, request: KnowledgeUrlImportRequest
    ) -> ImportedKnowledgeDocument:
        resource = await self.fetcher.download(str(request.url))
        parsed = await asyncio.to_thread(self.parser.parse, resource)
        content_sha256 = hashlib.sha256(resource.body).hexdigest()
        document = KnowledgeDocumentInput(
            title=request.title or parsed.title,
            content=parsed.content,
            knowledge_base_id=request.knowledge_base_id,
            allowed_principal_ids=request.allowed_principal_ids,
            public=request.public,
            source_url=resource.final_url,
            metadata=request.metadata,
        )
        document_id = self._stable_document_id(
            tenant_id=tenant_id,
            document=document,
            content_sha256=content_sha256,
        )
        indexed = await self.knowledge_service.index_document(
            tenant_id=tenant_id,
            document=document,
            document_id=document_id,
        )
        return ImportedKnowledgeDocument(
            **indexed.model_dump(),
            source_url=resource.final_url,
            content_type=parsed.content_type,
            filename=parsed.filename,
            content_sha256=content_sha256,
        )

    @staticmethod
    def _stable_document_id(
        *, tenant_id: str, document: KnowledgeDocumentInput, content_sha256: str
    ) -> str:
        identity = json.dumps(
            {
                "tenant_id": tenant_id,
                "content_sha256": content_sha256,
                "document": document.model_dump(mode="json"),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"doc_{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
