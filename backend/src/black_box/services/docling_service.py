"""Docling integration service (Block A Step 1/1b — real semantic chunking).

Thin abstraction over DS4SD Docling so the API layer never touches the
converter directly (Rules.md §2 SRP + §3 async). The heavy converter is
created lazily on first use and held on the app lifespan.

Usage (validated against docling 2.129.0, per official docs):
    converter = DocumentConverter(allowed_formats=[InputFormat.HTML, ...])
    stream    = DocumentStream(name="doc.html", stream=BytesIO(raw))
    result    = converter.convert(stream, max_file_size=..., max_num_pages=...)
    doc       = result.document                       # DoclingDocument
    chunker   = HybridChunker()
    chunks    = list(chunker.chunk(dl_doc=doc))       # list[DocChunk]
    enriched  = chunker.contextualize(chunk=chunk)    # heading-prepended text
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request, UploadFile, status

from black_box.core.config import get_settings
from black_box.schemas import ChunkOut, IngestResponse
from black_box.services.math_resolver import resolve_math_in_text
from black_box.services.table_validator import validate_tables

_UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_CONTENT

if TYPE_CHECKING:
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Internal structure-aware semantic chunk (serialized via ChunkOut)."""

    id: str
    text: str
    page: int | None = None
    heading: str | None = None
    tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class DoclingService:
    """Owns a lazily-initialized Docling DocumentConverter + HybridChunker."""

    def __init__(self) -> None:
        self._converter: DocumentConverter | None = None
        self._chunker: HybridChunker | None = None

    # -- lifecycle ----------------------------------------------------------

    async def _ensure_converter(self) -> DocumentConverter:
        """Boot the converter once with the configured format allowlist."""
        if self._converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.document_converter import DocumentConverter
            except ImportError as exc:  # pragma: no cover
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Docling is not installed — run `uv add docling[pdf,charts]`.",
                ) from exc

            settings = get_settings()
            try:
                allowed = [InputFormat[name] for name in settings.allowed_format_list]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Invalid ALLOWED_FORMATS: {settings.allowed_format_list}",
                ) from None

            # Sync construction in a threadpool; it may touch model warm-up.
            self._converter = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: DocumentConverter(allowed_formats=allowed),
            )
            logger.info(
                "Docling DocumentConverter initialized (formats=%s)",
                settings.allowed_format_list,
            )
        return self._converter

    def _ensure_chunker_sync(self) -> HybridChunker:
        """Return the shared HybridChunker (tokenizer-backed, loaded once).

        Not async: instantiation is cheap once the HF tokenizer is cached and
        happens inside the same executor used for conversion.
        """
        if self._chunker is None:
            from docling.chunking import HybridChunker

            # Default tokenizer = HuggingFaceTokenizer (all-MiniLM-L6-v2) —
            # aligned with the tokenizer Docling uses for its own chunk split.
            self._chunker = HybridChunker()
        return self._chunker

    async def aclose(self) -> None:
        self._converter = None
        self._chunker = None

    def accel(self) -> str | None:
        """Accelerator label for the /health probe (Rules.md §3)."""
        try:
            from docling.utils import accelerator

            return str(accelerator.get_accelerator())
        except Exception:  # noqa: BLE001 — probing only, degraded to None
            return None

    # -- ingestion ----------------------------------------------------------

    async def ingest(self, file: UploadFile) -> IngestResponse:
        """Parse an uploaded document into structure-aware semantic chunks."""
        converter = await self._ensure_converter()
        settings = get_settings()

        raw = await file.read()
        if len(raw) == 0:
            raise HTTPException(
                status_code=_UNPROCESSABLE,
                detail="Uploaded file is empty.",
            )
        if len(raw) > settings.max_upload_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds max upload size of {settings.max_upload_size / (1024 * 1024):.0f} MB.",
            )

        filename = file.filename or "document"
        doc_id = uuid.uuid4().hex

        # First-class Docling in-memory stream: no tempfile, no secrets on disk
        # (Rules.md §5). The allowed_formats allowlist + Docling's own format
        # detection from the stream name guard against disguised payloads.
        from docling.datamodel.base_models import DocumentStream

        stream = DocumentStream(name=filename, stream=BytesIO(raw))

        def _convert() -> tuple[Any, list[str]]:
            try:
                result = converter.convert(
                    stream,
                    max_file_size=settings.max_upload_size,
                    max_num_pages=settings.max_num_pages,
                )
            except Exception as exc:  # ConversionError & co. = unparseable input
                from docling.document_converter import ConversionError

                if isinstance(exc, ConversionError):
                    logger.info("Docling rejected upload %s: %s", filename, exc)
                    return None, [str(exc)]
                raise
            errors: list[str] = []
            if hasattr(result, "errors") and result.errors:
                errors = [str(e) for e in result.errors]
            return result, errors

        result, errors = await asyncio.get_running_loop().run_in_executor(
            None, _convert
        )
        doc = result.document if hasattr(result, "document") else None

        if errors:
            logger.warning("Docling conversion reported errors: %s", errors)
            raise HTTPException(
                status_code=_UNPROCESSABLE,
                detail=f"Document could not be parsed: {'; '.join(errors[:3])}",
            )
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Docling returned no document for the upload.",
            )

        # Real structure/table/equation-aware chunking (Block A Step 1b).
        chunker = self._ensure_chunker_sync()
        raw_chunks = await asyncio.get_running_loop().run_in_executor(
            None, lambda: list(chunker.chunk(dl_doc=doc))
        )

        chunks: list[Chunk] = []
        total_tokens = 0
        math_resolutions = []
        for idx, c in enumerate(raw_chunks):
            text = getattr(c, "text", "") or ""
            heading = self._headings_of(c)
            page = self._page_of(c)
            tokens = (
                chunker.tokenizer.count_tokens(text)
                if getattr(chunker, "tokenizer", None)
                else len(text.split())
            )
            total_tokens += tokens
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{idx}",
                    text=text,
                    page=page,
                    heading=heading,
                    tokens=tokens,
                    meta=self._meta_of(c),
                )
            )
            # 1c: resolve math symbols in this chunk (Block C consumes later)
            if resolution := resolve_math_in_text(text, context=heading or ""):
                math_resolutions.append(resolution)

        # 1d: validate every extracted table against the expected schema
        table_reports = await asyncio.get_running_loop().run_in_executor(
            None, validate_tables, doc
        )

        return IngestResponse(
            document_id=doc_id,
            format=file.content_type or "application/octet-stream",
            chunk_count=len(chunks),
            total_tokens=total_tokens,
            chunks=[ChunkOut(**c.__dict__) for c in chunks],
            tables=table_reports,
            math=math_resolutions,
        )

    # -- chunk introspection helpers -----------------------------------------

    @staticmethod
    def _headings_of(chunk: Any) -> str | None:
        meta = getattr(chunk, "meta", None)
        headings = getattr(meta, "headings", None) or []
        return " | ".join(headings) if headings else None

    @staticmethod
    def _page_of(chunk: Any) -> int | None:
        """Best-effort source page from doc-item provenance (PDFs only)."""
        meta = getattr(chunk, "meta", None)
        for item in getattr(meta, "doc_items", []) or []:
            prov = getattr(item, "prov", None) or []
            for p in prov:
                page = getattr(p, "page_no", None)
                if page is not None:
                    return int(page)
        return None

    @staticmethod
    def _meta_of(chunk: Any) -> dict[str, Any]:
        meta = getattr(chunk, "meta", None)
        labels: set[str] = set()
        for item in getattr(meta, "doc_items", []) or []:
            label = getattr(item, "label", None)
            if label is not None:
                labels.add(str(label))
        return {
            "headings": list(getattr(meta, "headings", None) or []),
            "captions": list(getattr(meta, "captions", None) or []),
            "doc_item_labels": sorted(labels),
        }


# -- FastAPI dependency wiring -------------------------------------------------


def get_docling_service(request: Request) -> DoclingService:
    """FastAPI dependency: serve the lifespan-managed DoclingService.

    Single ownership per worker (Rules.md §1 SRP): the lifespan creates the
    instance on `app.state.docling` and tears it down on shutdown; this
    dependency only *reads* it. A defensive fallback creates one if the
    lifespan never ran (e.g. unit tests with a bare TestClient).
    """
    service = getattr(request.app.state, "docling", None)
    if service is None:
        service = DoclingService()
        request.app.state.docling = service
    return service
