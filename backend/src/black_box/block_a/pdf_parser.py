"""Module 1 — ingestion front end for the Block A DAG.

Real Docling conversion (same converter/chunker as the FastAPI
`DoclingService`) but synchronous — the Block A pipeline is orchestrated
outside the request lifecycle (CLI / worker), so a sync adapter keeps the DAG
nodes plain functions (Rules.md §1 KISS). No duplication of logic: this is a
thin file-based entry into the same Docling API.
"""

from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path
from typing import Any

from black_box.core.config import get_settings

logger = logging.getLogger(__name__)


class UnsupportedPaperError(Exception):
    """Raised when the paper cannot be parsed into structured chunks."""


def parse_paper(
    source: str | Path,
    *,
    max_num_pages: int | None = None,
    max_file_size: int | None = None,
) -> list[dict[str, Any]]:
    """Parse a research paper into structure-aware chunk dicts (Stage Module 1).

    Returns a list of msgpack-safe chunk dicts:
        {"id", "text", "heading", "page", "tokens", "meta"}
    matching `black_box.schemas.ChunkOut` — the downstream stages consume
    text + heading only.
    """
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.document_converter import ConversionError, DocumentConverter

    settings = get_settings()
    max_num_pages = max_num_pages or settings.max_num_pages
    max_file_size = max_file_size or settings.max_upload_size

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"paper not found: {path}")

    raw = path.read_bytes()
    if len(raw) == 0:
        raise UnsupportedPaperError(f"paper is empty: {path.name}")

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.MD, InputFormat.HTML]
    )
    stream = DocumentStream(name=path.name, stream=io.BytesIO(raw))

    try:
        result = converter.convert(
            stream,
            max_file_size=max_file_size,
            max_num_pages=max_num_pages,
        )
    except ConversionError as exc:
        raise UnsupportedPaperError(
            f"conversion failed for {path.name}: {exc}"
        ) from exc

    doc = getattr(result, "document", None)
    if doc is None:
        raise UnsupportedPaperError(f"no document produced for {path.name}")

    chunker = HybridChunker()
    chunks: list[dict[str, Any]] = []
    doc_id = uuid.uuid4().hex
    for idx, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        text = getattr(chunk, "text", "") or ""
        if not text.strip():
            continue
        headings = list(getattr(getattr(chunk, "meta", None), "headings", None) or [])
        chunks.append(
            {
                "id": f"{doc_id}:{idx}",
                "text": text,
                "heading": " | ".join(headings) if headings else None,
                "page": _page_of(chunk),
                "tokens": _count_tokens(chunker, text),
                "meta": {
                    "headings": headings,
                    "doc_item_labels": _labels_of(chunk),
                },
            }
        )
    logger.info("parsed %s -> %d chunks", path.name, len(chunks))
    return chunks


def _page_of(chunk: Any) -> int | None:
    for item in getattr(getattr(chunk, "meta", None), "doc_items", []) or []:
        for prov in getattr(item, "prov", None) or []:
            page = getattr(prov, "page_no", None)
            if page is not None:
                return int(page)
    return None


def _labels_of(chunk: Any) -> list[str]:
    labels: set[str] = set()
    for item in getattr(getattr(chunk, "meta", None), "doc_items", []) or []:
        label = getattr(item, "label", None)
        if label is not None:
            labels.add(str(label))
    return sorted(labels)


def _count_tokens(chunker: Any, text: str) -> int:
    tokenizer = getattr(chunker, "tokenizer", None)
    if tokenizer is not None:
        try:
            return int(tokenizer.count_tokens(text))
        except Exception:  # noqa: BLE001 — tokenizer quirks degrade to words
            return len(text.split())
    return len(text.split())


__all__ = ["UnsupportedPaperError", "parse_paper"]
