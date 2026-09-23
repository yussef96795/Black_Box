"""P3 — live prompt/response observability (JSONL trace writer).

Every LLM stage writes a sequence of frames to
``<traces_dir>/<paper_id>.jsonl``:

    request        pre-call  — ts, stage, model, prompt_chars, prompt_hash
    response       post-call — ts, stage, model, final_model (post-validation
                               JSON), attempts (raw completions observed)
    error          post-call — ts, stage, model, error_type, detail, trace
    raw_response / raw_error / usage — per-attempt frames emitted by the
                               instructor event hooks attached in
                               `LLMEvaluator` (raw completions, retry counts,
                               token usage)

Hermetic by default (D3): nothing is written unless
``BLOCK_A_TRACE_ENABLED`` is set, so tests never leak files nor depend on
network. Enabling into a ``tmp_path`` keeps trace assertions offline.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def json_safe(value: Any) -> Any:
    """Coerce an arbitrary hook payload into a JSON-serializable shape.

    Pydantic models dump to their validated JSON; everything else falls back
    to ``str()`` so a malformed completion never breaks the trace writer.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.loads(json.dumps(value, default=str))
        except (TypeError, ValueError):
            return str(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except (TypeError, ValueError):
            return str(value)
    return str(value)


class TraceWriter:
    """Appends JSONL events to a single per-paper trace file (D3).

    The parent directory is created on first write — a disabled trace leaves
    nothing behind on disk.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def write(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str, sort_keys=True) + "\n")


__all__ = ["TraceWriter", "_now", "json_safe"]
