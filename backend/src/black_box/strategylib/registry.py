"""Component registry — the index over the executable strategylib codebase.

Resolves a component id (the vocabulary Block A extracted) to the native
importable module implementing it. The manifest.json is the *index* only;
the implementations are first-class Python modules under `components/`.
Plain JSON metadata could never hold executable, testable code — this is
the "JSON won't do the trick" boundary (Block B design, ponytail).

Lookups are pure (``load_registry`` is a cache keyed by path) so tests may
point at alternate manifests.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover — runtime dep, present in the venv
    np = None  # type: ignore[assignment]

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


@dataclass(frozen=True)
class Component:
    """A resolvable executable strategy component."""

    id: str
    name: str
    pillar: str
    implements: tuple[str, ...]
    params: dict[str, dict[str, float]]
    requires: dict[str, object]
    impl: str
    trust: str
    version: str

    def compute(self, data: dict[str, object], params: dict[str, float] | None = None):
        """Call the native implementation on a data dict + params."""
        module_name, _, func_name = self.impl.partition(":")
        fn = getattr(importlib.import_module(module_name), func_name)
        return fn(data, params or {})


@lru_cache(maxsize=8)
def load_registry(path: Path | None = None) -> dict[str, Component]:
    """Load the manifest into a {component_id: Component} index."""
    manifest_path = Path(path) if path else MANIFEST_PATH
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    index: dict[str, Component] = {}
    for entry in data.get("components", []):
        index[entry["id"]] = Component(
            id=entry["id"],
            name=entry["name"],
            pillar=entry["pillar"],
            implements=tuple(entry.get("implements", [])),
            params=entry.get("params", {}),
            requires=entry.get("requires", {}),
            impl=entry["impl"],
            trust=entry.get("trust", "paper_derived"),
            version=entry.get("version", "0.0.0"),
        )
    return index


def resolve(component_id: str, path: Path | None = None) -> Component | None:
    """Resolve a component id; None on registry miss (novel component path)."""
    return load_registry(path).get(component_id)


def by_pillar(pillar: str, path: Path | None = None) -> list[Component]:
    """All components for a pillar (entry/exit/sizing), id-sorted."""
    return sorted(
        (c for c in load_registry(path).values() if c.pillar == pillar),
        key=lambda c: c.id,
    )


def matching(primitive: str, path: Path | None = None) -> list[Component]:
    """Components that implement a primitive/trigger id (empty on miss)."""
    return [c for c in load_registry(path).values() if primitive in c.implements]


def append_manifest(entry: dict, path: Path | None = None) -> None:
    """Register a synthesized component in the index (idempotent)."""
    manifest_path = Path(path) if path else MANIFEST_PATH
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.setdefault("components", [])
    if any(c["id"] == entry["id"] for c in data["components"]):
        return
    data["components"].append(entry)
    manifest_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    load_registry.cache_clear()


__all__ = [
    "MANIFEST_PATH",
    "Component",
    "append_manifest",
    "by_pillar",
    "load_registry",
    "matching",
    "resolve",
]
