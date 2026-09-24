"""strategylib — native, self-extending strategy component library.

Index lives in ``manifest.json``; implementations are first-class importable
modules under ``components/``. Block B synthesizes novel components from
typed ASTs (``synthesize.py``) and registers them here as native code — a
paper-invented indicator becomes a reusable, testable, versioned component,
not a one-off blob inside one strategy.
"""

from black_box.strategylib.registry import (
    Component,
    append_manifest,
    by_pillar,
    load_registry,
    matching,
    resolve,
)
from black_box.strategylib.synthesize import interpret, synthesize_component

__all__ = [
    "Component",
    "append_manifest",
    "by_pillar",
    "interpret",
    "load_registry",
    "matching",
    "resolve",
    "synthesize_component",
]
