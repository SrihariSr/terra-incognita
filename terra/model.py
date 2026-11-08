"""Load one model at a time and hand it out.

The model code itself lives in terra/backends/ (hf.py for Qwen, mlx_backend.py
for Gemma 4, fake.py for tests); this module keeps the names older code
imports (`LM`, `Patch`, `BOS`, `MODEL_ID`, `get_lm`) and owns the single
"loaded model" slot.

Only one model may be in memory at a time (see terra/memory.py). get_lm(key)
therefore unloads whatever is loaded before loading `key`, and refuses if
another process already holds a model.
"""

from __future__ import annotations

import gc

from . import memory, registry
from .backends.base import BaseLM, Patch  # noqa: F401
from .backends.hf import BOS, MODEL_ID, HFLM  # noqa: F401

LM = BaseLM          # the interface type, for annotations

_LM: BaseLM | None = None
_KEY: str | None = None


def current_key() -> str | None:
    return _KEY if _LM is not None else None


def get_lm(key: str | None = None) -> BaseLM:
    """The loaded model for `key` (default TERRA_MODEL, else Qwen), loading it if needed."""
    global _LM, _KEY
    key = key or registry.startup_key()
    if _LM is not None and _KEY == key:
        return _LM
    if _LM is not None:
        unload_lm()
    memory.check_before_load(key)
    try:
        _LM = registry.load_backend(key)
    except Exception:
        memory.release()
        raise
    _KEY = key
    return _LM


def unload_lm() -> None:
    """Drop the loaded model and free its memory before anything else is loaded."""
    global _LM, _KEY
    if _LM is not None:
        _LM.unload()
    _LM = None
    _KEY = None
    gc.collect()
    memory.release()
