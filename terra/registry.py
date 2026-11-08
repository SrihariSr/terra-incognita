"""The models TERRA can use, and where each model's files live.

Every piece of code that reads or writes a per-model file (probe weights,
results, activations, figures, the political map) asks `paths(key)`, so the
layout is defined once, here.

Layout
------
Qwen keeps the files it has always had, so nothing on disk moves:
    models/probe_final.npz, results/*.json, activations/...
Every other model gets its own folders:
    models/<key>/, results/models/<key>/, activations/<key>/

A model is *available* when its probe file exists (the map and "New worlds"
work), and *editable* when its gate.json exists (Edit and Rewrite work).

TERRA_ARTEFACT_ROOT moves the whole layout to another folder (the tests use a
temporary one). TERRA_FAKE_LM=1 swaps every model for the fake backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import ROOT


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    backend: str        # "hf" (PyTorch) or "mlx"
    repo: str           # Hugging Face repo id
    approx_gb: float    # memory it needs once loaded


# Memory needs: the weights' total size from each repo's
# model.safetensors.index.json (Qwen bf16 15.2 GB; Gemma 12B 11.0 GB; Gemma 31B
# 28.8 GB, larger than "4-bit" suggests because its MLP weights are stored at
# 8 bits), plus about half a gigabyte of working memory.
MODELS: dict[str, ModelSpec] = {
    "qwen2.5-7b": ModelSpec("qwen2.5-7b", "Qwen2.5-7B-Instruct", "hf", "Qwen/Qwen2.5-7B-Instruct", 15.5),
    "gemma4-12b": ModelSpec("gemma4-12b", "Gemma 4 12B (4-bit)", "mlx", "mlx-community/gemma-4-12B-it-qat-4bit", 11.5),
    "gemma4-31b": ModelSpec("gemma4-31b", "Gemma 4 31B (4-bit)", "mlx", "mlx-community/gemma-4-31B-it-qat-4bit", 29.5),
}
DEFAULT_KEY = "qwen2.5-7b"


def spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; choose one of {sorted(MODELS)}")
    return MODELS[key]


def artefact_root() -> Path:
    return Path(os.environ.get("TERRA_ARTEFACT_ROOT", str(ROOT)))


def fake_mode() -> bool:
    return os.environ.get("TERRA_FAKE_LM") == "1"


def startup_key() -> str:
    return os.environ.get("TERRA_MODEL", DEFAULT_KEY)


@dataclass(frozen=True)
class ModelPaths:
    key: str
    models: Path      # probe weights and steering probes
    results: Path     # every results file and figure for this model
    acts: Path        # activation arrays

    @property
    def probe(self) -> Path:
        return self.models / "probe_final.npz"

    @property
    def steer_probes(self) -> Path:
        return self.models / "steer_probes.npz"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    def result(self, name: str) -> Path:
        return self.results / name


def paths(key: str) -> ModelPaths:
    """The one place that decides where a model's files live."""
    spec(key)
    root = artefact_root()
    if key == DEFAULT_KEY:
        return ModelPaths(key, root / "models", root / "results", root / "activations")
    return ModelPaths(key, root / "models" / key, root / "results" / "models" / key, root / "activations" / key)


def available(key: str) -> bool:
    return paths(key).probe.exists()


def editable(key: str) -> bool:
    return paths(key).result("gate.json").exists()


def build_command(key: str) -> str:
    return f"uv run python scripts/build_model.py --model {key} --stage all"


def load_backend(key: str):
    """Construct (load) the backend for a model. Callers must unload any other first."""
    s = spec(key)
    if fake_mode():
        from .backends.fake import FakeLM
        return FakeLM(key=key, display=s.display)
    if s.backend == "hf":
        from .backends.hf import HFLM
        return HFLM(s.repo, key=key, display=s.display)
    if s.backend == "mlx":
        from .backends.mlx_backend import MLXLM
        return MLXLM(s.repo, key=key, display=s.display)
    raise ValueError(f"unknown backend {s.backend!r}")
