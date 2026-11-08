"""Shared fixtures: a fake model, fake artefacts in a temporary folder, a test client.

Nothing here loads a real model or touches the real results/ and models/
folders: TERRA_ARTEFACT_ROOT points every per-model path at a temporary
folder, and TERRA_FAKE_LM=1 makes the registry hand out FakeLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRA_FAKE_LM", "1")
    monkeypatch.setenv("TERRA_ARTEFACT_ROOT", str(tmp_path))
    monkeypatch.delenv("TERRA_MODEL", raising=False)
    monkeypatch.delenv("TERRA_PATCH", raising=False)
    import terra.model as tm
    from terra.backends.fake import FakeLM, make_fake_artefacts
    tm.unload_lm()
    make_fake_artefacts(keys_built=["qwen2.5-7b", "gemma4-12b"], keys_editable=["qwen2.5-7b", "gemma4-12b"],
                        political_keys=["qwen2.5-7b"])
    FakeLM.reset_counters()
    yield tmp_path
    tm.unload_lm()
    FakeLM.reset_counters()


@pytest.fixture
def fake_lm(fake_root):
    from terra.backends.fake import FakeLM
    lm = FakeLM("qwen2.5-7b")
    yield lm
    lm.unload()


@pytest.fixture
def client(fake_root, monkeypatch):
    from fastapi.testclient import TestClient

    import terra.server as srv
    calls = []

    def fake_unload(*a, **k):
        calls.append(1)
        return ["gemma4:31b"]

    # Never talk to a real Ollama from the tests.
    monkeypatch.setattr(srv.ollama, "unload_ollama_models", fake_unload)
    with TestClient(srv.app) as c:
        c.ollama_calls = calls
        yield c
