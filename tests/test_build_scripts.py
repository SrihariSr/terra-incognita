"""The build stages that ask the model questions, run end to end on the fake model.

These caught nothing until a real build crashed: baseline_cities.run used
the name `key` both for the model key and for each city's answer key. The
fake model's answers are never correct, so the baseline test only checks the
file is written for the right model; the menu test then writes its own
baseline in which three cities pass, so the pair loop runs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_baseline_writes_to_the_models_own_folder(fake_root):
    from terra import registry
    from terra.backends.fake import FakeLM
    lm = FakeLM("gemma4-12b")
    out = script("baseline_cities").run("gemma4-12b", lm=lm)
    path = registry.paths("gemma4-12b").result("baseline_cities.json")
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["model_key"] == "gemma4-12b" and saved["n_cities"] == out["n_cities"] > 0
    # Qwen's own baseline file is untouched (it lives elsewhere).
    assert not registry.paths("qwen2.5-7b").result("baseline_cities.json").exists()


def test_pair_menu_runs_for_a_non_qwen_model(fake_root):
    from terra import registry
    from terra.backends.fake import FakeLM
    P = registry.paths("gemma4-12b")
    ok = {q: True for q in ("country", "currency", "language", "hemisphere", "season")}
    cities = [{"name": n, "all_correct": True, "correct": ok, "answers": {q: "x" for q in ok}}
              for n in ("Paris", "Sydney", "Tokyo")]
    P.result("baseline_cities.json").write_text(json.dumps({"cities": cities}))
    out = script("build_pair_menu").run("gemma4-12b", method="all", lm=FakeLM("gemma4-12b"))
    assert out["n_pairs"] == 6                       # every ordered pair of three cities in different countries
    assert P.result("pair_menu_alltoken.json").exists()
    assert out["model"] == "gemma4-12b" and out["method"] == "all"
