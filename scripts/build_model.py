"""Build every artefact the demo needs for one model.

    uv run python scripts/build_model.py --model gemma4-31b --stage all
    uv run python scripts/build_model.py --model gemma4-31b --stage extract3k
    uv run python scripts/build_model.py --model qwen2.5-7b --stage political

Stages, in order (each is skipped if its output already exists; --force redoes it):
  extract3k  activations for the 3k-place subset at every layer ("coords" prompt)
  sweep      probe error per layer on the 3k subset           -> probe_layers.json
  full       all 40,485 places at the best layer +-3, then a probe sweep on them
                                                               -> probe_layers_full.json
  final      final probe, bootstrap ensemble, out-of-fold believed positions
                                                               -> probe_final.npz, places_believed.json, probe_oof.npz
  fictional  the 50 invented names                             -> fictional.json
  political  the model's political map (CPU only)             -> political_map.geojson, political_stats.json
  gate       layer-band sweep and the ten-pair gate            -> gate.json
  menu       baseline questions, then the validated pair menu  -> baseline_cities.json, pair_menu_alltoken.json

The model is loaded only for stages that need it (extract3k, full, fictional,
gate, menu), once, and refused if another process (e.g. the server) already
holds a model. Files go to registry.paths(model); Qwen's stay where they
always were.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from terra import registry  # noqa: E402
from terra.model import get_lm, unload_lm  # noqa: E402

STAGES = ["extract3k", "sweep", "full", "final", "fictional", "political", "gate", "menu"]
NEEDS_MODEL = {"extract3k", "full", "fictional", "gate", "menu"}
FULL_HALF_WIDTH = 3


def script(name: str):
    """Import scripts/<name>.py as a module (the stage code lives there)."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def meta_n(P, tag: str, mode: str = "coords"):
    try:
        return json.loads((P.acts / f"{tag}_{mode}" / "meta.json").read_text())
    except (OSError, ValueError):
        return None


def done(key: str, stage: str) -> bool:
    P = registry.paths(key)
    if stage == "extract3k":
        m = meta_n(P, "3k")
        return bool(m) and m.get("n") == 3000
    if stage == "sweep":
        return P.result("probe_layers.json").exists()
    if stage == "full":
        return P.result("probe_layers_full.json").exists()
    if stage == "final":
        return all(p.exists() for p in (P.probe, P.result("places_believed.json"), P.result("probe_oof.npz")))
    if stage == "fictional":
        return P.result("fictional.json").exists()
    if stage == "political":
        return P.result("political_map.geojson").exists() and P.result("political_stats.json").exists()
    if stage == "gate":
        return P.result("gate.json").exists()
    if stage == "menu":
        return P.result("pair_menu_alltoken.json").exists()
    raise ValueError(stage)


def full_layers(key: str) -> list[int]:
    """Best layer of the 3k sweep and its neighbours (+-3), within the model's depth."""
    P = registry.paths(key)
    best = json.loads(P.result("probe_layers.json").read_text())["chosen"]["layer"]
    n_layers = len(meta_n(P, "3k")["layers"])
    return list(range(max(0, best - FULL_HALF_WIDTH), min(n_layers - 1, best + FULL_HALF_WIDTH) + 1))


def run_stage(key: str, stage: str, cell_deg: float, lm_box: dict, batch_size: int = 64) -> None:
    def lm():
        if "lm" not in lm_box:
            lm_box["lm"] = get_lm(key)
        return lm_box["lm"]

    P = registry.paths(key)
    if stage == "extract3k":
        script("extract_activations").run(key, "3k", ["coords"], None, batch_size=batch_size, lm=lm())
    elif stage == "sweep":
        script("fit_probes").run(key, "3k", ["coords"])
    elif stage == "full":
        layers = full_layers(key)
        print(f"[build] full set at layers {layers[0]}-{layers[-1]}")
        script("extract_activations").run(key, "full", ["coords"], layers, batch_size=batch_size, lm=lm())
        script("fit_probes").run(key, "full", ["coords"], str(P.result("probe_layers_full.json")))
    elif stage == "final":
        script("fit_final").run(key)
    elif stage == "fictional":
        script("fictional").run(key, lm=lm())
    elif stage == "political":
        from terra.politics import build_for_model
        s = build_for_model(key, cell_deg=cell_deg)
        print(f"[political] radius {s['settings']['radius_km']} km, land {s['model_land_fraction']:.0%} "
              f"(real {s['real_land_fraction']:.0%}), {s['n_countries_drawn']} countries drawn, "
              f"{s['share_of_model_land_in_right_country']:.0%} of the model's land in the right country")
    elif stage == "gate":
        g = script("gate").run(key, lm=lm())
        if g.get("decision") != "GO":
            print(f"[build] warning: gate decision {g.get('decision')} for {key}; edits may not work well")
    elif stage == "menu":
        # The baseline answers take a few minutes; reuse them when resuming.
        if not P.result("baseline_cities.json").exists():
            script("baseline_cities").run(key, lm=lm())
        else:
            print("[build] baseline_cities.json exists, reusing it (delete it to re-ask)")
        script("build_pair_menu").run(key, method="all", lm=lm())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(registry.MODELS))
    ap.add_argument("--stage", required=True, choices=STAGES + ["all"])
    ap.add_argument("--force", action="store_true", help="redo the stage even if its output exists")
    ap.add_argument("--cell-deg", type=float, default=1.0, help="political map grid (0.5 for a finer build)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="prompts per forward pass when reading activations (lower it if memory is short; "
                         "keep it the same when resuming, as a change restarts the extraction)")
    args = ap.parse_args()
    stages = STAGES if args.stage == "all" else [args.stage]
    lm_box: dict = {}
    t_all = time.time()
    try:
        for i, st in enumerate(stages):
            if done(args.model, st) and not args.force:
                print(f"[build {i + 1}/{len(stages)}] {st}: already built, skipping")
                continue
            t0 = time.time()
            print(f"[build {i + 1}/{len(stages)}] {st}: starting", flush=True)
            run_stage(args.model, st, args.cell_deg, lm_box, args.batch_size)
            print(f"[build {i + 1}/{len(stages)}] {st}: done in {(time.time() - t0) / 60:.1f} min "
                  f"(total {(time.time() - t_all) / 60:.1f} min)", flush=True)
    finally:
        if "lm" in lm_box:
            unload_lm()
    print(f"[build] {args.model}: available={registry.available(args.model)} "
          f"editable={registry.editable(args.model)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
