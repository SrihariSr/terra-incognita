"""Probe sweep: one ridge probe per layer, for each prompt style.

For every layer and prompt style we report the held-out great-circle error
under three ways of splitting the data into 5 folds:

* random:          places assigned to folds at random. Neighbouring places
                   (e.g. two London boroughs) can sit in train and test, so
                   this is the most flattering number.
* blocked_country: every place in a country is in the same fold, so the
                   probe is always tested on countries it never saw.
* blocked_cell:    the same, with 10 x 10 degree grid cells instead of
                   countries.

Every place gets an out-of-fold prediction (made by a probe that never saw
it), and we report the median error over all places. The layer and prompt
style are chosen by the blocked_country median (the honest number).

alpha (ridge strength) is chosen per layer by closed-form leave-one-out on
all places. This uses no test labels at prediction time beyond picking one
number per layer; the blocked splits would prefer *more* regularisation, so
if anything this makes the blocked numbers slightly worse, not better.

    uv run python scripts/fit_probes.py --set 3k --modes bare coords
    uv run python scripts/fit_probes.py --model gemma4-31b --set 3k --modes coords
Writes probe_layers.json in the model's results folder (results/ for Qwen)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import registry  # noqa: E402
from terra.activations import load  # noqa: E402
from terra.probe import (Standardiser, baseline_error, cross_val_predict, error_summary,  # noqa: E402
                         folds_grouped, folds_random, great_circle_km, grid_cells,
                         latlon_to_xyz, loo_alpha)


def run(key: str = registry.DEFAULT_KEY, set_name: str = "3k", modes=("bare", "coords"), out: str | None = None,
        layers=None) -> dict:
    """Probe error per layer for `key`; writes probe_layers.json (or `out`)."""
    P = registry.paths(key)
    out_path = Path(out) if out else P.result("probe_layers.json")
    places = pd.read_csv(P.acts / f"places_{set_name}.csv")
    Y = latlon_to_xyz(places["latitude"], places["longitude"])
    groups_country = places["iso"].fillna(places["country"]).astype(str).to_numpy()
    folds = {
        "random": folds_random(len(places), 5, seed=0),
        "blocked_country": folds_grouped(groups_country, 5, seed=0),
        "blocked_cell": folds_grouped(grid_cells(places["latitude"], places["longitude"], 10.0), 5, seed=0),
    }
    res = {"model": key, "set": set_name, "n_places": len(places), "folds": 5,
           "baseline_mean_direction": {k: error_summary(baseline_error(Y, f)) for k, f in folds.items()},
           "modes": {}}
    print("[baseline] predict-the-mean median km:",
          {k: round(v["median_km"]) for k, v in res["baseline_mean_direction"].items()})

    for mode in modes:
        meta = json.loads((P.acts / f"{set_name}_{mode}" / "meta.json").read_text())
        rows = []
        use = [L for L in meta["layers"] if layers is None or L in layers]
        t_start = time.time()
        for n_done, L in enumerate(use):
            t0 = time.time()
            pids, X = load(set_name, mode, L, root=P.acts)
            assert (pids == places["pid"].to_numpy()).all()
            st = Standardiser.fit(X.astype(np.float64))
            alpha, _ = loo_alpha(st(X.astype(np.float64)), Y - Y.mean(0))
            rec = {"layer": L, "alpha": alpha}
            for name, f in folds.items():
                pred = cross_val_predict(X, Y, f, alpha)
                rec[name] = error_summary(great_circle_km(pred, Y))
            rows.append(rec)
            eta = (time.time() - t_start) / (n_done + 1) * (len(use) - n_done - 1)
            print(f"[{mode} L{L:02d}] alpha={alpha:9.0f}  median km: random {rec['random']['median_km']:6.0f}  "
                  f"country {rec['blocked_country']['median_km']:6.0f}  cell {rec['blocked_cell']['median_km']:6.0f}"
                  f"  ({time.time() - t0:.1f}s, ETA {eta / 60:.1f} min)", flush=True)
        res["modes"][mode] = rows

    # Choose (mode, layer) by the country-blocked median error.
    best = min(((m, r) for m, rs in res["modes"].items() for r in rs),
               key=lambda mr: mr[1]["blocked_country"]["median_km"])
    res["chosen"] = {"mode": best[0], "layer": best[1]["layer"], "alpha": best[1]["alpha"],
                     "rule": "lowest blocked_country median great-circle error",
                     "random": best[1]["random"], "blocked_country": best[1]["blocked_country"],
                     "blocked_cell": best[1]["blocked_cell"]}
    for m, rs in res["modes"].items():
        b = min(rs, key=lambda r: r["blocked_country"]["median_km"])
        res[f"best_{m}"] = {"layer": b["layer"], "blocked_country_median_km": b["blocked_country"]["median_km"],
                            "random_median_km": b["random"]["median_km"]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"[chosen] {res['chosen']['mode']} layer {res['chosen']['layer']}: "
          f"random {res['chosen']['random']['median_km']:.0f} km, "
          f"country-blocked {res['chosen']['blocked_country']['median_km']:.0f} km -> {out_path}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=registry.DEFAULT_KEY, choices=sorted(registry.MODELS))
    ap.add_argument("--set", default="3k")
    ap.add_argument("--modes", nargs="+", default=["bare", "coords"])
    ap.add_argument("--out", default=None, help="default: <model results>/probe_layers.json")
    args = ap.parse_args()
    run(args.model, args.set, args.modes, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
