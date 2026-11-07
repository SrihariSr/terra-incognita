"""Fit the final probe on the full place set and precompute the map.

Steps
1. Load activations for every place at the chosen layer and prompt style
   (read from results/probe_layers.json unless given).
2. Pick ridge alpha by closed-form leave-one-out.
3. Out-of-fold predictions under random, country-blocked and cell-blocked
   5-fold splits. The random out-of-fold prediction is what we draw on the
   globe as "where the model believes this place is": each dot comes from a
   probe that never saw that place, so its error line is honest.
4. Fit one probe on all places (used for live /locate queries), plus a
   bootstrap ensemble of 20 probes, each fitted on a resample (with
   replacement) of the places. Applying all 20 to a new name gives a cloud of
   20 guesses; its spread shows how sure the *probe* is.
5. Save models/probe_final.npz, results/probe_final.json (held-out errors)
   and results/places_believed.json (about 10,000 places for the globe).

    uv run python scripts/fit_final.py [--mode bare --layer 24]
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

from terra import ACTS, RESULTS, ROOT  # noqa: E402
from terra.activations import load  # noqa: E402
from terra.data import load_countries, subset_3k  # noqa: E402
from terra.probe import (Probe, Standardiser, baseline_error, cross_val_predict, error_summary,  # noqa: E402
                         folds_grouped, folds_random, great_circle_km, grid_cells,
                         latlon_to_xyz, loo_alpha, xyz_to_latlon)

N_BOOT = 20
N_DISPLAY = 10_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="full")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()
    # Prefer the layer sweep on the full place set if it has been run.
    sweep_file = RESULTS / "probe_layers_full.json"
    if not sweep_file.exists():
        sweep_file = RESULTS / "probe_layers.json"
    chosen = json.loads(sweep_file.read_text())["chosen"]
    print(f"[final] layer choice from {sweep_file.name}")
    mode = args.mode or chosen["mode"]
    layer = args.layer if args.layer is not None else chosen["layer"]
    t0 = time.time()

    places = pd.read_csv(ACTS / f"places_{args.set}.csv")
    pids, X = load(args.set, mode, layer)
    assert (pids == places["pid"].to_numpy()).all()
    Y = latlon_to_xyz(places["latitude"], places["longitude"])
    print(f"[final] {len(places)} places, mode={mode}, layer={layer}")

    st = Standardiser.fit(X.astype(np.float64))
    alpha, scores = loo_alpha(st(X.astype(np.float64)), Y - Y.mean(0))
    print(f"[final] LOO alpha = {alpha:.0f} ({time.time() - t0:.0f}s)")

    groups_country = places["iso"].fillna(places["country"]).astype(str).to_numpy()
    folds = {
        "random": folds_random(len(places), 5, seed=0),
        "blocked_country": folds_grouped(groups_country, 5, seed=0),
        "blocked_cell": folds_grouped(grid_cells(places["latitude"], places["longitude"], 10.0), 5, seed=0),
    }
    oof, summaries = {}, {}
    for name, f in folds.items():
        oof[name] = cross_val_predict(X, Y, f, alpha)
        km = great_circle_km(oof[name], Y)
        summaries[name] = error_summary(km)
        # Also report the error on well-known places only (the large GeoNames
        # cities), since those are the ones people will look at in the demo.
        big = (places["source"] == "geonames").to_numpy()
        summaries[name + "_big_cities"] = error_summary(km[big])
        print(f"[final] {name}: median {summaries[name]['median_km']:.0f} km, "
              f"big cities {summaries[name + '_big_cities']['median_km']:.0f} km ({time.time() - t0:.0f}s)")

    # Final probe on everything, plus the bootstrap ensemble.
    final = Probe.fit(X, Y, alpha)
    A, c = final.raw_weights()
    rng = np.random.default_rng(0)
    boot_A, boot_c = [], []
    for k in range(N_BOOT):
        # Multinomial counts = how many times each place is drawn in a
        # resample of size n with replacement; used as row weights.
        w = rng.multinomial(len(X), np.full(len(X), 1.0 / len(X))).astype(np.float64)
        p = Probe.fit(X, Y, alpha, w=w)
        a_k, c_k = p.raw_weights()
        boot_A.append(a_k)
        boot_c.append(c_k)
        print(f"[final] bootstrap {k + 1}/{N_BOOT} ({time.time() - t0:.0f}s)", flush=True)
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    np.savez(models / "probe_final.npz", A=A.astype(np.float32), c=c.astype(np.float32),
             boot_A=np.stack(boot_A).astype(np.float32), boot_c=np.stack(boot_c).astype(np.float32),
             alpha=alpha, layer=layer, mode=mode)

    # Spread of the bootstrap ensemble on the training places, as a scale
    # reference for clouds shown on invented names.
    Xf = X.astype(np.float64)
    boot_pred = np.stack([Xf @ a + cc for a, cc in zip(boot_A, boot_c)])  # [B, n, 3]
    centre = boot_pred.mean(0)
    spread = np.median(great_circle_km(boot_pred, np.broadcast_to(centre, boot_pred.shape)), axis=0)

    # Display set for the globe.
    disp = subset_3k(places.assign(pid=places["pid"]), n=N_DISPLAY, seed=0)
    idx = disp["pid"].to_numpy()
    lat_b, lon_b = xyz_to_latlon(oof["random"][idx])
    km_r = great_circle_km(oof["random"][idx], Y[idx])
    continents = load_countries()["Continent"]
    pts = []
    for j, i in enumerate(idx):
        r = places.iloc[i]
        pts.append({"pid": int(i), "name": r["name"], "type": r["entity_subtype"],
                    "country": str(r["country"]).replace("_", " "),
                    "lat": round(float(r["latitude"]), 4), "lon": round(float(r["longitude"]), 4),
                    "blat": round(float(lat_b[j]), 4), "blon": round(float(lon_b[j]), 4),
                    "err_km": round(float(km_r[j]), 1), "big": bool(r["source"] == "geonames"),
                    "cont": continents.get(r["iso"], "?") if isinstance(r["iso"], str) else "?"})
    (RESULTS / "places_believed.json").write_text(json.dumps(
        {"mode": mode, "layer": layer, "note": "believed = 5-fold out-of-fold prediction (random split)",
         "points": pts}, ensure_ascii=False))

    # Per-place out-of-fold predictions for every place (for figures).
    np.savez(RESULTS / "probe_oof.npz", pid=places["pid"].to_numpy(),
             **{k: v.astype(np.float32) for k, v in oof.items()}, true=Y.astype(np.float32))

    out = {"set": args.set, "n_places": len(places), "mode": mode, "layer": layer, "alpha": alpha,
           "loo_scores": scores, "heldout": summaries, "n_bootstrap": N_BOOT,
           "baseline_mean_direction": error_summary(baseline_error(Y, folds["random"])),
           "pred_norm_median": float(np.median(np.linalg.norm(oof["random"], axis=1))),
           "bootstrap_spread_median_km": float(np.median(spread)),
           "n_display": len(pts), "seconds": time.time() - t0}
    (RESULTS / "probe_final.json").write_text(json.dumps(out, indent=2))
    print(f"[final] done in {time.time() - t0:.0f}s: random {summaries['random']['median_km']:.0f} km, "
          f"country-blocked {summaries['blocked_country']['median_km']:.0f} km")
    return 0


if __name__ == "__main__":
    sys.exit(main())
