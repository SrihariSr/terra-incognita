"""Batch evaluation of the tier-1 edit on fresh random city pairs.

This is the number for the rigour slide. Unlike the gate, the layer band is
fixed in advance (read from results/gate.json) and the pairs are drawn at
random, so there is no selection on the outcome.

Two pair sets, each drawn with a fixed seed from the cities the unedited
model answers correctly on all five ripple questions
(results/baseline_cities.json):

* cross: source and target in opposite hemispheres and different
  countries. All five questions can usually be scored.
* any:   any two cities in different countries. Hemisphere and season are
  often the same for both, so only the questions whose answer keys differ
  are scored ("countable").

For every pair we run the swap edit and the equal-norm random control, and
score each answer as flipped (matches the target, not the source), kept
(matches the source, not the target) or other.

A pair *passes* if at least 80% of its countable questions flip (i.e. 4 of 5
when all five count) and it has at least 3 countable questions.

    uv run python scripts/batch_edit_eval.py [--n 100] [--method last|all] [--seed 0] [--out ...]
Writes results/edit_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import RESULTS  # noqa: E402
from terra.model import get_lm  # noqa: E402
from terra.patch import band, generate_plans, plan_edit, plan_edit_all  # noqa: E402
from terra.ripple import QKEYS, make_key, prompts_for, score_edit  # noqa: E402

PASS_FRACTION = 0.8
MIN_COUNTABLE = 3


def draw_pairs(cities, n, cross, rng):
    pairs, seen = [], set()
    tries = 0
    while len(pairs) < n and tries < 100_000:
        tries += 1
        a, b = rng.choice(len(cities), 2, replace=False)
        s, t = cities[a], cities[b]
        if s["iso"] == t["iso"] or (s["name"], t["name"]) in seen:
            continue
        if cross and (s["latitude"] > 0) == (t["latitude"] > 0):
            continue
        seen.add((s["name"], t["name"]))
        pairs.append((s, t))
    return pairs


def summarise(rows, kind):
    sel = [r[kind] for r in rows if r["n_countable"] >= MIN_COUNTABLE]
    frac = [r["n_flipped"] / r["n_countable"] for r in sel]
    kept = [r["n_source"] / r["n_countable"] for r in sel]
    per_q = {}
    for q in QKEYS:
        vals = [r["per_question"][q]["flipped"] for r in sel if r["per_question"][q]["countable"]]
        per_q[q] = {"flip_rate": float(np.mean(vals)) if vals else None, "n": len(vals)}
    return {"n_pairs": len(sel), "pass_rate": float(np.mean([f >= PASS_FRACTION for f in frac])),
            "mean_fraction_flipped": float(np.mean(frac)), "mean_fraction_kept_source": float(np.mean(kept)),
            "per_question": per_q}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band", type=int, nargs=2, default=None)
    ap.add_argument("--out", default=str(RESULTS / "edit_eval.json"))
    ap.add_argument("--method", choices=["last", "all"], default="last",
                    help="patch only the last name token (tier 1 as gated) or every name token")
    args = ap.parse_args()
    t0 = time.time()
    gate = json.loads((RESULTS / "gate.json").read_text())
    layers = band(*(args.band or gate["best_band"]))
    base = json.loads((RESULTS / "baseline_cities.json").read_text())
    cities = [c for c in base["cities"] if c["all_correct"]]
    print(f"[eval] {len(cities)} baseline-correct cities, band {layers[0]}-{layers[-1]}")
    lm = get_lm()
    rng = np.random.default_rng(args.seed)
    out = {"model": lm.model_id, "band": [layers[0], layers[-1]], "seed": args.seed, "method": args.method,
           "pass_rule": f">= {PASS_FRACTION:.0%} of countable questions flipped, >= {MIN_COUNTABLE} countable",
           "n_pool_cities": len(cities), "sets": {}}
    for set_name, cross in (("cross", True), ("any", False)):
        pairs = draw_pairs(cities, args.n, cross, rng)
        rows = []
        for b0 in range(0, len(pairs), 4):
            chunk = pairs[b0:b0 + 4]
            plans = []
            for s, t in chunk:
                qs = prompts_for(s["name"])
                texts = [qs[q] for q in QKEYS]
                planner = plan_edit_all if args.method == "all" else plan_edit
                plans.append(planner(lm, texts, s["name"], t["name"], layers, mode="swap"))
                plans.append(planner(lm, texts, s["name"], t["name"], layers, mode="control"))
            outs = generate_plans(lm, plans, max_new_tokens=10)
            for i, (s, t) in enumerate(chunk):
                sk = make_key(s["name"], s["iso"], s["latitude"])
                tk = make_key(t["name"], t["iso"], t["latitude"])
                ed = score_edit(dict(zip(QKEYS, outs[2 * i])), sk, tk)
                ct = score_edit(dict(zip(QKEYS, outs[2 * i + 1])), sk, tk)
                rows.append({"source": s["name"], "target": t["name"], "n_countable": ed["n_countable"],
                             "edit": ed, "control": ct})
            print(f"[eval {set_name}] {len(rows)}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
        res = {"edit": summarise(rows, "edit"), "control": summarise(rows, "control"), "pairs": rows}
        out["sets"][set_name] = res
        print(f"[eval {set_name}] edit pass {res['edit']['pass_rate']:.2f} "
              f"(mean flipped {res['edit']['mean_fraction_flipped']:.2f}); control pass "
              f"{res['control']['pass_rate']:.2f} (mean flipped {res['control']['mean_fraction_flipped']:.2f})")
    out["seconds"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
