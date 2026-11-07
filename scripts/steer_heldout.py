"""Held-out check of the tier-2 setting chosen in results/steer_eval.json.

The steering setting (layer band and scale) was picked on the 10 gate pairs,
so its 9/10 is optimistic. Here the setting is fixed and we draw fresh
random cross-hemisphere pairs (seeded) from the baseline-correct cities,
excluding the gate pairs. For each pair we run three edits on the same
questions: tier-2 steering, the tier-1 swap, and a random-direction control
with the same per-layer size as the steering change.

Measured per pair:
* moved:  the hemisphere answer or the July-season answer flips to the
          target's (location changed);
* both moved: both flip;
* identity kept: the country answer still names the source's country AND
          the landmark answer shares a content word with the unedited one.
A pair counts as "moved, identity kept" if moved and identity kept.

    uv run python scripts/steer_heldout.py [--n 40]
Writes results/steer_heldout.json
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
from terra.companies import identity_kept  # noqa: E402
from terra.model import get_lm  # noqa: E402
from terra.patch import band, generate_plans, plan_edit, stable_seed  # noqa: E402
from terra.ripple import QKEYS, make_key, matches, prompts_for, score_edit  # noqa: E402
from terra.steer import plan_steer, random_like  # noqa: E402

LANDMARK = "What is the most famous landmark in {city}? Answer with just its name."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    st = json.loads((RESULTS / "steer_eval.json").read_text())
    cfg = st["best"]
    gate = json.loads((RESULTS / "gate.json").read_text())
    swap_band = band(*gate["best_band"])
    gate_pairs = {tuple(p) for p in gate["pairs"]}
    cities = [c for c in json.loads((RESULTS / "baseline_cities.json").read_text())["cities"] if c["all_correct"]]
    rng = np.random.default_rng(args.seed)
    pairs, seen = [], set()
    while len(pairs) < args.n:
        a, b = rng.choice(len(cities), 2, replace=False)
        s, t = cities[a], cities[b]
        if s["iso"] == t["iso"] or (s["latitude"] > 0) == (t["latitude"] > 0):
            continue
        if (s["name"], t["name"]) in gate_pairs or (s["name"], t["name"]) in seen:
            continue
        seen.add((s["name"], t["name"]))
        pairs.append((s, t))
    lm = get_lm()
    rows = []
    for s, t in pairs:
        qs = prompts_for(s["name"])
        texts = [qs[q] for q in QKEYS] + [LANDMARK.format(city=s["name"])]
        steer = plan_steer(lm, None, texts, s["name"], t["name"], layers=band(*cfg["band"]), scale=cfg["scale"])
        ctrl = random_like(lm, steer, stable_seed(s["name"], t["name"], "steer-heldout"))
        swap = plan_edit(lm, texts, s["name"], t["name"], swap_band, mode="swap")
        o_plain = generate_plans(lm, [[lm.chat(x) for x in texts]], max_new_tokens=12)[0]
        o_steer, o_ctrl = generate_plans(lm, [steer, ctrl], max_new_tokens=12)
        o_swap = generate_plans(lm, [swap], max_new_tokens=12)[0]
        sk = make_key(s["name"], s["iso"], s["latitude"])
        tk = make_key(t["name"], t["iso"], t["latitude"])
        rec = {"source": s["name"], "target": t["name"], "landmark_before": o_plain[5]}
        for name, o in (("steer", o_steer), ("control", o_ctrl), ("swap", o_swap)):
            ans = dict(zip(QKEYS, o[:5]))
            pq = score_edit(ans, sk, tk)["per_question"]
            hemi, seas = pq["hemisphere"]["flipped"], pq["season"]["flipped"]
            kept = matches(ans["country"], sk, "country") and identity_kept(o_plain[5], o[5])
            rec[name] = {"answers": ans, "landmark": o[5], "moved": hemi or seas, "both_moved": hemi and seas,
                         "identity_kept": kept, "moved_and_kept": (hemi or seas) and kept}
        rows.append(rec)
        print(f"[heldout] {s['name']} -> {t['name']}: steer moved={rec['steer']['moved']} kept={rec['steer']['identity_kept']} "
              f"| control moved={rec['control']['moved']} | swap kept={rec['swap']['identity_kept']}  "
              f"{rec['steer']['answers']['hemisphere']}/{rec['steer']['answers']['season']}/{rec['steer']['landmark']!r}",
              flush=True)

    def rate(kind, key):
        return float(np.mean([r[kind][key] for r in rows]))

    out = {"setting": cfg, "n_pairs": len(rows), "seed": args.seed,
           "note": "setting fixed from results/steer_eval.json; fresh cross-hemisphere pairs, gate pairs excluded",
           "rates": {k: {m: rate(k, m) for m in ("moved", "both_moved", "identity_kept", "moved_and_kept")}
                     for k in ("steer", "control", "swap")},
           "pairs": rows, "seconds": time.time() - t0}
    (RESULTS / "steer_heldout.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[heldout] {json.dumps(out['rates'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
