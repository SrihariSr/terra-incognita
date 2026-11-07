"""Tier 2 test: does probe-direction steering move a city but keep its identity?

On the gate pairs we sweep a few layer bands and scales. For each setting:
* ripple answers (country, currency, language, hemisphere, season), scored
  against the target city as in the gate;
* identity: "What is the most famous landmark in X?" The landmark is
  *kept* if the steered answer shares a content word with the unedited
  answer (Eiffel Tower stays Eiffel Tower).

Rule fixed in advance (ROADMAP M4): keep tier 2 as the "move, don't swap"
mode only if some setting flips the hemisphere or season answer while the
landmark is kept on at least half of the pairs. The chosen setting is the
one with the most such pairs (ties: smaller scale). For that setting we also
run a random-direction control with the same norm as the steering change.

    uv run python scripts/steer_eval.py
Writes results/steer_eval.json (and models/steer_probes.npz if missing)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import RESULTS  # noqa: E402
from terra.companies import identity_kept  # noqa: E402
from terra.data import city_info  # noqa: E402
from terra.model import Patch, get_lm  # noqa: E402
from terra.patch import EditPlan, band, generate_plans, plan_edit, stable_seed  # noqa: E402
from terra.ripple import QKEYS, make_key, prompts_for, score_edit  # noqa: E402
from terra.steer import PROBE_PATH, fit_layer_probes, plan_context_city, plan_steer  # noqa: E402

LANDMARK = "What is the most famous landmark in {city}? Answer with just its name."
BANDS = [(4, 7), (2, 9), (6, 10)]
SCALES = [1.0, 4.0, 8.0, 16.0, 32.0]


def key(c):
    i = city_info(c)
    return make_key(c, i["iso"], i["latitude"])


def main() -> int:
    t0 = time.time()
    if not PROBE_PATH.exists():
        fit_layer_probes(range(0, 14))
    gate = json.loads((RESULTS / "gate.json").read_text())
    pairs = [tuple(p) for p in gate["pairs"]]
    lm = get_lm()

    def texts_for(src):
        qs = prompts_for(src)
        return [qs[q] for q in QKEYS] + [LANDMARK.format(city=src)]

    # Unedited landmark answers and the tier-1 swap's landmark answers.
    swap_band = band(*gate["best_band"])
    ref = {}
    for s, t in pairs:
        plain = [lm.chat(x) for x in texts_for(s)]
        sw = plan_edit(lm, texts_for(s), s, t, swap_band, mode="swap")
        o_plain, o_swap = generate_plans(lm, [plain, sw], max_new_tokens=12)
        ref[(s, t)] = {"before": o_plain, "swap": o_swap}
        print(f"[ref] {s}->{t}: landmark before {o_plain[5]!r}, after swap {o_swap[5]!r}")

    sweep = []
    for a, b in BANDS:
        for sc in SCALES:
            rows = []
            for s, t in pairs:
                plan = plan_steer(lm, None, texts_for(s), s, t, layers=band(a, b), scale=sc)
                out = generate_plans(lm, [plan], max_new_tokens=12)[0]
                ans = dict(zip(QKEYS, out[:5]))
                score = score_edit(ans, key(s), key(t))
                pq = score["per_question"]
                kept = identity_kept(ref[(s, t)]["before"][5], out[5])
                moved = pq["hemisphere"]["flipped"] or pq["season"]["flipped"]
                rows.append({"source": s, "target": t, "answers": ans, "landmark": out[5],
                             "landmark_before": ref[(s, t)]["before"][5], "landmark_kept": kept,
                             "moved": moved, "n_flipped": score["n_flipped"],
                             "delta_norm_mean": float(np.mean(list(plan.delta_norms.values())))})
            good = sum(r["moved"] and r["landmark_kept"] for r in rows)
            rec = {"method": "steer", "band": [a, b], "scale": sc, "n_moved_and_kept": good,
                   "n_moved": sum(r["moved"] for r in rows), "n_kept": sum(r["landmark_kept"] for r in rows),
                   "mean_flipped": float(np.mean([r["n_flipped"] for r in rows])), "pairs": rows}
            sweep.append(rec)
            print(f"[steer] band {a}-{b} scale {sc:4.0f}: moved+kept {good}/10, moved {rec['n_moved']}, "
                  f"kept {rec['n_kept']}, mean flips {rec['mean_flipped']:.1f}, |dh| {rows[0]['delta_norm_mean']:.1f}",
                  flush=True)

    # Second candidate: context transplant (terra/steer.py:plan_context_city),
    # swept over the same bands; "scale" is recorded as 0 for this method.
    for a, b in BANDS + [(4, 11), (0, 7)]:
        rows = []
        for s, t in pairs:
            plan = plan_context_city(lm, texts_for(s), s, t, band(a, b))
            out = generate_plans(lm, [plan], max_new_tokens=12)[0]
            ans = dict(zip(QKEYS, out[:5]))
            score = score_edit(ans, key(s), key(t))
            pq = score["per_question"]
            kept = identity_kept(ref[(s, t)]["before"][5], out[5])
            moved = pq["hemisphere"]["flipped"] or pq["season"]["flipped"]
            rows.append({"source": s, "target": t, "answers": ans, "landmark": out[5],
                         "landmark_before": ref[(s, t)]["before"][5], "landmark_kept": kept,
                         "moved": moved, "n_flipped": score["n_flipped"],
                         "delta_norm_mean": float(np.mean(list(plan.delta_norms.values())))})
        good = sum(r["moved"] and r["landmark_kept"] for r in rows)
        rec = {"method": "context", "band": [a, b], "scale": 0.0, "n_moved_and_kept": good,
               "n_moved": sum(r["moved"] for r in rows), "n_kept": sum(r["landmark_kept"] for r in rows),
               "mean_flipped": float(np.mean([r["n_flipped"] for r in rows])), "pairs": rows}
        sweep.append(rec)
        print(f"[context] band {a}-{b}: moved+kept {good}/10, moved {rec['n_moved']}, kept {rec['n_kept']}, "
              f"mean flips {rec['mean_flipped']:.1f}", flush=True)

    best = max(sweep, key=lambda r: (r["n_moved_and_kept"], r.get("method", "steer") == "steer", -r["scale"]))
    keep = best["n_moved_and_kept"] >= len(pairs) / 2

    # Random-direction control at the chosen setting, same per-layer norm.
    ctrl = []
    for s, t in pairs:
        if best["method"] == "context":
            plan = plan_context_city(lm, texts_for(s), s, t, band(*best["band"]))
        else:
            plan = plan_steer(lm, None, texts_for(s), s, t, layers=band(*best["band"]), scale=best["scale"])
        src_states = torch.cat(lm.residuals_multi(plan.source_texts,
                                                  [[c for r, c in zip(plan.patch.rows, plan.patch.cols) if r == i]
                                                   for i in range(len(plan.source_texts))], plan.layers), 0)
        g = torch.Generator().manual_seed(stable_seed(s, t, "steer-control"))
        u = torch.randn(lm.d_model, generator=g)
        u = u / u.norm()
        vecs = {}
        for j, L in enumerate(plan.layers):
            clean = src_states[:, j]
            n = (plan.patch.vectors[L] - clean).norm(dim=-1, keepdim=True)
            vecs[L] = clean + n * u
        cplan = EditPlan(plan.source_texts, Patch(plan.patch.rows, plan.patch.cols, vecs), plan.delta_norms, plan.layers)
        out = generate_plans(lm, [cplan], max_new_tokens=12)[0]
        ans = dict(zip(QKEYS, out[:5]))
        sc = score_edit(ans, key(s), key(t))
        pq = sc["per_question"]
        ctrl.append({"source": s, "target": t, "answers": ans, "landmark": out[5],
                     "moved": pq["hemisphere"]["flipped"] or pq["season"]["flipped"],
                     "landmark_kept": identity_kept(ref[(s, t)]["before"][5], out[5]),
                     "n_flipped": sc["n_flipped"]})
    out = {
        "rule": "keep if some setting flips hemisphere or season while keeping the landmark on >= half the pairs",
        "pairs": pairs, "reference": {f"{s}->{t}": v for (s, t), v in ref.items()},
        "swap_landmark_kept": sum(identity_kept(v["before"][5], v["swap"][5]) for v in ref.values()),
        "sweep": sweep,
        "chosen": {"method": best["method"], "band": best["band"], "scale": best["scale"]} if keep else {},
        "best": {"method": best["method"], "band": best["band"], "scale": best["scale"],
                 "n_moved_and_kept": best["n_moved_and_kept"]},
        "keep": keep, "control": ctrl,
        "control_moved_and_kept": sum(r["moved"] and r["landmark_kept"] for r in ctrl),
        "seconds": time.time() - t0,
    }
    (RESULTS / "steer_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[steer] best {best['method']} {best['band']} x{best['scale']}: {best['n_moved_and_kept']}/10 moved+kept; "
          f"control {out['control_moved_and_kept']}/10; keep={keep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
