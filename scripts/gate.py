"""GATE: does the tier-1 edit work?

Procedure (from ROADMAP.md):
1. Take 10 well-known city pairs in different countries and hemispheres,
   chosen so that all five ripple answers differ between source and target.
2. Check that the unedited model answers all five questions correctly for
   every city involved. Pairs that fail are replaced from a reserve list.
3. Sweep several layer bands. For each band, patch every pair and count the
   pairs where at least 4 of 5 answers flip to the target city.
4. At the best band, run the equal-norm random-direction control.
5. GO if >= 7/10 pairs flip >= 4/5 answers AND the control flips >= 4/5 on
   at most 1/10 pairs.

Caveat recorded in the output: the band is chosen on these same 10 pairs,
so the gate number is optimistic. The honest edit success rate comes from
scripts/batch_edit_eval.py on fresh random pairs at the fixed band.

Usage:  uv run python scripts/gate.py [--model gemma4-31b]
Writes: results/gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import registry  # noqa: E402
from terra.data import city_info  # noqa: E402
from terra.model import get_lm  # noqa: E402
from terra.patch import ask, band  # noqa: E402
from terra.ripple import QKEYS, baseline_correct, make_key, prompts_for, score_edit  # noqa: E402

PAIRS = [
    ("Paris", "Sydney"),
    ("Tokyo", "Buenos Aires"),
    ("Berlin", "Cape Town"),
    ("Moscow", "Rio de Janeiro"),
    ("Seoul", "Santiago"),
    ("Cairo", "Lima"),
    ("Rome", "Johannesburg"),
    ("Madrid", "Melbourne"),
    ("Sydney", "Stockholm"),
    ("Buenos Aires", "Beijing"),
]
RESERVE = [
    ("Stockholm", "São Paulo"),
    ("Istanbul", "Perth"),
    ("Athens", "Montevideo"),
    ("Warsaw", "Wellington"),
    ("Melbourne", "Rome"),
    ("Lima", "Berlin"),
]

def bands_for(n_layers: int) -> list[tuple[int, int]]:
    """Layer bands to sweep, scaled to the model's depth.

    For Qwen's 28 layers this is exactly the original sweep: bands starting
    at 0, 4, 8, 12 and ending at 7, 11, 15, 19, 23, 27 (sevenths of the depth).
    """
    starts = [round(n_layers * k / 7) for k in range(4)]
    ends = [round(n_layers * k / 7) - 1 for k in range(2, 8)]
    return [(a, b) for a in starts for b in ends if b > a + 2]


BANDS = bands_for(28)


def key_for(city: str):
    info = city_info(city)
    return make_key(city, info["iso"], info["latitude"])


def run_edit(lm, src, tgt, layers, mode):
    qs = prompts_for(src)
    answers = ask(lm, [qs[q] for q in QKEYS], max_new_tokens=10, source=src, target=tgt,
                  layers=layers, mode=mode)
    return dict(zip(QKEYS, answers))


def run(key: str = registry.DEFAULT_KEY, lm=None, out_path: str | None = None) -> dict:
    """The ten-pair gate for `key`; writes gate.json in its results folder."""
    out_file = Path(out_path) if out_path else registry.paths(key).result("gate.json")
    t0 = time.time()
    lm = lm or get_lm(key)
    BANDS = bands_for(lm.n_layers)

    # ---- 1. baseline: does the clean model know every city?
    baseline = {}
    cities = sorted({c for p in PAIRS + RESERVE for c in p})
    for c in cities:
        qs = prompts_for(c)
        ans = dict(zip(QKEYS, ask(lm, [qs[q] for q in QKEYS], max_new_tokens=10)))
        ok = baseline_correct(ans, key_for(c))
        baseline[c] = {"answers": ans, "correct": ok, "all_correct": all(ok.values())}
        print(f"[baseline] {c:15s} {'OK ' if all(ok.values()) else 'BAD'} {ans}")

    pairs = []
    reserve = list(RESERVE)
    replaced = []
    for p in PAIRS:
        if baseline[p[0]]["all_correct"] and baseline[p[1]]["all_correct"]:
            pairs.append(p)
            continue
        while reserve:
            r = reserve.pop(0)
            if baseline[r[0]]["all_correct"] and baseline[r[1]]["all_correct"]:
                pairs.append(r)
                replaced.append({"dropped": p, "replacement": r})
                break
    print(f"[pairs] using {len(pairs)} pairs; replaced {replaced}")

    # Confirm all five questions are countable for each pair.
    for s, t in pairs:
        sk, tk = key_for(s), key_for(t)
        sc = score_edit({q: "" for q in QKEYS}, sk, tk)
        if sc["n_countable"] < 5:
            print(f"[warn] {s}->{t}: only {sc['n_countable']} countable questions")

    # ---- 2. band sweep with the swap edit
    sweep = []
    for a, b in BANDS:
        layers = band(a, b)
        res = []
        for s, t in pairs:
            ans = run_edit(lm, s, t, layers, "swap")
            sc = score_edit(ans, key_for(s), key_for(t))
            res.append({"source": s, "target": t, "answers": ans, "n_flipped": sc["n_flipped"],
                        "n_countable": sc["n_countable"], "n_source": sc["n_source"]})
        n_pass = sum(r["n_flipped"] >= 4 for r in res)
        mean_flip = sum(r["n_flipped"] for r in res) / len(res)
        sweep.append({"band": [a, b], "n_pairs_pass": n_pass, "mean_flipped": mean_flip, "pairs": res})
        print(f"[sweep] band {a:2d}-{b:2d}: {n_pass}/{len(res)} pairs flip >=4/5, mean flips {mean_flip:.2f}")

    # Best band: most passing pairs, then highest mean flips, then the
    # narrowest band (a smaller intervention is preferable).
    best = max(sweep, key=lambda r: (r["n_pairs_pass"], r["mean_flipped"], -(r["band"][1] - r["band"][0])))
    layers = band(*best["band"])
    print(f"[best] band {best['band']}")

    # ---- 3. control at the best band
    ctrl = []
    for s, t in pairs:
        ans = run_edit(lm, s, t, layers, "control")
        sc = score_edit(ans, key_for(s), key_for(t))
        ctrl.append({"source": s, "target": t, "answers": ans, "n_flipped": sc["n_flipped"],
                     "n_source": sc["n_source"]})
        print(f"[control] {s}->{t}: flipped {sc['n_flipped']}/5, kept source {sc['n_source']}/5  {ans}")
    ctrl_pass = sum(r["n_flipped"] >= 4 for r in ctrl)

    go = best["n_pairs_pass"] >= 7 and ctrl_pass <= 1
    out = {
        "model": lm.model_id,
        "criteria": "GO if >=7/10 pairs flip >=4/5 answers at the best band and the random "
                    "control flips >=4/5 on <=1/10 pairs",
        "pairs": pairs,
        "replaced": replaced,
        "baseline": baseline,
        "bands_tried": BANDS,
        "sweep": sweep,
        "best_band": best["band"],
        "edit_pairs_pass": best["n_pairs_pass"],
        "edit_mean_flipped": best["mean_flipped"],
        "control": ctrl,
        "control_pairs_pass": ctrl_pass,
        "control_mean_flipped": sum(r["n_flipped"] for r in ctrl) / len(ctrl),
        "decision": "GO" if go else "NO-GO",
        "caveat": "The band was selected on these same pairs, so this pass rate is optimistic; "
                  "see results/edit_eval.json for fresh pairs at the fixed band.",
        "seconds": time.time() - t0,
    }
    out["model"] = key
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[gate] {out['decision']}: edit {best['n_pairs_pass']}/{len(pairs)}, "
          f"control {ctrl_pass}/{len(pairs)} -> {out_file}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=registry.DEFAULT_KEY, choices=sorted(registry.MODELS))
    ap.add_argument("--out", default=None, help="default: <model results>/gate.json")
    args = ap.parse_args()
    run(args.model, out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
