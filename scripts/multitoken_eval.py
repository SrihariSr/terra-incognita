"""Experiment: patch every token of the name instead of only the last one.

Runs terra.patch.plan_edit_all (swap and its equal-norm control) on exactly
the pairs in results/edit_eval.json, at the same band, and compares pass
rates with the last-token swap recorded there. Same scoring, same rule.

Adoption rule, fixed before running: switch the demo to all-token patching
only if, on the cross-hemisphere set, its pass rate beats the last-token
swap by at least 3 percentage points AND its control pass rate is at most
2%. Otherwise the demo keeps last-token patching.

    uv run python scripts/multitoken_eval.py
Writes results/multitoken_eval.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import RESULTS  # noqa: E402
from terra.data import city_info  # noqa: E402
from terra.model import get_lm  # noqa: E402
from terra.patch import band, generate_plans, plan_edit_all  # noqa: E402
from terra.ripple import QKEYS, make_key, prompts_for, score_edit  # noqa: E402

PASS = 0.8
MIN_COUNTABLE = 3


def passes(sc):
    return sc["n_countable"] >= MIN_COUNTABLE and sc["n_flipped"] >= PASS * sc["n_countable"]


def main() -> int:
    t0 = time.time()
    ev = json.loads((RESULTS / "edit_eval.json").read_text())
    layers = band(*ev["band"])
    lm = get_lm()
    out = {"band": ev["band"], "rule": "adopt if cross pass rate +3 points or more and control pass <= 2%",
           "sets": {}}
    for set_name, data in ev["sets"].items():
        pairs = [p for p in data["pairs"] if p["n_countable"] >= MIN_COUNTABLE]
        rows = []
        for b0 in range(0, len(pairs), 4):
            chunk = pairs[b0:b0 + 4]
            plans = []
            for p in chunk:
                qs = prompts_for(p["source"])
                texts = [qs[q] for q in QKEYS]
                plans.append(plan_edit_all(lm, texts, p["source"], p["target"], layers, mode="swap"))
                plans.append(plan_edit_all(lm, texts, p["source"], p["target"], layers, mode="control"))
            outs = generate_plans(lm, plans, max_new_tokens=10)
            for i, p in enumerate(chunk):
                si, ti = city_info(p["source"]), city_info(p["target"])
                sk, tk = make_key(p["source"], si["iso"], si["latitude"]), make_key(p["target"], ti["iso"], ti["latitude"])
                ed = score_edit(dict(zip(QKEYS, outs[2 * i])), sk, tk)
                ct = score_edit(dict(zip(QKEYS, outs[2 * i + 1])), sk, tk)
                multi = len(lm.tok(" " + p["source"], add_special_tokens=False)["input_ids"]) > 1
                rows.append({"source": p["source"], "target": p["target"], "multi_token_source": multi,
                             "last_token_pass": passes(p["edit"]), "all_token_pass": passes(ed),
                             "all_token_control_pass": passes(ct), "all_token_answers": dict(zip(QKEYS, outs[2 * i]))})
            print(f"[multi {set_name}] {len(rows)}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
        multi = [r for r in rows if r["multi_token_source"]]
        out["sets"][set_name] = {
            "n": len(rows),
            "last_token_pass_rate": float(np.mean([r["last_token_pass"] for r in rows])),
            "all_token_pass_rate": float(np.mean([r["all_token_pass"] for r in rows])),
            "all_token_control_pass_rate": float(np.mean([r["all_token_control_pass"] for r in rows])),
            "n_multi_token_sources": len(multi),
            "multi_last_token_pass_rate": float(np.mean([r["last_token_pass"] for r in multi])) if multi else None,
            "multi_all_token_pass_rate": float(np.mean([r["all_token_pass"] for r in multi])) if multi else None,
            "pairs": rows,
        }
        s = out["sets"][set_name]
        print(f"[multi {set_name}] last-token {s['last_token_pass_rate']:.2f}, all-token {s['all_token_pass_rate']:.2f} "
              f"(control {s['all_token_control_pass_rate']:.2f}); multi-token sources: "
              f"{s['multi_last_token_pass_rate']} -> {s['multi_all_token_pass_rate']}")
    c = out["sets"]["cross"]
    out["adopt"] = (c["all_token_pass_rate"] - c["last_token_pass_rate"] >= 0.03
                    and c["all_token_control_pass_rate"] <= 0.02)
    out["seconds"] = time.time() - t0
    (RESULTS / "multitoken_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[multi] adopt = {out['adopt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
