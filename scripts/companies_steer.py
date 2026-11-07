"""Exploratory: does tier-2 steering move a company but keep its identity?

Uses the same (company, target city) moves as results/companies.json and the
city-tuned steering band from results/steer_eval.json, at a few scales. A
move passes if at least 2 of the countable finance answers flip (as in
companies_eval.py); identity is kept if "What is X best known for?" shares a
content word with the unedited answer.

This is a side experiment: the demo's company edit stays as chosen in
results/companies.json unless this clearly beats it.

    uv run python scripts/companies_steer.py
Writes results/companies_steer.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import RESULTS  # noqa: E402
from terra.companies import QUESTIONS, SCORED, identity_kept, questions_for, score  # noqa: E402
from terra.model import get_lm  # noqa: E402
from terra.patch import band, generate_plans  # noqa: E402
from terra.steer import plan_steer  # noqa: E402

SCALES = [4.0, 8.0, 16.0]


def main() -> int:
    t0 = time.time()
    co = json.loads((RESULTS / "companies.json").read_text())
    st = json.loads((RESULTS / "steer_eval.json").read_text())
    layers = band(*st["best"]["band"])
    by_name = {c["name"]: c for c in co["companies"]}
    edits = co["edits"]
    lm = get_lm()
    out_rows = {s: [] for s in SCALES}
    for e in edits:
        c = by_name[e["company"]]
        texts = questions_for(c["name"])
        plans = [plan_steer(lm, None, texts, c["name"], e["target_city"], layers=layers, scale=s) for s in SCALES]
        outs = generate_plans(lm, plans, max_new_tokens=16)
        for s, o in zip(SCALES, outs):
            ans = dict(zip(QUESTIONS, o))
            sc = score(ans, c["iso"], e["target_iso"])
            n_c = sum(sc[q]["countable"] for q in SCORED)
            n_f = sum(sc[q]["flipped"] for q in SCORED if sc[q]["countable"])
            out_rows[s].append({"company": c["name"], "target_city": e["target_city"], "answers": ans,
                                "n_countable": n_c, "n_flipped": n_f,
                                "identity_kept": identity_kept(c["before"]["identity"], ans["identity"])})
        print(f"[co-steer] {c['name']} -> {e['target_city']}: " +
              ", ".join(f"x{s:g} {r[-1]['n_flipped']}/{r[-1]['n_countable']} id={r[-1]['identity_kept']}"
                        for s, r in out_rows.items()), flush=True)
    summ = {}
    for s, rows in out_rows.items():
        sel = [r for r in rows if r["n_countable"] >= 2]
        summ[str(s)] = {"n": len(sel),
                        "pass_rate": float(np.mean([r["n_flipped"] >= min(2, r["n_countable"]) for r in sel])),
                        "identity_kept_rate": float(np.mean([r["identity_kept"] for r in sel])),
                        "pass_and_kept": float(np.mean([(r["n_flipped"] >= min(2, r["n_countable"])) and r["identity_kept"]
                                                        for r in sel]))}
    out = {"band": [layers[0], layers[-1]], "scales": SCALES, "summary": summ,
           "reference_city_swap": co["summary"]["city"], "rows": {str(k): v for k, v in out_rows.items()},
           "seconds": time.time() - t0}
    (RESULTS / "companies_steer.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[co-steer] {json.dumps(summ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
