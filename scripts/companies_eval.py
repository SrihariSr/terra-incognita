"""Company layer: read believed HQ locations and test moving companies.

1. Load companies with Wikidata HQs (inconsistent records dropped, see
   terra/companies.py).
2. Read where the probe places each company name; compare with the HQ.
3. Ask the finance questions unedited. A company is "known" if the model
   names its HQ country correctly. Only known companies are used further.
4. Move each known company to two target cities (seeded choice among
   TARGET_CITIES in other countries with a different currency), with both
   edit variants ("city" swap and "context" transplant) and the equal-norm
   random control. Score the three finance answers as flipped or not, and
   check whether "What is X best known for?" survives.
5. Choose the variant for the demo by a rule fixed here in advance: use
   "context" if its pass rate is within 0.10 of "city" and it keeps the
   company identity more often; otherwise use "city".

    uv run python scripts/companies_eval.py
Writes results/companies.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import RESULTS  # noqa: E402
from terra.companies import (QUESTIONS, SCORED, TARGET_CITIES, identity_kept, load_companies,  # noqa: E402
                             plan_context, questions_for, score, target_iso)
from terra.data import load_countries  # noqa: E402
from terra.locate import Locator  # noqa: E402
from terra.model import get_lm  # noqa: E402
from terra.patch import band, generate_plans, plan_edit  # noqa: E402
from terra.probe import great_circle_km, latlon_to_xyz  # noqa: E402

PASS = 2  # of the 3 scored finance answers must flip (when all 3 are countable)


def main() -> int:
    t0 = time.time()
    gate = json.loads((RESULTS / "gate.json").read_text())
    layers = band(*gate["best_band"])
    base = {c["name"]: c for c in json.loads((RESULTS / "baseline_cities.json").read_text())["cities"]}
    targets = [c for c in TARGET_CITIES if base.get(c, {}).get("all_correct")]
    print(f"[companies] target cities passing the city baseline: {targets}")
    lm = get_lm()
    loc = Locator(lm)
    cos = load_companies()
    kept = [c for c in cos if "dropped" not in c]
    print(f"[companies] {len(kept)} of {len(cos)} companies have consistent Wikidata records")

    # ---- reading: believed HQ positions
    locs = loc.locate([c["name"] for c in kept])
    for c, r in zip(kept, locs):
        c["blat"], c["blon"] = r["lat"], r["lon"]
        c["err_km"] = float(great_circle_km(latlon_to_xyz(r["lat"], r["lon"]), latlon_to_xyz(c["lat"], c["lon"])))
        c["believed_nearest"] = r["nearest"]

    # ---- baseline answers
    plans = [[lm.chat(t) for t in questions_for(c["name"])] for c in kept]
    outs = []
    for b0 in range(0, len(plans), 8):
        outs += generate_plans(lm, plans[b0:b0 + 8], max_new_tokens=16)
    for c, o in zip(kept, outs):
        c["before"] = dict(zip(QUESTIONS, o))
        sc = score(c["before"], c["iso"])
        c["baseline_correct"] = {q: sc[q]["matches_source"] for q in SCORED}
        c["known"] = bool(sc["hq_country"]["matches_source"])
        print(f"[baseline] {c['name']:26s} known={c['known']}  {c['before']['hq_country']!r} | "
              f"{c['before']['central_bank']!r} | {c['before']['currency']!r}")

    # ---- edits
    rng = np.random.default_rng(0)
    cur = load_countries()["CurrencyCode"]
    rows = []
    for c in [c for c in kept if c["known"]]:
        options = [t for t in targets if target_iso(t) != c["iso"] and cur[target_iso(t)] != cur[c["iso"]]]
        for city in rng.choice(options, size=min(2, len(options)), replace=False):
            city = str(city)
            texts = questions_for(c["name"])
            p_city = plan_edit(lm, texts, c["name"], city, layers, mode="swap")
            p_ctx = plan_context(lm, texts, c["name"], city, layers)
            p_ctrl = plan_edit(lm, texts, c["name"], city, layers, mode="control")
            o_city, o_ctx, o_ctrl = generate_plans(lm, [p_city, p_ctx, p_ctrl], max_new_tokens=16)
            rec = {"company": c["name"], "target_city": city, "target_iso": target_iso(city)}
            for name, o in (("city", o_city), ("context", o_ctx), ("control", o_ctrl)):
                ans = dict(zip(QUESTIONS, o))
                sc = score(ans, c["iso"], target_iso(city))
                n_c = sum(sc[q]["countable"] for q in SCORED)
                n_f = sum(sc[q]["flipped"] for q in SCORED if sc[q]["countable"])
                rec[name] = {"answers": ans, "n_countable": n_c, "n_flipped": n_f,
                             "identity_kept": identity_kept(c["before"]["identity"], ans["identity"])}
            rows.append(rec)
            print(f"[edit] {c['name']} -> {city}: city {rec['city']['n_flipped']}/{rec['city']['n_countable']} "
                  f"(id {rec['city']['identity_kept']}), context {rec['context']['n_flipped']}/"
                  f"{rec['context']['n_countable']} (id {rec['context']['identity_kept']}), "
                  f"control {rec['control']['n_flipped']}  | ctx: {rec['context']['answers']['central_bank']!r} "
                  f"/ {rec['context']['answers']['identity']!r}", flush=True)

    def summary(kind):
        sel = [r[kind] for r in rows if r[kind]["n_countable"] >= 2]
        need = lambda r: min(PASS, r["n_countable"])  # noqa: E731
        return {"n": len(sel),
                "pass_rate": float(np.mean([r["n_flipped"] >= need(r) for r in sel])) if sel else None,
                "mean_fraction_flipped": float(np.mean([r["n_flipped"] / r["n_countable"] for r in sel])) if sel else None,
                "identity_kept_rate": float(np.mean([r["identity_kept"] for r in sel])) if sel else None}

    summ = {k: summary(k) for k in ("city", "context", "control")}
    choose_ctx = (summ["context"]["pass_rate"] is not None and summ["city"]["pass_rate"] is not None
                  and summ["context"]["pass_rate"] >= summ["city"]["pass_rate"] - 0.10
                  and summ["context"]["identity_kept_rate"] > summ["city"]["identity_kept_rate"])
    known = [c for c in kept if c["known"]]
    errs = [c["err_km"] for c in kept]
    out = {
        "source": "data/companies_wikidata.json (Wikidata P159/P625/P17), central banks P1304",
        "n_companies": len(cos), "n_consistent": len(kept), "n_known": len(known),
        "dropped": [{"name": c["name"], "reason": c["dropped"]} for c in cos if "dropped" in c],
        "probe_hq_error_km": {"median": float(np.median(errs)), "mean": float(np.mean(errs)),
                              "known_median": float(np.median([c["err_km"] for c in known])) if known else None},
        "pass_rule": f">= {PASS} of the countable finance answers flip (of 3)",
        "summary": summ, "chosen_variant": "context" if choose_ctx else "city",
        "choice_rule": "context if pass rate within 0.10 of city and identity kept more often, else city",
        "targets": targets, "companies": kept, "edits": rows, "band": [layers[0], layers[-1]],
        "seconds": time.time() - t0,
    }
    (RESULTS / "companies.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[companies] known {len(known)}/{len(kept)}; summary {summ}; chosen {out['chosen_variant']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
