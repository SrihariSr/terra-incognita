"""Build the validated menu of edit pairs for the demo.

Take famous cities that the unedited model answers correctly on all five
ripple questions (results/baseline_cities.json). For every ordered pair in
different countries, run the tier-1 swap and the random control at the
gate band. A pair is *validated* if the edit flips at least 80% of the
countable answers (with at least 3 countable) and the control flips fewer
than half. Everything is cached so the demo answers instantly, and the
free question ("I'm in X for Christmas...") is precomputed for the
headline sources.

    uv run python scripts/build_pair_menu.py
Writes results/pair_menu.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import RESULTS  # noqa: E402
from terra.data import city_info  # noqa: E402
from terra.model import get_lm  # noqa: E402
from terra.patch import band, generate_plans, plan_edit, plan_edit_all  # noqa: E402
from terra.ripple import FREE_QUESTION, QKEYS, make_key, prompts_for, score_edit  # noqa: E402

DEMO_CITIES = ["Paris", "London", "Tokyo", "Berlin", "Rome", "Moscow", "Cairo", "Istanbul", "Athens",
               "Stockholm", "Beijing", "New York City", "Toronto", "Mexico City", "Lisbon", "Amsterdam",
               "Vienna", "Sydney", "Melbourne", "Buenos Aires", "Rio de Janeiro", "Cape Town",
               "Johannesburg", "Lima", "Santiago", "Montevideo", "Perth", "Auckland", "Wellington"]
HEADLINE_SOURCES = ["Paris", "London", "Tokyo", "New York City"]
HEADLINE_TARGETS = ["Sydney", "Buenos Aires", "Cape Town", "Rio de Janeiro", "Tokyo", "Paris", "Moscow",
                    "Melbourne", "Auckland", "Lima"]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["last", "all"], default="last")
    ap.add_argument("--out", default=str(RESULTS / "pair_menu.json"))
    args = ap.parse_args()
    planner = plan_edit_all if args.method == "all" else plan_edit
    t0 = time.time()
    gate = json.loads((RESULTS / "gate.json").read_text())
    layers = band(*gate["best_band"])
    base = {c["name"]: c for c in json.loads((RESULTS / "baseline_cities.json").read_text())["cities"]}
    cities = [c for c in DEMO_CITIES if base.get(c, {}).get("all_correct")]
    missing = [c for c in DEMO_CITIES if c not in cities]
    print(f"[menu] {len(cities)} demo cities pass baseline; not used: {missing}")
    lm = get_lm()
    pairs = [(s, t) for s in cities for t in cities
             if s != t and city_info(s)["iso"] != city_info(t)["iso"]]
    rows = []
    for b0 in range(0, len(pairs), 4):
        chunk = pairs[b0:b0 + 4]
        plans = []
        for s, t in chunk:
            qs = prompts_for(s)
            texts = [qs[q] for q in QKEYS]
            plans.append(planner(lm, texts, s, t, layers, mode="swap"))
            plans.append(planner(lm, texts, s, t, layers, mode="control"))
        outs = generate_plans(lm, plans, max_new_tokens=10)
        for i, (s, t) in enumerate(chunk):
            si, ti = city_info(s), city_info(t)
            sk, tk = make_key(s, si["iso"], si["latitude"]), make_key(t, ti["iso"], ti["latitude"])
            after, control = dict(zip(QKEYS, outs[2 * i])), dict(zip(QKEYS, outs[2 * i + 1]))
            se, sc = score_edit(after, sk, tk), score_edit(control, sk, tk)
            ok = (se["n_countable"] >= 3 and se["n_flipped"] >= 0.8 * se["n_countable"]
                  and sc["n_flipped"] < 0.5 * sc["n_countable"])
            rows.append({"source": s, "target": t, "mode": "swap", "before": base[s]["answers"],
                         "after": after, "control": control, "n_countable": se["n_countable"],
                         "n_flipped": se["n_flipped"], "control_flipped": sc["n_flipped"], "validated": ok})
        if (b0 // 4) % 10 == 0:
            print(f"[menu] {len(rows)}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
    valid = [r for r in rows if r["validated"]]
    print(f"[menu] {len(valid)}/{len(rows)} pairs validated")

    # Precompute the free question for the headline pairs.
    free = []
    vset = {(r["source"], r["target"]) for r in valid}
    jobs = [(s, t) for s in HEADLINE_SOURCES for t in HEADLINE_TARGETS if (s, t) in vset]
    for s, t in jobs:
        q = FREE_QUESTION.format(city=s)
        plan = planner(lm, [q], s, t, layers, mode="swap")
        before, after = generate_plans(lm, [[lm.chat(q)], plan], max_new_tokens=80)
        free.append({"question": q, "source": s, "target": t, "before": before[0], "after": after[0]})
        print(f"[free] {s} -> {t}: {after[0][:90]!r}", flush=True)

    menu_cities = [{"name": c, "lat": city_info(c)["latitude"], "lon": city_info(c)["longitude"],
                    "iso": city_info(c)["iso"]} for c in cities]
    out = {"band": [layers[0], layers[-1]], "method": args.method, "rule": ">= 80% of >= 3 countable answers flip; control < 50%",
           "cities": menu_cities, "n_pairs": len(rows), "n_validated": len(valid),
           "pairs": valid, "all_pairs": rows, "free_answers": free, "seconds": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[menu] wrote {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
