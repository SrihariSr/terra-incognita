"""Evaluate "Rewrite the world": three moves applied together.

Random worlds (seeded): three moves drawn from the cities the unedited model
answers correctly (results/baseline_cities.json), with three different
sources, three different targets, no city both a source and a target, and
each source in a different country from its target. For every world we run
terra.rewrite.run_rewrite in both modes (swap, every name token, and the
tier-2 "move" mode) together with their equal-norm random controls.

Reported:
* city level: share of moved cities whose ripple answers pass (>= 80% of
  countable answers flip) when all three edits are applied together;
* world level: share of worlds where all three cities pass;
* pair questions ("same hemisphere?", "which is further south?"): share of
  answers consistent with the rewritten world, and the flip rate on the
  questions whose right answer the edits change (countable). Only pair
  questions the unedited model answers correctly are scored.

Also precomputes three suggested worlds for the demo (with the free
question before and after) into results/rewrite_menu.json.

    uv run python scripts/rewrite_eval.py [--n 40] [--no-pad --out ... --menu-out ...]

Padding rule, fixed before the padded run: the multi-move mode pads shorter
target names (terra/patch.py:TARGET_PAD) if, on the same 40 worlds, the
padded version's pair-answer consistency is not lower than the unpadded one
and its city pass rate is at most 2 points lower. The unpadded run is kept as
results/rewrite_eval_nopad.json.
Writes results/rewrite_eval.json, results/rewrite_menu.json
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
from terra.patch import band, generate_plans, plan_edit_multi  # noqa: E402
from terra.companies import identity_kept  # noqa: E402
from terra.rewrite import free_question, run_rewrite  # noqa: E402

SUGGESTED = [
    {"name": "Paris, Tokyo and Rio trade places",
     "moves": [("Paris", "Sydney"), ("Tokyo", "London"), ("Rio de Janeiro", "Moscow")]},
    {"name": "Berlin, Sydney and Buenos Aires",
     "moves": [("Berlin", "Cape Town"), ("Sydney", "Stockholm"), ("Buenos Aires", "Cairo")]},
    {"name": "Swap Paris and Sydney",
     "moves": [("Paris", "Sydney"), ("Sydney", "Paris"), ("London", "Rio de Janeiro")]},
]


def draw_worlds(cities, n, rng):
    worlds, seen = [], set()
    while len(worlds) < n:
        idx = rng.choice(len(cities), 6, replace=False)
        src, tgt = [cities[i] for i in idx[:3]], [cities[i] for i in idx[3:]]
        if any(s["iso"] == t["iso"] for s, t in zip(src, tgt)):
            continue
        key = tuple((s["name"], t["name"]) for s, t in zip(src, tgt))
        if key in seen:
            continue
        seen.add(key)
        worlds.append([(s["name"], t["name"]) for s, t in zip(src, tgt)])
    return worlds


def scoreable(c):
    """Same rule as batch_edit_eval.py: a moved city is scored only if at least
    3 of its 5 ripple answers can tell source and target apart."""
    return "score_after" in c and c["score_after"]["n_countable"] >= 3


def summarise(results):
    cities = [c for r in results for c in r["cities"] if scoreable(c)]
    pairs = [p for r in results for p in r["pairs"] if p["before_correct"]]
    out = {
        "n_worlds": len(results), "n_moved_cities": len(cities),
        "city_pass_rate": float(np.mean([c["passes"] for c in cities])),
        "city_control_pass_rate": float(np.mean([
            c["score_control"]["n_countable"] >= 3 and c["score_control"]["n_flipped"] >= 0.8 * c["score_control"]["n_countable"]
            for c in cities])),
        # A world counts if every scoreable moved city in it passes.
        "world_all_cities_pass_rate": float(np.mean([all(c["passes"] for c in r["cities"] if scoreable(c))
                                                     for r in results])),
        "pair_questions": {},
    }
    # Move-mode measures (as in steer_heldout.py), for every city with a
    # countable hemisphere or season question: location moved = hemisphere or
    # season answer flipped; identity kept = country answer still the source's
    # country and the landmark answer shares a content word with the unedited one.
    locs = [c for r in results for c in r["cities"] if "score_after" in c and
            (c["score_after"]["per_question"]["hemisphere"]["countable"] or c["score_after"]["per_question"]["season"]["countable"])]

    def moved(c, kind):
        pq = c[kind]["per_question"]
        return pq["hemisphere"]["flipped"] or pq["season"]["flipped"]

    def kept(c, kind):
        pq = c[kind]["per_question"]
        lm_after = c["landmark"]["after" if kind == "score_after" else "control"]
        return pq["country"]["matches_source"] and identity_kept(c["landmark"]["before"], lm_after)

    out["location"] = {"n": len(locs),
                       "moved": float(np.mean([moved(c, "score_after") for c in locs])) if locs else None,
                       "identity_kept": float(np.mean([kept(c, "score_after") for c in locs])) if locs else None,
                       "moved_and_kept": float(np.mean([moved(c, "score_after") and kept(c, "score_after") for c in locs])) if locs else None,
                       "control_moved": float(np.mean([moved(c, "score_control") for c in locs])) if locs else None}
    for q in ("same_hemisphere", "further_south", "all"):
        sel = [p for p in pairs if q == "all" or p["key"] == q]
        scorable = [p for p in sel if p["key_after"] is not None]
        countable = [p for p in scorable if p["countable"]]
        out["pair_questions"][q] = {
            "n_scored": len(scorable),
            "after_consistent_rate": float(np.mean([p["after_correct"] for p in scorable])) if scorable else None,
            "control_matches_after_rate": float(np.mean([p["control_matches_after"] for p in scorable])) if scorable else None,
            "n_countable": len(countable),
            "countable_flip_rate": float(np.mean([p["after_correct"] for p in countable])) if countable else None,
            "countable_control_flip_rate": float(np.mean([p["control_matches_after"] for p in countable])) if countable else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--pad", action="store_true", help="pad shorter target names (tested, not adopted for scored questions)")
    ap.add_argument("--out", default=str(RESULTS / "rewrite_eval.json"))
    ap.add_argument("--menu-out", default=str(RESULTS / "rewrite_menu.json"))
    ap.add_argument("--summarise-only", action="store_true",
                    help="recompute the summary from the saved answers in results/rewrite_eval.json")
    args = ap.parse_args()
    if args.summarise_only:
        out = json.loads(Path(args.out).read_text())
        out["summary"] = {m: summarise(r) for m, r in out["worlds"].items()}
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(json.dumps({m: {k: v for k, v in sm.items() if k != "pair_questions"} for m, sm in out["summary"].items()}))
        return 0
    t0 = time.time()
    gate = json.loads((RESULTS / "gate.json").read_text())
    layers = band(*gate["best_band"])
    cities = [c for c in json.loads((RESULTS / "baseline_cities.json").read_text())["cities"] if c["all_correct"]]
    worlds = draw_worlds(cities, args.n, np.random.default_rng(args.seed))
    lm = get_lm()
    res = {"swap": [], "steer": []}
    for k, w in enumerate(worlds):
        for mode in ("swap", "steer"):
            res[mode].append(run_rewrite(lm, w, layers, mode=mode, all_tokens=True, pad=args.pad))
        sw = res["swap"][-1]
        print(f"[rewrite {k + 1}/{len(worlds)}] {w}: cities pass {[c.get('passes') for c in sw['cities']]}, "
              f"pairs consistent {sum(p['after_correct'] for p in sw['pairs'] if p['key_after'])}/"
              f"{sum(1 for p in sw['pairs'] if p['key_after'])} ({time.time() - t0:.0f}s)", flush=True)
    out = {"band": [layers[0], layers[-1]], "seed": args.seed, "pad": not args.no_pad,
           "patch": "every name token (swap); tier 2 (steer)",
           "summary": {m: summarise(r) for m, r in res.items()}, "worlds": res, "seconds": None}

    # Suggested worlds for the demo, cached with their free question.
    menu = []
    for wd in SUGGESTED:
        entry = {"name": wd["name"], "moves": [{"source": s, "target": t} for s, t in wd["moves"]], "results": {}}
        q = free_question([s for s, _ in wd["moves"]])
        for mode in ("swap", "steer"):
            r = run_rewrite(lm, wd["moves"], layers, mode=mode, all_tokens=True, pad=args.pad)
            if mode == "swap":
                plan = plan_edit_multi(lm, [q], wd["moves"], layers, all_tokens=True, pad=args.pad)
            else:
                from terra.steer import plan_steer_multi
                plan = plan_steer_multi(lm, [q], wd["moves"])
            b, a = generate_plans(lm, [[lm.chat(q)], plan], max_new_tokens=110)
            r["free"] = {"question": q, "before": b[0], "after": a[0]}
            entry["results"][mode] = r
        menu.append(entry)
        print(f"[suggested] {wd['name']}: {entry['results']['swap']['free']['after'][:100]!r}", flush=True)
    out["seconds"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    Path(args.menu_out).write_text(json.dumps({"band": [layers[0], layers[-1]], "pad": not args.no_pad, "worlds": menu},
                                              indent=2, ensure_ascii=False))
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
