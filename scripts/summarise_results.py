"""Collect the headline numbers from results/*.json into results/summary.md.

Every number quoted in STATUS.md, README.md, DEVPOST.md or on stage should be
copied from this table, which is generated from the results files (never
typed by hand).

    uv run python scripts/summarise_results.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terra import RESULTS  # noqa: E402


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def main() -> int:
    rows = []

    def add(what, value, src):
        rows.append((what, value, src))

    g = load("gate.json")
    if g:
        add("Gate: pairs flipping >= 4/5 ripple answers (best band)", f"{g['edit_pairs_pass']}/{len(g['pairs'])}", "gate.json")
        add("Gate: random control pairs flipping >= 4/5", f"{g['control_pairs_pass']}/{len(g['pairs'])}", "gate.json")
        add("Gate: chosen layer band", f"{g['best_band'][0]} to {g['best_band'][1]}", "gate.json")
        add("Gate decision", g["decision"], "gate.json")
    pl = load("probe_layers.json")
    if pl:
        c = pl["chosen"]
        add("3k sweep: chosen prompt and layer", f"{c['mode']}, layer {c['layer']}", "probe_layers.json")
        add("3k sweep: median error, random split", f"{c['random']['median_km']:,.0f} km", "probe_layers.json")
        add("3k sweep: median error, countries held out", f"{c['blocked_country']['median_km']:,.0f} km", "probe_layers.json")
        for m in pl["modes"]:
            b = pl[f"best_{m}"]
            add(f"3k sweep: best '{m}' prompt (countries held out)", f"{b['blocked_country_median_km']:,.0f} km at layer {b['layer']}", "probe_layers.json")
    pf = load("probe_final.json")
    if pf:
        h = pf["heldout"]
        add("Final probe: places", f"{pf['n_places']:,}", "probe_final.json")
        add("Final probe: prompt and layer", f"{pf['mode']}, layer {pf['layer']}", "probe_final.json")
        add("Final probe: median error, random 5-fold", f"{h['random']['median_km']:,.0f} km", "probe_final.json")
        add("Final probe: median error, countries held out", f"{h['blocked_country']['median_km']:,.0f} km", "probe_final.json")
        add("Final probe: median error, 10-degree cells held out", f"{h['blocked_cell']['median_km']:,.0f} km", "probe_final.json")
        add("Final probe: median error, large cities, random 5-fold", f"{h['random_big_cities']['median_km']:,.0f} km", "probe_final.json")
        add("Final probe: median error, large cities, countries held out", f"{h['blocked_country_big_cities']['median_km']:,.0f} km", "probe_final.json")
        add("Final probe: within 500 km (random split)", pct(h["random"]["within_500km"]), "probe_final.json")
        if "baseline_mean_direction" in pf:
            add("Baseline: always guess the average position", f"{pf['baseline_mean_direction']['median_km']:,.0f} km", "probe_final.json")
    bc = load("baseline_cities.json")
    if bc:
        add("Cities the unedited model answers all 5 correctly", f"{bc['n_all_correct']}/{bc['n_cities']}", "baseline_cities.json")
        add("Unedited hemisphere accuracy over those cities", pct(bc["per_question_accuracy"]["hemisphere"]), "baseline_cities.json")
    ev = load("edit_eval.json")
    if ev:
        for s, lab in (("cross", "opposite hemispheres"), ("any", "any two countries")):
            e, c = ev["sets"][s]["edit"], ev["sets"][s]["control"]
            add(f"Edit success, random pairs, {lab} (n={e['n_pairs']})", pct(e["pass_rate"]), "edit_eval.json")
            add(f"Random control success, same pairs, {lab}", pct(c["pass_rate"]), "edit_eval.json")
            add(f"Mean share of answers flipped, edit, {lab}", pct(e["mean_fraction_flipped"]), "edit_eval.json")
            add(f"Mean share of answers flipped, control, {lab}", pct(c["mean_fraction_flipped"]), "edit_eval.json")
        cq = ev["sets"]["cross"]["control"]["per_question"]
        add("Control: per-question flip rate, opposite hemispheres",
            ", ".join(f"{q} {pct(v['flip_rate'])}" for q, v in cq.items()), "edit_eval.json")
        eq = ev["sets"]["cross"]["edit"]["per_question"]
        add("Edit: per-question flip rate, opposite hemispheres",
            ", ".join(f"{q} {pct(v['flip_rate'])}" for q, v in eq.items()), "edit_eval.json")
    mt = load("multitoken_eval.json")
    if mt:
        for sn, lab in (("cross", "opposite hemispheres"), ("any", "any two countries")):
            v = mt["sets"][sn]
            add(f"All-token vs last-token patch, same pairs as edit_eval, {lab} (n={v['n']})",
                f"{pct(v['all_token_pass_rate'])} vs {pct(v['last_token_pass_rate'])} (all-token control {pct(v['all_token_control_pass_rate'])})",
                "multitoken_eval.json")
        add("All-token patching adopted by the pre-set rule", "yes" if mt["adopt"] else "no", "multitoken_eval.json")
    fa = load("edit_eval_fresh_alltoken.json")
    if fa:
        cq = fa["sets"]["cross"]["control"]["per_question"]
        add("Control (all-token, fresh pairs): per-question flip rate, opposite hemispheres",
            ", ".join(f"{q} {pct(v['flip_rate'])}" for q, v in cq.items()), "edit_eval_fresh_alltoken.json")
    for meth in ("alltoken", "lasttoken"):
        fe = load(f"edit_eval_fresh_{meth}.json")
        if fe:
            for sn, lab in (("cross", "opposite hemispheres"), ("any", "any two countries")):
                e, c = fe["sets"][sn]["edit"], fe["sets"][sn]["control"]
                add(f"Fresh pairs (seed {fe['seed']}), {meth.replace('token', '-token')} swap, {lab} (n={e['n_pairs']})",
                    f"{pct(e['pass_rate'])} (control {pct(c['pass_rate'])})", f"edit_eval_fresh_{meth}.json")
    hc = load("hemisphere_check.json")
    if hc:
        add("Hemisphere: model's spoken answer correct", pct(hc["spoken_hemisphere_accuracy"]) + f" of {hc['n_cities']} cities", "hemisphere_check.json")
        add("Hemisphere: probe (out-of-fold) puts city in the right hemisphere", pct(hc["probe_hemisphere_accuracy"]), "hemisphere_check.json")
        add("Hemisphere: spoken wrong but probe right", f"{hc['n_spoken_wrong_but_probe_right']}/{hc['n_spoken_wrong']}", "hemisphere_check.json")
    for name in ("pair_menu.json", "pair_menu_alltoken.json"):
        pm = load(name)
        if pm:
            meth = "every name token" if pm.get("method") == "all" else "last name token"
            add(f"Demo pairs validated, {meth} (edit works, control does not)", f"{pm['n_validated']}/{pm['n_pairs']}", name)
    fi = load("fictional.json")
    if fi:
        add("Invented names landing in the region their sound suggests", f"{fi['n_coherent']}/{fi['n_scored']}", "fictional.json")
        add("Same, chance level (shuffled expectations)", pct(fi.get("chance_rate_mean")), "fictional.json")
    co = load("companies.json")
    if co:
        add("(Earlier experiment, not in the demo) Companies with consistent Wikidata HQ", f"{co['n_consistent']}/{co['n_companies']}", "companies.json")
        add("(Earlier experiment) " + "Companies whose HQ country the model knows", f"{co['n_known']}/{co['n_consistent']}", "companies.json")
        add("(Earlier experiment) " + "Probe error on company names vs Wikidata HQ (median)", f"{co['probe_hq_error_km']['median']:,.0f} km", "companies.json")
        for k in ("city", "context", "control"):
            sm = co["summary"][k]
            add(f"(Earlier experiment) Company edit '{k}': pass rate (n={sm['n']})", pct(sm["pass_rate"]), "companies.json")
            add(f"(Earlier experiment) Company edit '{k}': identity kept", pct(sm["identity_kept_rate"]), "companies.json")
        add("(Earlier experiment) Company edit variant chosen", co["chosen_variant"], "companies.json")
        # Belief audit: unedited answers that differ from the home-country key.
        non_us = [c for c in co["companies"] if c.get("known") and c["iso"] != "US"]
        fed = [c["name"] for c in non_us if "federal reserve" in c["before"]["central_bank"].lower()]
        usd = [c["name"] for c in non_us
               if re.search(r"\b(us|u\.s\.|united states|american)\b", c["before"]["currency"].lower())
               and "dollar" in c["before"]["currency"].lower()]
        add("(Earlier experiment) Non-US companies the model ties to the Federal Reserve (unedited)", f"{len(fed)}/{len(non_us)}: {', '.join(fed)}", "companies.json")
        add("(Earlier experiment) Non-US companies the model says report in US dollars (unedited)", f"{len(usd)}/{len(non_us)}: {', '.join(usd)}", "companies.json")
    rw = load("rewrite_eval.json")
    if rw:
        for mode, lab in (("swap", "swap"), ("steer", "move (tier 2)")):
            sm = rw["summary"][mode]
            pq = sm["pair_questions"]
            add(f"Rewrite the world ({lab}), {sm['n_worlds']} random 3-move worlds: moved cities fully moved (control)",
                f"{pct(sm['city_pass_rate'])} ({pct(sm['city_control_pass_rate'])}) of {sm['n_moved_cities']}", "rewrite_eval.json")
            add(f"Rewrite the world ({lab}): worlds where all three cities fully moved", pct(sm["world_all_cities_pass_rate"]), "rewrite_eval.json")
            add(f"Rewrite the world ({lab}): pair answers consistent with the rewritten world (control)",
                f"{pct(pq['all']['after_consistent_rate'])} ({pct(pq['all']['control_matches_after_rate'])}), n={pq['all']['n_scored']}",
                "rewrite_eval.json")
            add(f"Rewrite the world ({lab}): pair answers that should change and do (control)",
                f"{pct(pq['all']['countable_flip_rate'])} ({pct(pq['all']['countable_control_flip_rate'])}), n={pq['all']['n_countable']}",
                "rewrite_eval.json")
            for q in ("same_hemisphere", "further_south"):
                add(f"Rewrite the world ({lab}): '{q}' consistent / should-change-and-does",
                    f"{pct(pq[q]['after_consistent_rate'])} (n={pq[q]['n_scored']}) / {pct(pq[q]['countable_flip_rate'])} (n={pq[q]['n_countable']})",
                    "rewrite_eval.json")
    cs = load("companies_steer.json")
    if cs:
        for sc, v in cs["summary"].items():
            add(f"(Earlier experiment) Company steering, scale {float(sc):g}: pass / identity kept / both (n={v['n']})",
                f"{pct(v['pass_rate'])} / {pct(v['identity_kept_rate'])} / {pct(v['pass_and_kept'])}", "companies_steer.json")
    st = load("steer_eval.json")
    if st:
        b = st["best"]
        add("Tier 2 steering: best setting", f"layers {b['band'][0]}-{b['band'][1]}, scale {b['scale']:g}", "steer_eval.json")
        add("Tier 2: pairs moved (hemisphere or season) with landmark kept", f"{b['n_moved_and_kept']}/{len(st['pairs'])}", "steer_eval.json")
        add("Tier 2: same for random control", f"{st['control_moved_and_kept']}/{len(st['pairs'])}", "steer_eval.json")
        add("Tier 1 swap: landmark kept", f"{st['swap_landmark_kept']}/{len(st['pairs'])}", "steer_eval.json")
        add("Tier 2 kept in demo", "yes" if st["keep"] else "no", "steer_eval.json")
    sh = load("steer_heldout.json")
    if sh:
        r = sh["rates"]
        for k, lab in (("steer", "tier 2 steering"), ("swap", "tier 1 swap"), ("control", "random control")):
            add(f"Held-out {sh['n_pairs']} pairs, {lab}: location moved / identity kept / both",
                f"{pct(r[k]['moved'])} / {pct(r[k]['identity_kept'])} / {pct(r[k]['moved_and_kept'])}", "steer_heldout.json")
    uc = load("ui_check.json")
    if uc:
        add("Browser check of the page", f"{uc['n_pass']}/{uc['n_checks']} checks pass ({uc['time']})", "ui_check.json")
    dc = load("demo_check.json")
    if dc:
        add("Demo check", f"{dc['n_pass']}/{dc['n_checks']} checks pass ({dc['time']})", "demo_check.json")

    lines = ["# Results summary", "", "Generated by `scripts/summarise_results.py` from the files in `results/`.", "",
             "| Measure | Value | Source |", "|---|---|---|"]
    lines += [f"| {a} | {b} | `results/{c}` |" for a, b, c in rows]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
