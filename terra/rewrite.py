"""Rewrite the world: move up to three cities at once and check the result.

The judge picks moves such as Paris -> Sydney, Tokyo -> Cape Town,
London -> Buenos Aires. All edits are applied together in one patched
forward pass (terra.patch.plan_edit_multi, or plan_steer_multi for the
"move, keep identity" mode). We then ask:

* the five ripple questions and the landmark question for every moved city;
* two questions for every pair of moved cities, whose right answer depends on
  *both* edits:
    - "Are Paris and Tokyo in the same hemisphere?"   (yes / no)
    - "Which city is further south, Paris or Tokyo?"  (one of the two)

Answer keys for the pair questions come from latitudes (GeoNames): before
the edit from the real cities, after the edit from the cities they were
moved onto. A pair question is *countable* only if the edit changes its
right answer, and "further south" is only scored when the two latitudes
differ by at least MIN_LAT_GAP degrees (closer than that, the question is
too fine for a fair test).

Under the swap edit the model may name the target city ("Sydney") when asked
about "Paris"; for "further south" an answer naming either a moved city or
the city it was moved onto counts as choosing that slot.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Sequence

from .data import city_info
from .backends.base import BaseLM as LM
from .patch import generate_plans, plan_edit_multi, stable_seed
from .ripple import LANDMARK_QUESTION, QKEYS, _mentions, make_key, prompts_for, score_edit

MAX_MOVES = 3
MIN_LAT_GAP = 5.0

PAIR_QUESTIONS = {
    "same_hemisphere": "Are {a} and {b} in the same hemisphere (northern or southern)? Answer yes or no.",
    "further_south": "Which city is further south, {a} or {b}? Answer with just the city name.",
}

FREE_TEMPLATE = {
    1: "I'm visiting {0} this July. What should I pack? Answer in two sentences.",
    2: "I'm visiting {0} and then {1} this July. What should I pack for each city? Answer in two short sentences.",
    3: "I'm visiting {0}, {1} and {2} this July. What should I pack for each city? Answer in three short sentences.",
}


def free_question(sources: Sequence[str]) -> str:
    return FREE_TEMPLATE[len(sources)].format(*sources)


UNSCORED = "no answer key for invented places: answers shown unscored"


def lat(name: str) -> float | None:
    """True latitude from GeoNames, or None for a name it does not know (invented places)."""
    info = city_info(name)
    return None if info is None else info["latitude"]


def pair_key(q: str, a_lat: float | None, b_lat: float | None):
    """Right answer for a pair question given the two latitudes ("a"/"b" for slots).

    None (unscored) when either place has no known latitude.
    """
    if a_lat is None or b_lat is None:
        return None
    if q == "same_hemisphere":
        return "yes" if (a_lat > 0) == (b_lat > 0) else "no"
    if abs(a_lat - b_lat) < MIN_LAT_GAP:
        return None
    return "a" if a_lat < b_lat else "b"


def read_pair_answer(q: str, answer: str, a_names: list[str], b_names: list[str]):
    """Parse a model answer into "yes"/"no" or slot "a"/"b" (None if unclear)."""
    t = answer.lower()
    if q == "same_hemisphere":
        y = re.search(r"\byes\b", t) is not None
        n = re.search(r"\bno\b", t) is not None
        return "yes" if y and not n else "no" if n and not y else None
    ha, hb = _mentions(answer, a_names), _mentions(answer, b_names)
    return "a" if ha and not hb else "b" if hb and not ha else None


def build_texts(moves: Sequence[tuple[str, str]]):
    """All prompts for one rewritten world, with labels to unpack the answers."""
    texts, labels = [], []
    for i, (s, _) in enumerate(moves):
        qs = prompts_for(s)
        for q in QKEYS:
            texts.append(qs[q])
            labels.append(("ripple", i, q))
        texts.append(LANDMARK_QUESTION.format(city=s))
        labels.append(("landmark", i, None))
    for i, j in combinations(range(len(moves)), 2):
        for q, tmpl in PAIR_QUESTIONS.items():
            texts.append(tmpl.format(a=moves[i][0], b=moves[j][0]))
            labels.append(("pair", (i, j), q))
    return texts, labels


def run_rewrite(lm: LM, moves: Sequence[tuple[str, str]], layers: Sequence[int], mode: str = "swap",
                all_tokens: bool = True, pad: bool = False, paths=None, target_latlons: dict | None = None) -> dict:
    """Before / after / control answers for a whole rewritten world.

    Moves may use any names. A city with no GeoNames answer key (an invented
    place) gets its answers unscored; pair questions involving it too.
    paths / target_latlons are only used by the steering mode (per-model
    probe files, and believed positions for invented targets).
    """
    moves = [(s.strip(), t.strip()) for s, t in moves]
    if not 1 <= len(moves) <= MAX_MOVES:
        raise ValueError(f"between 1 and {MAX_MOVES} moves")
    if len({s for s, _ in moves}) != len(moves):
        raise ValueError("each city can only be moved once")
    if any(s == t or not s or not t for s, t in moves):
        raise ValueError("source and target must differ and not be empty")
    texts, labels = build_texts(moves)
    if mode == "steer":
        from .steer import plan_steer_multi, random_like
        edit = plan_steer_multi(lm, texts, moves, paths=paths, target_latlons=target_latlons)
        ctrl = random_like(lm, edit, stable_seed(*[f"{s}>{t}" for s, t in moves], "steer-control"))
    else:
        edit = plan_edit_multi(lm, texts, moves, layers, mode="swap", all_tokens=all_tokens, pad=pad)
        ctrl = plan_edit_multi(lm, texts, moves, layers, mode="control", all_tokens=all_tokens, pad=pad)
    plain = [lm.chat(t) for t in texts]
    if edit.layers == ctrl.layers:
        before, after, control = generate_plans(lm, [plain, edit, ctrl], max_new_tokens=12)
    else:
        before, after = generate_plans(lm, [plain, edit], max_new_tokens=12)
        control = generate_plans(lm, [ctrl], max_new_tokens=12)[0]

    cities = []
    for i, (s, t) in enumerate(moves):
        rec = {"source": s, "target": t, "before": {}, "after": {}, "control": {}}
        cities.append(rec)
    pairs = {}
    for k, (kind, idx, q) in enumerate(labels):
        if kind == "ripple":
            for name, out in (("before", before), ("after", after), ("control", control)):
                cities[idx][name][q] = out[k]
        elif kind == "landmark":
            cities[idx]["landmark"] = {"before": before[k], "after": after[k], "control": control[k]}
        else:
            i, j = idx
            pairs.setdefault((i, j), {})[q] = {"question": texts[k], "before": before[k], "after": after[k],
                                               "control": control[k]}
    for rec in cities:
        si, ti = city_info(rec["source"]), city_info(rec["target"])
        if si and ti:
            sk = make_key(rec["source"], si["iso"], si["latitude"])
            tk = make_key(rec["target"], ti["iso"], ti["latitude"])
            rec["keys"] = {"source": sk.main, "target": tk.main}
            rec["score_after"] = score_edit(rec["after"], sk, tk)
            rec["score_control"] = score_edit(rec["control"], sk, tk)
            sc = rec["score_after"]
            rec["passes"] = sc["n_countable"] >= 3 and sc["n_flipped"] >= 0.8 * sc["n_countable"]
            rec["scored"] = True
        else:
            rec["scored"] = False
            rec["unscored_reason"] = UNSCORED
    pair_rows = []
    for (i, j), qs in pairs.items():
        (sa, ta), (sb, tb) = moves[i], moves[j]
        for q, rec in qs.items():
            kb = pair_key(q, lat(sa), lat(sb))
            ka = pair_key(q, lat(ta), lat(tb))
            names_a, names_b = [sa, ta], [sb, tb]
            row = {"key": q, "a": sa, "b": sb, "a_to": ta, "b_to": tb, **rec,
                   "key_before": kb, "key_after": ka,
                   "countable": kb is not None and ka is not None and kb != ka,
                   "read_before": read_pair_answer(q, rec["before"], names_a, names_b),
                   "read_after": read_pair_answer(q, rec["after"], names_a, names_b),
                   "read_control": read_pair_answer(q, rec["control"], names_a, names_b)}
            row["scored"] = ka is not None
            row["before_correct"] = kb is not None and row["read_before"] == kb
            row["after_correct"] = ka is not None and row["read_after"] == ka
            row["control_says_before"] = kb is not None and row["read_control"] == kb
            row["control_matches_after"] = ka is not None and row["read_control"] == ka
            pair_rows.append(row)
    return {"moves": [{"source": s, "target": t} for s, t in moves], "mode": mode,
            "cities": cities, "pairs": pair_rows, "free_question": free_question([s for s, _ in moves]),
            "scored": all(c["scored"] for c in cities)}
