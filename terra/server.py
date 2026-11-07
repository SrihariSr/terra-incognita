"""FastAPI backend for the demo.

Start with:
    uv run uvicorn terra.server:app --port 8000
then open http://localhost:8000

The model is loaded once at start-up and kept in memory. Every request that
touches the model goes through one lock, because a single GPU cannot run two
generations at once.

Endpoints (all JSON)
    GET  /api/health          model loaded? which band and probe layer?
    GET  /api/places          the ~10k believed positions for the globe
    GET  /api/menu            validated edit pairs, invented names, suggested worlds
    GET  /api/locate?name=    believed position of any name, with its cloud
    POST /api/edit            {source, target, mode} -> ripple answers before,
                              after, and under the random control, scored
    POST /api/ask             {question, source?, target?, moves?, mode?} -> free
                              answer before and after the edit(s)
    POST /api/rewrite         {moves: [{source, target}] (1 to 3), mode} -> every
                              move applied together: ripple answers per city and
                              questions about pairs of moved cities
    POST /api/company         (earlier experiment, not used by the page)
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import RESULTS, ROOT
from .data import city_info
from .locate import Locator
from .model import get_lm
from .patch import band, generate_plans, name_positions, plan_edit, plan_edit_all, stable_seed
from .patch import plan_edit_multi
from .rewrite import MAX_MOVES, run_rewrite
from .ripple import FREE_QUESTION, LANDMARK_QUESTION, QKEYS, make_key, prompts_for, score_edit

WEB = ROOT / "web"


@asynccontextmanager
async def lifespan(_app):
    """Load the model and every precomputed file once, before serving."""
    global S
    S = State()
    yield


app = FastAPI(title="TERRA INCOGNITA", lifespan=lifespan)


def _read(name: str, default=None):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else default


class State:
    """Everything loaded at start-up."""

    def __init__(self):
        t0 = time.time()
        self.lm = get_lm()
        self.locator = Locator(self.lm)
        gate = _read("gate.json")
        self.band = band(*gate["best_band"])
        # Patch method for city edits: every name token if results/multitoken_eval.json
        # adopted it and its validated menu exists, else the last token only (as gated).
        # TERRA_PATCH=last or TERRA_PATCH=all overrides.
        mt = _read("multitoken_eval.json", {})
        auto = "all" if mt.get("adopt") and (RESULTS / "pair_menu_alltoken.json").exists() else "last"
        self.method = os.environ.get("TERRA_PATCH", auto)
        self.planner = plan_edit_all if self.method == "all" else plan_edit
        menu_file = "pair_menu_alltoken.json" if self.method == "all" else "pair_menu.json"
        self.pair_menu = _read(menu_file, {"pairs": [], "cities": []})
        self.fictional = _read("fictional.json", {"menu": []})
        self.companies = _read("companies.json", {"companies": []})
        self.steer = _read("steer_eval.json", {"keep": False})
        base = _read("baseline_cities.json", {"cities": []})
        self.baseline = {c["name"]: c for c in base["cities"]}
        # Cache of finished edits so the demo is instant for menu pairs.
        self.edit_cache = {(p["source"], p["target"], "swap"): p for p in self.pair_menu.get("pairs", [])}
        for p in self.pair_menu.get("all_pairs", []):
            self.edit_cache.setdefault((p["source"], p["target"], "swap"), p)
        self.ask_cache = {(a["question"], a.get("source"), a.get("target"), "swap"): a
                          for a in self.pair_menu.get("free_answers", [])}
        self.locate_cache = {}
        # "Rewrite the world": suggested worlds precomputed by scripts/rewrite_eval.py.
        self.rewrite_menu = _read("rewrite_menu.json", {"worlds": []})
        self.rewrite_cache = {}
        for w in self.rewrite_menu.get("worlds", []):
            key = tuple((m["source"], m["target"]) for m in w["moves"])
            for mode, r in w.get("results", {}).items():
                self.rewrite_cache[(key, mode)] = r
                if "free" in r:
                    f = r["free"]
                    self.ask_cache[(f["question"], key, mode)] = {"question": f["question"], "before": f["before"],
                                                                  "after": f["after"], "moves": [list(k) for k in key]}
        print(f"[server] ready in {time.time() - t0:.1f}s; band {self.band[0]}-{self.band[-1]}; "
              f"patch method {self.method} ({menu_file})")


S: State | None = None


def key_for(name: str):
    info = city_info(name)
    if info is None:
        return None
    return make_key(name, info["iso"], info["latitude"])


# ------------------------------------------------------------------ reading

@app.get("/api/health")
def health():
    return {"ok": S is not None, "model": S.lm.model_id if S else None,
            "band": [S.band[0], S.band[-1]] if S else None,
            "probe_layer": S.locator.layer if S else None, "probe_mode": S.locator.mode if S else None,
            "patch_method": S.method if S else None}


@app.get("/api/places")
def places():
    return FileResponse(RESULTS / "places_believed.json", media_type="application/json")


@app.get("/api/menu")
def menu():
    return {
        "band": [S.band[0], S.band[-1]],
        "pairs": [{"source": p["source"], "target": p["target"]} for p in S.pair_menu.get("pairs", [])],
        "cities": S.pair_menu.get("cities", []),
        "fictional": S.fictional.get("menu", []),
        "worlds": [{"name": w["name"], "moves": w["moves"]} for w in S.rewrite_menu.get("worlds", [])],
        "max_moves": MAX_MOVES,
        "steer_available": bool(S.steer.get("keep")),
    }


@app.get("/api/locate")
def locate(name: str):
    name = name.strip()
    if not name or len(name) > 80:
        raise HTTPException(400, "name must be 1 to 80 characters")
    if name in S.locate_cache:
        return S.locate_cache[name]
    with S.lm.lock:
        r = S.locator.locate([name])[0]
    S.locate_cache[name] = r
    return r


# ------------------------------------------------------------------ editing

class EditReq(BaseModel):
    source: str
    target: str
    mode: str = "swap"   # "swap" (tier 1) or "steer" (tier 2, if validated)


def _run_edit(source: str, target: str, mode: str) -> dict:
    qs = prompts_for(source)
    texts = [qs[q] for q in QKEYS]
    if mode == "steer":
        from .steer import plan_move, random_like
        edit_plan = plan_move(S.lm, texts, source, target)
        # Control matched to the steering change (same layers and sizes).
        ctrl_plan = random_like(S.lm, edit_plan, stable_seed(source, target, "steer-control"))
    else:
        edit_plan = S.planner(S.lm, texts, source, target, S.band, mode="swap")
        ctrl_plan = S.planner(S.lm, texts, source, target, S.band, mode="control")
    plans = [[S.lm.chat(t) for t in texts], edit_plan]
    same_layers = edit_plan.layers == ctrl_plan.layers
    if same_layers:
        plans.append(ctrl_plan)
    outs = generate_plans(S.lm, plans, max_new_tokens=10)
    if not same_layers:
        outs.append(generate_plans(S.lm, [ctrl_plan], max_new_tokens=10)[0])
    before, after, control = (dict(zip(QKEYS, o)) for o in outs)
    return {"source": source, "target": target, "mode": mode,
            "before": before, "after": after, "control": control}


def _edited_location(source: str, target: str, mode: str) -> dict:
    """Where the location probe reads `source` *after* the edit.

    We build the same patch for the probe's own prompt ("What are the lat/lon
    coordinates of Paris"), run it with the patch applied, and read the probe
    layer at the name token. This is a measurement, not an animation target
    chosen by hand.
    """
    from .activations import TEMPLATES
    if mode != "swap":
        return {}  # only measured for the tier-1 swap
    text = TEMPLATES[S.locator.mode].format(name=source)
    plan = S.planner(S.lm, [text], source, target, S.band, mode="swap", wrap=False)
    pos = [[c for r, c in zip(plan.patch.rows, plan.patch.cols) if r == 0]]
    h = S.lm.residuals_multi(plan.source_texts, pos, [S.locator.layer], patch=plan.patch)[0][:, 0]
    loc = S.locator.from_hidden(h.numpy().astype("float64"))[-1]
    return {"lat": loc["lat"], "lon": loc["lon"], "nearest": loc["nearest"], "measured_for": "swap"}


def _landmark(source: str, target: str, mode: str) -> dict:
    """Unscored identity check: the landmark answer before, after and under control."""
    q = [LANDMARK_QUESTION.format(city=source)]
    if mode == "steer":
        from .steer import plan_move, random_like
        ed = plan_move(S.lm, q, source, target)
        ct = random_like(S.lm, ed, stable_seed(source, target, "steer-control"))
    else:
        ed = S.planner(S.lm, q, source, target, S.band, mode="swap")
        ct = S.planner(S.lm, q, source, target, S.band, mode="control")
    plans = [[S.lm.chat(q[0])], ed] + ([ct] if ed.layers == ct.layers else [])
    outs = generate_plans(S.lm, plans, max_new_tokens=12)
    if ed.layers != ct.layers:
        outs.append(generate_plans(S.lm, [ct], max_new_tokens=12)[0])
    return {"before": outs[0][0], "after": outs[1][0], "control": outs[2][0]}


@app.post("/api/edit")
def edit(req: EditReq):
    src, tgt = req.source.strip(), req.target.strip()
    if not src or not tgt or src == tgt:
        raise HTTPException(400, "need two different names")
    cached = S.edit_cache.get((src, tgt, req.mode))
    if cached is None:
        with S.lm.lock:
            t0 = time.time()
            cached = _run_edit(src, tgt, req.mode)
            cached["seconds"] = time.time() - t0
        S.edit_cache[(src, tgt, req.mode)] = cached
    if "landmark" not in cached:
        with S.lm.lock:
            cached["landmark"] = _landmark(src, tgt, req.mode)
    res = dict(cached)
    try:
        with S.lm.lock:
            res["edited_location"] = _edited_location(src, tgt, req.mode)
    except Exception as exc:  # the readout is a nice-to-have; never fail the edit for it
        res["edited_location_error"] = str(exc)
    sk, tk = key_for(src), key_for(tgt)
    if sk and tk:
        res["keys"] = {"source": sk.main, "target": tk.main}
        res["score_after"] = score_edit(res["after"], sk, tk)
        res["score_control"] = score_edit(res["control"], sk, tk)
        res["score_before"] = score_edit(res["before"], sk, tk)
    for who, name in (("source_info", src), ("target_info", tgt)):
        info = city_info(name)
        if info:
            res[who] = info
    res["band"] = [S.band[0], S.band[-1]]
    return res


class Move(BaseModel):
    source: str
    target: str


class AskReq(BaseModel):
    question: str
    source: str | None = None
    target: str | None = None
    moves: list[Move] | None = None    # "Rewrite the world": several moves at once
    mode: str = "swap"
    max_new_tokens: int = 80


@app.post("/api/ask")
def ask(req: AskReq):
    q = req.question.strip()
    if not q or len(q) > 400:
        raise HTTPException(400, "question must be 1 to 400 characters")
    n_tok = max(8, min(req.max_new_tokens, 160))
    if req.moves:
        return _ask_multi(q, [(m.source.strip(), m.target.strip()) for m in req.moves], req.mode, n_tok)
    ck = (q, req.source, req.target, req.mode)
    if ck in S.ask_cache:
        return S.ask_cache[ck]
    res = {"question": q, "source": req.source, "target": req.target}
    with S.lm.lock:
        t0 = time.time()
        before_text = [S.lm.chat(q)]
        if req.source and req.target:
            if not name_positions(S.lm, S.lm.chat(q), req.source):
                res["note"] = f"The question does not mention {req.source}, so the edit has nothing to act on."
                out = generate_plans(S.lm, [before_text], max_new_tokens=n_tok)
                res["before"] = res["after"] = out[0][0]
            else:
                if req.mode == "steer":
                    from .steer import plan_move
                    plan = plan_move(S.lm, [q], req.source, req.target)
                else:
                    plan = S.planner(S.lm, [q], req.source, req.target, S.band, mode="swap")
                out = generate_plans(S.lm, [before_text, plan], max_new_tokens=n_tok)
                res["before"], res["after"] = out[0][0], out[1][0]
        else:
            res["before"] = generate_plans(S.lm, [before_text], max_new_tokens=n_tok)[0][0]
        res["seconds"] = time.time() - t0
    S.ask_cache[ck] = res
    return res


def _ask_multi(q: str, moves: list[tuple[str, str]], mode: str, n_tok: int) -> dict:
    """Free question with every move of a rewritten world applied together."""
    from .patch import multi_substitute
    key = tuple(moves)
    ck = (q, key, mode)
    if ck in S.ask_cache:
        return S.ask_cache[ck]
    res = {"question": q, "moves": [list(m) for m in moves], "mode": mode}
    with S.lm.lock:
        t0 = time.time()
        if not multi_substitute(q, moves)[1]:
            res["note"] = "The question mentions none of the moved cities, so the edits have nothing to act on."
            out = generate_plans(S.lm, [[S.lm.chat(q)]], max_new_tokens=n_tok)
            res["before"] = res["after"] = out[0][0]
        else:
            if mode == "steer":
                from .steer import plan_steer_multi
                plan = plan_steer_multi(S.lm, [q], moves)
            else:
                plan = plan_edit_multi(S.lm, [q], moves, S.band, mode="swap", all_tokens=S.method == "all", pad=True)
            out = generate_plans(S.lm, [[S.lm.chat(q)], plan], max_new_tokens=n_tok)
            res["before"], res["after"] = out[0][0], out[1][0]
        res["seconds"] = time.time() - t0
    S.ask_cache[ck] = res
    return res


class RewriteReq(BaseModel):
    moves: list[Move]
    mode: str = "swap"


@app.post("/api/rewrite")
def rewrite(req: RewriteReq):
    """Apply up to three moves together; ripple and pair questions before/after/control."""
    moves = [(m.source.strip(), m.target.strip()) for m in req.moves]
    if not 1 <= len(moves) <= MAX_MOVES:
        raise HTTPException(400, f"give between 1 and {MAX_MOVES} moves")
    if len({s for s, _ in moves}) != len(moves) or any(s == t or not s or not t for s, t in moves):
        raise HTTPException(400, "each city can be moved once, onto a different city")
    key = (tuple(moves), req.mode)
    if key not in S.rewrite_cache:
        with S.lm.lock:
            t0 = time.time()
            r = run_rewrite(S.lm, moves, S.band, mode=req.mode, all_tokens=S.method == "all")
            r["seconds"] = time.time() - t0
        S.rewrite_cache[key] = r
    res = dict(S.rewrite_cache[key])
    res["info"] = {n: city_info(n) for m in moves for n in m if city_info(n)}
    return res


class CompanyReq(BaseModel):
    company: str
    target_city: str


@app.post("/api/company")
def company(req: CompanyReq):
    from .companies import run_company_edit
    with S.lm.lock:
        t0 = time.time()
        res = run_company_edit(S.lm, S.band, req.company, req.target_city)
        res["seconds"] = time.time() - t0
    return res


@app.get("/api/rigour")
def rigour():
    """Headline numbers, each read from its results file (never typed in)."""
    R = {}
    pf = _read("probe_final.json")
    if pf:
        R["probe"] = {"random_median_km": pf["heldout"]["random"]["median_km"],
                      "country_median_km": pf["heldout"]["blocked_country"]["median_km"],
                      "cell_median_km": pf["heldout"]["blocked_cell"]["median_km"],
                      "baseline_median_km": pf.get("baseline_mean_direction", {}).get("median_km"),
                      "layer": pf["layer"], "n_places": pf["n_places"]}
    # Headline edit numbers for the method the demo uses: the all-token method
    # is reported on fresh pairs it was never looked at on.
    ev = _read("edit_eval_fresh_alltoken.json") if S.method == "all" else None
    ev = ev or _read("edit_eval.json")
    if ev:
        R["edit_source"] = "results/edit_eval_fresh_alltoken.json" if ev.get("method") == "all" else "results/edit_eval.json"
        R["edit_method"] = ev.get("method", "last")
        R["edit"] = {"cross_pass": ev["sets"]["cross"]["edit"]["pass_rate"],
                     "cross_ctrl_pass": ev["sets"]["cross"]["control"]["pass_rate"],
                     "cross_n": ev["sets"]["cross"]["edit"]["n_pairs"],
                     "any_pass": ev["sets"]["any"]["edit"]["pass_rate"],
                     "any_ctrl_pass": ev["sets"]["any"]["control"]["pass_rate"],
                     "any_n": ev["sets"]["any"]["edit"]["n_pairs"]}
    fi = _read("fictional.json")
    if fi:
        R["fiction"] = {"n_coherent": fi["n_coherent"], "n_scored": fi["n_scored"],
                        "chance": fi.get("chance_rate_mean")}
    rw = _read("rewrite_eval.json")
    if rw:
        sw = rw["summary"]["swap"]
        R["rewrite"] = {"n_worlds": sw["n_worlds"], "city_pass": sw["city_pass_rate"],
                        "city_ctrl_pass": sw["city_control_pass_rate"], "world_pass": sw["world_all_cities_pass_rate"],
                        "pair_consistent": sw["pair_questions"]["all"]["after_consistent_rate"],
                        "pair_ctrl": sw["pair_questions"]["all"]["control_matches_after_rate"],
                        "pair_n": sw["pair_questions"]["all"]["n_scored"]}
    sh = _read("steer_heldout.json")
    if sh:
        R["steer"] = {"n": sh["n_pairs"], "moved": sh["rates"]["steer"]["moved"],
                      "kept": sh["rates"]["steer"]["identity_kept"], "both": sh["rates"]["steer"]["moved_and_kept"],
                      "ctrl_moved": sh["rates"]["control"]["moved"], "swap_kept": sh["rates"]["swap"]["identity_kept"]}
    hc = _read("hemisphere_check.json")
    if hc:
        R["hemisphere"] = {"spoken": hc["spoken_hemisphere_accuracy"], "probe": hc["probe_hemisphere_accuracy"],
                           "n": hc["n_cities"], "n_wrong": hc["n_spoken_wrong"],
                           "n_wrong_probe_right": hc["n_spoken_wrong_but_probe_right"],
                           "examples": [{k: e[k] for k in ("name", "true_lat", "probe_lat", "said")}
                                        for e in hc["examples"] if e["probe_correct"]][:3]}
    figs = RESULTS / "figures"
    order = ["how_it_works.png", "edit_vs_control.png", "rewrite_world.png", "swap_vs_move.png",
             "probe_by_layer.png", "band_sweep.png", "error_map.png", "believed_map.png"]
    have = {p.name for p in figs.glob("*.png")} if figs.exists() else set()
    R["figures"] = [f for f in order if f in have] + sorted(have - set(order))
    return R


@app.get("/api/free_question")
def free_question(city: str = "Paris"):
    return {"question": FREE_QUESTION.format(city=city)}


@app.exception_handler(Exception)
def _err(request, exc):
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


# Static files last, so /api routes win.
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=RESULTS / "figures"), name="figures")
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
