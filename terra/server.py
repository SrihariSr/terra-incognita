"""FastAPI backend for the demo.

Start with:
    uv run uvicorn terra.server:app --port 8000                       # Qwen
    TERRA_MODEL=gemma4-31b uv run uvicorn terra.server:app --port 8000
then open http://localhost:8000

One model is loaded at a time and kept in memory. Every request that touches
the model goes through one lock, because a single GPU cannot run two
generations at once. Switching models (POST /api/model) unloads the current
one completely before loading the next, so two models are never in memory
together; while it happens, model routes answer 503 "switching models".

Endpoints (all JSON)
    GET  /api/health          what is loaded, its probe layer and edit band
    GET  /api/models          every model: built? editable? loaded?
    POST /api/model           {key} switch model
    GET  /api/places          the ~10k believed positions for the globe
    GET  /api/menu            validated pairs, menu cities, invented names, worlds
    GET  /api/locate?name=    believed position of any name, with its cloud
    POST /api/edit            {source, target, mode} -> ripple answers before,
                              after and under the random control (any names;
                              names without an answer key are shown unscored)
    POST /api/ask             {question, source?, target?, moves?, mode?}
    POST /api/rewrite         {moves: [{source, target}] (1 to 3), mode}
    GET  /api/political       the model's political map and its statistics
    GET  /api/rigour          headline numbers, each from a results file
    GET  /figures/<name>      the loaded model's figures
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ROOT, ollama, registry
from .activations import template_text
from .data import city_info
from .locate import Locator
from .model import get_lm, unload_lm
from .patch import AlignmentError, band, generate_plans, name_positions, plan_edit, plan_edit_all, stable_seed
from .patch import plan_edit_multi
from .rewrite import MAX_MOVES, UNSCORED, run_rewrite
from .ripple import FREE_QUESTION, LANDMARK_QUESTION, QKEYS, make_key, prompts_for, score_edit

WEB = ROOT / "web"


# ----------------------------------------------------------------- state

class ModelState:
    """Everything that belongs to one loaded model: the backend, its probe,
    its menus and every cache. A new object per model, so caches never mix."""

    def __init__(self, key: str, lm):
        t0 = time.time()
        self.key = key
        self.spec = registry.spec(key)
        self.paths = registry.paths(key)
        self.lm = lm
        self.locator = Locator(lm, path=self.paths.probe)
        gate = self.read("gate.json")
        self.editable = gate is not None
        self.band = band(*gate["best_band"]) if gate else None
        # Patch method for city edits: every name token if the all-token menu
        # exists (for Qwen, also if results/multitoken_eval.json adopted it),
        # else the last token only (as gated). TERRA_PATCH=last|all overrides.
        mt = self.read("multitoken_eval.json", {})
        auto = "all" if mt.get("adopt", True) and self.paths.result("pair_menu_alltoken.json").exists() else "last"
        self.method = os.environ.get("TERRA_PATCH", auto)
        self.planner = plan_edit_all if self.method == "all" else plan_edit
        menu_file = "pair_menu_alltoken.json" if self.method == "all" else "pair_menu.json"
        self.pair_menu = self.read(menu_file, {"pairs": [], "cities": []})
        self.fictional = self.read("fictional.json", {"menu": []})
        self.steer = self.read("steer_eval.json", {"keep": False})
        self.steer_ok = bool(self.steer.get("keep")) and self.paths.steer_probes.exists()
        # Caches (finished edits keyed by (source, target, mode), etc.).
        self.edit_cache = {(p["source"], p["target"], "swap"): p for p in self.pair_menu.get("pairs", [])}
        for p in self.pair_menu.get("all_pairs", []):
            self.edit_cache.setdefault((p["source"], p["target"], "swap"), p)
        self.ask_cache = {(a["question"], a.get("source"), a.get("target"), "swap"): a
                          for a in self.pair_menu.get("free_answers", [])}
        self.locate_cache: dict[str, dict] = {}
        self.rewrite_menu = self.read("rewrite_menu.json", {"worlds": []})
        self.rewrite_cache = {}
        for w in self.rewrite_menu.get("worlds", []):
            k = tuple((m["source"], m["target"]) for m in w["moves"])
            for mode, r in w.get("results", {}).items():
                self.rewrite_cache[(k, mode)] = r
                if "free" in r:
                    f = r["free"]
                    self.ask_cache[(f["question"], k, mode)] = {"question": f["question"], "before": f["before"],
                                                                "after": f["after"], "moves": [list(x) for x in k]}
        print(f"[server] {key} ready in {time.time() - t0:.1f}s; editable={self.editable}; "
              f"band {self.band}; patch method {self.method} ({menu_file})")

    def read(self, name: str, default=None):
        p = self.paths.result(name)
        return json.loads(p.read_text()) if p.exists() else default

    def close(self) -> None:
        """Drop every reference to the model so its memory can be freed."""
        self.locator.lm = None
        self.lm = None
        self.edit_cache.clear()
        self.ask_cache.clear()
        self.locate_cache.clear()
        self.rewrite_cache.clear()

    def locate_cached(self, name: str) -> dict:
        if name not in self.locate_cache:
            self.locate_cache[name] = self.locator.locate([name])[0]
        return self.locate_cache[name]


class Server:
    def __init__(self):
        self.lock = threading.Lock()      # one model computation (or switch) at a time
        self.current: ModelState | None = None
        self.switching = False
        self.switch_msg = ""
        self.error: str | None = None
        self.ollama_unloaded: list[str] = []


S = Server()


def _pick_startup_key() -> str | None:
    key = registry.startup_key()
    if registry.available(key):
        return key
    for k in registry.MODELS:
        if registry.available(k):
            print(f"[server] {key} is not built; starting with {k} instead")
            return k
    return None


@asynccontextmanager
async def lifespan(_app):
    """Load the start-up model and its files once, before serving."""
    S.error = None
    S.switching = False
    key = _pick_startup_key()
    if key is None:
        S.error = "no model is built yet: run scripts/build_model.py"
    else:
        try:
            S.current = ModelState(key, get_lm(key))
        except Exception as exc:  # e.g. another process holds a model
            S.error = f"could not load {key}: {exc}"
            print(f"[server] {S.error}")
    yield
    if S.current is not None:
        S.current.close()
        S.current = None
    unload_lm()


app = FastAPI(title="TERRA INCOGNITA", lifespan=lifespan)


def cur() -> ModelState:
    """The loaded model, or 503 while switching / when none is loaded."""
    if S.switching:
        raise HTTPException(503, f"switching models: {S.switch_msg}")
    if S.current is None:
        raise HTTPException(503, S.error or "no model loaded")
    return S.current


@contextmanager
def using(M: ModelState):
    """Hold the model lock, and fail cleanly if the model changed meanwhile.

    Planning errors (a name that cannot be found or aligned in a prompt) are
    the caller's input problem, so they become 400 with the reason.
    """
    with S.lock:
        if S.current is not M or S.switching:
            raise HTTPException(503, "the model changed while this request waited; please retry")
        try:
            yield
        except (AlignmentError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc


def need_editable(M: ModelState) -> None:
    if not M.editable:
        raise HTTPException(409, f"{M.spec.display} has no edit gate yet: run "
                                 f"`uv run python scripts/build_model.py --model {M.key} --stage gate` "
                                 f"and `--stage menu`")


def key_for(name: str):
    info = city_info(name)
    if info is None:
        return None
    return make_key(name, info["iso"], info["latitude"])


def latlon_for(M: ModelState, name: str) -> tuple[float, float]:
    """True position from GeoNames, or (for invented places) where the model believes it is."""
    info = city_info(name)
    if info is not None:
        return info["latitude"], info["longitude"]
    r = M.locate_cached(name)
    return r["lat"], r["lon"]


def place_info(M: ModelState, name: str) -> dict:
    info = city_info(name)
    if info is not None:
        return dict(info, known=True)
    r = M.locate_cached(name)
    return {"name": name, "known": False, "latitude": r["lat"], "longitude": r["lon"],
            "note": "no GeoNames entry: position is where the model believes it is"}


def model_status() -> dict:
    c = S.current
    return {"loaded": c.key if c else None, "switching": S.switching, "switch_msg": S.switch_msg,
            "error": S.error, "ollama_unloaded": S.ollama_unloaded}


# ------------------------------------------------------------------ models

@app.get("/api/health")
def health():
    c = S.current
    return {"ok": c is not None and not S.switching, **model_status(),
            "key": c.key if c else None, "display": c.spec.display if c else None,
            "model": c.lm.model_id if c and c.lm else None,
            "band": [c.band[0], c.band[-1]] if c and c.band else None,
            "probe_layer": c.locator.layer if c else None, "probe_mode": c.locator.mode if c else None,
            "patch_method": c.method if c else None, "editable": c.editable if c else False}


@app.get("/api/models")
def models():
    out = []
    for k, s in registry.MODELS.items():
        out.append({"key": k, "display": s.display, "backend": s.backend, "repo": s.repo,
                    "approx_gb": s.approx_gb, "available": registry.available(k),
                    "editable": registry.editable(k), "loaded": S.current is not None and S.current.key == k,
                    "build_command": registry.build_command(k)})
    return {"models": out, **model_status()}


class ModelReq(BaseModel):
    key: str


@app.post("/api/model")
def switch_model(req: ModelReq):
    """Unload the current model completely, then load `key`."""
    key = req.key
    if key not in registry.MODELS:
        raise HTTPException(404, f"unknown model {key!r}")
    if not registry.available(key):
        raise HTTPException(409, f"{registry.spec(key).display} is not built yet: run "
                                 f"`{registry.build_command(key)}`")
    if S.current is not None and S.current.key == key and not S.switching:
        return {"ok": True, "unchanged": True, **model_status()}
    with S.lock:
        if S.switching:
            raise HTTPException(409, "already switching models")
        old = S.current.key if S.current else None
        old_name = registry.spec(old).display if old else "nothing"
        S.switching = True
        S.switch_msg = f"Unloading {old_name}, loading {registry.spec(key).display}"
        t0 = time.time()
        try:
            # 1-2. Drop every reference to the current model and free its memory.
            if S.current is not None:
                S.current.close()
                S.current = None
            unload_lm()
            # 3. Ask Ollama (if running) to unload its models too.
            S.ollama_unloaded = ollama.unload_ollama_models()
            # 4. Load the new model, its probe and its menus.
            S.current = ModelState(key, get_lm(key))
            S.error = None
        except Exception as exc:
            S.error = f"could not load {key}: {type(exc).__name__}: {exc}"
            if old and old != key:
                try:
                    unload_lm()
                    S.current = ModelState(old, get_lm(old))
                    S.error += f"; restored {old}"
                except Exception as exc2:
                    S.error += f"; restoring {old} also failed: {exc2}"
            raise HTTPException(500, S.error) from exc
        finally:
            S.switching = False
            S.switch_msg = ""
    return {"ok": True, "seconds": time.time() - t0, **model_status()}


# ------------------------------------------------------------------ reading

@app.get("/api/places")
def places():
    M = cur()
    p = M.paths.result("places_believed.json")
    if not p.exists():
        raise HTTPException(404, "places_believed.json not built for this model")
    return FileResponse(p, media_type="application/json")


@app.get("/api/menu")
def menu():
    M = cur()
    return {
        "model": M.key, "display": M.spec.display, "editable": M.editable,
        "not_editable_reason": None if M.editable else
        f"No edit gate for {M.spec.display} yet: run `uv run python scripts/build_model.py --model {M.key} "
        f"--stage gate` and `--stage menu`.",
        "band": [M.band[0], M.band[-1]] if M.band else None,
        "pairs": [{"source": p["source"], "target": p["target"]} for p in M.pair_menu.get("pairs", [])],
        "cities": M.pair_menu.get("cities", []),
        "fictional": M.fictional.get("menu", []),
        "worlds": [{"name": w["name"], "moves": w["moves"]} for w in M.rewrite_menu.get("worlds", [])],
        "max_moves": MAX_MOVES,
        "steer_available": M.steer_ok,
    }


@app.get("/api/locate")
def locate(name: str):
    M = cur()
    name = name.strip()
    if not name or len(name) > 80:
        raise HTTPException(400, "name must be 1 to 80 characters")
    if name in M.locate_cache:
        return M.locate_cache[name]
    with using(M):
        return M.locate_cached(name)


@app.get("/api/political")
def political():
    M = cur()
    gj, st = M.read("political_map.geojson"), M.read("political_stats.json")
    if gj is None or st is None:
        raise HTTPException(404, f"political map not built for {M.spec.display}: run "
                                 f"`uv run python scripts/build_model.py --model {M.key} --stage political`")
    return {"model": M.key, "display": M.spec.display, "map": gj, "stats": st}


# ------------------------------------------------------------------ editing

class EditReq(BaseModel):
    source: str
    target: str
    mode: str = "swap"   # "swap" (tier 1) or "steer" (tier 2, if validated)


def _plan_move(M: ModelState, texts, source: str, target: str):
    from .steer import plan_move
    return plan_move(M.lm, texts, source, target, paths=M.paths, target_latlon=latlon_for(M, target))


def _run_edit(M: ModelState, source: str, target: str, mode: str) -> dict:
    qs = prompts_for(source)
    texts = [qs[q] for q in QKEYS]
    if mode == "steer":
        from .steer import random_like
        edit_plan = _plan_move(M, texts, source, target)
        # Control matched to the steering change (same layers and sizes).
        ctrl_plan = random_like(M.lm, edit_plan, stable_seed(source, target, "steer-control"))
    else:
        edit_plan = M.planner(M.lm, texts, source, target, M.band, mode="swap")
        ctrl_plan = M.planner(M.lm, texts, source, target, M.band, mode="control")
    plans = [[M.lm.chat(t) for t in texts], edit_plan]
    same_layers = edit_plan.layers == ctrl_plan.layers
    if same_layers:
        plans.append(ctrl_plan)
    outs = generate_plans(M.lm, plans, max_new_tokens=10)
    if not same_layers:
        outs.append(generate_plans(M.lm, [ctrl_plan], max_new_tokens=10)[0])
    before, after, control = (dict(zip(QKEYS, o)) for o in outs)
    return {"source": source, "target": target, "mode": mode,
            "before": before, "after": after, "control": control}


def _edited_location(M: ModelState, source: str, target: str, mode: str) -> dict:
    """Where the location probe reads `source` *after* the edit.

    We build the same patch for the probe's own prompt ("What are the lat/lon
    coordinates of Paris"), run it with the patch applied, and read the probe
    layer at the name token. This is a measurement, not an animation target
    chosen by hand.
    """
    if mode != "swap":
        return {}  # only measured for the tier-1 swap
    text = template_text(M.lm, M.locator.mode, source)
    plan = M.planner(M.lm, [text], source, target, M.band, mode="swap", wrap=False)
    pos = [[c for r, c in zip(plan.patch.rows, plan.patch.cols) if r == 0]]
    h = M.lm.residuals_multi(plan.source_texts, pos, [M.locator.layer], patch=plan.patch)[0][:, 0]
    loc = M.locator.from_hidden(h.numpy().astype("float64"))[-1]
    return {"lat": loc["lat"], "lon": loc["lon"], "nearest": loc["nearest"], "measured_for": "swap"}


def _landmark(M: ModelState, source: str, target: str, mode: str) -> dict:
    """Unscored identity check: the landmark answer before, after and under control."""
    q = [LANDMARK_QUESTION.format(city=source)]
    if mode == "steer":
        from .steer import random_like
        ed = _plan_move(M, q, source, target)
        ct = random_like(M.lm, ed, stable_seed(source, target, "steer-control"))
    else:
        ed = M.planner(M.lm, q, source, target, M.band, mode="swap")
        ct = M.planner(M.lm, q, source, target, M.band, mode="control")
    plans = [[M.lm.chat(q[0])], ed] + ([ct] if ed.layers == ct.layers else [])
    outs = generate_plans(M.lm, plans, max_new_tokens=12)
    if ed.layers != ct.layers:
        outs.append(generate_plans(M.lm, [ct], max_new_tokens=12)[0])
    return {"before": outs[0][0], "after": outs[1][0], "control": outs[2][0]}


@app.post("/api/edit")
def edit(req: EditReq):
    M = cur()
    need_editable(M)
    src, tgt = req.source.strip(), req.target.strip()
    if not src or not tgt:
        raise HTTPException(400, "give both a source and a target name")
    if src == tgt:
        raise HTTPException(400, "source and target are the same name")
    if req.mode == "steer" and not M.steer_ok:
        raise HTTPException(409, f"the move (steer) mode is not available for {M.spec.display}")
    cache_key = (src, tgt, req.mode)
    cached = M.edit_cache.get(cache_key)
    live = cached is None
    if live:
        with using(M):
            t0 = time.time()
            cached = _run_edit(M, src, tgt, req.mode)
            cached["seconds"] = time.time() - t0
        M.edit_cache[cache_key] = cached
    if "landmark" not in cached:
        with using(M):
            cached["landmark"] = _landmark(M, src, tgt, req.mode)
    res = dict(cached)
    res["cached"] = not live
    try:
        with using(M):
            res["edited_location"] = _edited_location(M, src, tgt, req.mode)
    except HTTPException:
        raise
    except Exception as exc:  # the readout is a nice-to-have; never fail the edit for it
        res["edited_location_error"] = str(exc)
    sk, tk = key_for(src), key_for(tgt)
    if sk and tk:
        res["scored"] = True
        res["keys"] = {"source": sk.main, "target": tk.main}
        res["score_after"] = score_edit(res["after"], sk, tk)
        res["score_control"] = score_edit(res["control"], sk, tk)
        res["score_before"] = score_edit(res["before"], sk, tk)
    else:
        res["scored"] = False
        res["unscored_reason"] = UNSCORED
    try:
        with using(M):
            res["source_info"] = place_info(M, src)
            res["target_info"] = place_info(M, tgt)
    except HTTPException:
        raise
    except Exception:
        pass
    res["band"] = [M.band[0], M.band[-1]]
    res["model"] = M.key
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
    M = cur()
    q = req.question.strip()
    if not q or len(q) > 400:
        raise HTTPException(400, "question must be 1 to 400 characters")
    n_tok = max(8, min(req.max_new_tokens, 160))
    if req.moves:
        need_editable(M)
        return _ask_multi(M, q, [(m.source.strip(), m.target.strip()) for m in req.moves], req.mode, n_tok)
    src = (req.source or "").strip() or None
    tgt = (req.target or "").strip() or None
    if src and tgt:
        need_editable(M)
        if src == tgt:
            raise HTTPException(400, "source and target are the same name")
    ck = (q, src, tgt, req.mode)
    if ck in M.ask_cache:
        return M.ask_cache[ck]
    res = {"question": q, "source": src, "target": tgt, "model": M.key}
    with using(M):
        t0 = time.time()
        before_text = [M.lm.chat(q)]
        if src and tgt:
            if not name_positions(M.lm, M.lm.chat(q), src):
                res["note"] = f"The question does not mention {src}, so the edit has nothing to act on."
                out = generate_plans(M.lm, [before_text], max_new_tokens=n_tok)
                res["before"] = res["after"] = out[0][0]
            else:
                if req.mode == "steer":
                    plan = _plan_move(M, [q], src, tgt)
                else:
                    plan = M.planner(M.lm, [q], src, tgt, M.band, mode="swap")
                out = generate_plans(M.lm, [before_text, plan], max_new_tokens=n_tok)
                res["before"], res["after"] = out[0][0], out[1][0]
        else:
            res["before"] = generate_plans(M.lm, [before_text], max_new_tokens=n_tok)[0][0]
        res["seconds"] = time.time() - t0
    M.ask_cache[ck] = res
    return res


def _ask_multi(M: ModelState, q: str, moves: list[tuple[str, str]], mode: str, n_tok: int) -> dict:
    """Free question with every move of a rewritten world applied together."""
    from .patch import multi_substitute
    if any(not s or not t or s == t for s, t in moves):
        raise HTTPException(400, "each move needs two different, non-empty names")
    k = tuple(moves)
    ck = (q, k, mode)
    if ck in M.ask_cache:
        return M.ask_cache[ck]
    res = {"question": q, "moves": [list(m) for m in moves], "mode": mode, "model": M.key}
    with using(M):
        t0 = time.time()
        if not multi_substitute(q, moves)[1]:
            res["note"] = "The question mentions none of the moved cities, so the edits have nothing to act on."
            out = generate_plans(M.lm, [[M.lm.chat(q)]], max_new_tokens=n_tok)
            res["before"] = res["after"] = out[0][0]
        else:
            if mode == "steer":
                from .steer import plan_steer_multi
                plan = plan_steer_multi(M.lm, [q], moves, paths=M.paths,
                                        target_latlons={t: latlon_for(M, t) for _, t in moves})
            else:
                # Padding shorter targets only for free questions (DECISIONS.md, 17:10).
                plan = plan_edit_multi(M.lm, [q], moves, M.band, mode="swap", all_tokens=M.method == "all", pad=True)
            out = generate_plans(M.lm, [[M.lm.chat(q)], plan], max_new_tokens=n_tok)
            res["before"], res["after"] = out[0][0], out[1][0]
        res["seconds"] = time.time() - t0
    M.ask_cache[ck] = res
    return res


class RewriteReq(BaseModel):
    moves: list[Move]
    mode: str = "swap"


@app.post("/api/rewrite")
def rewrite(req: RewriteReq):
    """Apply up to three moves together; ripple and pair questions before/after/control."""
    M = cur()
    need_editable(M)
    moves = [(m.source.strip(), m.target.strip()) for m in req.moves]
    if not 1 <= len(moves) <= MAX_MOVES:
        raise HTTPException(400, f"give between 1 and {MAX_MOVES} moves")
    if any(not s or not t for s, t in moves):
        raise HTTPException(400, "every move needs a source and a target name")
    if any(s == t for s, t in moves):
        raise HTTPException(400, "a move has the same source and target")
    if len({s for s, _ in moves}) != len(moves):
        raise HTTPException(400, "each place can only be moved once in a rewritten world")
    if req.mode == "steer" and not M.steer_ok:
        raise HTTPException(409, f"the move (steer) mode is not available for {M.spec.display}")
    k = (tuple(moves), req.mode)
    live = k not in M.rewrite_cache
    if live:
        with using(M):
            t0 = time.time()
            tl = {t: latlon_for(M, t) for _, t in moves} if req.mode == "steer" else None
            r = run_rewrite(M.lm, moves, M.band, mode=req.mode, all_tokens=M.method == "all",
                            paths=M.paths, target_latlons=tl)
            r["seconds"] = time.time() - t0
        M.rewrite_cache[k] = r
    res = dict(M.rewrite_cache[k])
    res["cached"] = not live
    res["info"] = {n: city_info(n) for m in moves for n in m if city_info(n)}
    res["model"] = M.key
    return res


class CompanyReq(BaseModel):
    company: str
    target_city: str


@app.post("/api/company")
def company(req: CompanyReq):
    """Earlier experiment (company layer), kept for the record; not used by the page."""
    from .companies import run_company_edit
    M = cur()
    need_editable(M)
    with using(M):
        t0 = time.time()
        res = run_company_edit(M.lm, M.band, req.company, req.target_city)
        res["seconds"] = time.time() - t0
    return res


@app.get("/api/rigour")
def rigour():
    """Headline numbers for the loaded model, each read from its results file."""
    M = cur()
    rel = os.path.relpath(M.paths.results, ROOT) if str(M.paths.results).startswith(str(ROOT)) else str(M.paths.results)
    R = {"model": M.key, "display": M.spec.display, "results_dir": rel}
    pf = M.read("probe_final.json")
    if pf:
        R["probe"] = {"random_median_km": pf["heldout"]["random"]["median_km"],
                      "country_median_km": pf["heldout"]["blocked_country"]["median_km"],
                      "cell_median_km": pf["heldout"]["blocked_cell"]["median_km"],
                      "baseline_median_km": pf.get("baseline_mean_direction", {}).get("median_km"),
                      "layer": pf["layer"], "n_places": pf["n_places"]}
    # Headline edit numbers for the method the demo uses: the all-token method
    # is reported on fresh pairs it was never looked at on.
    ev = M.read("edit_eval_fresh_alltoken.json") if M.method == "all" else None
    ev_name = "edit_eval_fresh_alltoken.json" if ev else "edit_eval.json"
    ev = ev or M.read("edit_eval.json")
    if ev:
        R["edit_source"] = f"{rel}/{ev_name}"
        R["edit_method"] = ev.get("method", "last")
        R["edit"] = {"cross_pass": ev["sets"]["cross"]["edit"]["pass_rate"],
                     "cross_ctrl_pass": ev["sets"]["cross"]["control"]["pass_rate"],
                     "cross_n": ev["sets"]["cross"]["edit"]["n_pairs"],
                     "any_pass": ev["sets"]["any"]["edit"]["pass_rate"],
                     "any_ctrl_pass": ev["sets"]["any"]["control"]["pass_rate"],
                     "any_n": ev["sets"]["any"]["edit"]["n_pairs"]}
    g = M.read("gate.json")
    if g:
        R["gate"] = {"decision": g.get("decision"), "pass": g.get("edit_pairs_pass"),
                     "control_pass": g.get("control_pairs_pass"), "band": g.get("best_band"),
                     "n": len(g.get("pairs", []))}
    fi = M.read("fictional.json")
    if fi:
        R["fiction"] = {"n_coherent": fi["n_coherent"], "n_scored": fi["n_scored"],
                        "chance": fi.get("chance_rate_mean")}
    rw = M.read("rewrite_eval.json")
    if rw:
        sw = rw["summary"]["swap"]
        R["rewrite"] = {"n_worlds": sw["n_worlds"], "city_pass": sw["city_pass_rate"],
                        "city_ctrl_pass": sw["city_control_pass_rate"], "world_pass": sw["world_all_cities_pass_rate"],
                        "pair_consistent": sw["pair_questions"]["all"]["after_consistent_rate"],
                        "pair_ctrl": sw["pair_questions"]["all"]["control_matches_after_rate"],
                        "pair_n": sw["pair_questions"]["all"]["n_scored"]}
    sh = M.read("steer_heldout.json")
    if sh:
        R["steer"] = {"n": sh["n_pairs"], "moved": sh["rates"]["steer"]["moved"],
                      "kept": sh["rates"]["steer"]["identity_kept"], "both": sh["rates"]["steer"]["moved_and_kept"],
                      "ctrl_moved": sh["rates"]["control"]["moved"], "swap_kept": sh["rates"]["swap"]["identity_kept"]}
    hc = M.read("hemisphere_check.json")
    if hc:
        R["hemisphere"] = {"spoken": hc["spoken_hemisphere_accuracy"], "probe": hc["probe_hemisphere_accuracy"],
                           "n": hc["n_cities"], "n_wrong": hc["n_spoken_wrong"],
                           "n_wrong_probe_right": hc["n_spoken_wrong_but_probe_right"],
                           "examples": [{k: e[k] for k in ("name", "true_lat", "probe_lat", "said")}
                                        for e in hc["examples"] if e["probe_correct"]][:3]}
    ps = M.read("political_stats.json")
    if ps:
        R["political"] = {"right_country": ps.get("share_of_model_land_in_right_country"),
                          "n_drawn": ps.get("n_countries_drawn"), "radius_km": ps.get("settings", {}).get("radius_km"),
                          "median_displacement_km": ps.get("median_displacement_km_20plus_places")}
    figs = M.paths.figures
    order = ["how_it_works.png", "edit_vs_control.png", "rewrite_world.png", "swap_vs_move.png",
             "political_map.png", "probe_by_layer.png", "band_sweep.png", "error_map.png", "believed_map.png"]
    have = {p.name for p in figs.glob("*.png")} if figs.exists() else set()
    R["figures"] = [f for f in order if f in have] + sorted(have - set(order))
    return R


_FIG_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.(png|jpg|svg)$")


@app.get("/figures/{name}")
def figure(name: str):
    M = cur()
    if not _FIG_NAME.match(name):
        raise HTTPException(400, "bad figure name")
    p = M.paths.figures / name
    if not p.exists():
        raise HTTPException(404, f"no figure {name} for {M.spec.display}")
    return FileResponse(p)


@app.get("/api/free_question")
def free_question(city: str = "Paris"):
    return {"question": FREE_QUESTION.format(city=city)}


@app.exception_handler(Exception)
def _err(request, exc):
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


# Static files last, so /api routes win.
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
