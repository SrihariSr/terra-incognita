"""Tier 2: move a place without swapping its identity (probe-direction steering).

The tier-1 swap replaces Paris's hidden state with Sydney's, so the model
thinks the word "Paris" *means* Sydney (it would also lose the Eiffel Tower).
Here we try a smaller, targeted change: shift the hidden state only along
the directions the location probe reads, so that the probe would now read
the target location, and leave every other direction alone.

Maths. At layer L the probe reads xyz = A^T h + c, with A a d x 3 matrix
(d = 3,584). We want a change dh with A^T dh = dxyz (the desired change in
probe reading). Among all such dh, the one with the smallest length is

    dh = A (A^T A)^(-1) dxyz            (the pseudo-inverse of A^T)

which only moves h inside the 3-dimensional space spanned by A's columns.
Because a probe can read a direction the model itself relies on only
weakly, we multiply dh by a scale s and sweep s.

dxyz: ridge probes shrink their outputs toward the centre of the sphere, so
the current reading xyz_now usually has length below 1. We aim for the
target direction at the *same* length, dxyz = |xyz_now| * u_target - xyz_now,
so the edit rotates the reading instead of also inflating it.

As with the swap, the change is written at the last token of the name,
during the prefill only, at each layer of a band (each layer uses its own
probe, and the vector written is the clean state plus s * dh).
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import torch

from . import RESULTS, ROOT
from .data import city_info
from .backends.base import BaseLM as LM, Patch
from .patch import EditPlan, name_positions
from .probe import Probe, Standardiser, latlon_to_xyz, loo_alpha

PROBE_PATH = ROOT / "models" / "steer_probes.npz"


def fit_layer_probes(layers, tag: str = "3k", mode: str = "coords") -> None:
    """Fit one ridge probe per layer on the 3k activations; save raw weights."""
    import pandas as pd

    from . import ACTS
    from .activations import load
    places = pd.read_csv(ACTS / f"places_{tag}.csv")
    Y = latlon_to_xyz(places["latitude"], places["longitude"])
    A, c = {}, {}
    for L in layers:
        _, X = load(tag, mode, L)
        st = Standardiser.fit(X.astype(np.float64))
        alpha, _ = loo_alpha(st(X.astype(np.float64)), Y - Y.mean(0))
        p = Probe.fit(X, Y, alpha)
        a, cc = p.raw_weights()
        A[f"A{L}"], c[f"c{L}"] = a.astype(np.float32), cc.astype(np.float32)
        print(f"[steer] probe layer {L}: alpha {alpha:.0f}")
    PROBE_PATH.parent.mkdir(exist_ok=True)
    np.savez(PROBE_PATH, layers=np.array(list(layers)), **A, **c)


@lru_cache(maxsize=4)
def _layer_probes(path: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    z = np.load(path)
    return {int(L): (z[f"A{L}"].astype(np.float64), z[f"c{L}"].astype(np.float64)) for L in z["layers"]}


def layer_probes(paths=None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-layer probes for steering; `paths` is a registry.ModelPaths (default Qwen)."""
    return _layer_probes(str(paths.steer_probes if paths is not None else PROBE_PATH))


def steer_config(paths=None) -> dict:
    p = (paths.results if paths is not None else RESULTS) / "steer_eval.json"
    if p.exists():
        cfg = json.loads(p.read_text()).get("chosen", {})
        if cfg:
            return cfg
    return {"band": [4, 7], "scale": 8.0}


def plan_steer(lm: LM, locator, user_texts, source: str, target: str | None = None,
               target_latlon: tuple[float, float] | None = None, layers=None, scale: float | None = None,
               paths=None) -> EditPlan:
    """Build a steering patch moving `source` toward `target` (a city name or lat/lon).

    For a target with no GeoNames entry (an invented place) pass
    target_latlon, e.g. where the model believes that place is.
    """
    cfg = steer_config(paths)
    layers = list(layers if layers is not None else range(cfg["band"][0], cfg["band"][1] + 1))
    scale = float(scale if scale is not None else cfg["scale"])
    if target_latlon is None:
        info = city_info(target)
        if info is None:
            raise ValueError(f"unknown target city {target!r}")
        target_latlon = (info["latitude"], info["longitude"])
    u_t = latlon_to_xyz(*target_latlon)
    probes = layer_probes(paths)
    src = [lm.chat(t) for t in user_texts]
    pos = [name_positions(lm, t, source) for t in src]
    h = torch.cat(lm.residuals_multi(src, pos, layers), 0).numpy().astype(np.float64)  # [n, L, d]
    vecs, norms = {}, {}
    for j, L in enumerate(layers):
        A, c = probes[L]
        now = h[:, j] @ A + c                                   # current probe reading, [n, 3]
        r = np.linalg.norm(now, axis=1, keepdims=True)
        dxyz = r * u_t[None] - now                              # rotate, keep length
        pinv = A @ np.linalg.inv(A.T @ A)                       # d x 3
        dh = dxyz @ pinv.T                                      # [n, d]
        vecs[L] = torch.from_numpy(h[:, j] + scale * dh).float()
        norms[L] = float(np.linalg.norm(scale * dh, axis=1).mean())
    rows = [r for r, ps in enumerate(pos) for _ in ps]
    cols = [p for ps in pos for p in ps]
    return EditPlan(source_texts=src, patch=Patch(rows=rows, cols=cols, vectors=vecs),
                    delta_norms=norms,
                    layers=layers)


def plan_context_city(lm: LM, user_texts, source: str, target: str, layers) -> EditPlan:
    """Alternative tier 2 ("context transplant"), tested alongside steering.

    Read the source name's own last token inside a phrase that places it in
    the target's country, "the city in Australia called Paris", and write
    that state into the plain prompt at the same layers. The vector still
    belongs to the token " Paris", so it may keep Paris's identity while
    carrying the new location. The phrase itself never appears in the prompt
    the model answers.
    """
    from .data import load_countries
    from .patch import substitute
    country = load_countries()["Country"][city_info(target)["iso"]]
    layers = list(layers)
    src = [lm.chat(t) for t in user_texts]
    ctx = [lm.chat(substitute(t, source, f"the city in {country} called {source}")) for t in user_texts]
    src_pos = [name_positions(lm, t, source) for t in src]
    ctx_pos = [name_positions(lm, t, source) for t in ctx]
    reads = lm.residuals_multi(src + ctx, src_pos + ctx_pos, layers)
    h_src = torch.cat(reads[:len(src)], 0)
    h_ctx = torch.cat(reads[len(src):], 0)
    rows = [r for r, ps in enumerate(src_pos) for _ in ps]
    cols = [p for ps in src_pos for p in ps]
    norms = (h_ctx - h_src).norm(dim=-1)
    return EditPlan(source_texts=src,
                    patch=Patch(rows=rows, cols=cols, vectors={L: h_ctx[:, j].contiguous() for j, L in enumerate(layers)}),
                    delta_norms={L: float(norms[:, j].mean()) for j, L in enumerate(layers)}, layers=layers)


def plan_move(lm: LM, user_texts, source: str, target: str, paths=None,
              target_latlon: tuple[float, float] | None = None) -> EditPlan:
    """The tier-2 edit chosen by scripts/steer_eval.py (steering or context)."""
    cfg = steer_config(paths)
    layers = list(range(cfg["band"][0], cfg["band"][1] + 1))
    if cfg.get("method") == "context":
        return plan_context_city(lm, user_texts, source, target, layers)
    return plan_steer(lm, None, user_texts, source, target, target_latlon=target_latlon, layers=layers,
                      scale=cfg.get("scale", 8.0), paths=paths)


def random_like(lm: LM, plan: EditPlan, seed: int) -> EditPlan:
    """Control for any edit plan: same slots and per-layer change size, random direction.

    At each patched layer L the plan moves the clean residual h_L by some
    vector of length n_L. The control writes h_L + n_L * u instead, with one
    seeded random unit vector u shared by all layers.
    """
    pos = [[c for r, c in zip(plan.patch.rows, plan.patch.cols) if r == i] for i in range(len(plan.source_texts))]
    clean = torch.cat(lm.residuals_multi(plan.source_texts, pos, plan.layers), 0)
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(lm.d_model, generator=g)
    u = u / u.norm()
    vecs = {}
    for j, L in enumerate(plan.layers):
        n = (plan.patch.vectors[L] - clean[:, j]).norm(dim=-1, keepdim=True)
        vecs[L] = clean[:, j] + n * u
    return EditPlan(plan.source_texts, Patch(plan.patch.rows, plan.patch.cols, vecs), plan.delta_norms, plan.layers)


def plan_steer_multi(lm: LM, user_texts, moves, layers=None, scale: float | None = None, paths=None,
                     target_latlons: dict | None = None) -> EditPlan:
    """Tier-2 steering for several moves at once (one forward pass).

    Each mention of a moved city (its last name token) is pushed along the
    probe directions toward its own target's position, exactly as in
    plan_steer; texts that mention no moved city are left alone.
    """
    from .patch import multi_substitute
    cfg = steer_config(paths)
    layers = list(layers if layers is not None else range(cfg["band"][0], cfg["band"][1] + 1))
    scale = float(scale if scale is not None else cfg["scale"])
    u_targets = []
    for _, t in moves:
        if target_latlons and t in target_latlons:
            u_targets.append(latlon_to_xyz(*target_latlons[t]))
            continue
        info = city_info(t)
        if info is None:
            raise ValueError(f"unknown target city {t!r}")
        u_targets.append(latlon_to_xyz(info["latitude"], info["longitude"]))
    probes = layer_probes(paths)
    src, pos, slot_move = [], [], []
    for u in user_texts:
        a_txt = lm.chat(u)
        off = a_txt.rfind(u)
        _, matches = multi_substitute(u, moves)
        ps = []
        for i, (sa, sb), _ in matches:
            ps.append(lm.span_tokens(a_txt, (off + sa, off + sb))[-1])
            slot_move.append(i)
        src.append(a_txt)
        pos.append(ps)
    rows = [r for r, ps in enumerate(pos) for _ in ps]
    cols = [p for ps in pos for p in ps]
    if not rows:
        return EditPlan(source_texts=src, patch=Patch(rows=[], cols=[], vectors={}), delta_norms={}, layers=layers)
    h = torch.cat(lm.residuals_multi(src, pos, layers), 0).numpy().astype(np.float64)   # [n_slots, L, d]
    U = np.stack([u_targets[i] for i in slot_move])                                   # [n_slots, 3]
    vecs, norms = {}, {}
    for j, L in enumerate(layers):
        A, c = probes[L]
        now = h[:, j] @ A + c
        r = np.linalg.norm(now, axis=1, keepdims=True)
        dxyz = r * U - now
        dh = dxyz @ (A @ np.linalg.inv(A.T @ A)).T
        vecs[L] = torch.from_numpy(h[:, j] + scale * dh).float()
        norms[L] = float(np.linalg.norm(scale * dh, axis=1).mean())
    return EditPlan(source_texts=src, patch=Patch(rows=rows, cols=cols, vectors=vecs), delta_norms=norms, layers=layers)
