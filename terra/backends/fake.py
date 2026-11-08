"""A fake model for tests and for running the server without real weights.

TERRA_FAKE_LM=1 makes the registry hand out FakeLM instead of a real model,
so the server, the patch planner and the API can be exercised in seconds on
any machine. Nothing here is a language model: residual vectors are small
deterministic random vectors built from token ids, and "answers" are short
strings that change whenever the patch changes. That is enough to test the
plumbing (positions, alignment, patching in prefill, unscored answers, model
switching), not the science.

FakeLM counts how many instances are loaded, unloaded and alive at once, so
tests can check that the server never holds two models in memory.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .base import BaseLM, Patch, derive_user_turn

_TOKEN = re.compile(r"<\|[a-z_]+\|>|<s>| ?[^\W\d_]+| ?\d+| ?[^\w\s]|\s+", re.U)


class FakeTokenizer:
    """Splits on words and punctuation, with character offsets like a fast tokenizer."""

    bos_token = "<s>"
    eos_token = "<|end|>"
    pad_token = "<|pad|>"
    all_special_tokens = ["<s>", "<|end|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>"]

    def __init__(self):
        self.vocab: dict[str, int] = {t: i for i, t in enumerate(self.all_special_tokens)}
        self.inv: dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.pad_token_id = self.vocab[self.pad_token]
        self.eos_token_id = self.vocab[self.eos_token]

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            i = 100 + int(hashlib.sha1(tok.encode()).hexdigest()[:6], 16) % 90_000
            while i in self.inv and self.inv[i] != tok:
                i += 1
            self.vocab[tok] = i
            self.inv[i] = tok
        return self.vocab[tok]

    def __call__(self, text, return_offsets_mapping: bool = False, add_special_tokens: bool = False, **kw):
        ids, offs = [], []
        for m in _TOKEN.finditer(text):
            ids.append(self._id(m.group(0)))
            offs.append((m.start(), m.end()))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        toks = [self.inv.get(int(i), "") for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in self.all_special_tokens]
        return "".join(toks)

    def get_vocab(self):
        return dict(self.vocab)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        user = messages[-1]["content"]
        return (f"<s><|system|>You are a fake model that knows Paris.<|end|>\n<|user|>\n{user}<|end|>\n"
                + ("<|assistant|>\n" if add_generation_prompt else ""))


class FakeLM(BaseLM):
    """Implements the backend interface on tiny deterministic tensors."""

    loads = 0
    unloads = 0
    live = 0
    max_live = 0
    events: list = []          # ("load" | "unload", key) in order

    def __init__(self, key: str = "fake", display: str = "Fake model", d_model: int = 16, n_layers: int = 6):
        super().__init__()
        FakeLM.loads += 1
        FakeLM.live += 1
        FakeLM.events.append(("load", key))
        FakeLM.max_live = max(FakeLM.max_live, FakeLM.live)
        self.key = key
        self.display = display
        self.model_id = f"fake/{key}"
        self.d_model = d_model
        self.n_layers = n_layers
        self.tok = FakeTokenizer()
        self.bos = self.tok.bos_token
        self.chat_kwargs = {}
        self.user_turn = derive_user_turn(self.chat)
        self.loaded = True

    @classmethod
    def reset_counters(cls):
        cls.loads = cls.unloads = cls.live = cls.max_live = 0
        cls.events = []

    def unload(self) -> None:
        if self.loaded:
            self.loaded = False
            FakeLM.unloads += 1
            FakeLM.live -= 1
            FakeLM.events.append(("unload", self.key))

    # ---------------------------------------------------------- fake states

    def _emb(self, tid: int) -> np.ndarray:
        return np.random.default_rng(tid).normal(size=self.d_model)

    def _states(self, ids: list[int], row_patch: dict[int, dict[int, np.ndarray]]) -> np.ndarray:
        """[n_layers, T, d]: token embedding grown with depth plus a causal mean.

        row_patch: position -> {layer: vector}. A patched slot keeps the patched
        vector (scaled with depth) at every later layer, like a real edit that
        later layers build on.
        """
        T = len(ids)
        e = np.stack([self._emb(i) for i in ids]) if T else np.zeros((0, self.d_model))
        cum = np.cumsum(e, axis=0) / np.arange(1, T + 1)[:, None] if T else e
        H = np.zeros((self.n_layers, T, self.d_model))
        for L in range(self.n_layers):
            H[L] = e * (1 + 0.1 * L) + 0.05 * L * cum
        for p, by_layer in row_patch.items():
            last = None
            for L in range(self.n_layers):
                if L in by_layer:
                    H[L, p] = by_layer[L]
                    last = L
                elif last is not None:
                    H[L, p] = by_layer[last] * (1 + 0.1 * L) / (1 + 0.1 * last)
        return H

    @staticmethod
    def _row_patch(patch: Patch | None, row: int) -> dict[int, dict[int, np.ndarray]]:
        out: dict[int, dict[int, np.ndarray]] = {}
        if patch is None:
            return out
        for k, (r, c) in enumerate(zip(patch.rows, patch.cols)):
            if r == row:
                for L, v in patch.vectors.items():
                    out.setdefault(c, {})[L] = v[k].float().numpy().astype(np.float64)
        return out

    # ------------------------------------------------------------ interface

    def residuals_multi(self, texts: Sequence[str], positions: Sequence[Sequence[int]],
                        layers: Sequence[int] | None = None, batch_size: int = 32,
                        patch: Patch | None = None) -> list[torch.Tensor]:
        assert self.loaded, "model was unloaded"
        layers = list(range(self.n_layers)) if layers is None else list(layers)
        out = []
        for r, (t, ps) in enumerate(zip(texts, positions)):
            H = self._states(self.tok(t)["input_ids"], self._row_patch(patch, r))
            if ps:
                out.append(torch.tensor(np.stack([H[layers, p] for p in ps]), dtype=torch.float32))
            else:
                out.append(torch.zeros(0, len(layers), self.d_model))
        return out

    def generate(self, texts: Sequence[str], max_new_tokens: int = 16, patch: Patch | None = None) -> list[str]:
        assert self.loaded, "model was unloaded"
        out = []
        for r, t in enumerate(texts):
            rp = self._row_patch(patch, r)
            sig = hashlib.sha1((t + json.dumps({str(p): {str(L): np.round(v, 3).tolist() for L, v in d.items()}
                                                for p, d in sorted(rp.items())})).encode()).hexdigest()[:6]
            out.append(f"Fake answer {sig}")
        return out


# ------------------------------------------------------------------ artefacts

FAKE_PLACES = [
    ("Paris", 48.85, 2.35, "France", "FR", 49.5, 3.1),
    ("Sydney", -33.87, 151.21, "Australia", "AU", -31.0, 148.0),
    ("London", 51.51, -0.13, "United Kingdom", "GB", 52.0, 1.0),
    ("Tokyo", 35.69, 139.69, "Japan", "JP", 36.5, 137.0),
    ("Berlin", 52.52, 13.40, "Germany", "DE", 51.0, 12.0),
    ("Cape Town", -33.93, 18.42, "South Africa", "ZA", -30.0, 20.0),
    ("Lima", -12.04, -77.03, "Peru", "PE", -10.0, -75.0),
    ("Buenos Aires", -34.61, -58.38, "Argentina", "AR", -33.0, -60.0),
]


def make_fake_artefacts(keys_built: Sequence[str], keys_editable: Sequence[str], d_model: int = 16,
                        n_layers: int = 6, political_keys: Sequence[str] = ()) -> None:
    """Write the small files the server needs, under registry.paths(key) for each key.

    Uses TERRA_ARTEFACT_ROOT (set it to a temporary folder first).
    """
    from .. import registry
    rng = np.random.default_rng(0)
    for key in keys_built:
        P = registry.paths(key)
        P.models.mkdir(parents=True, exist_ok=True)
        P.results.mkdir(parents=True, exist_ok=True)
        np.savez(P.probe, A=rng.normal(size=(d_model, 3)).astype(np.float32) * 0.1,
                 c=np.zeros(3, np.float32), boot_A=rng.normal(size=(4, d_model, 3)).astype(np.float32) * 0.1,
                 boot_c=np.zeros((4, 3), np.float32), alpha=1.0, layer=n_layers - 1, mode="coords")
        pts = [{"pid": i, "name": n, "type": "City", "country": c, "lat": la, "lon": lo, "blat": bla,
                "blon": blo, "err_km": 300.0, "big": True, "cont": "EU"}
               for i, (n, la, lo, c, iso, bla, blo) in enumerate(FAKE_PLACES)]
        _dump(P.results / "places_believed.json", {"mode": "coords", "layer": n_layers - 1, "points": pts})
        _dump(P.results / "fictional.json", {"menu": ["Brindlemoor", "Krasnovik"], "n_coherent": 1, "n_scored": 2,
                                              "chance_rate_mean": 0.1, "results": []})
        _dump(P.results / "probe_final.json", {
            "layer": n_layers - 1, "n_places": len(pts), "mode": "coords",
            "heldout": {"random": {"median_km": 800.0}, "blocked_country": {"median_km": 2000.0},
                        "blocked_cell": {"median_km": 1200.0}},
            "baseline_mean_direction": {"median_km": 6400.0}})
        if key in keys_editable:
            _dump(P.results / "gate.json", {"best_band": [1, 3], "pairs": [["Paris", "Sydney"]],
                                            "decision": "GO", "edit_pairs_pass": 1, "control_pairs_pass": 0})
            cities = [{"name": n, "lat": la, "lon": lo, "iso": iso} for n, la, lo, c, iso, _, _ in FAKE_PLACES]
            ans = {q: "x" for q in ("country", "currency", "language", "hemisphere", "season")}
            pair = {"source": "Paris", "target": "Sydney", "mode": "swap", "before": ans, "after": ans,
                    "control": ans, "validated": True}
            menu = {"band": [1, 3], "method": "all", "cities": cities, "pairs": [pair], "all_pairs": [pair],
                    "free_answers": [], "n_pairs": 1, "n_validated": 1}
            _dump(P.results / "pair_menu_alltoken.json", menu)
            _dump(P.results / "pair_menu.json", dict(menu, method="last"))
        if key in political_keys:
            square = {"type": "Polygon", "coordinates": [[[0, 40], [10, 40], [10, 50], [0, 50], [0, 40]]]}
            _dump(P.results / "political_map.geojson", {"type": "FeatureCollection", "features": [
                {"type": "Feature", "geometry": square,
                 "properties": {"ISO_A2": "FR", "name": "France", "MAPCOLOR7": 1}}]})
            _dump(P.results / "political_stats.json", {"countries": [{"iso": "FR", "name": "France",
                                                                       "displacement_km": 120.0, "precision": 0.6,
                                                                       "recall": 0.5, "direction": "north"}],
                                                        "most_displaced": [], "vanished": [], "imagined": []})


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
