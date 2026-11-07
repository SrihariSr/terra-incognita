"""Record the residual stream at the last token of each place name.

Following Gurnee & Tegmark (2023), we read the model's internal state at the
last token of the name. Two prompt styles are compared:

* "bare":   <|endoftext|>Paris
* "coords": <|endoftext|>What are the lat/lon coordinates of Paris

The vector at the last name token, at layer L, is a 3,584-number summary of
everything the model has worked out about the name up to that depth. If a
straight-line (linear) map from that vector to coordinates works on places
the probe has never seen, the model is storing location in a linearly
readable way.

Storage: one float16 .npy file per layer, shape [n_places, 3584], plus an
index.csv giving the place id (pid) for each row. float16 halves the disk
size; the probes are fitted in float32 after loading.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import ACTS
from .model import BOS, LM

TEMPLATES = {
    "bare": BOS + "{name}",
    "coords": BOS + "What are the lat/lon coordinates of {name}",
}


def prompt_and_position(lm: LM, name: str, mode: str) -> tuple[str, int]:
    """The prompt text for a name, and the index of its last name token."""
    text = TEMPLATES[mode].format(name=name)
    start = text.rfind(name)
    toks = lm.span_tokens(text, (start, start + len(name)))
    return text, toks[-1]


def extract(lm: LM, places: pd.DataFrame, mode: str, layers: Sequence[int], tag: str,
            batch_size: int = 64) -> Path:
    """Write activations for `places` to activations/<tag>_<mode>/L<layer>.npy."""
    out_dir = ACTS / f"{tag}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    texts, pos = [], []
    for n in places["name"]:
        t, p = prompt_and_position(lm, n, mode)
        texts.append(t)
        pos.append(p)
    # Sort by length so each batch has similar lengths (less padding, faster).
    order = np.argsort([len(t) for t in texts])
    arrays = {L: np.zeros((len(texts), lm.d_model), dtype=np.float16) for L in layers}
    t0 = time.time()
    for b0 in range(0, len(order), batch_size):
        idx = order[b0:b0 + batch_size]
        h = lm.residuals([texts[i] for i in idx], [pos[i] for i in idx], layers, batch_size=batch_size)
        for j, L in enumerate(layers):
            arrays[L][idx] = h[:, j].numpy().astype(np.float16)
        if (b0 // batch_size) % 20 == 0:
            done = b0 + len(idx)
            print(f"[extract {tag}/{mode}] {done}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    for L, a in arrays.items():
        np.save(out_dir / f"L{L:02d}.npy", a)
    places[["pid", "name"]].to_csv(out_dir / "index.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps({
        "model": lm.model_id, "mode": mode, "template": TEMPLATES[mode], "layers": list(layers),
        "n": len(texts), "seconds": time.time() - t0}, indent=2))
    print(f"[extract {tag}/{mode}] wrote {len(layers)} layers x {len(texts)} to {out_dir} "
          f"in {time.time() - t0:.0f}s")
    return out_dir


def load(tag: str, mode: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (pids, float32 activations) for one layer."""
    d = ACTS / f"{tag}_{mode}"
    pids = pd.read_csv(d / "index.csv")["pid"].to_numpy()
    X = np.load(d / f"L{layer:02d}.npy").astype(np.float32)
    return pids, X
