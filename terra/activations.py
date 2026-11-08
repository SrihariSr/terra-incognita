"""Record the residual stream at the last token of each place name.

Following Gurnee & Tegmark (2023), we read the model's internal state at the
last token of the name. Two prompt styles are compared:

* "bare":   <BOS>Paris
* "coords": <BOS>What are the lat/lon coordinates of Paris

<BOS> is the model's beginning-of-sequence text: Qwen has none, so we use
<|endoftext|> in its place; Gemma uses its real <bos>. (Without it, a
one-token name would sit at position 0, which models use as an "attention
sink" with an abnormal residual vector.)

The vector at the last name token, at layer L, is a d_model-number summary of
everything the model has worked out about the name up to that depth. If a
straight-line (linear) map from that vector to coordinates works on places
the probe has never seen, the model is storing location in a linearly
readable way.

Storage: one float16 .npy file per layer, shape [n_places, d_model], plus an
index.csv giving the place id (pid) for each row. float16 halves the disk
size; the probes are fitted in float32 after loading.

Extraction is resumable: arrays are written in place (memory-mapped) and a
progress file records how many batches are done, so a stopped run carries on
where it left off.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import ACTS
from .backends.base import BaseLM as LM
from .backends.hf import BOS as QWEN_BOS

TEMPLATE_BODIES = {
    "bare": "{name}",
    "coords": "What are the lat/lon coordinates of {name}",
}
# Qwen's full templates, kept for older code and for the Qwen meta files.
TEMPLATES = {k: QWEN_BOS + v for k, v in TEMPLATE_BODIES.items()}


def template_text(lm: LM, mode: str, name: str) -> str:
    """The prompt for a name in a given style, with this model's BOS text."""
    return (getattr(lm, "bos", None) or "") + TEMPLATE_BODIES[mode].format(name=name)


def prompt_and_position(lm: LM, name: str, mode: str) -> tuple[str, int]:
    """The prompt text for a name, and the index of its last name token."""
    text = template_text(lm, mode, name)
    start = text.rfind(name)
    toks = lm.span_tokens(text, (start, start + len(name)))
    return text, toks[-1]


def out_dir_for(tag: str, mode: str, root: Path = ACTS) -> Path:
    return Path(root) / f"{tag}_{mode}"


def is_complete(tag: str, mode: str, layers: Sequence[int], n: int, root: Path = ACTS) -> bool:
    """True if a finished extraction with these layers and size already exists."""
    d = out_dir_for(tag, mode, root)
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        return False
    return meta.get("n") == n and set(layers) <= set(meta.get("layers", [])) and \
        all((d / f"L{L:02d}.npy").exists() for L in layers)


def extract(lm: LM, places: pd.DataFrame, mode: str, layers: Sequence[int], tag: str,
            batch_size: int = 64, root: Path = ACTS, checkpoint_every: int = 10) -> Path:
    """Write activations for `places` to <root>/<tag>_<mode>/L<layer>.npy (resumable)."""
    layers = list(layers)
    out_dir = out_dir_for(tag, mode, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts, pos = [], []
    for n in places["name"]:
        t, p = prompt_and_position(lm, n, mode)
        texts.append(t)
        pos.append(p)
    N = len(texts)
    # Sort by length so each batch has similar lengths (less padding, faster).
    order = np.argsort([len(t) for t in texts], kind="stable")
    sig = hashlib.sha1(("|".join(texts) + str(layers) + str(batch_size) + lm.model_id).encode()).hexdigest()
    prog_file = out_dir / "progress.json"
    start_batch = 0
    try:
        prog = json.loads(prog_file.read_text())
        if prog.get("sig") == sig:
            start_batch = int(prog.get("done_batches", 0))
    except (OSError, ValueError):
        pass
    mode_flag = "r+" if start_batch > 0 else "w+"
    arrays = {L: np.lib.format.open_memmap(out_dir / f"L{L:02d}.npy", mode=mode_flag, dtype=np.float16,
                                           shape=(N, lm.d_model)) for L in layers}
    n_batches = (N + batch_size - 1) // batch_size
    if start_batch:
        print(f"[extract {tag}/{mode}] resuming at batch {start_batch}/{n_batches}", flush=True)
    t0 = time.time()
    for b in range(start_batch, n_batches):
        idx = order[b * batch_size:(b + 1) * batch_size]
        h = lm.residuals([texts[i] for i in idx], [pos[i] for i in idx], layers, batch_size=batch_size)
        for j, L in enumerate(layers):
            arrays[L][idx] = h[:, j].numpy().astype(np.float16)
        done = b + 1
        if done % checkpoint_every == 0 or done == n_batches:
            for a in arrays.values():
                a.flush()
            prog_file.write_text(json.dumps({"sig": sig, "done_batches": done, "n_batches": n_batches}))
            rate = (time.time() - t0) / max(1, done - start_batch)
            eta = rate * (n_batches - done)
            print(f"[extract {tag}/{mode}] {min(done * batch_size, N)}/{N} places "
                  f"({time.time() - t0:.0f}s, ETA {eta / 60:.1f} min)", flush=True)
    for a in arrays.values():
        a.flush()
    del arrays
    places[["pid", "name"]].to_csv(out_dir / "index.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps({
        "model": lm.model_id, "mode": mode, "template": template_text(lm, mode, "{name}"), "layers": layers,
        "n": N, "seconds": time.time() - t0}, indent=2))
    prog_file.unlink(missing_ok=True)
    print(f"[extract {tag}/{mode}] wrote {len(layers)} layers x {N} to {out_dir} in {time.time() - t0:.0f}s")
    return out_dir


def load(tag: str, mode: str, layer: int, root: Path = ACTS) -> tuple[np.ndarray, np.ndarray]:
    """Return (pids, float32 activations) for one layer."""
    d = out_dir_for(tag, mode, root)
    pids = pd.read_csv(d / "index.csv")["pid"].to_numpy()
    X = np.load(d / f"L{layer:02d}.npy").astype(np.float32)
    return pids, X
