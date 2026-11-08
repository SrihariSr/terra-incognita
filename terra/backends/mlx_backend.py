"""MLX backend: Gemma 4 (4-bit builds from mlx-community) on Apple silicon.

Why MLX and not Ollama: the map and the edits need the residual stream at
every layer, and the ability to overwrite it in the middle of a forward pass.
Ollama only offers chat, generation and embeddings over HTTP, so it cannot be
used here. MLX models are ordinary Python modules, so we can reach inside.

How we reach inside without hooks
---------------------------------
PyTorch has forward hooks; MLX does not. Instead we wrap every decoder layer
in a small module, `_LayerTap`, that calls the original layer and then looks
at its output h (the residual stream after that layer, shape
[batch, tokens, d_model]):

* when *recording*, it keeps h at the requested (row, token) slots;
* when *patching*, during the prefill only (more than one token), it writes
  our vectors into those slots before passing h on.

Gemma 4's decoder layer returns a tuple (h, shared_kv, offset); we only touch
h. The model reads attributes such as `layer.layer_type` from each layer, so
the wrapper forwards unknown attribute lookups to the layer it wraps.

Two MLX details make this cheap:
* MLX is lazy: arrays are only computed when evaluated. When we only need
  layer 21, we evaluate the recorded layer-21 slots and MLX never computes
  the layers after it.
* Reading needs no padding tricks: we pad batches on the *right*. With a
  causal mask a token never looks at tokens after it, so pads appended at the
  end cannot change any earlier token's residual stream.

Generation is greedy with a key/value cache, one prompt at a time (no
padding, so no attention mask is needed), patched in the prefill only, as in
the Hugging Face backend.

Exchange format: residuals leave as float32 torch CPU tensors, and patch
vectors arrive as torch tensors (converted to MLX here), so terra/patch.py and
terra/steer.py work unchanged.
"""

from __future__ import annotations

import gc
import time
from typing import Sequence

import numpy as np
import torch

from .base import BaseLM, Patch, derive_user_turn, strip_thinking

# Tokens that end a turn in the chat formats we support (only those present
# in the vocabulary are used).
STOP_STRINGS = ["<turn|>", "<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>"]


class _TapState:
    """What the layer wrappers should do on the next forward pass."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.record: dict[int, tuple[list[int], list[int]]] = {}   # layer -> (rows, cols)
        self.recorded: dict[int, object] = {}                        # layer -> mx.array [n, d]
        self.patch: dict[int, tuple[list[int], list[int], object]] = {}  # layer -> (rows, cols, vecs)

    def on_layer(self, idx: int, h):
        import mlx.core as mx
        if idx in self.patch and h.shape[1] > 1:          # prefill only
            rows, cols, vecs = self.patch[idx]
            h = mx.array(h)                                # copy before writing
            h[mx.array(rows), mx.array(cols)] = vecs.astype(h.dtype)
        if idx in self.record:
            rows, cols = self.record[idx]
            self.recorded[idx] = h[mx.array(rows), mx.array(cols)]
        return h


def _make_tap_class():
    import mlx.nn as nn

    class _LayerTap(nn.Module):
        """Calls the wrapped decoder layer, then lets _TapState read or patch h."""

        def __init__(self, inner, idx: int, state: _TapState):
            super().__init__()
            self.inner = inner          # a Module, so stored in the module tree
            self._idx = idx             # plain attributes, not parameters
            self._state = state

        def __call__(self, x, *args, **kwargs):
            out = self.inner(x, *args, **kwargs)
            if isinstance(out, tuple):
                return (self._state.on_layer(self._idx, out[0]),) + tuple(out[1:])
            return self._state.on_layer(self._idx, out)

        def __getattr__(self, key):
            # Forward e.g. `layer_type`, `self_attn` to the wrapped layer.
            inner = dict.get(self, "inner")
            if key != "inner" and inner is not None and key not in self:
                try:
                    return getattr(inner, key)
                except AttributeError:
                    pass
            return super().__getattr__(key)

    return _LayerTap


def find_layer_list(model) -> list:
    """The Python list holding the decoder layers, for the model families we use."""
    for path in ("layers", "model.layers", "language_model.model.layers", "language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, list) and obj:
            return obj
    raise ValueError("could not find the decoder layers in this MLX model")


def config_hidden_size(model, backbone) -> int | None:
    """Width of the residual stream (d_model) from the model's own config.

    mlx-lm keeps it in different places per family: Gemma 4's wrapper has
    args.text_config["hidden_size"], its text model has config.hidden_size,
    and most text-only models have args.hidden_size. None if not found.
    """
    for obj, attr in ((backbone, "config"), (backbone, "args"), (model, "args"),
                      (getattr(model, "language_model", None), "args")):
        cfg = getattr(obj, attr, None) if obj is not None else None
        if cfg is None:
            continue
        for c in (cfg, getattr(cfg, "text_config", None)):
            if c is None:
                continue
            v = c.get("hidden_size") if isinstance(c, dict) else getattr(c, "hidden_size", None)
            if isinstance(v, int) and v > 0:
                return v
    return None


def find_backbone(model):
    """The transformer without its output head (returns hidden states, not logits)."""
    lm = getattr(model, "language_model", None)
    if lm is not None and getattr(lm, "model", None) is not None:
        return lm.model
    if getattr(model, "model", None) is not None:
        return model.model
    return model


def resolve_local(repo: str) -> str:
    """Path of a repo already in the local Hugging Face cache; never downloads."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception as exc:
        raise FileNotFoundError(
            f"{repo} is not in the local Hugging Face cache. Download it first with:\n"
            f"    uv run hf download {repo}") from exc


class MLXLM(BaseLM):
    """Gemma 4 (or any mlx-lm text model) behind the shared backend interface."""

    def __init__(self, repo: str, key: str = "", display: str = "", model=None, tokenizer=None,
                 eos_ids: Sequence[int] | None = None, hidden_size: int | None = None):
        """Load `repo` from the local cache, or wrap a given (model, tokenizer) pair (tests)."""
        super().__init__()
        import mlx.core as mx
        self.mx = mx
        self.model_id = repo
        self.key = key
        self.display = display or repo
        t0 = time.time()
        if model is None:
            from mlx_lm import load
            path = resolve_local(repo)
            # Gemma 4 repos are tagged image-text; mlx-lm's gemma4 module loads
            # the language model and drops vision and audio weights.
            loaded = load(path)                     # (model, tokenizer) or, in some versions, (model, tokenizer, config)
            model, wrapper = loaded[0], loaded[1]
            tokenizer = getattr(wrapper, "_tokenizer", wrapper)
            eos_ids = set(getattr(wrapper, "eos_token_ids", []) or [])
        self.model = model
        self.tok = tokenizer
        self.backbone = find_backbone(model)
        self.layers = find_layer_list(model)
        self.n_layers = len(self.layers)
        self.state = _TapState()
        Tap = _make_tap_class()
        for i in range(self.n_layers):
            if not isinstance(self.layers[i], Tap):
                self.layers[i] = Tap(self.layers[i], i, self.state)
        # The padding id must exist before any forward pass (the width
        # fallback below runs one).
        self.pad_id = getattr(self.tok, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = 0
        self.d_model = hidden_size or config_hidden_size(model, self.backbone) or self._infer_hidden_size()
        # Thinking off: Gemma 4's template then emits an empty thought channel.
        self.chat_kwargs = {"enable_thinking": False}
        self.bos = getattr(self.tok, "bos_token", None) or ""
        try:
            self.user_turn = derive_user_turn(self.chat)
        except Exception:
            self.user_turn = ""
        vocab = self.tok.get_vocab() if hasattr(self.tok, "get_vocab") else {}
        self.eos_ids = set(eos_ids or [])
        if getattr(self.tok, "eos_token_id", None) is not None:
            self.eos_ids.add(self.tok.eos_token_id)
        for s in STOP_STRINGS:
            if s in vocab:
                self.eos_ids.add(vocab[s])
        self.special = list(getattr(self.tok, "all_special_tokens", []) or [])
        print(f"[model] loaded {repo} with MLX ({self.n_layers} layers, d={self.d_model}) "
              f"in {time.time() - t0:.1f}s; user turn marker {self.user_turn!r}")

    def _infer_hidden_size(self) -> int:
        """Run one token through the backbone with a recorder on layer 0."""
        mx = self.mx
        self.state.reset()
        self.state.record = {0: ([0], [0])}
        self.backbone(mx.array([[self.pad_id]]))
        h = self.state.recorded[0]
        mx.eval(h)
        self.state.reset()
        return int(h.shape[-1])

    def unload(self) -> None:
        """Drop the weights and give the unified memory back."""
        self.layers = None
        self.backbone = None
        self.model = None
        self.state.reset()
        gc.collect()
        try:
            self.mx.clear_cache()
        except AttributeError:
            self.mx.metal.clear_cache()

    # ----------------------------------------------------------------- helpers

    def _ids(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def _patch_for_rows(self, patch: Patch | None, row_map: dict[int, int]):
        """Patch entries for the given original rows, re-indexed to batch rows."""
        mx = self.mx
        out = {}
        if patch is None or not patch.rows:
            return out
        keep = [k for k, r in enumerate(patch.rows) if r in row_map]
        if not keep:
            return out
        rows = [row_map[patch.rows[k]] for k in keep]
        cols = [patch.cols[k] for k in keep]
        for L, v in patch.vectors.items():
            vecs = mx.array(v[keep].float().numpy())
            out[L] = (rows, cols, vecs)
        return out

    @staticmethod
    def _to_torch(a) -> torch.Tensor:
        import mlx.core as mx
        return torch.from_numpy(np.array(a.astype(mx.float32)))

    # ---------------------------------------------------------------- reading

    def residuals_multi(self, texts: Sequence[str], positions: Sequence[Sequence[int]],
                        layers: Sequence[int] | None = None, batch_size: int = 16,
                        patch: Patch | None = None) -> list[torch.Tensor]:
        """Residual stream at several tokens per text, for each requested layer.

        Same contract as HFLM.residuals_multi. Batches are right-padded (see
        the module docstring for why that is exact under a causal mask).
        """
        mx = self.mx
        layers = list(range(self.n_layers)) if layers is None else list(layers)
        out: list[torch.Tensor] = []
        for b0 in range(0, len(texts), batch_size):
            bt = list(texts[b0:b0 + batch_size])
            bp = positions[b0:b0 + batch_size]
            ids = [self._ids(t) for t in bt]
            T = max(len(x) for x in ids)
            batch = mx.array([x + [self.pad_id] * (T - len(x)) for x in ids])
            rows = [r for r, ps in enumerate(bp) for _ in ps]
            cols = [p for ps in bp for p in ps]
            self.state.reset()
            self.state.patch = self._patch_for_rows(patch, {b0 + r: r for r in range(len(bt))})
            if rows:
                self.state.record = {L: (rows, cols) for L in layers}
            self.backbone(batch)
            if rows:
                rec = [self.state.recorded[L] for L in layers]
                mx.eval(*rec)                       # computes only what is needed
                stacked = torch.stack([self._to_torch(a) for a in rec], dim=1)   # [n, L, d]
            else:
                stacked = torch.zeros(0, len(layers), self.d_model)
            self.state.reset()
            k = 0
            for ps in bp:
                out.append(stacked[k:k + len(ps)])
                k += len(ps)
        return out

    # ------------------------------------------------------------- generation

    def generate(self, texts: Sequence[str], max_new_tokens: int = 16,
                 patch: Patch | None = None) -> list[str]:
        """Greedy decoding, one prompt at a time, patched in the prefill only."""
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        answers = []
        for r, text in enumerate(texts):
            ids = self._ids(text)
            cache = make_prompt_cache(self.model)
            self.state.reset()
            self.state.patch = self._patch_for_rows(patch, {r: 0})
            logits = self.model(mx.array([ids]), cache=cache)       # prefill (patched)
            self.state.reset()                                        # decode steps untouched
            new: list[int] = []
            for _ in range(max_new_tokens):
                nxt = int(mx.argmax(logits[0, -1]).item())
                if nxt in self.eos_ids:
                    break
                new.append(nxt)
                logits = self.model(mx.array([[nxt]]), cache=cache)
            raw = self.tok.decode(new, skip_special_tokens=False)
            answers.append(strip_thinking(raw, self.special))
        return answers
