"""Load Qwen2.5-7B-Instruct once and give simple tools to read and write its
residual stream.

Background: what is the "residual stream"?
------------------------------------------
A transformer keeps one vector per token (here 3,584 numbers long). Each of
the 28 layers reads that vector, computes an update with attention and an
MLP, and *adds* the update back on. So the vector after layer i is

    h_i = h_{i-1} + attention_i(...) + mlp_i(...)

and this running sum is called the residual stream. Everything the model
"knows" about a token at a given depth has to live in h_i, which is why we
read it (to fit probes) and overwrite it (to edit beliefs).

In Hugging Face's Qwen2 code, `model.model.layers[i]` is one decoder layer.
A PyTorch forward hook on it sees the layer's output, which is h_i (either a
bare tensor, or a tuple whose first element is h_i, depending on the
transformers version). We handle both.

Two operations are built here:

1. `residuals(...)`: run prompts and record h_i at chosen token positions
   for chosen layers (used to fit probes and to capture "target" states).
2. `generate(...)`: greedy decoding, optionally with a *patch*: during the
   prefill (the single forward pass over the whole prompt) we replace h_i at
   one token position with a vector we supply, for a band of layers. Later
   generated tokens attend to the cached keys and values computed from the
   patched states, so the edit carries into the answer without touching the
   prompt text or the weights.
"""

from __future__ import annotations

import os

# The demo must run with Wi-Fi off. The model is already in the local
# Hugging Face cache, so tell the libraries never to contact the Hub (without
# this, loading can stall on network time-outs when offline). Set
# TERRA_ALLOW_NETWORK=1 to download the model the first time.
if os.environ.get("TERRA_ALLOW_NETWORK") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import threading
import time
from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Qwen has no beginning-of-sequence token. The very first token of any
# sequence becomes an "attention sink" with an abnormally large residual
# norm, so a one-token name placed at position 0 would give a strange
# vector. We start bare-name prompts with <|endoftext|> to play the role of
# BOS (as Llama's <s> did in Gurnee & Tegmark's setup).
BOS = "<|endoftext|>"


@dataclass
class Patch:
    """Overwrite the residual stream at some (row, token) slots, for some layers.

    rows, cols: parallel lists. Entry k says "in batch row rows[k], at token
                index cols[k] of that row's *unpadded* prompt".
    vectors:    dict layer -> tensor [len(rows), d_model]; entry k is the new
                residual vector written into slot k at that layer.
    """

    rows: list[int]
    cols: list[int]
    vectors: dict[int, torch.Tensor]


class LM:
    """A thin wrapper around the Hugging Face model with hook utilities."""

    def __init__(self, model_id: str = MODEL_ID, dtype: torch.dtype = torch.bfloat16,
                 device: str | None = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        except Exception as exc:  # bf16 not supported: fall back to fp16
            print(f"[model] {dtype} failed ({exc}); falling back to float16")
            dtype = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        self.model.eval()
        self.dtype = dtype
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        # Generation on one GPU is not thread safe; the server serialises
        # requests through this lock.
        self.lock = threading.Lock()
        print(f"[model] loaded {model_id} on {device} in {dtype} "
              f"({self.n_layers} layers, d={self.d_model}) in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ text

    def chat(self, user_text: str) -> str:
        """Wrap a user message in Qwen's chat template, ready for the reply."""
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True)

    def span_tokens(self, text: str, span: tuple[int, int]) -> list[int]:
        """Token indices whose characters overlap text[span[0]:span[1]].

        We use the fast tokenizer's offset mapping: for every token it tells
        us which characters of the original string it came from.
        """
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        a, b = span
        idx = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < b and e > a and e > s]
        if not idx:
            raise ValueError(f"span {span} not found in tokens of {text!r}")
        return idx

    def name_tokens(self, text: str, name: str) -> list[int]:
        """Token indices of the *last* occurrence of `name` inside `text`."""
        a = text.rfind(name)
        if a < 0:
            raise ValueError(f"{name!r} not in {text!r}")
        return self.span_tokens(text, (a, a + len(name)))

    def last_name_token(self, text: str, name: str) -> int:
        return self.name_tokens(text, name)[-1]

    # --------------------------------------------------------------- batching

    def _encode(self, texts: Sequence[str], left_pad: bool):
        """Tokenise a batch. Returns ids, mask and each row's pad offset.

        Left padding (pads at the start) is needed for generation so that every
        row's last real token sits at the same final position. For reading
        states we also use left padding, and simply shift positions by the
        number of pad tokens in that row.
        """
        self.tok.padding_side = "left" if left_pad else "right"
        enc = self.tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=False)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        lengths = mask.sum(1).tolist()
        T = ids.shape[1]
        offsets = [T - L for L in lengths] if left_pad else [0] * len(texts)
        return ids, mask, offsets

    # ------------------------------------------------------------------ hooks

    def _hook_layers(self, fn_by_layer: dict[int, callable]):
        """Register a forward hook on each listed layer; return the handles."""
        handles = []
        for i, fn in fn_by_layer.items():
            def hook(module, args, output, _fn=fn, _i=i):
                h = output[0] if isinstance(output, tuple) else output
                new_h = _fn(_i, h)
                if new_h is None:
                    return None
                if isinstance(output, tuple):
                    return (new_h,) + tuple(output[1:])
                return new_h
            handles.append(self.layers[i].register_forward_hook(hook))
        return handles

    # ---------------------------------------------------------------- reading

    def _patch_hooks(self, patch: Patch | None, offs: list[int]):
        """Hooks that write patch vectors into the prefill pass only."""
        if patch is None or not patch.rows:
            return []
        rows_t = torch.tensor(patch.rows, device=self.device)
        cols_t = torch.tensor([offs[r] + c for r, c in zip(patch.rows, patch.cols)], device=self.device)
        vecs = {i: v.to(self.device, self.dtype) for i, v in patch.vectors.items()}

        def put(i, h):
            if h.shape[1] == 1:
                return None  # a decode step (one new token): leave untouched
            h = h.clone()
            h[rows_t, cols_t] = vecs[i]
            return h

        return self._hook_layers({i: put for i in patch.vectors})

    @torch.inference_mode()
    def residuals_multi(self, texts: Sequence[str], positions: Sequence[Sequence[int]],
                        layers: Sequence[int] | None = None, batch_size: int = 32,
                        patch: Patch | None = None) -> list[torch.Tensor]:
        """Residual stream at several tokens per text, for each requested layer.

        texts:     raw strings (already including any chat template / BOS).
        positions: for each text, a list of token indices (unpadded).
        patch:     optional edit applied while reading (only valid when the
                   whole batch fits in one chunk, i.e. len(texts) <= batch_size).
        Returns, per text, a float32 CPU tensor [len(positions[i]), len(layers), d].
        """
        if layers is None:
            layers = list(range(self.n_layers))
        layers = list(layers)
        if patch is not None:
            assert len(texts) <= batch_size
        out: list[torch.Tensor] = []
        for b0 in range(0, len(texts), batch_size):
            bt = list(texts[b0:b0 + batch_size])
            bp = positions[b0:b0 + batch_size]
            ids, mask, offs = self._encode(bt, left_pad=True)
            rows = torch.tensor([r for r, ps in enumerate(bp) for _ in ps], dtype=torch.long, device=self.device)
            cols = torch.tensor([offs[r] + p for r, ps in enumerate(bp) for p in ps], dtype=torch.long,
                                device=self.device)
            store: dict[int, torch.Tensor] = {}

            def grab(i, h):
                store[i] = h[rows, cols].float().cpu()
                return None

            # Patch hooks are registered first so that, on a shared layer,
            # the reading hook sees the already-patched state.
            handles = self._patch_hooks(patch, offs) + self._hook_layers({i: grab for i in layers})
            try:
                # position_ids count only real tokens, so left padding does
                # not shift the rotary position encoding of the prompt.
                pos_ids = (mask.cumsum(1) - 1).clamp(min=0)
                # model.model is the transformer without the output head: we
                # only need hidden states, not next-token logits.
                self.model.model(input_ids=ids, attention_mask=mask, position_ids=pos_ids,
                                 use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            stacked = torch.stack([store[i] for i in layers], dim=1)  # [n_entries, L, d]
            k = 0
            for ps in bp:
                out.append(stacked[k:k + len(ps)])
                k += len(ps)
        return out

    def residuals(self, texts: Sequence[str], positions: Sequence[int],
                  layers: Sequence[int] | None = None, batch_size: int = 32) -> torch.Tensor:
        """One token per text. Returns [N, len(layers), d_model] float32."""
        outs = self.residuals_multi(texts, [[p] for p in positions], layers, batch_size)
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------- generation

    @torch.inference_mode()
    def generate(self, texts: Sequence[str], max_new_tokens: int = 16,
                 patch: Patch | None = None) -> list[str]:
        """Greedy decoding for a batch of prompts, with an optional patch.

        The patch is applied only during the prefill pass (when the hidden
        state has more than one position). During the following decode steps
        the model sees one new token at a time and reads the patched prompt
        through its key/value cache.
        """
        ids, mask, offs = self._encode(texts, left_pad=True)
        handles = self._patch_hooks(patch, offs)
        try:
            out = self.model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tok.pad_token_id)
        finally:
            for h in handles:
                h.remove()
        new = out[:, ids.shape[1]:]
        return [self.tok.decode(r, skip_special_tokens=True).strip() for r in new]


_LM: LM | None = None


def get_lm() -> LM:
    """Load the model on first use and keep it for the life of the process."""
    global _LM
    if _LM is None:
        _LM = LM()
    return _LM
