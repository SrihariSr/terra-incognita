"""The interface every model backend implements.

The rest of TERRA (probes, patching, steering, the server) only talks to a
model through these members:

    chat(user_text)            wrap a user message in the model's chat template
    span_tokens(text, span)    token indices covering a character span
    residuals(...)             residual stream at one token per text
    residuals_multi(texts, positions, layers, patch=None)
                               residual stream at several tokens per text
    generate(texts, max_new_tokens, patch=None)
                               greedy decoding, optionally patched in prefill
    d_model, n_layers, model_id, lock, tok
    bos                        text to put before bare-name prompts
    user_turn                  marker that starts the user's message in a chat
    unload()                   drop the weights so the memory can be reused

Exchange format: residual vectors cross the interface as float32 *torch CPU
tensors*, whatever the backend computes in internally. That keeps
terra/patch.py and terra/steer.py identical for every model.

What a "residual stream" is, and what patching does, is explained in
terra/backends/hf.py.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Sequence

import torch


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


SENTINEL = "TERRA"


def derive_user_turn(render_chat) -> str:
    """Find the text that opens the user's message in a chat template.

    We render a chat whose user message is a sentinel string and look at what
    comes just before it. The marker is everything from the last "<" in that
    prefix, e.g. "<|im_start|>user\\n" (Qwen) or "<|turn>user\\n" (Gemma 4).
    Names are only searched for after this marker, so a system prompt that
    happens to mention a name (Qwen's says "created by Alibaba Cloud") is
    never patched.
    """
    text = render_chat(SENTINEL)
    i = text.find(SENTINEL)
    if i < 0:
        return ""
    prefix = text[:i]
    j = prefix.rfind("<")
    return prefix[j:] if j >= 0 else prefix[-24:]


class BaseLM:
    """Shared text helpers. Backends implement residuals_multi, generate, unload."""

    key: str = ""
    display: str = ""
    model_id: str = ""
    n_layers: int = 0
    d_model: int = 0
    bos: str = ""
    user_turn: str = ""
    # Extra keyword arguments for apply_chat_template (e.g. thinking off).
    chat_kwargs: dict = {}

    def __init__(self):
        # Generation on one GPU is not thread safe; callers serialise
        # requests through this lock.
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ text

    def chat(self, user_text: str) -> str:
        """Wrap a user message in the chat template, ready for the reply."""
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True, **self.chat_kwargs)

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

    def residuals(self, texts: Sequence[str], positions: Sequence[int],
                  layers: Sequence[int] | None = None, batch_size: int = 32) -> torch.Tensor:
        """One token per text. Returns [N, len(layers), d_model] float32."""
        outs = self.residuals_multi(texts, [[p] for p in positions], layers, batch_size)
        return torch.cat(outs, dim=0)

    # ------------------------------------------------- implemented per backend

    def residuals_multi(self, texts, positions, layers=None, batch_size: int = 32,
                        patch: Patch | None = None) -> list[torch.Tensor]:
        raise NotImplementedError

    def generate(self, texts, max_new_tokens: int = 16, patch: Patch | None = None) -> list[str]:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError


_THINK = [re.compile(r"<\|channel>.*?<channel\|>", re.S),   # Gemma 4 thought channel
          re.compile(r"<\|channel>.*$", re.S),               # an unfinished one
          re.compile(r"<think>.*?</think>", re.S),
          re.compile(r"<think>.*$", re.S)]


def strip_thinking(text: str, special_tokens: Sequence[str] = ()) -> str:
    """Remove any "thinking" channel and leftover special tokens from output."""
    for pat in _THINK:
        text = pat.sub("", text)
    for t in sorted(special_tokens, key=len, reverse=True):
        if t:
            text = text.replace(t, "")
    return text.strip()
