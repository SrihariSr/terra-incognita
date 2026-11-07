"""Editing the model's belief about where a place is, by activation patching.

Tier 1: activation patching ("swap")
------------------------------------
Say the user asks "Which country is Paris in?". To make the model believe
Paris is Sydney *without changing the words*, we:

1. Write the same question with the target name: "Which country is Sydney
   in?". Run it and record the residual stream h_L(Sydney) at the last token
   of " Sydney", for every layer L in a band (say layers 0 to 15).
2. Run the real prompt. During the prefill only, at the last token of
   " Paris", overwrite the residual stream at each band layer with the
   recorded h_L(Sydney).
3. Let the model generate. Every later token reads the (patched) key/value
   cache, so the answer is computed from a "Paris" slot that now carries
   Sydney's internal representation.

The question template and every other token are untouched, so any change in
the answer is caused by what is stored at the name's position.

The control
-----------
A big enough random push to any vector can break a model's answer, so we
need to show the effect comes from the *direction* of the patch, not its
size. For each layer L, the patch moves the residual by the vector

    delta_L = h_L(target) - h_L(source)        (from clean runs).

The control instead writes h_L(source) + ||delta_L|| * u, where u is one
random unit vector (seeded, the same for every layer of this edit). So at
every layer the control moves the residual exactly as far as the real
patch does, but in a random direction. If the control also flipped the
answers, our edit would be "just noise"; it should not.

Tier 2 (probe-direction steering) lives in steer.py.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

import torch

from .model import LM, Patch


USER_TURN = "<|im_start|>user"


def name_positions(lm: LM, text: str, name: str) -> list[int]:
    """Index of the last token of every whole-word occurrence of `name`.

    In a chat-wrapped prompt only the user's message is searched: Qwen's
    default system prompt says "created by Alibaba Cloud", and we must not
    patch that mention when editing the company Alibaba.
    """
    start = max(text.rfind(USER_TURN), 0)
    out = []
    for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text):
        if m.start() >= start:
            out.append(lm.span_tokens(text, (m.start(), m.end()))[-1])
    return out


def substitute(text: str, source: str, target: str) -> str:
    """Replace whole-word occurrences of source with target."""
    return re.sub(r"(?<!\w)" + re.escape(source) + r"(?!\w)", target, text)


def stable_seed(*parts: str) -> int:
    """A seed that depends only on the strings (not on Python's hash salt)."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class EditPlan:
    """Everything needed to run an edit or its control on a batch of prompts."""

    source_texts: list[str]          # chat-wrapped prompts mentioning the source
    patch: Patch                     # the swap (or control) patch
    delta_norms: dict[int, float]    # mean ||delta_L|| per layer (for reporting)
    layers: list[int]


def plan_edit(lm: LM, user_texts: Sequence[str], source: str, target: str,
              layers: Sequence[int], mode: str = "swap", seed: int | None = None,
              wrap: bool = True) -> EditPlan:
    """Build the patch that makes the model read `source` as `target`.

    user_texts: user messages that mention `source` (e.g. the ripple
                questions). Each is wrapped in the chat template.
    mode:       "swap" for the tier-1 edit, "control" for the equal-norm
                random-direction control.
    wrap:       wrap each text in the chat template (False for raw prompts,
                e.g. the probe's "What are the lat/lon coordinates of" prompt).
    """
    layers = list(layers)
    fmt = lm.chat if wrap else (lambda t: t)
    src = [fmt(t) for t in user_texts]
    tgt = [fmt(substitute(t, source, target)) for t in user_texts]
    src_pos = [name_positions(lm, t, source) for t in src]
    tgt_pos = [name_positions(lm, t, target) for t in tgt]
    for a, b, t in zip(src_pos, tgt_pos, user_texts):
        if len(a) != len(b) or not a:
            raise ValueError(f"could not align {source!r} and {target!r} in {t!r}")
    # One reading pass over both sets of prompts: the clean source states
    # (needed for the control and for reporting) and the target states.
    reads = lm.residuals_multi(src + tgt, src_pos + tgt_pos, layers)
    h_src = torch.cat(reads[:len(src)], 0)   # [n_slots, L, d]
    h_tgt = torch.cat(reads[len(src):], 0)
    delta = h_tgt - h_src
    norms = delta.norm(dim=-1)               # [n_slots, L]

    if mode == "swap":
        new = h_tgt
    elif mode == "control":
        if seed is None:
            seed = stable_seed(source, target, "control")
        g = torch.Generator().manual_seed(seed)
        u = torch.randn(lm.d_model, generator=g)
        u = u / u.norm()
        new = h_src + norms.unsqueeze(-1) * u
    else:
        raise ValueError(mode)

    rows = [r for r, ps in enumerate(src_pos) for _ in ps]
    cols = [p for ps in src_pos for p in ps]
    vectors = {L: new[:, j].contiguous() for j, L in enumerate(layers)}
    return EditPlan(source_texts=src, patch=Patch(rows=rows, cols=cols, vectors=vectors),
                    delta_norms={L: float(norms[:, j].mean()) for j, L in enumerate(layers)},
                    layers=layers)


def ask(lm: LM, user_texts: Sequence[str], max_new_tokens: int = 12, source: str | None = None,
        target: str | None = None, layers: Sequence[int] | None = None, mode: str = "swap") -> list[str]:
    """Ask questions, optionally with `source` edited to read as `target`."""
    if source is None or target is None or mode == "none":
        return lm.generate([lm.chat(t) for t in user_texts], max_new_tokens=max_new_tokens)
    plan = plan_edit(lm, user_texts, source, target, layers, mode=mode)
    return lm.generate(plan.source_texts, max_new_tokens=max_new_tokens, patch=plan.patch)


def generate_plans(lm: LM, plans: Sequence[EditPlan | list[str]], max_new_tokens: int = 12) -> list[list[str]]:
    """Run several edits (or unedited prompt lists) in one batched generate call.

    Each element is either an EditPlan, or a plain list of chat-wrapped texts
    to run without any patch. Returns one list of answers per element.
    """
    texts, rows, cols, sizes = [], [], [], []
    vec_parts: dict[int, list[torch.Tensor]] = {}
    layer_sets = {tuple(p.layers) for p in plans if isinstance(p, EditPlan)}
    all_layers = sorted({L for ls in layer_sets for L in ls})
    for p in plans:
        off = len(texts)
        if isinstance(p, EditPlan):
            texts += p.source_texts
            rows += [off + r for r in p.patch.rows]
            cols += p.patch.cols
            for L in all_layers:
                if L in p.patch.vectors:
                    vec_parts.setdefault(L, []).append(p.patch.vectors[L])
                else:
                    raise ValueError("all plans in one batch must patch the same layers")
            sizes.append(len(p.source_texts))
        else:
            texts += list(p)
            sizes.append(len(p))
    patch = Patch(rows=rows, cols=cols, vectors={L: torch.cat(v, 0) for L, v in vec_parts.items()}) if rows else None
    outs = lm.generate(texts, max_new_tokens=max_new_tokens, patch=patch)
    res, k = [], 0
    for n in sizes:
        res.append(outs[k:k + n])
        k += n
    return res


def band(a: int, b: int) -> list[int]:
    """Layers a..b inclusive."""
    return list(range(a, b + 1))


# ---------------------------------------------------------------------------
# Experimental (not used by the demo unless results/multitoken_eval.json says
# it is better): patch *every* token of the name, not only the last one.
#
# Multi-word names such as "Buenos Aires" are several tokens. The tier-1 swap
# overwrites only the last one (" Aires"), so the earlier token (" Buenos")
# still carries the source's identity and later layers can read it directly.
# Here we overwrite every source name token. Token counts differ between
# names, so we align from the right: the source's last token gets the
# target's last token, the one before gets the target's one before, and any
# extra leading source tokens get the target's first token.

def name_spans(lm: LM, text: str, name: str) -> list[list[int]]:
    """Token indices of every whole-word occurrence of `name` in the user message."""
    start = max(text.rfind(USER_TURN), 0)
    out = []
    for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text):
        if m.start() >= start:
            out.append(lm.span_tokens(text, (m.start(), m.end())))
    return out


def plan_edit_all(lm: LM, user_texts: Sequence[str], source: str, target: str,
                  layers: Sequence[int], mode: str = "swap", seed: int | None = None,
                  wrap: bool = True) -> EditPlan:
    """Like plan_edit, but overwrite every token of the source name ("swap" or "control")."""
    layers = list(layers)
    fmt = lm.chat if wrap else (lambda t: t)
    src = [fmt(t) for t in user_texts]
    tgt = [fmt(substitute(t, source, target)) for t in user_texts]
    rows, cols, src_pos, tgt_pos = [], [], [], []
    for r, (a, b) in enumerate(zip(src, tgt)):
        sa, sb = name_spans(lm, a, source), name_spans(lm, b, target)
        if len(sa) != len(sb) or not sa:
            raise ValueError(f"could not align {source!r} and {target!r}")
        sp, tp = [], []
        for ta, tb in zip(sa, sb):
            for k in range(len(ta)):                       # k counts from the right
                j = max(len(tb) - len(ta) + k, 0)
                sp.append(ta[k])
                tp.append(tb[j])
                rows.append(r)
                cols.append(ta[k])
        src_pos.append(sp)
        tgt_pos.append(tp)
    reads = lm.residuals_multi(src + tgt, src_pos + tgt_pos, layers)
    h_src = torch.cat(reads[:len(src)], 0)
    h_tgt = torch.cat(reads[len(src):], 0)
    norms = (h_tgt - h_src).norm(dim=-1)
    if mode == "swap":
        new = h_tgt
    elif mode == "control":
        g = torch.Generator().manual_seed(seed if seed is not None else stable_seed(source, target, "control-all"))
        u = torch.randn(lm.d_model, generator=g)
        new = h_src + norms.unsqueeze(-1) * (u / u.norm())
    else:
        raise ValueError(mode)
    return EditPlan(source_texts=src,
                    patch=Patch(rows=rows, cols=cols, vectors={L: new[:, j].contiguous() for j, L in enumerate(layers)}),
                    delta_norms={L: float(norms[:, j].mean()) for j, L in enumerate(layers)}, layers=layers)


# ---------------------------------------------------------------------------
# "Rewrite the world": several moves applied together.
#
# With moves Paris -> Sydney, Tokyo -> Cape Town, a question such as "Are
# Paris and Tokyo in the same hemisphere?" mentions two moved cities. We
# substitute *all* moves at once to get the target text ("Are Sydney and
# Cape Town in the same hemisphere?"), read the target states there (so each
# target's state is computed in the already-rewritten context), and write
# them over every moved city's tokens in the original prompt, in one forward
# pass. Substitution is simultaneous, so swapping two cities (Paris -> Sydney
# and Sydney -> Paris) also works.

def multi_substitute(text: str, moves: Sequence[tuple[str, str]]):
    """Replace every source with its target in one pass.

    Returns (new_text, matches) where each match is
    (move index, (start, end) in text, (start, end) in new_text).
    """
    mapping = {s: (i, t) for i, (s, t) in enumerate(moves)}
    names = sorted(mapping, key=len, reverse=True)            # longest first
    pat = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in names) + r")(?!\w)")
    out, matches, pos = [], [], 0
    new_len = 0
    for m in pat.finditer(text):
        i, tgt = mapping[m.group(1)]
        out.append(text[pos:m.start()])
        new_len += m.start() - pos
        a = new_len
        out.append(tgt)
        new_len += len(tgt)
        matches.append((i, (m.start(), m.end()), (a, new_len)))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), matches


# Neutral words placed before a shorter target name so it spans as many tokens
# as the source ("Rio de Janeiro" -> "city of Moscow"). Without this, every
# token of "Rio de Janeiro" receives Moscow's state and a sentence listing
# several cities reads "Moscow and Moscow".
TARGET_PAD = {1: "the ", 2: "city of ", 3: "the city of "}


def n_name_tokens(lm: LM, name: str) -> int:
    return len(lm.tok(" " + name, add_special_tokens=False)["input_ids"])


def pad_targets(lm: LM, moves: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    out = []
    for s, t in moves:
        k = min(n_name_tokens(lm, s) - n_name_tokens(lm, t), 3)
        out.append((s, TARGET_PAD[k] + t if k > 0 else t))
    return out


def plan_edit_multi(lm: LM, user_texts: Sequence[str], moves: Sequence[tuple[str, str]],
                    layers: Sequence[int], mode: str = "swap", all_tokens: bool = True,
                    wrap: bool = True, pad: bool = False) -> EditPlan:
    """Apply every move at once ("swap") or its equal-norm random control ("control").

    Texts that mention no moved city are left unpatched. all_tokens=True
    patches every token of each name (aligned from the right), as in
    plan_edit_all; False patches only the last token, as in plan_edit.
    pad=True pads shorter target names (TARGET_PAD) so each source token gets
    its own target state. Scored questions use pad=False (results/rewrite_eval*.json:
    padding lowered pair consistency); the server uses pad=True only for free
    questions, where it removes "Moscow and Moscow" duplicates.
    """
    layers = list(layers)
    if pad and all_tokens:
        seeds = list(moves)                       # control seeds use the plain names
        moves = pad_targets(lm, moves)
    else:
        seeds = list(moves)
    fmt = lm.chat if wrap else (lambda t: t)
    src_texts, tgt_texts = [], []
    rows, cols, slot_move, src_pos, tgt_pos = [], [], [], [], []
    for r, u in enumerate(user_texts):
        new_u, matches = multi_substitute(u, moves)
        a_txt, b_txt = fmt(u), fmt(new_u)
        off_a, off_b = a_txt.rfind(u), b_txt.rfind(new_u)
        src_texts.append(a_txt)
        tgt_texts.append(b_txt)
        sp, tp = [], []
        for i, (sa, sb), (ta, tb) in matches:
            s_tok = lm.span_tokens(a_txt, (off_a + sa, off_a + sb))
            t_tok = lm.span_tokens(b_txt, (off_b + ta, off_b + tb))
            if not all_tokens:
                s_tok, t_tok = s_tok[-1:], t_tok[-1:]
            for k in range(len(s_tok)):
                j = max(len(t_tok) - len(s_tok) + k, 0)
                sp.append(s_tok[k])
                tp.append(t_tok[j])
                rows.append(r)
                cols.append(s_tok[k])
                slot_move.append(i)
        src_pos.append(sp)
        tgt_pos.append(tp)
    if not rows:
        return EditPlan(source_texts=src_texts, patch=Patch(rows=[], cols=[], vectors={}), delta_norms={}, layers=layers)
    # Only rows with slots need reading; empty position lists are fine.
    reads = lm.residuals_multi(src_texts + tgt_texts, src_pos + tgt_pos, layers)
    h_src = torch.cat(reads[:len(src_texts)], 0)
    h_tgt = torch.cat(reads[len(src_texts):], 0)
    norms = (h_tgt - h_src).norm(dim=-1)                       # [n_slots, L]
    if mode == "swap":
        new = h_tgt
    elif mode == "control":
        # One random direction per move (seeded by that move), scaled to the
        # real patch's size at each slot and layer.
        dirs = []
        for i, (s, t) in enumerate(seeds):
            g = torch.Generator().manual_seed(stable_seed(s, t, "control"))
            u = torch.randn(lm.d_model, generator=g)
            dirs.append(u / u.norm())
        U = torch.stack([dirs[i] for i in slot_move])          # [n_slots, d]
        new = h_src + norms.unsqueeze(-1) * U.unsqueeze(1)
    else:
        raise ValueError(mode)
    return EditPlan(source_texts=src_texts,
                    patch=Patch(rows=rows, cols=cols, vectors={L: new[:, j].contiguous() for j, L in enumerate(layers)}),
                    delta_norms={L: float(norms[:, j].mean()) for j, L in enumerate(layers)}, layers=layers)
