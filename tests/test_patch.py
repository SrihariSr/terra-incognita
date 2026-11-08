"""Patch planning with invented names and names of different token lengths."""

from __future__ import annotations

import pytest
import torch

from terra.patch import (AlignmentError, band, multi_substitute, name_positions, name_spans, pad_targets,
                         plan_edit, plan_edit_all, plan_edit_multi)


def target_states(lm, text, name, layers):
    """Residual stream at every token of `name` in a chat-wrapped `text`."""
    chat = lm.chat(text)
    spans = name_spans(lm, chat, name)
    return lm.residuals_multi([chat], [spans[0]], layers)[0], spans[0]


def test_user_turn_marker_found(fake_lm):
    assert fake_lm.user_turn.startswith("<|user|>")


def test_system_prompt_mention_is_ignored(fake_lm):
    # The fake system prompt says "knows Paris"; only the user's Paris counts.
    chat = fake_lm.chat("Which country is Paris in?")
    assert chat.count("Paris") == 2
    assert len(name_positions(fake_lm, chat, "Paris")) == 1


def test_invented_short_source_long_target(fake_lm):
    layers = band(1, 3)
    plan = plan_edit_all(fake_lm, ["Which country is Brindlemoor in?"], "Brindlemoor", "New Krasnovik City", layers)
    src_chat = plan.source_texts[0]
    src_tok = name_spans(fake_lm, src_chat, "Brindlemoor")[0]
    assert len(src_tok) == 1                      # one source token
    assert plan.patch.cols == src_tok
    # Aligned from the right: the only source token gets the target's last token.
    tgt, tgt_tok = target_states(fake_lm, "Which country is New Krasnovik City in?", "New Krasnovik City", layers)
    assert len(tgt_tok) == 3
    for j, L in enumerate(layers):
        assert torch.allclose(plan.patch.vectors[L][0], tgt[-1, j])


def test_invented_long_source_short_target(fake_lm):
    layers = band(1, 3)
    plan = plan_edit_all(fake_lm, ["Which country is New Krasnovik City in?"], "New Krasnovik City", "Brindlemoor",
                         layers)
    assert len(plan.patch.cols) == 3              # every source token is patched
    tgt, _ = target_states(fake_lm, "Which country is Brindlemoor in?", "Brindlemoor", layers)
    for k in range(3):                            # extra leading tokens get the target's first token
        assert torch.allclose(plan.patch.vectors[layers[0]][k], tgt[0, 0])


def test_last_token_plan_and_control_have_equal_size(fake_lm):
    layers = band(1, 3)
    texts = ["Which country is Paris in?"]
    swap = plan_edit(fake_lm, texts, "Paris", "Zyxville", layers, mode="swap")
    ctrl = plan_edit(fake_lm, texts, "Paris", "Zyxville", layers, mode="control")
    clean = fake_lm.residuals_multi(swap.source_texts, [swap.patch.cols], layers)[0][0]
    for j, L in enumerate(layers):
        d_swap = (swap.patch.vectors[L][0] - clean[j]).norm()
        d_ctrl = (ctrl.patch.vectors[L][0] - clean[j]).norm()
        assert torch.isclose(d_swap, d_ctrl, rtol=1e-4)


def test_name_not_in_prompt_raises_alignment_error(fake_lm):
    with pytest.raises(AlignmentError):
        plan_edit(fake_lm, ["What is the weather like?"], "Paris", "Sydney", band(1, 2))
    with pytest.raises(AlignmentError):
        plan_edit_all(fake_lm, ["What is the weather like?"], "Paris", "Sydney", band(1, 2))


def test_multi_substitute_swaps_simultaneously():
    text, matches = multi_substitute("Is Paris south of Sydney?", [("Paris", "Sydney"), ("Sydney", "Paris")])
    assert text == "Is Sydney south of Paris?"
    assert [m[0] for m in matches] == [0, 1]


def test_plan_edit_multi_with_invented_names(fake_lm):
    moves = [("Brindlemoor", "Sydney"), ("Paris", "New Krasnovik City")]
    texts = ["Are Brindlemoor and Paris in the same hemisphere?", "What is the weather?"]
    plan = plan_edit_multi(fake_lm, texts, moves, band(1, 2))
    # Row 0 mentions both moved places (1 + 1 tokens); row 1 mentions none.
    assert sorted(set(plan.patch.rows)) == [0]
    assert len(plan.patch.rows) == 2
    ctrl = plan_edit_multi(fake_lm, texts, moves, band(1, 2), mode="control")
    assert ctrl.patch.cols == plan.patch.cols


def test_pad_targets_gives_each_source_token_a_target(fake_lm):
    assert pad_targets(fake_lm, [("Rio de Janeiro", "Moscow")]) == [("Rio de Janeiro", "city of Moscow")]
    assert pad_targets(fake_lm, [("Paris", "Buenos Aires")]) == [("Paris", "Buenos Aires")]


def test_patch_is_visible_when_reading(fake_lm):
    layers = band(1, 3)
    plan = plan_edit_all(fake_lm, ["Which country is Paris in?"], "Paris", "Sydney", layers)
    pos = [plan.patch.cols]
    read = fake_lm.residuals_multi(plan.source_texts, pos, layers, patch=plan.patch)[0]
    for j, L in enumerate(layers):
        assert torch.allclose(read[0, j], plan.patch.vectors[L][0])


def test_generation_changes_only_with_a_patch(fake_lm):
    layers = band(1, 3)
    plan = plan_edit_all(fake_lm, ["Which country is Paris in?"], "Paris", "Brindlemoor", layers)
    plain = fake_lm.generate(plan.source_texts)
    again = fake_lm.generate(plan.source_texts)
    edited = fake_lm.generate(plan.source_texts, patch=plan.patch)
    assert plain == again and plain != edited
