"""Gemma 4 text handling with the real tokenizer and chat template.

Only the tokenizer files are read (a few MB); the model weights are never
loaded. The decoder is the tiny random MLX model from test_mlx_backend, so
these tests check prompt formatting, finding the name's tokens, and patch
alignment, not what Gemma would answer. Skipped when the Gemma repos are not
in the local Hugging Face cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
transformers = pytest.importorskip("transformers")

import test_mlx_backend as toy_module  # noqa: E402

from terra.activations import template_text  # noqa: E402
from terra.backends.base import Patch, strip_thinking  # noqa: E402
from terra.backends.mlx_backend import MLXLM, resolve_local  # noqa: E402
from terra.patch import name_positions, name_spans, plan_edit_all, plan_edit_multi  # noqa: E402
from terra.ripple import prompts_for  # noqa: E402

REPOS = ["mlx-community/gemma-4-12B-it-qat-4bit", "mlx-community/gemma-4-31B-it-qat-4bit"]


@pytest.fixture(scope="module", params=REPOS)
def gemma_text(request):
    try:
        path = resolve_local(request.param)
    except Exception:
        pytest.skip(f"{request.param} not in the local cache")
    tok = transformers.AutoTokenizer.from_pretrained(path)
    toy_module.V = len(tok)               # the toy embedding must cover Gemma's vocabulary
    mx.random.seed(0)
    model = toy_module.ToyModel()
    gen = json.loads((Path(path) / "generation_config.json").read_text())
    eos = gen.get("eos_token_id")
    lm = MLXLM(request.param, key="check", display="check", model=model, tokenizer=tok,
               eos_ids=eos if isinstance(eos, list) else [eos], hidden_size=toy_module.D)
    yield lm, tok
    toy_module.V = 100_000


def test_chat_format(gemma_text):
    lm, tok = gemma_text
    assert lm.user_turn == "<|turn>user\n"
    assert lm.bos == "<bos>"
    chat = lm.chat("Which country is Paris in?")
    # One <bos>, the question inside the user turn, and thinking switched off
    # (an empty thought channel before the answer).
    assert chat.startswith("<bos><|turn>user\nWhich country is Paris in?")
    assert chat.endswith("<|channel>thought\n<channel|>")
    assert sum(1 for i in lm._ids(chat) if i == tok.bos_token_id) == 1
    assert {1, 106} <= lm.eos_ids


def test_name_tokens_for_real_multiword_and_invented_names(gemma_text):
    lm, tok = gemma_text
    for name, last in [("Paris", " Paris"), ("Rio de Janeiro", " Janeiro"), ("Brindlemoor", "moor")]:
        text = lm.chat(f"Which country is {name} in?")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        spans = name_spans(lm, text, name)
        assert len(spans) == 1
        assert "".join(tok.decode([ids[i]]) for i in spans[0]).strip() == name
        assert tok.decode([ids[name_positions(lm, text, name)[-1]]]) == last


def test_extraction_prompt_starts_with_one_bos(gemma_text):
    lm, tok = gemma_text
    text = template_text(lm, "coords", "Paris")
    assert text == "<bos>What are the lat/lon coordinates of Paris"
    assert lm._ids(text)[0] == tok.bos_token_id


def test_patch_plans_align(gemma_text):
    lm, _ = gemma_text
    texts = list(prompts_for("Rio de Janeiro").values())[:2]
    plan = plan_edit_all(lm, texts, "Rio de Janeiro", "Moscow", [1, 2], mode="swap")
    assert plan.layers == [1, 2]
    multi = plan_edit_multi(lm, ["Is Rio de Janeiro further south than Brindlemoor? Answer yes or no."],
                            [("Rio de Janeiro", "Moscow"), ("Brindlemoor", "Paris")], [1, 2])
    assert multi.layers == [1, 2]
    assert isinstance(lm.generate([lm.chat("Which country is Paris in?")], max_new_tokens=3)[0], str)


def test_thinking_text_is_stripped(gemma_text):
    lm, _ = gemma_text
    assert strip_thinking("<|channel>thought\nhmm<channel|>France<turn|>", lm.special) == "France"


# ---------------------------------------------------------------------------
# The real Gemma 4 architecture from mlx-lm, built from each repo's own
# config.json but shrunk to width 64 and six layers, with random weights.
# This runs the backend through Gemma's actual layer classes, masks (sliding
# and full attention) and cache, which the toy model cannot.

@pytest.fixture(scope="module", params=REPOS)
def tiny_gemma(request):
    try:
        path = Path(resolve_local(request.param))
    except Exception:
        pytest.skip(f"{request.param} not in the local cache")
    from mlx_lm.models import gemma4
    cfg = json.loads((path / "config.json").read_text())
    t = cfg["text_config"]
    t.update(hidden_size=64, intermediate_size=128, num_hidden_layers=6, layer_types=t["layer_types"][:6],
             num_attention_heads=2, num_key_value_heads=1, num_global_key_value_heads=1,
             head_dim=32, global_head_dim=64)
    cfg["model_type"] = "gemma4"          # mlx-lm maps the 12B's "gemma4_unified" to this class too
    for k in ("vision_config", "audio_config", "quantization", "quantization_config"):
        cfg.pop(k, None)
    mx.random.seed(0)
    model = gemma4.Model(gemma4.ModelArgs.from_dict(cfg))
    mx.eval(model.parameters())
    tok = transformers.AutoTokenizer.from_pretrained(path)
    # No hidden_size given: the backend must read it from the config, as it
    # does when loading the real model.
    return MLXLM(request.param, key="tiny", display="tiny", model=model, tokenizer=tok, eos_ids=[1, 106, 50])


def test_tiny_gemma_width_and_layers(tiny_gemma):
    lm = tiny_gemma
    assert lm.d_model == 64 and lm.n_layers == 6
    assert lm.layers[5].layer_type == "full_attention"       # forwarded through the tap


def test_tiny_gemma_reading_padding_and_patch(tiny_gemma):
    import torch
    lm = tiny_gemma
    text = lm.chat("Which country is Rio de Janeiro in?")
    short = lm.chat("Paris?")
    h = lm.residuals_multi([text], [[5, 6]], layers=[0, 3, 5])[0]
    assert tuple(h.shape) == (2, 3, 64) and h.dtype == torch.float32
    alone = lm.residuals_multi([short], [[4]], layers=[2, 5])[0]
    batched = lm.residuals_multi([short, text], [[4], [5]], layers=[2, 5])[0]
    assert torch.allclose(alone, batched, atol=1e-4)
    vec = torch.full((1, 64), 0.5)
    got = lm.residuals_multi([text], [[5]], layers=[1], patch=Patch(rows=[0], cols=[5], vectors={1: vec}))[0]
    assert torch.allclose(got[0, 0], vec[0], atol=1e-2)


def test_tiny_gemma_generation(tiny_gemma):
    lm = tiny_gemma
    out = lm.generate([lm.chat("Which country is Paris in?")], max_new_tokens=3)
    assert isinstance(out[0], str) and lm.state.patch == {}
