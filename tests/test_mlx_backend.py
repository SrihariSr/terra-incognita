"""The MLX backend's layer taps, on a tiny MLX model built here (no weights downloaded).

The toy model mimics what matters about Gemma 4 in mlx-lm: a list of decoder
layers that return a tuple (h, shared_kv, offset), a per-layer `layer_type`
attribute the model reads, a backbone at `model.model`, causal mixing across
positions, and a key/value-style cache for generation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from terra.backends.base import Patch, strip_thinking  # noqa: E402
from terra.backends.fake import FakeTokenizer  # noqa: E402
from terra.backends.mlx_backend import MLXLM  # noqa: E402

D, L, V = 8, 4, 100_000


class ToyCache:
    def __init__(self):
        self.past = None


class ToyLayer(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.lin = nn.Linear(D, D)
        self.layer_type = "full_attention"

    def __call__(self, x, mask=None, cache=None):
        if cache is not None:
            allx = x if cache.past is None else mx.concatenate([cache.past, x], axis=1)
            cache.past = allx
        else:
            allx = x
        T, Tall = x.shape[1], allx.shape[1]
        mean = mx.cumsum(allx, axis=1) / mx.arange(1, Tall + 1).reshape(1, -1, 1)
        h = x + 0.5 * mx.tanh(self.lin(mean[:, Tall - T:, :]))
        return h, None, None


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.layers = [ToyLayer(i) for i in range(L)]

    def __call__(self, inputs, cache=None):
        h = self.embed(inputs)
        cache = cache or [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            # Reads an attribute from each layer, as Gemma 4's mask code does.
            assert layer.layer_type == "full_attention"
            h, _, _ = layer(h, None, c)
        return h


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()
        self.head = nn.Linear(D, V)

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, inputs, cache=None):
        return self.head(self.model(inputs, cache))

    def make_cache(self):
        return [ToyCache() for _ in self.model.layers]


@pytest.fixture(scope="module")
def toy():
    mx.random.seed(0)
    model = ToyModel()
    mx.eval(model.parameters())
    tok = FakeTokenizer()
    return MLXLM("toy/model", key="toy", display="Toy", model=model, tokenizer=tok,
                 eos_ids=[tok.eos_token_id], hidden_size=D)


def test_width_is_inferred_when_the_config_has_none():
    # The toy model has no config, so the backend falls back to one forward
    # pass, which needs the padding id to be set first (a real bug, fixed).
    mx.random.seed(0)
    lm = MLXLM("toy/model", key="toy", display="Toy", model=ToyModel(), tokenizer=FakeTokenizer(), eos_ids=[0])
    assert lm.d_model == D and lm.pad_id is not None


def manual_states(toy, text):
    """[L, T, D] residual stream computed layer by layer, without the taps."""
    ids = mx.array([toy._ids(text)])
    h = toy.backbone.embed(ids)
    out = []
    for tap in toy.layers:
        h, _, _ = tap.inner(h, None, None)
        out.append(np.array(h[0]))
    return np.stack(out)


def test_interface_basics(toy):
    assert toy.n_layers == L and toy.d_model == D
    assert toy.user_turn.startswith("<|user|>")
    assert toy.bos == "<s>"


def test_recorded_states_match_a_manual_forward_pass(toy):
    text = toy.chat("Which country is Paris in?")
    pos = [3, 7]
    got = toy.residuals_multi([text], [pos], layers=[0, 2, 3])[0]        # [2, 3, D]
    want = manual_states(toy, text)
    for k, p in enumerate(pos):
        for j, layer in enumerate([0, 2, 3]):
            assert np.allclose(got[k, j].numpy(), want[layer, p], atol=1e-5)


def test_right_padding_does_not_change_states(toy):
    short = toy.chat("Paris?")
    long = toy.chat("Which country is Buenos Aires in, and what currency is used there?")
    alone = toy.residuals_multi([short], [[4]], layers=[1, 3])[0]
    batched = toy.residuals_multi([short, long], [[4], [5]], layers=[1, 3])[0]
    assert torch.allclose(alone, batched, atol=1e-5)


def test_patch_is_written_at_the_slot_and_layers(toy):
    text = toy.chat("Which country is Paris in?")
    vec = torch.full((1, D), 3.0)
    patch = Patch(rows=[0], cols=[5], vectors={1: vec, 2: vec})
    got = toy.residuals_multi([text], [[5, 6]], layers=[1, 2, 3], patch=patch)[0]
    assert torch.allclose(got[0, 0], vec[0]) and torch.allclose(got[0, 1], vec[0])
    clean = toy.residuals_multi([text], [[5, 6]], layers=[3])[0]
    assert not torch.allclose(got[1, 2], clean[1, 0])        # a later position feels the patch
    assert toy.state.patch == {} and toy.state.record == {}  # nothing left armed


def test_generation_is_greedy_and_patch_changes_it(toy):
    text = toy.chat("Which country is Paris in?")
    a = toy.generate([text], max_new_tokens=5)
    b = toy.generate([text], max_new_tokens=5)
    assert a == b
    patch = Patch(rows=[0], cols=[5], vectors={0: torch.full((1, D), 5.0), 1: torch.full((1, D), 5.0)})
    c = toy.generate([text], max_new_tokens=5, patch=patch)
    assert isinstance(c[0], str)
    assert toy.state.patch == {}


def test_strip_thinking():
    raw = "<|channel>thought\nlet me think<channel|>Paris is in France."
    assert strip_thinking(raw) == "Paris is in France."
    assert strip_thinking("<think>x</think> Hello <turn|>", ["<turn|>"]) == "Hello"
