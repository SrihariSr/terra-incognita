"""Edits and rewrites with places that have no GeoNames answer key."""

from __future__ import annotations

from terra.patch import band
from terra.rewrite import UNSCORED, lat, pair_key, run_rewrite


def test_lat_and_pair_key_degrade_to_unscored():
    assert lat("Brindlemoor") is None
    assert lat("Paris") is not None
    assert pair_key("same_hemisphere", None, 10.0) is None
    assert pair_key("further_south", 10.0, None) is None
    assert pair_key("same_hemisphere", 10.0, -10.0) == "no"


def test_rewrite_with_an_invented_place_is_partly_unscored(fake_lm):
    r = run_rewrite(fake_lm, [("Brindlemoor", "Krasnovik"), ("Paris", "Sydney")], band(1, 3))
    invented, real = r["cities"]
    assert invented["scored"] is False and invented["unscored_reason"] == UNSCORED
    assert "score_after" not in invented
    assert real["scored"] is True and "score_after" in real
    assert r["scored"] is False
    # Both pair questions involve the invented place, so they are unscored.
    assert len(r["pairs"]) == 2
    assert all(p["key_after"] is None and p["scored"] is False for p in r["pairs"])
    # Answers are still produced before, after and under the control.
    assert set(invented["before"]) == set(invented["after"]) == set(invented["control"])


def test_rewrite_rejects_identical_or_empty():
    import pytest

    from terra.backends.fake import FakeLM
    lm = FakeLM("qwen2.5-7b")
    try:
        with pytest.raises(ValueError):
            run_rewrite(lm, [("Paris", "Paris")], band(1, 2))
        with pytest.raises(ValueError):
            run_rewrite(lm, [("Paris", "")], band(1, 2))
    finally:
        lm.unload()
