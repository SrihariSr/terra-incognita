"""Sanity tests for the maths and the answer scoring (no model needed)."""

from __future__ import annotations

import numpy as np

from terra.probe import Probe, folds_grouped, great_circle_km, latlon_to_xyz, loo_alpha, xyz_to_latlon
from terra.ripple import _mentions, countable, make_key, score_edit


def test_geometry():
    # Paris to Sydney is about 16,960 km along the surface.
    d = great_circle_km(latlon_to_xyz(48.8566, 2.3522), latlon_to_xyz(-33.8688, 151.2093))
    assert 16_900 < d < 17_000, d
    # Two points either side of the dateline are close, not 359 degrees apart.
    d = great_circle_km(latlon_to_xyz(-17.0, 179.9), latlon_to_xyz(-17.0, -179.9))
    assert d < 25, d
    # Round trip lat/lon -> xyz -> lat/lon.
    lat, lon = xyz_to_latlon(latlon_to_xyz(-33.87, 151.21))
    assert abs(lat + 33.87) < 1e-9 and abs(lon - 151.21) < 1e-9


def test_ridge_recovers_linear_map():
    # If coordinates really are a linear function of the features, the probe
    # should recover them almost exactly.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 50))
    W = rng.normal(size=(50, 3))
    Y = X @ W
    alpha, _ = loo_alpha((X - X.mean(0)) / X.std(0), Y - Y.mean(0))
    p = Probe.fit(X, Y, alpha=1e-3)
    assert np.abs(p.predict_xyz(X) - Y).max() < 1e-2
    # Raw weights reproduce the standardised predictions.
    A, c = p.raw_weights()
    assert np.allclose(X @ A + c, p.predict_xyz(X))
    assert alpha <= 10, alpha  # almost no regularisation needed for noise-free data


def test_grouped_folds_keep_groups_together():
    groups = np.array(["US"] * 50 + ["FR"] * 10 + ["JP"] * 10 + ["BR"] * 5)
    f = folds_grouped(groups, 5, seed=0)
    for g in np.unique(groups):
        assert len(set(f[groups == g])) == 1


def test_scoring():
    paris = make_key("Paris", "FR", 48.85)
    sydney = make_key("Sydney", "AU", -33.87)
    assert _mentions("Australian Dollar", sydney.words["currency"])
    assert not _mentions("Euro", sydney.words["currency"])
    assert all(countable(paris, sydney, q) for q in ["country", "currency", "language", "hemisphere", "season"])
    after = {"country": "Australia", "currency": "Australian Dollar", "language": "English",
             "hemisphere": "Southern", "season": "Winter"}
    sc = score_edit(after, paris, sydney)
    assert sc["n_flipped"] == 5 and sc["n_countable"] == 5
    toronto = make_key("Toronto", "CA", 43.65)
    assert not countable(toronto, sydney, "currency")  # both "Dollar"
    assert countable(toronto, sydney, "hemisphere")  # north vs south
    # Word boundaries: "Niger" must not match "Nigeria".
    assert not _mentions("Nigeria", ["niger"])
