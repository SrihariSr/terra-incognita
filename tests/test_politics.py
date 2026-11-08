"""The political map on a synthetic world of three fake countries.

Country AA: believed exactly where it is.
Country BB: every place believed 10 degrees east of the truth (half its width),
            so about half its territory should overlap the real country.
Country CC: only three places, too few to keep any land.
"""

from __future__ import annotations

import numpy as np
import pytest

from terra.politics import build_political_map, great_circle_km, latlon_to_xyz


def square(lon0, lat0, size):
    return {"type": "Polygon", "coordinates": [[[lon0, lat0], [lon0 + size, lat0], [lon0 + size, lat0 + size],
                                                [lon0, lat0 + size], [lon0, lat0]]]}


@pytest.fixture(scope="module")
def world():
    real = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": square(0, 0, 20), "properties": {"ISO_A2": "AA", "MAPCOLOR7": 2, "ADMIN": "Aland"}},
        {"type": "Feature", "geometry": square(40, 0, 20), "properties": {"ISO_A2": "BB", "MAPCOLOR7": 5, "ADMIN": "Bland"}},
        {"type": "Feature", "geometry": square(-60, 30, 10), "properties": {"ISO_A2": "CC", "MAPCOLOR7": 1, "ADMIN": "Cland"}},
    ]}
    g = np.arange(0.25, 20, 0.5)
    lat, lon = np.meshgrid(g, g, indexing="ij")
    lat, lon = lat.ravel(), lon.ravel()
    true_a = latlon_to_xyz(lat, lon)
    true_b = latlon_to_xyz(lat, lon + 40)
    bel_b = latlon_to_xyz(lat, lon + 50)
    true_c = latlon_to_xyz([33, 35, 37], [-55, -55, -55])
    believed = np.concatenate([true_a, bel_b, true_c])
    true = np.concatenate([true_a, true_b, true_c])
    iso = ["AA"] * len(lat) + ["BB"] * len(lat) + ["CC"] * 3
    gj, stats = build_political_map(believed, true, iso, real, cell_deg=1.0, k=12, radius_km=150)
    return gj, {r["iso"]: r for r in stats["countries"]}, stats


def test_displacement(world):
    _, by, _ = world
    assert by["AA"]["displacement_km"] < 5
    expected = great_circle_km(by["BB"]["true_centroid"], by["BB"]["believed_centroid"])
    assert by["BB"]["displacement_km"] == pytest.approx(expected, rel=0.01)
    assert 1000 < by["BB"]["displacement_km"] < 1150          # 10 degrees of longitude near 10 N
    assert by["BB"]["direction"] == "east"


def test_precision_and_recall(world):
    _, by, _ = world
    assert by["AA"]["precision"] > 0.8 and by["AA"]["recall"] > 0.8
    assert 0.35 < by["BB"]["precision"] < 0.65
    assert 0.35 < by["BB"]["recall"] < 0.65


def test_small_country_vanishes_and_geojson(world):
    gj, by, stats = world
    assert by["CC"]["n_cells"] == 0
    isos = {f["properties"]["ISO_A2"] for f in gj["features"]}
    assert isos == {"AA", "BB"}
    props = {f["properties"]["ISO_A2"]: f["properties"] for f in gj["features"]}
    assert props["BB"]["MAPCOLOR7"] == 5
    assert stats["most_displaced"] == [] or stats["most_displaced"][0]["iso"] == "BB"
    # No polygon crosses the antimeridian.
    for f in gj["features"]:
        coords = np.array([c for ring in _rings(f["geometry"]) for c in ring])
        assert coords[:, 0].min() >= -180 and coords[:, 0].max() <= 180


def test_radius_is_chosen_from_candidates_when_not_given():
    real = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": square(0, 0, 20), "properties": {"ISO_A2": "AA", "MAPCOLOR7": 2}}]}
    g = np.arange(0.25, 20, 0.5)
    lat, lon = np.meshgrid(g, g, indexing="ij")
    xyz = latlon_to_xyz(lat.ravel(), lon.ravel())
    _, stats = build_political_map(xyz, xyz, ["AA"] * len(xyz), real, cell_deg=2.0)
    assert stats["settings"]["radius_km"] in [r["radius_km"] for r in stats["radius_table"]]
    assert 0 < stats["model_land_fraction"] < 0.2


def _rings(geom):
    if geom["type"] == "Polygon":
        return geom["coordinates"]
    return [r for poly in geom["coordinates"] for r in poly]
