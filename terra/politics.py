"""The model's political map: the world's countries where the model believes they are.

The probe misses by hundreds of kilometres on average. Instead of hiding
that, we draw it: every place is put where the model's hidden state says it
is (the out-of-fold probe reading, so a probe that never saw the place), and
each place carries its real country. From that cloud of labelled points we
draw borders.

Steps
-----
1. Grid. Cover the globe with cells of `cell_deg` degrees (1 by default).
   Each cell centre becomes a point on the unit sphere, like the places.
2. Neighbours. For each cell, find the k = 12 nearest believed places. We
   compare chord lengths |a - b| between unit vectors; for points on a sphere
   the chord is 2 sin(theta / 2) for an angle theta between them, which grows
   with theta, so the nearest by chord are the nearest along the surface.
3. Land or sea in the model's world. A cell is land if its k-th neighbour is
   closer than R km: enough places are believed to be near it. R is chosen
   from a list of candidates so that the share of the globe that is land in
   the model's world is as close as possible to the real share on the same
   grid (Antarctica left out of both, since almost no places are there).
4. Country. Each land cell takes the distance-weighted vote of its k
   neighbours' real countries (weight 1 / distance), then one majority-filter
   pass replaces a cell's label by the label held by at least 5 of the 9
   cells in its 3 x 3 neighbourhood, which removes speckle.
5. Areas. A cell's area is proportional to cos(latitude): cells shrink
   towards the poles. Every area figure below uses that weight.
6. Polygons. Each country's cells are joined into shapes (shapely), lightly
   simplified, and pieces smaller than 3 cells are dropped. Cells stop at the
   +-180 degree line, so no shape crosses it.

Stats, per country:
* displacement: great-circle distance from the centroid of the true
  positions of its places to the centroid of their believed positions;
* precision: the share of the model's territory for the country that lies
  inside the real country;
* recall: the share of the real country that the model gives to it.
Real countries are rasterised on the same grid by testing cell centres
against web/vendor/countries.geojson.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

EARTH_RADIUS_KM = 6371.0
RADIUS_CANDIDATES = (200, 250, 300, 350, 400, 450, 500, 600, 700, 800)
COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


# ------------------------------------------------------------------ geometry

def latlon_to_xyz(lat, lon) -> np.ndarray:
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def xyz_to_latlon(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz = xyz / np.clip(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-12, None)
    return np.degrees(np.arcsin(np.clip(xyz[..., 2], -1, 1))), np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))


def km_to_chord(km: float) -> float:
    """Chord length on the unit sphere for a surface distance in km."""
    return 2.0 * np.sin(km / EARTH_RADIUS_KM / 2.0)


def great_circle_km(a_latlon, b_latlon) -> float:
    a, b = latlon_to_xyz(*a_latlon), latlon_to_xyz(*b_latlon)
    return float(EARTH_RADIUS_KM * np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def bearing_word(a_latlon, b_latlon) -> str:
    """Compass direction of travel from a to b (initial great-circle bearing)."""
    la1, lo1 = np.radians(a_latlon)
    la2, lo2 = np.radians(b_latlon)
    y = np.sin(lo2 - lo1) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)
    deg = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return COMPASS[int((deg + 22.5) // 45) % 8]


def mean_direction(xyz: np.ndarray):
    """Centroid of points on a sphere: the normalised mean vector, as lat/lon."""
    m = xyz.mean(0)
    lat, lon = xyz_to_latlon(m)
    return float(lat), float(lon)


def make_grid(cell_deg: float):
    """Cell centres (lat, lon) of a regular grid, shape [n_lat, n_lon]."""
    lats = np.arange(-90 + cell_deg / 2, 90, cell_deg)
    lons = np.arange(-180 + cell_deg / 2, 180, cell_deg)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    return LAT, LON


# ------------------------------------------------------------- real borders

def feature_iso(props: dict) -> str | None:
    """ISO alpha-2 of a Natural Earth feature (its ISO_A2 is "-99" for France and Norway)."""
    for k in ("ISO_A2", "ISO_A2_EH", "WB_A2"):
        v = props.get(k)
        if isinstance(v, str) and len(v) == 2 and v.isalpha():
            return v.upper()
    try:
        from .data import iso_for
        return iso_for(str(props.get("ADMIN") or props.get("NAME") or ""))
    except Exception:
        return None


def rasterise_real(real_geojson: dict, LAT: np.ndarray, LON: np.ndarray):
    """ISO code of the real country containing each cell centre (None for sea)."""
    import shapely
    from shapely.geometry import shape
    geoms, isos = [], []
    for f in real_geojson["features"]:
        iso = feature_iso(f.get("properties", {}))
        if iso is None:
            continue
        geoms.append(shape(f["geometry"]))
        isos.append(iso)
    pts = shapely.points(LON.ravel(), LAT.ravel())
    tree = shapely.STRtree(geoms)
    pi, gi = tree.query(pts, predicate="within")
    out = np.full(LAT.size, None, dtype=object)
    out[pi] = np.asarray(isos, dtype=object)[gi]
    return out.reshape(LAT.shape)


def real_colours(real_geojson: dict) -> dict:
    return {feature_iso(f["properties"]): f["properties"].get("MAPCOLOR7")
            for f in real_geojson["features"] if feature_iso(f["properties"])}


def real_names(real_geojson: dict) -> dict:
    return {feature_iso(f["properties"]): f["properties"].get("ADMIN") or f["properties"].get("NAME")
            for f in real_geojson["features"] if feature_iso(f["properties"])}


# ---------------------------------------------------------------- the model

def vote(nbr_idx: np.ndarray, nbr_dist: np.ndarray, labels: np.ndarray, n_codes: int) -> np.ndarray:
    """Distance-weighted vote: each neighbour adds 1 / distance to its country's score."""
    w = 1.0 / (nbr_dist + 1e-4)
    lab = labels[nbr_idx]                                  # [n_cells, k]
    scores = np.zeros((len(lab), n_codes))
    rows = np.repeat(np.arange(len(lab)), lab.shape[1])
    np.add.at(scores, (rows, lab.ravel()), w.ravel())
    return scores.argmax(1)


def majority_filter(grid: np.ndarray, min_votes: int = 5) -> np.ndarray:
    """One pass: a land cell takes the label held by >= min_votes of its 3 x 3 block.

    Sea is -1 and never changes. Longitude wraps around; latitude does not.
    """
    H, W = grid.shape
    padded = np.full((H + 2, W), -1, dtype=grid.dtype)
    padded[1:-1] = grid
    blocks = [np.roll(padded, -dx, axis=1)[1 + dy:H + 1 + dy] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    stack = np.stack(blocks)                               # [9, H, W]
    out = grid.copy()
    for i, j in zip(*np.nonzero(grid >= 0)):
        votes = Counter(int(v) for v in stack[:, i, j] if v >= 0)
        lab, n = votes.most_common(1)[0]
        if n >= min_votes and lab != grid[i, j]:
            out[i, j] = lab
    return out


def country_shapes(grid: np.ndarray, code: int, cell_deg: float, min_cells: int = 3, simplify: float = 0.25):
    """Union of one country's cells as a shapely geometry (None if nothing is left)."""
    import shapely
    from shapely.geometry import box
    H, W = grid.shape
    boxes = []
    for i in range(H):
        row = grid[i] == code
        if not row.any():
            continue
        lat0 = -90 + i * cell_deg
        # Merge runs of consecutive cells in a row first (far fewer shapes to union).
        j = 0
        while j < W:
            if row[j]:
                k = j
                while k + 1 < W and row[k + 1]:
                    k += 1
                boxes.append(box(-180 + j * cell_deg, lat0, -180 + (k + 1) * cell_deg, lat0 + cell_deg))
                j = k + 1
            else:
                j += 1
    if not boxes:
        return None
    geom = shapely.union_all(boxes)
    parts = list(getattr(geom, "geoms", [geom]))
    keep = [p for p in parts if p.area / (cell_deg ** 2) >= min_cells - 1e-9]
    if not keep:
        return None
    geom = shapely.union_all(keep).simplify(simplify * cell_deg, preserve_topology=True)
    return None if geom.is_empty else geom


def build_political_map(believed_xyz: np.ndarray, true_xyz: np.ndarray, iso: list, real_geojson: dict,
                        cell_deg: float = 1.0, k: int = 12, radius_km: float | None = None,
                        candidates=RADIUS_CANDIDATES, names: dict | None = None, model_name: str = "the model"):
    """Return (GeoJSON FeatureCollection, stats dict). See the module docstring."""
    from scipy.spatial import cKDTree
    from shapely.geometry import mapping

    iso = np.asarray([i if isinstance(i, str) and i else None for i in iso], dtype=object)
    ok = np.array([i is not None for i in iso])
    bxyz = believed_xyz[ok] / np.linalg.norm(believed_xyz[ok], axis=1, keepdims=True)
    txyz = true_xyz[ok] / np.linalg.norm(true_xyz[ok], axis=1, keepdims=True)
    iso = iso[ok]
    codes = sorted(set(iso))
    code_of = {c: n for n, c in enumerate(codes)}
    labels = np.array([code_of[c] for c in iso])

    LAT, LON = make_grid(cell_deg)
    cells = latlon_to_xyz(LAT.ravel(), LON.ravel())
    w = np.cos(np.radians(LAT.ravel()))
    cell_km2 = (np.pi * EARTH_RADIUS_KM / 180 * cell_deg) ** 2   # area of a cell at the equator
    k_eff = min(k, len(bxyz))
    dist, idx = cKDTree(bxyz).query(cells, k=k_eff)
    dist = dist.reshape(len(cells), k_eff)
    idx = idx.reshape(len(cells), k_eff)
    kth = dist[:, -1]

    real = rasterise_real(real_geojson, LAT, LON).ravel()
    not_ant = np.array([r != "AQ" for r in real])
    real_land = np.array([r is not None for r in real]) & not_ant
    real_frac = float((w * real_land).sum() / (w * not_ant).sum())

    table = []
    for R in candidates:
        land = (kth < km_to_chord(R)) & not_ant
        table.append({"radius_km": R, "land_fraction": float((w * land).sum() / (w * not_ant).sum())})
    if radius_km is None:
        radius_km = min(table, key=lambda r: abs(r["land_fraction"] - real_frac))["radius_km"]
    land = kth < km_to_chord(radius_km)

    lab = np.full(len(cells), -1)
    lab[land] = vote(idx[land], dist[land], labels, len(codes))
    grid = majority_filter(lab.reshape(LAT.shape))
    lab = grid.ravel()
    model_frac = float((w * (lab >= 0) * not_ant).sum() / (w * not_ant).sum())

    colours = real_colours(real_geojson)
    ne_names = real_names(real_geojson)
    names = names or {}
    features, rows = [], []
    for c in codes:
        n = code_of[c]
        in_model = lab == n
        in_real = real == c
        model_area = float((w * in_model).sum() * cell_km2)
        real_area = float((w * in_real).sum() * cell_km2)
        overlap = float((w * (in_model & in_real)).sum() * cell_km2)
        sel = labels == n
        b_cent = mean_direction(bxyz[sel])
        t_cent = mean_direction(txyz[sel])
        disp = great_circle_km(t_cent, b_cent)
        # Where does the misplaced territory lie, in reality?
        outside = in_model & ~in_real
        if outside.any():
            where = Counter("sea" if r is None else r for r in real[outside]).most_common(1)[0][0]
        else:
            where = None
        name = names.get(c) or ne_names.get(c) or c
        row = {"iso": c, "name": name, "n_places": int(sel.sum()), "n_cells": int(in_model.sum()),
               "believed_centroid": [round(b_cent[0], 3), round(b_cent[1], 3)],
               "true_centroid": [round(t_cent[0], 3), round(t_cent[1], 3)],
               "displacement_km": round(disp, 1), "direction": bearing_word(t_cent, b_cent),
               "model_area_km2": round(model_area), "real_area_km2": round(real_area),
               "overlap_km2": round(overlap),
               "precision": round(overlap / model_area, 4) if model_area > 0 else None,
               "recall": round(overlap / real_area, 4) if real_area > 0 else None,
               "imagined_km2": round(model_area - overlap),
               "imagined_mostly_over": None if where is None else ("sea" if where == "sea" else
                                                                     names.get(where) or ne_names.get(where) or where)}
        rows.append(row)
        geom = country_shapes(grid, n, cell_deg)
        if geom is None:
            continue
        features.append({"type": "Feature", "geometry": mapping(geom), "properties": {
            "ISO_A2": c, "name": name, "n_places": row["n_places"],
            "believed_centroid": row["believed_centroid"], "true_centroid": row["true_centroid"],
            "displacement_km": row["displacement_km"], "direction": row["direction"],
            "precision": row["precision"], "recall": row["recall"],
            "model_area_km2": row["model_area_km2"],
            "MAPCOLOR7": colours.get(c) or (sum(map(ord, c)) % 7 + 1)}})

    total_model = float((w * (lab >= 0)).sum() * cell_km2)
    total_overlap = float(sum(r["overlap_km2"] for r in rows))
    big = [r for r in rows if r["n_places"] >= 20]
    stats = {
        "model": model_name,
        "source": "out-of-fold probe readings (random 5-fold) of every place, labelled with its real country",
        "caveat": ("Countries with many places in the list (e.g. the US, the UK, India, Australia) win more "
                   "neighbour votes and grow; countries with few places shrink or vanish. The map shows the "
                   "model's beliefs as read by a linear probe, filtered through the place list's coverage."),
        "settings": {"cell_deg": cell_deg, "k": k_eff, "radius_km": radius_km, "majority_filter": "3x3, >= 5 of 9",
                     "min_piece_cells": 3, "radius_rule": "land share closest to the real land share (Antarctica excluded)"},
        "radius_table": table,
        "real_land_fraction": round(real_frac, 4),
        "model_land_fraction": round(model_frac, 4),
        "n_places": int(len(labels)),
        "n_countries_with_places": len(codes),
        "n_countries_drawn": len(features),
        "share_of_model_land_in_right_country": round(total_overlap / total_model, 4) if total_model else None,
        "median_displacement_km_20plus_places": float(np.median([r["displacement_km"] for r in big])) if big else None,
        "most_displaced": sorted(big, key=lambda r: -r["displacement_km"])[:15],
        "vanished": [{"iso": r["iso"], "name": r["name"], "n_places": r["n_places"]}
                     for r in sorted(rows, key=lambda r: -r["n_places"]) if r["n_places"] >= 10 and r["n_cells"] == 0],
        "imagined": [{"iso": r["iso"], "name": r["name"], "imagined_km2": r["imagined_km2"],
                      "precision": r["precision"], "mostly_over": r["imagined_mostly_over"]}
                     for r in sorted(rows, key=lambda r: -r["imagined_km2"])[:15] if r["imagined_km2"] > 0],
        "countries": rows,
    }
    return {"type": "FeatureCollection", "features": features}, stats


def build_for_model(key: str, cell_deg: float = 1.0, k: int = 12, radius_km: float | None = None) -> dict:
    """Read a model's out-of-fold positions, build its map, write the two files."""
    from . import ROOT, registry
    from .data import all_places, load_countries
    P = registry.paths(key)
    z = np.load(P.result("probe_oof.npz"))
    places = all_places().set_index("pid")
    pid = z["pid"]
    iso = places.loc[pid, "iso"].tolist()
    real = json.loads((ROOT / "web" / "vendor" / "countries.geojson").read_text())
    names = load_countries()["Country"].to_dict()
    gj, stats = build_political_map(z["random"], z["true"], iso, real, cell_deg=cell_deg, k=k,
                                    radius_km=radius_km, names=names, model_name=registry.spec(key).display)
    P.result("political_map.geojson").write_text(json.dumps(gj))
    P.result("political_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats
