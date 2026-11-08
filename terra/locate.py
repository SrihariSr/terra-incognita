"""Where does the model believe a name is? Works for any string.

One forward pass with the chosen prompt template gives the residual stream
at the last token of the name, at the chosen layer. The final probe maps it
to a point on the unit sphere (xyz = h @ A + c, then normalised), which we
convert to latitude and longitude.

The bootstrap ensemble (20 probes fitted on resampled places) gives 20 such
points. Their spread is the "uncertainty cloud" on the globe. It measures
how much the *probe* depends on which places it was trained on; a wide cloud
means the reading is fragile, not that the model "feels unsure".

To describe the answer in words we also report the nearest real GeoNames
city (population above 15,000) to the believed point.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import ROOT
from .activations import prompt_and_position
from .data import load_cities, load_countries
from .backends.base import BaseLM as LM
from .probe import great_circle_km, latlon_to_xyz, xyz_to_latlon


class Locator:
    def __init__(self, lm: LM, path=ROOT / "models" / "probe_final.npz"):
        z = np.load(path)
        self.lm = lm
        self.A = z["A"].astype(np.float64)
        self.c = z["c"].astype(np.float64)
        self.boot_A = z["boot_A"].astype(np.float64)
        self.boot_c = z["boot_c"].astype(np.float64)
        self.layer = int(z["layer"])
        self.mode = str(z["mode"])

    def hidden(self, names: list[str]) -> np.ndarray:
        texts, pos = zip(*[prompt_and_position(self.lm, n, self.mode) for n in names])
        h = self.lm.residuals(list(texts), list(pos), [self.layer])
        return h[:, 0].numpy().astype(np.float64)

    def from_hidden(self, h: np.ndarray) -> list[dict]:
        xyz = h @ self.A + self.c
        lat, lon = xyz_to_latlon(xyz)
        cloud = np.einsum("nd,bdk->nbk", h, self.boot_A) + self.boot_c[None]  # [n, B, 3]
        clat, clon = xyz_to_latlon(cloud)
        out = []
        for i in range(len(h)):
            spread = great_circle_km(cloud[i], np.broadcast_to(xyz[i], cloud[i].shape))
            near = nearest_city(float(lat[i]), float(lon[i]))
            out.append({
                "lat": float(lat[i]), "lon": float(lon[i]),
                # Length of the raw probe output: well-known places land near 1,
                # names the probe cannot place land nearer the centre (0).
                "norm": float(np.linalg.norm(xyz[i])),
                "cloud": [[float(a), float(b)] for a, b in zip(clat[i], clon[i])],
                "cloud_median_km": float(np.median(spread)),
                "nearest": near,
            })
        return out

    def locate(self, names: list[str]) -> list[dict]:
        res = self.from_hidden(self.hidden(names))
        for n, r in zip(names, res):
            r["name"] = n
        return res


@lru_cache(maxsize=1)
def _city_xyz():
    c = load_cities()
    c = c[c["population"] >= 15000].reset_index(drop=True)
    return c, latlon_to_xyz(c["latitude"].astype(float), c["longitude"].astype(float))


def nearest_city(lat: float, lon: float) -> dict:
    c, xyz = _city_xyz()
    p = latlon_to_xyz(lat, lon)
    i = int(np.argmax(xyz @ p))
    iso = c.loc[i, "country_code"]
    countries = load_countries()
    return {"name": c.loc[i, "name"], "iso": iso,
            "country": countries["Country"].get(iso, iso),
            "continent": countries["Continent"].get(iso, "?"),
            "km": float(great_circle_km(xyz[i], p))}
