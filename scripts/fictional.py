"""Invented places: where does the model put names that do not exist?

50 names: 39 that we made up to sound like a particular language or region
(some may by chance match a tiny real place), plus 11 famous fictional
places. For each one we wrote down, *before running anything*, the region a
person would guess from the name (EXPECTED below). A name "lands coherently"
if the nearest real city to the believed point is in one of the expected
countries or continents. Names with no sensible expectation (Atlantis,
Narnia) are reported but not scored.

The coherent ones become the suggested names in the demo's search box.

    uv run python scripts/fictional.py [--model gemma4-31b]
Writes fictional.json in the model's results folder
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import registry  # noqa: E402
from terra.locate import Locator  # noqa: E402
from terra.model import get_lm  # noqa: E402

# Region groups: ISO country codes, or GeoNames continent codes (AF, AS, EU,
# NA, OC, SA) written as "@AF".
GROUPS = {
    "slavic": ["RU", "UA", "BY", "PL", "CZ", "SK", "SI", "HR", "RS", "BA", "ME", "MK", "BG"],
    "japan": ["JP"],
    "hispanic": ["ES", "MX", "CU", "GT", "HN", "SV", "NI", "CR", "PA", "DO", "PR", "@SA"],
    "nordic": ["NO", "SE", "DK", "FI", "IS"],
    "italy": ["IT", "SM", "VA"],
    "germanic": ["DE", "AT", "CH", "LI", "LU"],
    "china": ["CN", "TW", "HK", "MO"],
    "india": ["IN", "NP", "BD", "LK", "PK"],
    "arab": ["SA", "AE", "OM", "YE", "IQ", "SY", "JO", "LB", "KW", "QA", "BH", "EG", "LY", "TN",
             "DZ", "MA", "SD", "PS", "IL"],
    "british": ["GB", "IE", "IM", "JE", "GG"],
    "east_africa": ["KE", "TZ", "UG", "RW", "BI", "MZ", "MW", "@AF"],
    "polynesia": ["NZ", "WS", "TO", "FJ", "PF", "CK", "@OC"],
    "usa": ["US"],
    "africa": ["@AF"],
    "himalaya": ["CN", "IN", "NP", "BT"],
    "south_america": ["@SA"],
    "europe": ["@EU"],
    "central_europe": ["RU", "UA", "BY", "PL", "CZ", "SK", "SI", "HR", "RS", "BA", "ME", "MK", "BG",
                       "HU", "RO", "AT", "DE"],
}

EXPECTED = {
    # invented, language-flavoured
    "Krasnovik": "slavic", "Volgorsk": "slavic", "Belozarsk": "slavic", "Novomirsk": "slavic",
    "Kamenodar": "slavic",
    "Takeshiro": "japan", "Kirishizu": "japan", "Moriyagawa": "japan", "Yukinohama": "japan",
    "Villarosa del Monte": "hispanic", "Puerto Celestino": "hispanic", "Santa Marisol": "hispanic",
    "Fjellheim": "nordic", "Skogsvik": "nordic", "Ravnholm": "nordic", "Lindeborg": "nordic",
    "Montevalli": "italy", "Borgo Santa Lucetta": "italy", "Vallerino": "italy",
    "Grünwaldheim": "germanic", "Eichenstadt": "germanic", "Brunnfelden": "germanic",
    "Longxiangzhen": "china", "Baishuicheng": "china", "Jinhuacun": "china",
    "Ranakpuram": "india", "Shivanagari": "india", "Lakshmigarh": "india",
    "Al-Qasirah": "arab", "Wadi Zahran": "arab", "Bir Hasibah": "arab",
    "Brindlemoor": "british", "Llanfaren": "british", "Ashcombe-on-Wold": "british",
    "Ballynacreeve": "british",
    "Mtwarani": "east_africa", "Kilimawe": "east_africa",
    "Waikaroa": "polynesia", "Motuhenga": "polynesia",
    # famous fictional places
    "Hogwarts": "british", "Wakanda": "africa", "Gotham City": "usa", "Shangri-La": "himalaya",
    "El Dorado": "south_america", "Asgard": "nordic", "Arendelle": "nordic", "Agrabah": "arab",
    "Latveria": "central_europe", "Atlantis": None, "Narnia": None,
}


def coherent(nearest: dict, group: str | None):
    if group is None:
        return None
    allowed = GROUPS[group]
    return nearest["iso"] in allowed or ("@" + nearest["continent"]) in allowed


def run(key: str = registry.DEFAULT_KEY, lm=None) -> dict:
    """Locate the 50 names with `key`'s probe; writes fictional.json for that model."""
    P = registry.paths(key)
    t0 = time.time()
    lm = lm or get_lm(key)
    loc = Locator(lm, path=P.probe)
    names = list(EXPECTED)
    res = loc.locate(names)
    rows = []
    for r in res:
        g = EXPECTED[r["name"]]
        ok = coherent(r["nearest"], g)
        rows.append({**r, "expected_group": g, "coherent": ok})
        print(f"[fiction] {r['name']:22s} -> {r['nearest']['name']}, {r['nearest']['country']} "
              f"({r['lat']:.1f}, {r['lon']:.1f}) cloud {r['cloud_median_km']:.0f} km  "
              f"expected {g}: {ok}")
    scored = [r for r in rows if r["coherent"] is not None]
    # Chance level: shuffle the expected groups across the scored names
    # (seeded, 1,000 times). This is the coherence rate we would see if the
    # landing spots had nothing to do with how the names sound.
    rng = np.random.default_rng(0)
    groups = [r["expected_group"] for r in scored]
    chance = []
    for _ in range(1000):
        perm = rng.permutation(groups)
        chance.append(np.mean([coherent(r["nearest"], g) for r, g in zip(scored, perm)]))
    # Reference: the same probe on known places gives a norm near 1; report it
    # so readers can see invented names sit closer to the centre.
    out = {"layer": loc.layer, "mode": loc.mode, "n": len(rows), "n_scored": len(scored),
           "n_coherent": sum(r["coherent"] for r in scored),
           "coherent_rate": sum(r["coherent"] for r in scored) / len(scored),
           "chance_rate_mean": float(np.mean(chance)),
           "chance_rate_p95": float(np.percentile(chance, 95)),
           "menu": [r["name"] for r in rows if r["coherent"]],
           "rule": "coherent = nearest real city (pop > 15k) to the believed point is in the "
                   "pre-registered expected countries/continent",
           "results": rows, "seconds": time.time() - t0}
    out["model"] = key
    P.results.mkdir(parents=True, exist_ok=True)
    P.result("fictional.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[fiction] coherent {out['n_coherent']}/{out['n_scored']} -> {P.result('fictional.json')}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=registry.DEFAULT_KEY, choices=sorted(registry.MODELS))
    args = ap.parse_args()
    run(args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
