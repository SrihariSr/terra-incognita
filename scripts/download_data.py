"""Download the raw place data into data/ and print row counts.

Sources
-------
1. world_place.csv from github.com/wesg52/world-models (MIT licence).
   This is the place list Gurnee & Tegmark (2023) used to show that language
   models represent space linearly. 39,585 rows with name, type, country,
   latitude and longitude.
2. GeoNames cities15000.zip: every city in the world with population above
   15,000, with coordinates, country code and population. We use it to
   confirm a city's country and to choose well-known cities for edits.
3. GeoNames countryInfo.txt: one row per country with its currency and
   languages. These become the answer keys for the "ripple" questions.
4. GeoNames iso-languagecodes.txt: maps language codes to language names.

Files that already exist are not downloaded again (so this script is safe
to re-run, and never overwrites good data with a failed download).
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCES = {
    "world_place.csv": "https://raw.githubusercontent.com/wesg52/world-models/main/data/entity_datasets/world_place.csv",
    "cities15000.zip": "https://download.geonames.org/export/dump/cities15000.zip",
    "countryInfo.txt": "https://download.geonames.org/export/dump/countryInfo.txt",
    # ISO 639 language codes to language names, so the "Languages" column of
    # countryInfo.txt (e.g. "fr-FR,frp,br") can become words like "French".
    "iso-languagecodes.txt": "https://download.geonames.org/export/dump/iso-languagecodes.txt",
}

# Column names for the GeoNames "geoname" table (documented in the GeoNames
# readme). The file has no header row.
GEONAMES_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]


def fetch(name: str, url: str) -> Path:
    """Download url to data/name unless it is already there."""
    out = DATA / name
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {name} already present ({out.stat().st_size:,} bytes)")
        return out
    print(f"[get ] {url}")
    r = requests.get(url, timeout=120, headers={"User-Agent": "terra-incognita-hackathon"})
    r.raise_for_status()
    # Write to a temporary name first, then rename, so a half-finished
    # download never looks like a complete file.
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.rename(out)
    print(f"[done] {name} ({out.stat().st_size:,} bytes)")
    return out


def main() -> int:
    DATA.mkdir(exist_ok=True)
    paths = {name: fetch(name, url) for name, url in SOURCES.items()}

    # Unzip cities15000.txt next to the zip if needed.
    txt = DATA / "cities15000.txt"
    if not txt.exists():
        with zipfile.ZipFile(paths["cities15000.zip"]) as z:
            z.extract("cities15000.txt", DATA)

    # Row counts, so we can see at a glance that the downloads are complete.
    places = pd.read_csv(paths["world_place.csv"])
    print(f"world_place.csv: {len(places):,} rows, columns = {list(places.columns)}")

    cities = pd.read_csv(txt, sep="\t", header=None, names=GEONAMES_COLUMNS,
                         dtype={"country_code": str}, keep_default_na=False,
                         low_memory=False)
    print(f"cities15000.txt: {len(cities):,} rows")

    # countryInfo.txt starts with many comment lines beginning with '#'. The
    # last comment line is the header.
    lines = paths["countryInfo.txt"].read_text(encoding="utf-8").splitlines()
    header = [l for l in lines if l.startswith("#ISO")][0].lstrip("#").split("\t")
    body = [l for l in lines if l and not l.startswith("#")]
    countries = pd.read_csv(io.StringIO("\n".join(body)), sep="\t", header=None,
                            names=header, keep_default_na=False)
    print(f"countryInfo.txt: {len(countries):,} rows, columns = {header}")

    langs = pd.read_csv(paths["iso-languagecodes.txt"], sep="\t", keep_default_na=False)
    print(f"iso-languagecodes.txt: {len(langs):,} rows, columns = {list(langs.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
