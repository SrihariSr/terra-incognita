"""Load the place lists and the GeoNames country facts.

Three kinds of data are used throughout the project:

* Places with true coordinates (what we fit probes on and draw on the globe).
  These come from Gurnee & Tegmark's world_place.csv, plus the largest
  GeoNames cities, because some famous cities ("Paris", "Sydney", "Buenos
  Aires") are missing from world_place.csv under their plain names.
* Cities with their country code (to build the ripple answer keys).
* Country facts (currency, languages) for the answer keys.

Geometry helpers (unit-sphere coordinates and great-circle distance) live in
probe.py, next to the maths that uses them.
"""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
import pandas as pd

from . import DATA

GEONAMES_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]


@lru_cache(maxsize=1)
def load_world_places() -> pd.DataFrame:
    """Gurnee & Tegmark's place list with a few tidy columns added."""
    df = pd.read_csv(DATA / "world_place.csv")
    df = df.dropna(subset=["name", "latitude", "longitude"]).copy()
    df["name"] = df["name"].astype(str).str.strip()
    df["country"] = df["country"].fillna("Unknown").astype(str)
    df["source"] = "world_place"
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_cities() -> pd.DataFrame:
    """GeoNames cities with population above 15,000."""
    df = pd.read_csv(DATA / "cities15000.txt", sep="\t", header=None, names=GEONAMES_COLUMNS,
                     dtype={"country_code": str}, keep_default_na=False, low_memory=False)
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    return df


@lru_cache(maxsize=1)
def load_countries() -> pd.DataFrame:
    """GeoNames countryInfo.txt as a table indexed by ISO code."""
    lines = (DATA / "countryInfo.txt").read_text(encoding="utf-8").splitlines()
    header = [l for l in lines if l.startswith("#ISO")][0].lstrip("#").split("\t")
    body = [l for l in lines if l and not l.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(body)), sep="\t", header=None, names=header,
                     keep_default_na=False, dtype=str)
    return df.set_index("ISO")


@lru_cache(maxsize=1)
def language_names() -> dict[str, str]:
    """ISO 639 code (1, 2 or 3 letters) -> English language name."""
    df = pd.read_csv(DATA / "iso-languagecodes.txt", sep="\t", keep_default_na=False, dtype=str)
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        name = r["Language Name"].split(";")[0].split("(")[0].strip()
        for col in ("ISO 639-1", "ISO 639-2", "ISO 639-3"):
            code = r[col].strip()
            if code and code not in out:
                out[code] = name
    return out


def country_languages(iso: str) -> list[str]:
    """Language names for a country, in GeoNames' order (main language first)."""
    codes = load_countries().loc[iso, "Languages"].split(",")
    names = language_names()
    out = []
    for c in codes:
        base = c.split("-")[0].strip()
        n = names.get(base)
        if n and n not in out:
            out.append(n)
    return out


@lru_cache(maxsize=1)
def country_name_to_iso() -> dict[str, str]:
    """Map world_place.csv country strings (e.g. 'United_States') to ISO codes."""
    c = load_countries()
    m = {name.lower(): iso for iso, name in c["Country"].items()}
    extra = {
        "england": "GB", "scotland": "GB", "wales": "GB", "northern ireland": "GB",
        "united states": "US", "czech republic": "CZ", "republic of ireland": "IE",
        "people's republic of china": "CN", "republic of the congo": "CG",
        "democratic republic of the congo": "CD", "ivory coast": "CI",
        "the gambia": "GM", "gambia": "GM", "the bahamas": "BS", "bahamas": "BS",
        "east timor": "TL", "macedonia": "MK", "north macedonia": "MK",
        "burma": "MM", "myanmar": "MM", "vatican city": "VA", "eswatini": "SZ",
        "swaziland": "SZ", "palestine": "PS", "state of palestine": "PS",
        "federated states of micronesia": "FM", "cape verde": "CV",
        "são tomé and príncipe": "ST", "taiwan": "TW", "kosovo": "XK",
        "south korea": "KR", "north korea": "KP", "russia": "RU",
        "georgia (country)": "GE", "georgia": "GE",
        "netherlands": "NL", "united states of america": "US", "kingdom of denmark": "DK",
        "bosnia & herzegovina": "BA", "bosnia and herzegovina": "BA", "dr congo": "CD",
        "mandatory palestine": "PS",
    }
    # world_place.csv sometimes gives an English or Welsh county instead of a
    # country. These are the common ones; all are in the United Kingdom.
    for county in ["derbyshire", "devon", "cambridgeshire", "north yorkshire", "kent",
                   "oxfordshire", "norfolk", "surrey", "lancashire", "hampshire", "suffolk",
                   "essex", "somerset", "west sussex", "east sussex", "cornwall", "dorset",
                   "wiltshire", "gloucestershire", "cumbria", "lincolnshire", "shropshire",
                   "herefordshire", "worcestershire", "warwickshire", "staffordshire",
                   "northumberland", "west yorkshire", "south yorkshire", "east riding of yorkshire",
                   "nottinghamshire", "leicestershire", "northamptonshire", "buckinghamshire",
                   "hertfordshire", "bedfordshire", "berkshire", "cheshire", "durham",
                   "county durham", "isle of wight", "greater manchester", "merseyside",
                   "greater london", "london", "rutland", "tyne and wear", "west midlands"]:
        extra[county] = "GB"
    m.update(extra)
    return m


def iso_for(country: str) -> str | None:
    return country_name_to_iso().get(country.replace("_", " ").strip().lower())


@lru_cache(maxsize=1)
def big_cities(min_population: int = 500_000) -> pd.DataFrame:
    """Large GeoNames cities, in the same column layout as world_place.

    Used both to fill gaps in the place map and as the menu of edit cities.
    When two cities share a name we keep the most populous one (so "London"
    is London, England, not London, Ontario).
    """
    c = load_cities()
    c = c[c["population"] >= min_population].sort_values("population", ascending=False)
    c = c.drop_duplicates(subset=["name"], keep="first")
    countries = load_countries()["Country"]
    out = pd.DataFrame({
        "name": c["name"].values,
        "entity_subtype": "City",
        "country": [countries.get(cc, cc) for cc in c["country_code"]],
        "country_code": c["country_code"].values,
        "latitude": c["latitude"].astype(float).values,
        "longitude": c["longitude"].astype(float).values,
        "page_views": np.nan,
        "population": c["population"].values,
        "source": "geonames",
    })
    return out.reset_index(drop=True)


def all_places() -> pd.DataFrame:
    """world_place.csv plus large GeoNames cities whose names it lacks.

    Each place gets a stable integer `pid` (its row number here), which is
    the index into every activation file.
    """
    wp = load_world_places()
    gn = big_cities()
    gn = gn[~gn["name"].isin(set(wp["name"]))]
    df = pd.concat([wp, gn], ignore_index=True, sort=False)
    df["iso"] = [iso_for(c) if s == "world_place" else cc
                 for c, s, cc in zip(df["country"], df["source"], df.get("country_code", pd.Series([None] * len(df))))]
    df["pid"] = np.arange(len(df))
    keep = ["pid", "name", "entity_subtype", "country", "iso", "latitude", "longitude",
            "page_views", "population", "is_test", "source"]
    return df[keep]


def subset_3k(df: pd.DataFrame, n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A spatially spread sample of about n places for the first probe sweep.

    Method: cut the globe into 15 x 15 degree cells. Every large GeoNames
    city is kept (they are the cities we will edit), then we fill the rest
    by going round the cells in turn, each time taking the next most-viewed
    place in that cell. Round-robin over cells stops the US, UK and India
    (which dominate world_place.csv) from swamping the sample. Cities and
    other kinds of place (landmarks, lakes, buildings) are mixed roughly
    two to one.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["cell"] = ((df["latitude"] + 90) // 15).astype(int) * 100 + ((df["longitude"] + 180) // 15).astype(int)
    big = df[df["source"] == "geonames"]
    rest = df[df["source"] == "world_place"].copy()
    rest["kind"] = np.where(rest["entity_subtype"] == "PopulatedPlace", "city", "other")
    # Random tie-breaker so equal page views do not always resolve the same way.
    rest["r"] = rng.random(len(rest))
    rest = rest.sort_values(["page_views", "r"], ascending=[False, True])
    chosen = list(big["pid"])
    budget = {"city": int((n - len(chosen)) * 2 / 3), "other": (n - len(chosen)) - int((n - len(chosen)) * 2 / 3)}
    for kind, k in budget.items():
        pool = {cell: list(g["pid"]) for cell, g in rest[rest["kind"] == kind].groupby("cell")}
        taken = 0
        while taken < k and any(pool.values()):
            for cell in sorted(pool):
                if pool[cell] and taken < k:
                    chosen.append(pool[cell].pop(0))
                    taken += 1
    return df[df["pid"].isin(chosen)].drop(columns=["cell"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _city_index() -> dict[str, dict]:
    """name or ASCII name -> the most populous GeoNames city with that name."""
    c = load_cities().sort_values("population", ascending=False)
    idx: dict[str, dict] = {}
    for _, r in c.iterrows():
        rec = {"name": r["name"], "iso": r["country_code"], "latitude": float(r["latitude"]),
               "longitude": float(r["longitude"]), "population": int(r["population"])}
        for key in (r["name"], r["asciiname"]):
            if key and key not in idx:
                idx[key] = rec
    return idx


def city_info(name: str) -> dict | None:
    """Country code, coordinates and population of a city, from GeoNames."""
    return _city_index().get(name)
