"""Fetch company headquarters and countries' central banks from Wikidata.

For each company label below we ask Wikidata for the item's headquarters
location (property P159), that location's coordinates (P625) and country
(P17, with its ISO code P297). We also fetch every country's central bank
(P1304). Results are cached to data/companies_wikidata.json and
data/central_banks_wikidata.json so the demo never needs the network.

Only the *company names* are typed here; every headquarters, coordinate and
central bank comes from Wikidata.

    uv run python scripts/fetch_companies.py [--refresh]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terra import DATA  # noqa: E402

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "terra-incognita-hackathon/0.1 (research demo)",
           "Accept": "application/sparql-results+json"}

# (short name used in prompts, exact English Wikidata label)
COMPANIES = [
    ("Apple", "Apple Inc."), ("Microsoft", "Microsoft"), ("Amazon", "Amazon"), ("Walmart", "Walmart"),
    ("Coca-Cola", "The Coca-Cola Company"), ("Boeing", "Boeing"), ("Goldman Sachs", "Goldman Sachs"),
    ("JPMorgan Chase", "JPMorgan Chase"), ("Tesla", "Tesla, Inc."), ("Nike", "Nike, Inc."),
    ("Toyota", "Toyota"), ("Sony", "Sony"), ("Nintendo", "Nintendo"), ("Honda", "Honda"),
    ("Samsung Electronics", "Samsung Electronics"), ("Hyundai", "Hyundai Motor Company"),
    ("Alibaba", "Alibaba Group"), ("Tencent", "Tencent"), ("Huawei", "Huawei"),
    ("Infosys", "Infosys"), ("Tata Consultancy Services", "Tata Consultancy Services"),
    ("Reliance Industries", "Reliance Industries"),
    ("Nestlé", "Nestlé"), ("Roche", "Hoffmann-La Roche"), ("Novartis", "Novartis"), ("UBS", "UBS"),
    ("Siemens", "Siemens"), ("Volkswagen", "Volkswagen"), ("BMW", "BMW"), ("SAP", "SAP"),
    ("Deutsche Bank", "Deutsche Bank"), ("Adidas", "Adidas"),
    ("LVMH", "LVMH"), ("TotalEnergies", "TotalEnergies"), ("Airbus", "Airbus"), ("L'Oréal", "L'Oréal"),
    ("HSBC", "HSBC"), ("Barclays", "Barclays"), ("BP", "BP"), ("Unilever", "Unilever"),
    ("Shell", "Shell plc"), ("Rolls-Royce", "Rolls-Royce Holdings"),
    ("Novo Nordisk", "Novo Nordisk"), ("Nokia", "Nokia"), ("Ericsson", "Ericsson"), ("IKEA", "IKEA"),
    ("Spotify", "Spotify"), ("ING", "ING Group"), ("Heineken", "Heineken N.V."),
    ("Banco Santander", "Banco Santander"), ("Zara", "Inditex"),
    ("Petrobras", "Petrobras"), ("Vale", "Vale"), ("Embraer", "Embraer"),
    ("BHP", "BHP"), ("Qantas", "Qantas"), ("Commonwealth Bank", "Commonwealth Bank"),
    ("Naspers", "Naspers"), ("Saudi Aramco", "Saudi Aramco"), ("Emirates", "Emirates"),
    ("Gazprom", "Gazprom"), ("Air New Zealand", "Air New Zealand"), ("Shopify", "Shopify"),
]


def sparql(q: str) -> list[dict]:
    for attempt in range(3):
        try:
            r = requests.get(ENDPOINT, params={"query": q, "format": "json"}, headers=HEADERS, timeout=90)
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as exc:  # retry politely
            print(f"[wikidata] attempt {attempt + 1} failed: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Wikidata query failed")


def main() -> int:
    out_co = DATA / "companies_wikidata.json"
    out_cb = DATA / "central_banks_wikidata.json"
    refresh = "--refresh" in sys.argv
    if refresh or not out_co.exists():
        values = " ".join(f'"{lab}"@en' for _, lab in COMPANIES)
        q = f"""
        SELECT ?item ?label ?hq ?hqLabel ?coord ?iso ?coIso ?links WHERE {{
          VALUES ?label {{ {values} }}
          ?item rdfs:label ?label .
          ?item wdt:P159 ?hq .
          ?item wikibase:sitelinks ?links .
          ?hq wdt:P625 ?coord .
          OPTIONAL {{ ?hq wdt:P17 ?country . ?country wdt:P297 ?iso . }}
          OPTIONAL {{ ?item wdt:P17 ?coCountry . ?coCountry wdt:P297 ?coIso . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}"""
        rows = sparql(q)
        by_label: dict[str, list] = {}
        for b in rows:
            lab = b["label"]["value"]
            coord = b["coord"]["value"]  # "Point(lon lat)"
            lon, lat = map(float, coord.replace("Point(", "").replace(")", "").split())
            by_label.setdefault(lab, []).append({
                "qid": b["item"]["value"].rsplit("/", 1)[-1], "hq": b["hqLabel"]["value"],
                "lat": lat, "lon": lon, "iso": b.get("iso", {}).get("value"),
                "company_iso": b.get("coIso", {}).get("value"), "sitelinks": int(b["links"]["value"])})
        companies = []
        for short, lab in COMPANIES:
            cands = by_label.get(lab, [])
            if not cands:
                print(f"[wikidata] no HQ for {lab}")
                continue
            # Several items can share a label (e.g. a national subsidiary); keep
            # the one with the most Wikipedia sitelinks, i.e. the main company.
            # Among its headquarters values, prefer one in the company's own
            # country (Tencent lists a Cayman registered office and Shenzhen).
            cands.sort(key=lambda c: (-c["sitelinks"], c["iso"] != c["company_iso"], c["iso"] is None))
            best = cands[0]
            companies.append({"name": short, "label": lab, **best})
        out_co.write_text(json.dumps({"source": "Wikidata SPARQL (P159, P625, P17/P297)",
                                      "fetched": time.strftime("%Y-%m-%d %H:%M"),
                                      "companies": companies}, indent=2, ensure_ascii=False))
        print(f"[wikidata] {len(companies)} companies -> {out_co}")
    if not out_cb.exists():
        q = """
        SELECT ?iso ?bank ?bankLabel WHERE {
          ?c wdt:P297 ?iso . ?c wdt:P1304 ?bank .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }"""
        rows = sparql(q)
        banks: dict[str, list[str]] = {}
        for b in rows:
            banks.setdefault(b["iso"]["value"], []).append(b["bankLabel"]["value"])
        out_cb.write_text(json.dumps({"source": "Wikidata SPARQL (P1304 central bank)",
                                      "fetched": time.strftime("%Y-%m-%d %H:%M"),
                                      "banks": banks}, indent=2, ensure_ascii=False))
        print(f"[wikidata] central banks for {len(banks)} countries -> {out_cb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
