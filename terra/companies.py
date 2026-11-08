"""The company layer: the same read-and-edit tools, applied to firms.

Reading: the location probe was fitted on place names only. We apply it,
unchanged, to company names ("What are the lat/lon coordinates of Toyota")
and compare the believed point with the headquarters recorded in Wikidata.

Editing: two ways to move a company, compared by measurement in
scripts/companies_eval.py.

* "city" (the roadmap's tier-1 recipe): at the company name's last token,
  write the residual stream of a *city* in the target country, captured from
  the same question with the city in place of the company. This is the same
  swap as for cities, so it may also replace the company's identity.
* "context" (context transplant): capture the residual stream of the
  company's own last token from a prompt where the name is preceded by
  "the <City>-based company", and write that into the plain prompt. The
  vector still belongs to the company's own token, so the hope is that the
  identity is kept and only the location moves. The words "<City>-based"
  never appear in the prompt the model answers.

Answer keys: headquarters country from Wikidata (P159 -> P17), central bank
from Wikidata (P1304) plus the European Central Bank for euro countries
(currency from GeoNames), currency from GeoNames.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import torch

from . import DATA
from .data import city_info, load_countries
from .locate import nearest_city
from .backends.base import BaseLM as LM, Patch
from .patch import EditPlan, generate_plans, name_positions, plan_edit
from .ripple import _mentions, country_words, currency_words

QUESTIONS = {
    "hq_country": "In which country is {company} headquartered? Answer with just the country name.",
    "central_bank": "Which central bank's interest-rate decisions matter most for {company}? "
                    "Answer with just the name of the central bank.",
    "currency": "In which currency does {company} mainly report its earnings? Answer with just the currency name.",
    "identity": "What is {company} best known for? Answer in five words or fewer.",
}
SCORED = ["hq_country", "central_bank", "currency"]

# Cities companies can be moved to (each must pass the city baseline).
TARGET_CITIES = ["Tokyo", "London", "Sydney", "Mumbai", "São Paulo", "Berlin", "Toronto", "Seoul",
                 "Johannesburg", "Shanghai", "Paris", "Moscow"]


@lru_cache(maxsize=1)
def central_banks() -> dict[str, list[str]]:
    return json.loads((DATA / "central_banks_wikidata.json").read_text())["banks"]


def bank_words(iso: str) -> list[str]:
    words = []
    for b in central_banks().get(iso, []):
        b = b.lower()
        words.append(b)
        words.append(re.sub(r"\s+system$", "", b))           # "federal reserve system" -> "federal reserve"
        words.append(re.sub(r"^the\s+", "", b))
    if load_countries().loc[iso, "CurrencyCode"] == "EUR":
        words += ["european central bank", "ecb"]
    return sorted(set(w for w in words if w))


def load_companies() -> list[dict]:
    """Wikidata companies, with inconsistent records removed.

    Dropped: items with fewer than 10 Wikipedia sitelinks (a subsidiary or
    duplicate item was matched, not the main company), and items whose HQ
    country differs from the company's own country (e.g. a registered office
    in a tax haven). A missing HQ country is filled from the nearest GeoNames
    city to the HQ coordinates.
    """
    raw = json.loads((DATA / "companies_wikidata.json").read_text())["companies"]
    out = []
    for c in raw:
        c = dict(c)
        if c.get("sitelinks", 0) < 10:
            c["dropped"] = "fewer than 10 sitelinks (not the main item)"
        if c["iso"] is None:
            c["iso"] = nearest_city(c["lat"], c["lon"])["iso"]
            c["iso_from"] = "nearest GeoNames city to HQ coordinates"
        if c.get("company_iso") and c["company_iso"] != c["iso"]:
            c["dropped"] = f"HQ country {c['iso']} differs from company country {c['company_iso']}"
        c["hq_country"] = load_countries()["Country"].get(c["iso"], c["iso"])
        out.append(c)
    return out


def keys_for_iso(iso: str) -> dict[str, list[str]]:
    return {"hq_country": country_words(iso), "central_bank": bank_words(iso),
            "currency": currency_words(iso)}


def score(answers: dict[str, str], src_iso: str, tgt_iso: str | None = None) -> dict:
    ks = keys_for_iso(src_iso)
    kt = keys_for_iso(tgt_iso) if tgt_iso else None
    out = {}
    for q in SCORED:
        a = answers[q]
        rec = {"answer": a, "matches_source": _mentions(a, ks[q])}
        if kt is not None:
            rec["matches_target"] = _mentions(a, kt[q])
            rec["countable"] = not (set(ks[q]) & set(kt[q]))
            rec["flipped"] = rec["matches_target"] and not rec["matches_source"]
        out[q] = rec
    return out


def identity_kept(before: str, after: str) -> bool:
    """Rough check that the 'best known for' answer survived the edit:
    the edited answer shares at least one content word (4+ letters) with the
    unedited one."""
    stop = {"with", "from", "that", "this", "known", "best", "their", "company", "products", "and"}
    wb = {w for w in re.findall(r"[a-z]{4,}", before.lower()) if w not in stop}
    wa = {w for w in re.findall(r"[a-z]{4,}", after.lower()) if w not in stop}
    return bool(wb & wa)


def plan_context(lm: LM, user_texts: list[str], company: str, city: str, layers: list[int]) -> EditPlan:
    """Context transplant: the company's own token, read in "the <city>-based company <name>"."""
    src = [lm.chat(t) for t in user_texts]
    tgt = [lm.chat(t.replace(company, f"the {city}-based company {company}")) for t in user_texts]
    src_pos = [name_positions(lm, t, company) for t in src]
    tgt_pos = [name_positions(lm, t, company) for t in tgt]
    reads = lm.residuals_multi(src + tgt, src_pos + tgt_pos, layers)
    h_src = torch.cat(reads[:len(src)], 0)
    h_tgt = torch.cat(reads[len(src):], 0)
    rows = [r for r, ps in enumerate(src_pos) for _ in ps]
    cols = [p for ps in src_pos for p in ps]
    norms = (h_tgt - h_src).norm(dim=-1)
    return EditPlan(source_texts=src,
                    patch=Patch(rows=rows, cols=cols, vectors={L: h_tgt[:, j].contiguous() for j, L in enumerate(layers)}),
                    delta_norms={L: float(norms[:, j].mean()) for j, L in enumerate(layers)}, layers=layers)


def questions_for(company: str) -> list[str]:
    return [QUESTIONS[q].format(company=company) for q in QUESTIONS]


def run_company_edit(lm: LM, layers: list[int], company: str, city: str, variant: str | None = None) -> dict:
    """Before/after answers for one company moved to one city."""
    if variant is None:
        variant = _chosen_variant()
    texts = questions_for(company)
    if variant == "context":
        plan = plan_context(lm, texts, company, city, layers)
    else:
        plan = plan_edit(lm, texts, company, city, layers, mode="swap")
    outs = generate_plans(lm, [[lm.chat(t) for t in texts], plan], max_new_tokens=16)
    before, after = dict(zip(QUESTIONS, outs[0])), dict(zip(QUESTIONS, outs[1]))
    return {"company": company, "target_city": city, "variant": variant,
            "questions": [{"key": q, "question": QUESTIONS[q].format(company=company),
                           "before": before[q], "after": after[q]} for q in QUESTIONS],
            "before": before, "after": after}


def _chosen_variant() -> str:
    from . import RESULTS
    p = RESULTS / "companies.json"
    if p.exists():
        return json.loads(p.read_text()).get("chosen_variant", "city")
    return "city"


def target_iso(city: str) -> str:
    return city_info(city)["iso"]
