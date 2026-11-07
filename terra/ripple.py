"""The five "ripple" questions and their automatic scoring.

When we move Paris to Sydney inside the model, a consistent edit should
change *every* downstream fact together: the country, the currency, the
language, the hemisphere, and (because seasons flip between hemispheres)
whether July is summer or winter. These five questions are asked with the
chat template, decoded greedily, and kept short so that a keyword match is
a fair way to score them.

Answer keys are built from GeoNames, not typed by hand:
* country:   GeoNames country name, plus a short list of common aliases
             ("USA", "UK", ...).
* currency:  GeoNames CurrencyName (e.g. "Euro", "Yen", "Rand") and the ISO
             currency code (e.g. "EUR").
* language:  every language GeoNames lists for the country. The *first*
             one is treated as the main language.
* hemisphere: north if the true latitude is above 0, else south.
* season in July: summer in the north, winter in the south.

Scoring an edited answer for the pair (source -> target):
* A question is *countable* only if the source and target keys can be told
  apart (e.g. Toronto -> Sydney: both "Dollar", so currency is not countable).
* An answer *flips* if it matches the target's key and does not match the
  source's key.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .data import country_languages, load_countries

QUESTIONS = {
    "country": "Which country is {city} in? Answer with just the country name.",
    "currency": "What currency is used in {city}? Answer with just the currency name.",
    "language": "What is the main language spoken in {city}? Answer with just the language.",
    "hemisphere": "Is {city} in the Northern or Southern Hemisphere? Answer with one word.",
    "season": "In {city}, is July a summer or winter month? Answer with one word.",
}
QKEYS = list(QUESTIONS)

FREE_QUESTION = "I'm in {city} for Christmas, what should I pack? Answer in two sentences."

# Shown in the demo but not scored: does the city keep its identity?
LANDMARK_QUESTION = "What is the most famous landmark in {city}? Answer with just its name."

# Common ways a country is named that GeoNames' official name does not cover.
COUNTRY_ALIASES = {
    "US": ["united states", "usa", "u.s.", "u.s.a.", "america", "united states of america"],
    "GB": ["united kingdom", "uk", "u.k.", "britain", "great britain", "england", "scotland", "wales",
           "northern ireland"],
    "KR": ["south korea", "republic of korea", "korea"],
    "KP": ["north korea", "dprk"],
    "RU": ["russia", "russian federation"],
    "CN": ["china", "people's republic of china", "prc"],
    "TW": ["taiwan", "republic of china"],
    "CZ": ["czech republic", "czechia"],
    "NL": ["netherlands", "the netherlands", "holland"],
    "AE": ["united arab emirates", "uae"],
    "CD": ["democratic republic of the congo", "dr congo", "drc", "congo-kinshasa"],
    "CG": ["republic of the congo", "congo-brazzaville"],
    "CI": ["ivory coast", "côte d'ivoire", "cote d'ivoire"],
    "MM": ["myanmar", "burma"],
    "VN": ["vietnam", "viet nam"],
    "IR": ["iran"],
    "SY": ["syria"],
    "LA": ["laos"],
    "BO": ["bolivia"],
    "VE": ["venezuela"],
    "TZ": ["tanzania"],
    "MK": ["north macedonia", "macedonia"],
    "TR": ["turkey", "türkiye", "turkiye"],
}

# Common names for currencies, keyed by ISO currency code, beyond GeoNames'
# CurrencyName. Used both ways: an answer saying "rouble" matches RUB.
CURRENCY_ALIASES = {
    "RUB": ["ruble", "rouble"],
    "CNY": ["yuan", "renminbi", "rmb"],
    "GBP": ["pound", "sterling"],
    "EUR": ["euro"],
    "JPY": ["yen"],
    "KRW": ["won"],
    "INR": ["rupee"],
    "BRL": ["real", "reais"],
    "ZAR": ["rand"],
    "PEN": ["sol", "soles", "nuevo sol"],
    "MXN": ["peso"],
    "ARS": ["peso"],
    "CLP": ["peso"],
    "COP": ["peso"],
    "SEK": ["krona", "kronor"],
    "NOK": ["krone", "kroner"],
    "DKK": ["krone", "kroner"],
    "PLN": ["zloty", "złoty"],
    "TRY": ["lira"],
    "CHF": ["franc"],
    "THB": ["baht"],
    "IDR": ["rupiah"],
    "VND": ["dong", "đồng"],
    "VES": ["bolivar", "bolívar"],
    "UAH": ["hryvnia"],
    "EGP": ["pound"],
    "NGN": ["naira"],
    "KES": ["shilling"],
    "SAR": ["riyal"],
    "AED": ["dirham"],
    "MAD": ["dirham"],
    "ILS": ["shekel", "sheqel"],
    "PKR": ["rupee"],
    "BDT": ["taka"],
    "PHP": ["peso", "piso"],
    "MYR": ["ringgit"],
    "HUF": ["forint"],
    "CZK": ["koruna", "crown"],
}

# Language name aliases (what a model might say vs. what GeoNames calls it).
LANGUAGE_ALIASES = {
    "Chinese": ["chinese", "mandarin"],
    "Yue Chinese": ["cantonese"],
    "Persian": ["persian", "farsi"],
    "Modern Greek": ["greek"],
    "Swahili": ["swahili", "kiswahili"],
    "Malay": ["malay", "bahasa malaysia"],
    "Indonesian": ["indonesian", "bahasa indonesia"],
    "Filipino": ["filipino", "tagalog"],
    "Tagalog": ["tagalog", "filipino"],
    "Panjabi": ["punjabi", "panjabi"],
    "Norwegian": ["norwegian"],
    "Norwegian Bokmål": ["norwegian"],
}


def _norm(text: str) -> str:
    # NFC so that "đồng" typed with combining accents equals the precomposed form.
    text = unicodedata.normalize("NFC", text).lower()
    return " " + re.sub(r"[^\w\s'.-]", " ", text) + " "


def _mentions(text: str, words: list[str]) -> bool:
    """Whole-word, case-insensitive match of any phrase in `words`."""
    t = _norm(text)
    for w in words:
        w = unicodedata.normalize("NFC", w).lower().strip()
        if not w:
            continue
        if re.search(r"(?<![\w])" + re.escape(w) + r"(?![\w])", t):
            return True
    return False


@dataclass
class Key:
    """Accepted answers for one city, per question."""

    city: str
    iso: str
    latitude: float
    words: dict[str, list[str]] = field(default_factory=dict)
    main: dict[str, str] = field(default_factory=dict)  # one canonical answer per question


def country_words(iso: str) -> list[str]:
    name = load_countries().loc[iso, "Country"]
    return [name.lower()] + COUNTRY_ALIASES.get(iso, [])


def currency_words(iso: str) -> list[str]:
    c = load_countries().loc[iso]
    code = c["CurrencyCode"]
    words = [c["CurrencyName"].lower(), code.lower()] + CURRENCY_ALIASES.get(code, [])
    # "Yuan Renminbi" -> also accept each word on its own.
    words += [w for w in c["CurrencyName"].lower().split() if len(w) > 2]
    return sorted(set(words))


def language_words(iso: str) -> list[str]:
    out = []
    for name in country_languages(iso):
        out.append(name.lower())
        out += LANGUAGE_ALIASES.get(name, [])
    return out


def make_key(city: str, iso: str, latitude: float) -> Key:
    c = load_countries().loc[iso]
    north = latitude > 0
    langs = country_languages(iso)
    k = Key(city=city, iso=iso, latitude=latitude)
    k.words = {
        "country": country_words(iso),
        "currency": currency_words(iso),
        "language": language_words(iso),
        "hemisphere": ["north", "northern"] if north else ["south", "southern"],
        "season": ["summer"] if north else ["winter"],
    }
    k.main = {
        "country": c["Country"],
        "currency": f'{c["CurrencyName"]} ({c["CurrencyCode"]})',
        "language": langs[0] if langs else "?",
        "hemisphere": "Northern" if north else "Southern",
        "season": "Summer" if north else "Winter",
    }
    return k


def matches(answer: str, key: Key, q: str) -> bool:
    return _mentions(answer, key.words[q])


def countable(src: Key, tgt: Key, q: str) -> bool:
    """Can an answer to q tell the source and target apart?

    For the language question we require that the source's main language is
    not one of the target's languages and vice versa; for the rest we require
    that the two keys share no accepted word.
    """
    if q == "language":
        a = LANGUAGE_ALIASES.get(src.main["language"], [src.main["language"].lower()])
        b = LANGUAGE_ALIASES.get(tgt.main["language"], [tgt.main["language"].lower()])
        return not (set(a) & set(tgt.words["language"])) and not (set(b) & set(src.words["language"]))
    return not (set(src.words[q]) & set(tgt.words[q]))


def score_edit(answers: dict[str, str], src: Key, tgt: Key) -> dict:
    """Per-question flip results for an edited (or control) run."""
    per_q = {}
    for q in QKEYS:
        a = answers[q]
        per_q[q] = {
            "answer": a,
            "countable": countable(src, tgt, q),
            "matches_target": matches(a, tgt, q),
            "matches_source": matches(a, src, q),
        }
        per_q[q]["flipped"] = per_q[q]["matches_target"] and not per_q[q]["matches_source"]
    n_count = sum(v["countable"] for v in per_q.values())
    n_flip = sum(v["flipped"] for v in per_q.values() if v["countable"])
    n_kept = sum(v["matches_source"] and not v["matches_target"] for v in per_q.values() if v["countable"])
    return {"per_question": per_q, "n_countable": n_count, "n_flipped": n_flip, "n_source": n_kept}


def baseline_correct(answers: dict[str, str], key: Key) -> dict[str, bool]:
    return {q: matches(answers[q], key, q) for q in QKEYS}


def prompts_for(city: str) -> dict[str, str]:
    return {q: t.format(city=city) for q, t in QUESTIONS.items()}
