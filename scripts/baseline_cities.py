"""Which cities does the unedited model know?

For every GeoNames city with population of at least 1,000,000 (at most 6
per country, largest first, so that China and India do not fill the list),
ask the five ripple questions with no edit and score them against the
GeoNames answer keys. Only cities answered correctly on all five are used
as edit sources or targets anywhere else in the project.

    uv run python scripts/baseline_cities.py
Writes results/baseline_cities.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import RESULTS  # noqa: E402
from terra.data import big_cities, city_info  # noqa: E402
from terra.model import get_lm  # noqa: E402
from terra.ripple import QKEYS, baseline_correct, make_key, prompts_for  # noqa: E402

# Famous cities below 1M people (or whose GeoNames population is the city
# proper only) that we also want available for the demo.
EXTRA = ["Paris", "Sydney", "Melbourne", "Stockholm", "Wellington", "Auckland", "Amsterdam",
         "Lisbon", "Athens", "Montevideo", "Cape Town", "Perth", "Oslo", "Helsinki", "Copenhagen",
         "Dublin", "Edinburgh", "Reykjavik", "Vienna", "Prague", "Geneva", "Brisbane",
         # demo menu and company target cities (scripts/build_pair_menu.py, terra/companies.py)
         "London", "Tokyo", "Berlin", "Rome", "Moscow", "Cairo", "Istanbul", "Beijing", "New York City",
         "Toronto", "Mexico City", "Buenos Aires", "Rio de Janeiro", "Johannesburg", "Lima", "Santiago",
         "Mumbai", "São Paulo", "Seoul", "Shanghai"]


def main() -> int:
    t0 = time.time()
    lm = get_lm()
    c = big_cities(min_population=1_000_000)
    c = c.groupby("country_code", group_keys=False).head(6)
    names = [n for n in dict.fromkeys(list(c["name"]) + EXTRA) if city_info(n) is not None]
    rows = []
    batch = 10
    for b0 in range(0, len(names), batch):
        chunk = names[b0:b0 + batch]
        texts = []
        for n in chunk:
            qs = prompts_for(n)
            texts += [lm.chat(qs[q]) for q in QKEYS]
        outs = lm.generate(texts, max_new_tokens=10)
        for i, n in enumerate(chunk):
            info = city_info(n)
            ans = dict(zip(QKEYS, outs[i * 5:(i + 1) * 5]))
            key = make_key(n, info["iso"], info["latitude"])
            ok = baseline_correct(ans, key)
            rows.append({"name": n, "iso": info["iso"], "latitude": info["latitude"],
                         "longitude": info["longitude"], "population": info["population"],
                         "answers": ans, "correct": ok, "all_correct": all(ok.values()),
                         "key": key.main})
        print(f"[baseline] {b0 + len(chunk)}/{len(names)}  ({time.time() - t0:.0f}s)", flush=True)
    good = [r for r in rows if r["all_correct"]]
    per_q = {q: sum(r["correct"][q] for r in rows) / len(rows) for q in QKEYS}
    out = {"model": lm.model_id, "n_cities": len(rows), "n_all_correct": len(good),
           "per_question_accuracy": per_q, "cities": rows, "seconds": time.time() - t0}
    (RESULTS / "baseline_cities.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[baseline] {len(good)}/{len(rows)} cities all-correct; per question {per_q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
