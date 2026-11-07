"""Does the model's internal map disagree with what it says?

The unedited model answers "Southern" for some northern cities (Mumbai,
Seoul, Guangzhou in results/baseline_cities.json). For every baseline city,
compare its spoken hemisphere answer with the sign of the latitude the
location probe reads from its hidden state. We use the out-of-fold
prediction (results/probe_oof.npz, random 5-fold), so each city is read by a
probe that never saw it during fitting.

    uv run python scripts/hemisphere_check.py
Writes results/hemisphere_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from terra import ACTS, RESULTS  # noqa: E402
from terra.probe import xyz_to_latlon  # noqa: E402


def main() -> int:
    base = json.loads((RESULTS / "baseline_cities.json").read_text())["cities"]
    places = pd.read_csv(ACTS / "places_full.csv")
    z = np.load(RESULTS / "probe_oof.npz")
    lat_oof, _ = xyz_to_latlon(z["random"])
    pid_of = {}
    for pid, name in zip(places["pid"], places["name"]):
        pid_of.setdefault(name, pid)
    rows = []
    for c in base:
        # Same name must also be the same place (within 2 degrees).
        if c["name"] not in pid_of or abs(places.loc[pid_of[c["name"]], "latitude"] - c["latitude"]) > 2:
            continue
        r = {"lat": float(lat_oof[pid_of[c["name"]]])}
        said = c["answers"]["hemisphere"].lower()
        said_n = "north" in said
        said_s = "south" in said
        rows.append({"name": c["name"], "true_lat": c["latitude"], "probe_lat": r["lat"],
                     "said": c["answers"]["hemisphere"], "said_correct": c["correct"]["hemisphere"],
                     "probe_correct": (r["lat"] > 0) == (c["latitude"] > 0),
                     "said_north": said_n, "said_south": said_s})
    wrong_said = [r for r in rows if not r["said_correct"] and (r["said_north"] or r["said_south"])]
    out = {
        "n_cities": len(rows),
        "spoken_hemisphere_accuracy": sum(r["said_correct"] for r in rows) / len(rows),
        "probe_hemisphere_accuracy": sum(r["probe_correct"] for r in rows) / len(rows),
        "n_spoken_wrong": len(wrong_said),
        "n_spoken_wrong_but_probe_right": sum(r["probe_correct"] for r in wrong_said),
        "examples": sorted(wrong_said, key=lambda r: -abs(r["true_lat"]))[:15],
        "rows": rows,
    }
    (RESULTS / "hemisphere_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[hemisphere] spoken accuracy {out['spoken_hemisphere_accuracy']:.0%}, probe sign accuracy "
          f"{out['probe_hemisphere_accuracy']:.0%}; spoken wrong {out['n_spoken_wrong']}, of which probe right "
          f"{out['n_spoken_wrong_but_probe_right']}")
    for r in out["examples"][:8]:
        print(f"   {r['name']:20s} true {r['true_lat']:6.1f}  probe {r['probe_lat']:6.1f}  said {r['said']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
