"""Extract last-name-token activations.

    uv run python scripts/extract_activations.py --set 3k --modes bare coords
    uv run python scripts/extract_activations.py --set full --modes bare --layers 10 11 12

--set 3k   : the spatially spread 3,000-place subset (terra.data.subset_3k)
--set full : every place (world_place.csv plus large GeoNames cities)
The place table used is saved next to the activations as places.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import ACTS  # noqa: E402
from terra.activations import extract  # noqa: E402
from terra.data import all_places, subset_3k  # noqa: E402
from terra.model import get_lm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["3k", "full"], default="3k")
    ap.add_argument("--modes", nargs="+", default=["bare", "coords"])
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    places = all_places()
    if args.set == "3k":
        places = subset_3k(places, n=3000, seed=0)
    ACTS.mkdir(exist_ok=True)
    places.to_csv(ACTS / f"places_{args.set}.csv", index=False)
    lm = get_lm()
    layers = args.layers if args.layers else list(range(lm.n_layers))
    for mode in args.modes:
        extract(lm, places, mode, layers, tag=args.set, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
