"""Extract last-name-token activations.

    uv run python scripts/extract_activations.py --set 3k --modes bare coords
    uv run python scripts/extract_activations.py --set full --modes coords --layers 20 21 22
    uv run python scripts/extract_activations.py --model gemma4-31b --set 3k --modes coords

--set 3k   : the spatially spread 3,000-place subset (terra.data.subset_3k)
--set full : every place (world_place.csv plus large GeoNames cities)
--model    : which model (default qwen2.5-7b); files go to registry.paths(model).acts
The place table used is saved next to the activations as places_<set>.csv.
Extraction is resumable: run the same command again after a stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terra import registry  # noqa: E402
from terra.activations import extract, is_complete  # noqa: E402
from terra.data import all_places, subset_3k  # noqa: E402
from terra.model import get_lm  # noqa: E402


def place_table(set_name: str):
    places = all_places()
    if set_name == "3k":
        places = subset_3k(places, n=3000, seed=0)
    return places


def run(key: str = registry.DEFAULT_KEY, set_name: str = "3k", modes=("coords",), layers=None,
        batch_size: int = 64, lm=None, force: bool = False) -> None:
    """Extract activations for `key`; skips modes that are already complete."""
    P = registry.paths(key)
    places = place_table(set_name)
    P.acts.mkdir(parents=True, exist_ok=True)
    places.to_csv(P.acts / f"places_{set_name}.csv", index=False)
    for mode in modes:
        if layers is not None and not force and is_complete(set_name, mode, layers, len(places), root=P.acts):
            print(f"[extract] {key} {set_name}/{mode} already complete, skipping")
            continue
        lm = lm or get_lm(key)
        use_layers = list(layers) if layers is not None else list(range(lm.n_layers))
        if not force and is_complete(set_name, mode, use_layers, len(places), root=P.acts):
            print(f"[extract] {key} {set_name}/{mode} already complete, skipping")
            continue
        extract(lm, places, mode, use_layers, tag=set_name, batch_size=batch_size, root=P.acts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=registry.DEFAULT_KEY, choices=sorted(registry.MODELS))
    ap.add_argument("--set", choices=["3k", "full"], default="3k")
    ap.add_argument("--modes", nargs="+", default=["bare", "coords"])
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="redo even if complete")
    args = ap.parse_args()
    run(args.model, args.set, args.modes, args.layers or None, args.batch_size, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
