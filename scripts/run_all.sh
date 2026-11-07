#!/usr/bin/env bash
# Rebuild every result from scratch, in dependency order.
# Each step is one script; each writes its own file in results/.
# Only one step at a time loads the model (it needs about 15 GB of memory).
# Total time on an M4 Max: roughly an hour.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="uv run python"

$PY scripts/download_data.py                       # data/ (skips files already present)
$PY scripts/fetch_companies.py                     # data/companies_wikidata.json, central banks
$PY scripts/gate.py                                # results/gate.json (layer band for the edit)
$PY scripts/extract_activations.py --set 3k --modes bare coords
$PY scripts/fit_probes.py --set 3k --modes bare coords            # results/probe_layers.json
$PY scripts/extract_activations.py --set full --modes coords --layers 20 21 22 23 24 25 26 27
$PY scripts/fit_probes.py --set full --modes coords --out results/probe_layers_full.json
$PY scripts/fit_final.py                           # models/probe_final.npz, results/probe_final.json, places_believed.json
$PY scripts/baseline_cities.py                     # results/baseline_cities.json
$PY scripts/batch_edit_eval.py                     # results/edit_eval.json
$PY scripts/build_pair_menu.py                     # results/pair_menu.json
$PY scripts/fictional.py                           # results/fictional.json
$PY scripts/companies_eval.py                      # results/companies.json (earlier experiment, not in the demo)
$PY scripts/steer_eval.py                          # results/steer_eval.json, models/steer_probes.npz
$PY scripts/steer_heldout.py                       # results/steer_heldout.json (tier 2 on fresh pairs)
$PY scripts/companies_steer.py                     # results/companies_steer.json (earlier experiment, negative)
$PY scripts/hemisphere_check.py                    # results/hemisphere_check.json (no model needed)
$PY scripts/multitoken_eval.py                     # results/multitoken_eval.json (all-token vs last-token)
$PY scripts/batch_edit_eval.py --method all --seed 2 --out results/edit_eval_fresh_alltoken.json
$PY scripts/batch_edit_eval.py --method last --seed 2 --out results/edit_eval_fresh_lasttoken.json
$PY scripts/build_pair_menu.py --method all --out results/pair_menu_alltoken.json
$PY scripts/rewrite_eval.py                        # results/rewrite_eval.json (unpadded, used)
$PY scripts/rewrite_eval.py --pad --out results/rewrite_eval_pad.json --menu-out results/rewrite_menu_pad.json
# then merge padded free answers into results/rewrite_menu.json (see DECISIONS.md 17:10)
$PY scripts/how_it_works_figure.py                 # results/figures/how_it_works.png
$PY scripts/make_figures.py                        # results/figures/*.png
$PY scripts/demo_check.py                          # results/demo_check.json (starts its own server)
# Browser check needs the server running in another terminal:
#   uv run uvicorn terra.server:app --port 8000   then   $PY scripts/ui_check.py
$PY scripts/summarise_results.py                   # results/summary.md (every headline number)
$PY scripts/write_demo_md.py                       # DEMO.md
$PY scripts/test_core.py                           # maths and scoring sanity tests
