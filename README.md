# TERRA INCOGNITA

Every language model carries a map of the Earth in its hidden states. This project reads that map out of Qwen2.5-7B-Instruct, lets you look up where the model believes any name is (real or invented), and edits the map: move Paris onto Sydney inside the model and watch its answers about Paris change together, while a random edit of the same size passes on none of the test pairs.

Built at WHACK'26 for the Marshall Wace "New Worlds" track. Everything runs locally on a Mac, offline.

## What it does

1. **Reads the map.** Ridge probes map the model's residual stream at the last token of a place name to a point on the unit sphere. About 10,000 places are drawn where the model believes they are, each placed by a probe that never saw that place, with lines to their true positions.
2. **New worlds.** Type any name ("Brindlemoor", "Takeshiro", "Hogwarts"). One forward pass, and a dot lands where the model believes the place is, with a cloud from 20 bootstrap probes.
3. **Edits the map.** Drag Paris onto Sydney. We overwrite the model's hidden state for the name " Paris" at layers 4 to 7 with the state it has for " Sydney" (every token of the name, aligned from the right) (the prompt text is unchanged), then ask five fixed questions (country, currency, main language, hemisphere, whether July is summer or winter), a landmark question and any free question. A second mode ("move: Paris stays Paris") shifts the state only along the location probe's directions, so the city keeps its identity while its hemisphere and season change.
4. **Rewrite the world.** Move up to three cities at once. All the edits are applied together in one pass, so per-city questions and questions about two moved cities ("Are Paris and Tokyo in the same hemisphere?", "Which is further south?") answer for the rewritten world, as do free questions that mention several of them.
5. **The model's political map.** Every place is put at its believed position with its real country, and a nearest-neighbour vote on a 1-degree grid draws the borders that result (`terra/politics.py`). Map tab, Borders: Real / The model's / Both; click a country for how far it has moved.
6. **Any name in an edit.** Every name you locate joins "Your places" and can be a source or target in Edit and Rewrite; any of the 10,000 dots can be clicked and used. Answers about invented places are shown unscored (no answer key).
7. **Several models.** The Model menu switches between Qwen2.5-7B-Instruct (Hugging Face, PyTorch) and Gemma 4 12B and 31B (4-bit, MLX). Only one model is in memory at a time. A model appears once its probe is built, and its Edit and Rewrite tabs once its edit gate is built.
8. **Rigour.** Held-out probe error in km under random, country-blocked and cell-blocked splits; edit success on random city pairs against an equal-norm random-direction control; a chance baseline for the invented names.

The measured numbers are in [`results/summary.md`](results/summary.md), generated from the results files by `scripts/summarise_results.py`. The main ones:

| Measure | Value | Source |
|---|---|---|
| Probe median error, places the probe never saw (random 5-fold, 40,485 places, layer 21) | 829 km | `results/probe_final.json` |
| Same, whole countries held out | 1,986 km | `results/probe_final.json` |
| Same, 10-degree grid cells held out | 1,177 km | `results/probe_final.json` |
| Always guessing the average position | 6,430 km | `results/probe_final.json` |
| Swap edit success, 100 fresh random cross-hemisphere city pairs (every name token patched) | 95% | `results/edit_eval_fresh_alltoken.json` |
| Random-direction control of equal size, same pairs | 0% | `results/edit_eval_fresh_alltoken.json` |
| Swap edit success, 81 fresh random pairs in any two countries (control) | 95% (0%) | `results/edit_eval_fresh_alltoken.json` |
| Same fresh pairs, last name token only (the version that passed the gate) | 87% and 86% | `results/edit_eval_fresh_lasttoken.json` |
| "Move" mode (tier 2), 40 fresh pairs: location moved / identity kept / both | 82% / 72% / 60% | `results/steer_heldout.json` |
| Same-size random control for the move mode: location moved | 0% | `results/steer_heldout.json` |
| Invented names landing in the region their sound suggests (chance) | 31/48 (7%) | `results/fictional.json` |
| Rewrite the world, 40 random three-move worlds: moved cities fully moved (control) | 97% of 96 (0%) | `results/rewrite_eval.json` |
| Same worlds: answers about two moved cities consistent with the rewritten world (control) | 73% of 163 (45%) | `results/rewrite_eval.json` |
| Model's spoken hemisphere answer correct / probe reads the right hemisphere | 72% / 95% of 270 cities | `results/hemisphere_check.json` |
| Political map: countries that keep land (of those with places) / share of the model's land in the right country | 93 of 222 / 28% | `results/political_stats.json` |

All numbers are for Qwen2.5-7B-Instruct. The Gemma models' numbers go to `results/models/<key>/` once built.

Success means at least 80% of the scoreable ripple answers switch to the target city.

## How it works

| Step | Code | Output |
|---|---|---|
| Download places (Gurnee & Tegmark's `world_place.csv`), GeoNames cities and country facts | `scripts/download_data.py` | `data/` |
| Gate: does activation patching flip the five answers? | `scripts/gate.py` | `results/gate.json` |
| Activations at the last name token, all layers, 3,000 places, two prompt styles | `scripts/extract_activations.py` | `activations/` |
| One probe per layer, three cross-validation schemes | `scripts/fit_probes.py` | `results/probe_layers.json` |
| Full place set (40,485 places) at the best layers; final probe and bootstrap | `scripts/fit_final.py` | `models/probe_final.npz`, `results/probe_final.json`, `results/places_believed.json` |
| Which cities the unedited model knows | `scripts/baseline_cities.py` | `results/baseline_cities.json` |
| Edit on random pairs vs the random control | `scripts/batch_edit_eval.py` | `results/edit_eval.json` |
| Validated pair menu for the demo | `scripts/build_pair_menu.py` | `results/pair_menu.json` |
| Invented and fictional names | `scripts/fictional.py` | `results/fictional.json` |
| Rewrite the world: random 3-move worlds and suggested demo worlds | `scripts/rewrite_eval.py` | `results/rewrite_eval.json`, `results/rewrite_menu.json` |
| One-page "how it works" figure | `scripts/how_it_works_figure.py` | `results/figures/how_it_works.png` |
| Companies (earlier experiment, removed from the demo when the judges said finance was not needed) | `scripts/fetch_companies.py`, `scripts/companies_eval.py` | `results/companies.json` |
| Tier 2 steering test, then the same setting on fresh pairs | `scripts/steer_eval.py`, `scripts/steer_heldout.py` | `results/steer_eval.json`, `results/steer_heldout.json` |
| Tier 2 on companies (earlier experiment, did not work) | `scripts/companies_steer.py` | `results/companies_steer.json` |
| Spoken vs probed hemisphere | `scripts/hemisphere_check.py` | `results/hemisphere_check.json` |
| Headline numbers table | `scripts/summarise_results.py` | `results/summary.md` |
| Figures | `scripts/make_figures.py` | `results/figures/` |
| End-to-end check of the demo API | `scripts/demo_check.py` | `results/demo_check.json` |
| Browser check of the page (Playwright, dev only) | `scripts/ui_check.py` | `results/ui_check.json`, `results/figures/screenshots/` |

`scripts/run_all.sh` runs the whole pipeline in order.

**Probe.** Targets are unit-sphere coordinates (x, y, z), not latitude and longitude, so the dateline is not a cliff. Features are standardised; ridge strength is chosen by closed-form leave-one-out. Error is great-circle distance. See `terra/probe.py` for the maths, written to be read.

**Edit (tier 1).** Prefill-only activation patching at the name over a band of layers (`terra/patch.py`). The gate was passed patching only the last token of the name; patching every token of the name (aligned from the right) was then tested with a pre-set rule, adopted, and re-measured on fresh pairs. The control writes the clean source state plus a random unit vector scaled to the patch's own size at each layer.

**Scoring.** Answer keys come from GeoNames (country, currency, languages; hemisphere and season from latitude). A question only counts for a pair if the two cities' keys differ; an answer flips if it matches the target and not the source (`terra/ripple.py`).

## Running the demo

```bash
uv sync                                   # first time only (needs the network)
uv run uvicorn terra.server:app --port 8000
# open http://localhost:8000
```

First-time model download: `uv run hf download Qwen/Qwen2.5-7B-Instruct` (about 15 GB). After that everything runs offline. The server patches every token of a city's name by default; `TERRA_PATCH=last` switches back to last-token patching (the version that passed the gate).

Other models (Apple silicon, MLX; download once, then offline):

```bash
uv run hf download mlx-community/gemma-4-12B-it-qat-4bit
uv run python scripts/build_model.py --model gemma4-12b --stage all     # resumable; prints progress and ETA
TERRA_MODEL=gemma4-12b uv run uvicorn terra.server:app --port 8000      # or switch in the page
```

The same for `gemma4-31b` (`mlx-community/gemma-4-31B-it-qat-4bit`). `scripts/build_model.py --help` lists the stages. Stop the server before building: only one process may hold a model.

Tests (no model needed: a small fake model stands in): `uv run python -m pytest -q tests`.

See `DEMO.md` for the rehearsal script and the validated menus.

## Prior work

- Gurnee & Tegmark (2023), *Language Models Represent Space and Time*, arXiv 2310.02207. Linear probes for place coordinates and static maps; we use their place list.
- Chen et al. (2023), causal interventions on spatial representations in small models, arXiv 2312.16257.

Our additions: invented places, live causal edits on a 7B chat model with measured ripple consistency and a matched random control, several edits applied together ("rewrite the world"), a move-without-swap edit, the model's political map, and an interactive offline demo that can switch between models.

## Limits

- The probe reads a linear projection of the hidden state. A good probe shows the information is there and linearly readable; it does not show the model uses exactly that direction.
- The tier-1 edit swaps the whole representation at the name token, so the model also takes on the target's identity (Paris "becomes" Sydney, landmarks included). The tier-2 move keeps identity but succeeds on 60% of fresh pairs.
- Patching only the last token fails mostly on multi-word names (New York City, Buenos Aires); patching every token fixes most of these (`results/multitoken_eval.json`).
- Scoring is by keyword match on short answers, which can miss correct paraphrases.
- The political map depends on the place list, not only on the model: countries with many places grow, countries with few shrink or vanish (`results/political_stats.json`, `caveat`).
