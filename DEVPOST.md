# TERRA INCOGNITA: reading and rewriting the map inside a language model

*Every number below is copied from `results/summary.md`, which `scripts/summarise_results.py` generates from the results files named in each row. Screenshots: `results/figures/screenshots/`. Figures, including a one-page explainer (`how_it_works.png`): `results/figures/`.*

## Inspiration

Old maps marked the unexplored as *terra incognita*. Language models carry a map of the Earth that nobody drew: no one gave Qwen coordinates, yet it can tell you Lima is south of Tokyo. We wanted to see that map directly, find out what the model does with places that do not exist, and then redraw it and watch whether the model's other beliefs follow.

## What it does

- **The map.** A 3D globe with 10,000 places drawn where Qwen2.5-7B-Instruct *believes* they are, read straight from its hidden states, with lines to where they really are.
- **New worlds.** Type any name, real or invented. "Krasnovik" lands in Poland, "Moriyagawa" in southern Japan, "Ballynacreeve" in Ireland: one forward pass, and a dot lands where the model believes the place is, with an uncertainty cloud.
- **The edit.** Drag Paris onto Sydney. We change the model's internal activations, not the prompt, and a panel re-asks five questions about Paris: country, currency, main language, hemisphere, and whether July is summer or winter. Ask "I'm in Paris for Christmas, what should I pack?" and the edited model adds sunscreen and a hat.
- **Move, don't swap.** A second mode moves a city without changing what it is: Berlin keeps Germany, the euro, German and the Brandenburg Gate, but the model now says it is in the Southern Hemisphere with winter in July.
- **Rewrite the world.** Move up to three cities at once (Paris to Sydney, Tokyo to London, Rio de Janeiro to Moscow). All the edits are applied together, so questions about several moved cities ("Are Paris and Tokyo in the same hemisphere?", "Which is further south?") and free questions that mention them all answer for the rewritten world.
- **Rigour.** Held-out errors, success on random pairs and random worlds against a random control of the same size, and a chance baseline for invented names, all on screen with the file each number comes from.

## Relation to New Worlds

The track asks for new worlds; we look for the one that already exists inside a model, and then make new ones from it. The model's internal map is a world of its own: mostly faithful, sometimes wrong (it says Dublin is in the Southern Hemisphere, while its hidden map puts Dublin at 56°N), and it extends past the real Earth, since invented names get coordinates too. Editing turns that map into a place to run counterfactual worlds: move Paris to Sydney, or rewrite three cities at once, and measure whether the model's beliefs about currencies, languages, seasons and the relations between cities stay coherent in the world it now believes in.

## How we built it

- **Reading.** We record the residual stream at the last token of each place name (prompt: "What are the lat/lon coordinates of <name>", after Gurnee & Tegmark) for 40,485 places: their `world_place.csv` plus large GeoNames cities. Ridge regression maps the 3,584-number state to a point on the unit sphere (x, y, z), so the dateline is not a cliff; we convert back to latitude and longitude for display. One probe per layer; the layer (21 of 28) is chosen by the error on *countries the probe never saw*. Every dot on the globe is placed by a probe that never saw that place.
- **Uncertainty.** 20 probes fitted on bootstrap resamples give the cloud around each located name.
- **Editing (swap).** Activation patching during the prefill only: run "Which country is Sydney in?", copy the hidden state at the name " Sydney" for layers 4 to 7, and write it over the state at " Paris" when the model reads "Which country is Paris in?". The answer is generated from the patched cache. The band was chosen by a sweep on 10 gate pairs (patching the last name token), then fixed. Patching every token of a multi-word name (aligned from the right) was then tested with a rule written in advance, adopted, and re-measured on fresh pairs.
- **Editing (move).** Probe-direction steering: shift the hidden state at the name only along the three directions the location probe reads (the minimum-length change that makes the probe read the target's position, times a scale), at layers 6 to 10.
- **Rewriting the world.** Every move is substituted at once to build the target prompt ("Are Sydney and London in the same hemisphere?"), so each target's state is computed in the already-rewritten context, and all of them are written over the moved cities' tokens in one forward pass. Two cities can even swap places.
- **Controls.** At the same tokens and layers, we move the hidden state by a random vector with exactly the same length as the real edit at each layer. If size alone broke answers, the control would flip them too.
- **Scoring.** Answer keys from GeoNames (country, currency, languages) and latitude (hemisphere, season, which city is further south). A question only counts if the edit changes its right answer, and only cities the unedited model gets right are used.
- **Stack.** PyTorch forward hooks on the Hugging Face model in bf16 on Apple-silicon MPS, NumPy ridge with closed-form leave-one-out, FastAPI, globe.gl. Everything runs locally and offline on one MacBook. An automated check drives every endpoint (`results/demo_check.json`) and a headless browser drives the page, including drag and drop (`results/ui_check.json`), with the server sandboxed off the internet.

## Results

| Measure | Value | Source |
|---|---|---|
| Probe median error, places the probe never saw (random 5-fold, 40,485 places) | 829 km | `results/probe_final.json` |
| Same, whole countries held out | 1,986 km | `results/probe_final.json` |
| Same, 10-degree grid cells held out | 1,177 km | `results/probe_final.json` |
| Baseline: always guess the average position | 6,430 km | `results/probe_final.json` |
| Swap edit success, 100 fresh random cross-hemisphere city pairs (>= 80% of answers flip) | 95% | `results/edit_eval_fresh_alltoken.json` |
| Random control of equal size, same pairs | 0% | `results/edit_eval_fresh_alltoken.json` |
| Swap edit success, 81 fresh random pairs in any two countries (control) | 95% (0%) | `results/edit_eval_fresh_alltoken.json` |
| Move mode, 40 fresh pairs: location moved / identity kept / both | 82% / 72% / 60% | `results/steer_heldout.json` |
| Same for its random control / for the swap | moved 0% / identity kept 0% | `results/steer_heldout.json` |
| Rewrite the world, 40 random three-move worlds: moved cities whose answers all move (control) | 97% of 96 (0%) | `results/rewrite_eval.json` |
| Same worlds: answers about two moved cities that fit the rewritten world (control) | 73% of 163 (45%) | `results/rewrite_eval.json` |
| Same worlds: of those, answers the edits should change and do (control) | 67% of 67 (15%) | `results/rewrite_eval.json` |
| Invented names landing in the region their sound suggests | 31 of 48 (chance 7%) | `results/fictional.json` |

Findings we did not expect:

- **The map knows more than the mouth.** Asked "Is Dublin in the Northern or Southern Hemisphere?", the unedited model says "Southern". Over 270 large cities its spoken answer is right 72% of the time, but the probe reading its hidden state puts the city in the right hemisphere 95% of the time, including 69 of the 76 cities it answers wrongly (`results/hemisphere_check.json`). Part of this is probably a bias towards the last option in the question.
- **Edit too late and the world goes inconsistent.** Patching at layer 8 or later still changes country, currency and language, but the July-season answer flips on at most 1 of the 10 gate pairs: "Paris is in Australia, and July is summer" (`results/gate.json`, `results/figures/band_sweep.png`). Location facts are read off the name early.
- **Editing several cities is easier than reasoning about them together.** With three moves applied at once, 97% of moved cities change all their own answers, but only 73% of answers to questions about two moved cities fit the rewritten world, and 67% of those whose right answer should change do (random control: 15%). The model updates each city's facts but does not always carry the new positions into comparisons (`results/rewrite_eval.json`).

## Challenges

- Qwen has no beginning-of-sequence token, so a one-word name at position 0 becomes an "attention sink" with a strange hidden state; we prepend `<|endoftext|>`.
- The model gets some cities wrong (it puts Madrid and Seoul in the Southern Hemisphere), so every pair is filtered to cities it answers correctly first.
- Patching only the last token of "Buenos Aires" leaves " Buenos" telling the model where it is; patching every token of the name lifted fresh-pair success from 87% to 95% (`results/edit_eval_fresh_*.json`). When several moved cities share one sentence, a long name moved onto a short one ("Rio de Janeiro" onto "Moscow") made the model say "Moscow and Moscow". Padding the short name ("city of Moscow") removes the duplicate, but on 40 random worlds it lowered pair-answer consistency (71% vs 73%, `results/rewrite_eval_pad.json`), so we use it only for free questions.
- The random control is not perfectly silent: on fresh cross-hemisphere pairs it flips the July-season answer 45% of the time and the hemisphere answer 16%, because a disturbed model drifts towards "Summer" and "Northern". Country, currency and language never flip under it, so no control pair passes (`results/edit_eval_fresh_alltoken.json`).
- One idea that did not work, reported as such: a "context transplant" edit (reading "Paris" inside "the city in Australia called Paris") moved nothing (0 of 10, `results/steer_eval.json`).

## Prior work (and what is new here)

- Gurnee & Tegmark (2023), *Language Models Represent Space and Time* (arXiv 2310.02207): linear probes for coordinates and static maps. We use their place list and prompt.
- Chen et al. (2023), causal interventions on spatial representations in small models (arXiv 2312.16257).

New here: invented places; live causal edits on a 7B chat model with measured consistency across five ripple questions and a matched random control; a move-without-swap edit tested on held-out pairs; several edits applied together, checked on questions that involve more than one moved city; and an interactive, offline demo.

## Limits

- A linear probe shows location is linearly readable; it does not prove the model uses exactly that direction.
- The swap edit replaces the whole representation of the name, so Paris takes on Sydney's identity too (it names the Sydney Opera House). The move mode keeps identity, but works on 60% of fresh pairs, not all.
- Answers are scored by keyword match on short answers, which can miss correct paraphrases.
- Probe errors are medians: some places are thousands of kilometres off, especially when whole countries are held out.

## What's next

Larger models and other kinds of fact (dates, people, organisations) with the same read-edit-check loop; finding out why answers about two edited cities lag behind answers about one; and a "belief diff" tool that lists where a model's internal map disagrees with the record.
