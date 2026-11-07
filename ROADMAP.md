# ROADMAP: TERRA INCOGNITA

Times are targets for Saturday 7 Nov to Sunday 8 Nov. Submissions close **11:00 Sunday**. If a milestone slips by more than an hour, stop and cut scope using the "Cut first" lists, do not push everything later.

---

## M0. Setup (by 12:45)

- `uv init`, Python 3.11+, deps: torch, transformers, accelerate, numpy, scikit-learn, pandas, fastapi, uvicorn, huggingface_hub.
- Start the model download immediately, in its own terminal: `hf download Qwen/Qwen2.5-7B-Instruct` (older CLI: `huggingface-cli download ...`). About 15 GB.
- `scripts/download_data.py`: fetch `world_place.csv`, GeoNames `cities15000.zip` and `countryInfo.txt` into `data/`. Print row counts.
- Repo layout from CLAUDE.md, plus empty `DECISIONS.md` and `STATUS.md`. No git commands.
- If the model is not already downloaded or downloading, start `hf download Qwen/Qwen2.5-7B-Instruct` in the background and carry on with work that does not need it.

## M1. Probe and patching harness (by 14:00)

- `terra/model.py`: load the model once on MPS in bf16; helper to get residual-stream states at the last name token for a batch of names, all layers.
- Extract activations for a **3,000-place subset** (mix of cities and landmarks, spread across continents). Save as fp16 `.npy` per layer in `activations/`.
- `terra/probe.py`: ridge to unit-sphere xyz per layer, random and spatially blocked splits, median great-circle error in km per layer. Write `results/probe_layers.json`.
- `terra/patch.py`: prefill-only residual patching at the last name token over a layer band, and the equal-norm random control.
- `terra/ripple.py`: the five auto-scored questions plus answer keys from GeoNames; baseline check that the unedited model answers correctly.

## GATE (14:30): does the edit work?

`scripts/gate.py`, writing `results/gate.json`:
- Pick 10 well-known city pairs in different countries and hemispheres (e.g. Paris to Sydney, Tokyo to Buenos Aires, London to Cape Town), all answered correctly at baseline.
- Sweep a few layer bands. For the best band, apply the patch and score the five ripple questions.
- **GO** if at least **7 of 10 pairs** flip at least **4 of 5** answers to the target city, **and** the random control flips at least 4 of 5 on **at most 1 of 10**.
- Record the exact numbers in `STATUS.md` and the decision in `DECISIONS.md`.
- **GO:** continue straight to M2.
- **NO-GO, first time:** spend at most 45 minutes on recovery: try the "What are the lat/lon coordinates of " template, patching all name tokens instead of only the last, and a wider layer band. Rerun the gate.
- **NO-GO after recovery:** do not keep pushing. Switch to the fallback at the bottom of this file, write its own milestones into `ROADMAP_FALLBACK.md`, and build it to the same rules. Keep the TERRA code as it is.

## M2. The map (by 17:00)

- Extract activations for the full place set (or ~10k if time is tight) at the chosen layer plus neighbours.
- Fit final probe and bootstrap ensemble. Precompute believed positions for every place; save `results/places_believed.json` and the held-out error summary.
- Backend `GET /places`, `GET /locate?name=` (one forward pass, returns lat/lon plus the ensemble cloud).
- Frontend: globe.gl with believed points, error arcs for a sample, search box for any name, uncertainty cloud.
- **Invented places test:** run 50 invented names (mix of language-flavoured ones like "Krasnovik", "Takeshiro", plus "Hogwarts", "Atlantis", "Wakanda"). Save `results/fictional.json`. Keep a menu of the ones that land coherently; seed the input box with them.

## M3. The edit, on screen (by 21:00)

- Backend `POST /edit {source, target}` returns before and after answers for the ripple questions, plus the random-control result; `POST /ask {question, edit?}` for free questions. Cache "before" answers.
- Frontend: drag a city's dot onto another city (snap to the nearest validated target), the dot animates across the globe, and the ripple panel fills in before and after with a consistency meter.
- `scripts/batch_edit_eval.py` (can run overnight): 50 random pairs plus controls, writing `results/edit_eval.json` with the success rate. This is the number for the rigour slide.
- Build the **validated pair menu** from pairs that pass.

## M4. Rewrite the world and upgrade (by 01:00)

*Updated Sat 7 Nov, 16:10: the judges confirmed the project does not need to be finance-related, so the company layer is replaced by the "Rewrite the world" mode. See DECISIONS.md.*

- **"Rewrite the world" mode:** the judge can move up to three cities at once. The edits are applied together (one patched forward pass, every moved city patched at once), and both the ripple panel and free questions reflect all of them: per-city ripple questions, plus questions that involve two moved cities (e.g. "Are Paris and Tokyo in the same hemisphere?"), whose correct answer depends on both edits. Measure it on random triples of moves against the equal-norm random control (`results/rewrite_eval.json`).
- **Tier 2 partial edit** (probe-direction steering): try it on the gate pairs. If it moves hemisphere or season answers while the city keeps its landmarks on at least half the pairs, add it as the "move, don't swap" mode. Otherwise drop it and say so.

## M5. Demo freeze (by 04:00)

- Everything precomputed and cached; test with Wi-Fi off.
- `results/` figures for the rigour slide: probe error by layer, held-out error map, edit success against control.
- **"How it works" figure:** a one-page figure in `results/figures/` explaining, in plain language, reading the map (the probe) and editing it (patching).
- **"Relation to New Worlds" paragraph:** a short paragraph in `DEVPOST.md` on how the project fits the track.
- Run the whole demo path end to end with a script (`scripts/demo_check.py`) that hits every endpoint the demo uses and checks the answers; write `results/demo_check.json`. Write `DEMO.md` with exact start-up commands and the validated menus (pairs, invented names, suggested rewritten worlds) for the humans to rehearse.
- **Feature freeze.** After this, only bug fixes and polish.

## M6. Polish and submit (04:00 to 10:30)

- Visual polish, loading states, error handling for odd inputs.
- Draft the Devpost text in `DEVPOST.md` (what it does, how it works, measured results with their source files, honest prior-work note) and save screenshots or figures for it in `results/figures/`. The humans submit it by **10:30**.

---

## Demo script (5 minutes)

1. **Hook (15 s):** "Every language model carries a map of the Earth that nobody drew. We pulled it out, and then we learned to edit it."
2. **The map (45 s):** the globe of believed locations, with error lines. Mention the held-out error in km.
3. **New worlds (60 s):** the judge types an invented place and sees where it lands.
4. **The edit (90 s):** the judge drags Paris onto Sydney. The ripple panel flips. The judge asks their own question.
5. **Rewrite the world (45 s):** the judge moves up to three cities at once and asks one question that involves them all (e.g. "I'm visiting Paris, Tokyo and London this July. What should I pack?").
6. **Rigour (30 s):** edit success rate against the random control, held-out error. Prior work acknowledged.
7. **Close (15 s).**

## Cut first (if behind)

1. Tier 2 partial edit.
2. Rewrite the world mode (keep single-city edits).
3. Full 39k places (stay at 10k or 3k).
4. Bootstrap clouds (show a single dot).

Never cut: the tier-1 edit with its random control, the ripple panel, the invented-place search, offline mode.

---

## Fallback if the gate fails: NULLSPACE (Brevan Howard track)

An AI researcher that can tell a real discovery from a lucky one.
- Claude proposes hypotheses as JSON in a small typed DSL (lags, ratios, ranks, regime conditions over a panel), never free code, so every test is counted and lookahead is impossible by construction.
- Stats engine: Newey-West t-statistics, stationary block bootstrap, deflated Sharpe ratio, online FDR control (LORD++ or e-LOND) over every test the agent runs, and a hash-committed holdout.
- Data: Ken French factors and portfolios, yfinance daily prices, football-data.co.uk results and odds.
- Demo: the judge secretly plants a weak signal in the data, or nothing. A naive agent and NULLSPACE run side by side; the naive one "finds" several things, NULLSPACE reports only what survives. Then replay the search on 1,000 null worlds to show the measured false discovery rate.
- Data downloads (yfinance, Ken French, football-data.co.uk) must be cached to `data/` so the demo runs offline. The hypothesis proposer may use the Claude API only if an `ANTHROPIC_API_KEY` is already set in the environment; otherwise use a fixed, seeded generator of DSL hypotheses and log that choice.