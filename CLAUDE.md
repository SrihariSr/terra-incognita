# TERRA INCOGNITA: project brief for Claude Code

WHACK'26 hackathon, Marshall Wace track ("New Worlds"). Hacking ends **11:00 Sunday 8 Nov**; judging is a ~5-minute live demo to Marshall Wace from 12:30. Read `ROADMAP.md` for the timeline and milestones, and work through it in order.

## What we are building

Every language model carries a hidden map of the Earth in its activations. We:

1. **Read it.** Fit linear probes from a 7B model's hidden states to coordinates, and plot ~10k places on a 3D globe where the model *believes* they are (with error lines to where they really are).
2. **Explore new worlds.** The judge types any name, real or invented ("Hogwarts", "Brindlemoor"), and a dot lands where the model believes it is, with an uncertainty cloud.
3. **Edit it.** The judge drags Paris onto Sydney. We change the model's internal activations (not the prompt), and a panel re-asks fixed "ripple" questions (country, currency, language, hemisphere, season in July) before and after, plus one free question ("I'm in Paris for Christmas, what should I pack?").
4. **Bridge to finance.** A company layer: place firms by the HQ the model believes they have, move one, and ask a finance question. Message: LLMs used by funds hold invisible, editable beliefs; this is a way to read and audit them.
5. **Rigour slide.** Held-out probe error in km, edit success rate over many city pairs, and a random-direction control edit of equal size that should *not* change answers.

Prior work to acknowledge honestly: Gurnee & Tegmark 2023 (probes, static maps; arXiv 2310.02207) and Chen et al. 2023 (causal interventions on spatial representations in small models; arXiv 2312.16257). Our additions: invented places, live causal edits on a 7B model with measured ripple consistency, the company layer, and an interactive demo.

## Fixed technical decisions (change one only if it blocks progress, and log why in DECISIONS.md)

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (ungated, 28 layers, hidden size 3584), bf16 on PyTorch MPS (fall back to fp16 if bf16 fails). Fallback model: `meta-llama/Llama-3.1-8B-Instruct` only if HF access is already granted. Load once, keep in memory in the server process.
- **Hooks:** plain PyTorch forward hooks on `model.model.layers[i]` (the output tuple's first element is the residual stream). No TransformerLens/nnsight unless plain hooks block us.
- **Place data:** `data/entity_datasets/world_place.csv` from github.com/wesg52/world-models (MIT; 39,585 rows; columns `name, entity_subtype, country, latitude, longitude, ..., is_test`). Raw URL: `https://raw.githubusercontent.com/wesg52/world-models/main/data/entity_datasets/world_place.csv`. Plus GeoNames `cities15000.zip` and `countryInfo.txt` from `https://download.geonames.org/export/dump/` (countryInfo gives currency and languages per country, used to auto-score ripple answers).
- **Reading activations:** last token of the place name, following Gurnee & Tegmark (their prompts were the bare name, or "What are the lat/lon coordinates of " + name). Start with the bare name; compare against one template by held-out error and keep the better one.
- **Probe:** ridge regression from standardised hidden states to **3D unit-sphere coordinates** (x, y, z), not lat/lon (lat/lon breaks at the dateline, where the Sydney demo sits). Convert back to lat/lon for display. One probe per layer; choose the layer by held-out median great-circle error. Use a **spatially blocked split** (hold out whole countries or grid cells) as well as the random split, and report both; the blocked number is the honest one.
- **Uncertainty:** bootstrap ensemble of ridge probes (e.g. 20 resamples) on the chosen layer; show the spread as a cloud on the globe.
- **Edit, tier 1 (guaranteed path):** activation patching. Run the *target* city's name through the same prompt, capture its residual stream at its last name token for a band of layers, and during the *prefill only* of the source prompt replace the source's last-name-token residual with it at those layers. Generation then reads the patched KV cache. Sweep the layer band; record which band works.
- **Edit, tier 2 (jaw-drop upgrade, only if time allows):** partial edit by probe-direction steering, adding `W_pinv @ (target_xyz - current_xyz)` (scaled) at the name token, so the city keeps its identity (Eiffel Tower) but moves hemisphere. Keep only if it measurably works.
- **Control:** a random vector with the same norm as the tier-1 patch delta, applied at the same positions and layers. It must not flip answers.
- **Ripple questions:** asked via the chat template, greedy decoding, short answers, auto-scored by keyword match against GeoNames-derived answer keys: country, currency, main language, hemisphere (from latitude), July season (summer/winter from hemisphere). Only use cities the *unedited* model answers correctly.
- **Backend:** FastAPI. **Frontend:** single static page with globe.gl. Vendor all JS files locally (no CDN at demo time).
- **Precompute** everything possible (activations, probe outputs, the 10k-dot map, "before" answers, validated pair menu). The demo must run with Wi-Fi off.

## Rules for working

- **You are running unattended.** Nobody will answer questions. Do not stop to ask; make the most reasonable decision yourself, write it down in `DECISIONS.md` (time, decision, reason), and keep going. Work through every milestone in `ROADMAP.md` to the end.
- **No git.** Do not run `git commit`, `git add`, `git push`, or any other git command that changes the repository. Never run `git reset --hard` or `git clean -fdx`.
- **Snapshots instead of commits.** When a milestone works end to end, copy the working code (not data or activations) to `snapshots/<milestone>/` before starting the next one, so a broken change can always be undone by copying back.
- **Measure before claiming.** Every number shown on stage, in the README or on Devpost must come from a results file in `results/` produced by a script in the repo. Never type a number from memory. If something fails or is slower than hoped, record it plainly in `DECISIONS.md`.
- **Never break the demo path.** Once a milestone works end to end, later work goes behind a flag or in a separate module until it also works.
- **Data safety:** never delete or overwrite anything in `data/`, `activations/` or `models/` except to regenerate it with a script.
- **Code style:** complete files, never partial snippets. Detailed inline comments in plain language that explain the mathematics (probes, great-circle distance, unit-sphere coordinates, patching) as if teaching it. Seed every random choice.
- **Writing style:** British spelling, no em dashes, no buzzwords, no overclaiming.
- **Stuck for more than ~30 minutes on one problem:** take the simplest workaround that keeps the demo path working, log it in `DECISIONS.md`, and move on.
- **No paid APIs.** Everything runs locally on the Mac.
- **Progress log:** keep `STATUS.md` up to date at every milestone: what works, the key numbers (with the results file they come from), what is next, and anything the user must know.

## Repo layout (target)

```
terra/            Python package: data.py, model.py, activations.py, probe.py,
                  patch.py, ripple.py, companies.py, server.py
scripts/          one script per pipeline step (download, extract, fit, gate, batch eval)
web/              index.html, app.js, style.css, vendor/ (globe.gl and deps)
snapshots/        known-good copies of the code at each milestone
STATUS.md         progress log for the humans; DECISIONS.md: every judgement call made
results/          JSON/CSV/figures that every claimed number comes from
data/ activations/ models/   gitignored
```