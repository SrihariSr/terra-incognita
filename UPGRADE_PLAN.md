# UPGRADE_PLAN.md: TERRA INCOGNITA v2

Four changes to the existing, working demo. You are running unattended: do not stop to ask, log every judgement call in `DECISIONS.md`, keep `STATUS.md` current, and follow every rule in `CLAUDE.md` (no git commands of any kind, complete files, plain-language comments that explain the maths, British spelling, no em dashes).

**Before touching anything:** copy the current code (`terra/`, `scripts/`, `web/`, `pyproject.toml`) to `snapshots/pre_v2/`. The current demo must keep working for Qwen at every step; if a change breaks it, fix it or roll back from the snapshot.

**Testing scope (the user's instruction):** code-level tests only. Do NOT download any model, do NOT load any real model, do NOT run the extraction or evaluation pipelines, do NOT run Playwright or browser tests. The user will run everything on the real models. The one exception is building the Qwen political map (Part 4), which needs only files already on disk and the CPU. See "Testing" at the end.

---

## Part 1. Fix city selection on the globe

Diagnosis from reading `web/app.js`:

1. The country-border layer (`polygonsData`, altitude 0.0015) covers all land, so a click on land hits a polygon and `onGlobeClick` never fires. Only ocean clicks reach `pickPlace`.
2. `pointsMerge(true)` merges the 10k points into one mesh, which disables per-point clicks (keep the merge, it is needed for speed).
3. City markers (`.city-marker`): a press on a *source* marker starts a drag, and `endDrag` returns early when the drop is under 300 km from home ("a click, not a drag"), so a click does nothing. *Target* markers have no listener at all.
4. The 300 km snap radius in `pickPlace` is fixed, so it is too small when zoomed out and too large when zoomed in.

Fix:

- Route clicks from every layer to one handler: `onGlobeClick`, `onPolygonClick` and `onPointClick` should all call `pickPlace(lat, lng)` with the click's globe coordinates. Check the callback signatures in the vendored `web/vendor/globe.gl.min.js` (recent versions pass `(obj, event, {lat, lng, altitude})`); if a callback gives no coordinates, convert the event's screen position with `globe.toGlobeCoords`.
- Make the snap radius depend on the screen, not on kilometres: pick the nearest believed point within about 14 pixels of the click, by projecting candidate points with `globe.getScreenCoords` (or convert 14 px to km from the current camera altitude). Search all 10k places.
- City markers: treat a press and release that moves less than about 6 pixels as a click, on every marker (source, target and custom). A click selects that place.
- **What "selected" means:** show the existing info card (true position, believed position, error) plus two buttons, **Use as source** and **Use as target**, which fill the Edit tab's fields (or add a move in the Rewrite tab when that tab is open). Highlight the selected marker or point. Any of the 10k places can be selected and used, not just the validated menu cities.

## Part 2. Swaps with any name, including invented ones

The backend already supports this: `/api/edit`, `/api/ask` and `plan_edit_all` accept any names, and `plan_edit_all` aligns names of different token lengths from the right. The frontend is what blocks it: the source/target dropdowns, the drag-snap and the Rewrite tab only use validated menu cities.

Build:

- **Custom places.** Every name located in the "New worlds" box (invented or real) becomes a custom place: a draggable marker with a distinct style at its believed position, listed under "Your places" with a remove button. Keep custom places per model in memory, and also in `localStorage` wrapped in try/catch.
- **Free-text source and target.** Replace the Edit tab's two dropdowns with text inputs backed by a `<datalist>` (validated menu cities first, then custom places, then big cities from the places list). Any text is allowed. Do the same for each move row in the Rewrite tab.
- **Drag rules.** Any marker (menu city, custom place, or a selected point) can be dragged. On release, snap to the nearest marker of any kind within about 40 pixels; if there is none, snap to the nearest big city in the places list. Validated pairs keep using the cache; anything else runs live, with a "computing live" status message.
- **Scoring when there is no answer key.** Invented places and landmarks have no GeoNames answer key, so `key_for` returns `None`. The ripple panel must then show the before, after and control answers unscored, labelled "no answer key for invented places: answers shown unscored", with the meters reading "unscored" instead of a number. Check `terra/rewrite.py` (`lat(name)`, `pair_key`) and any other place that assumes `city_info(name)` exists, and make it degrade to unscored instead of raising.
- **Guards:** reject an edit only if source and target are identical or empty. If a name cannot be aligned in a prompt, return a clear 400 error that the frontend shows.

## Part 3. Model switcher: Qwen 2.5 7B, Gemma 4 12B, Gemma 4 31B

**Important constraint:** the map and the edits need per-layer hidden states and the ability to overwrite them. Ollama's API does not expose either (it offers chat, generation and embeddings only), so the user's Ollama copies cannot be used for this. Load Gemma through **MLX** on Apple silicon, from 4-bit builds:
- Gemma 4 12B: `mlx-community/gemma-4-12B-it-qat-4bit`
- Gemma 4 31B: `mlx-community/gemma-4-31B-it-qat-4bit`

The Mac has 36 GB of unified memory. Qwen in bf16 takes about 15 GB; Gemma 4 31B at 4 bits about 17.5 GB. **Only one model may be in memory at a time.** Add `mlx-lm` (and `mlx-vlm` only if needed, see below) and `shapely` to `pyproject.toml` with `uv add`.

### 3a. One backend interface

The rest of the code uses only these members of `LM`: `chat`, `span_tokens`, `residuals`, `residuals_multi(texts, positions, layers, patch=None)`, `generate(texts, max_new_tokens, patch=None)`, `d_model`, `n_layers`, `model_id`, `lock`, and `tok` (once). Make that the explicit interface:

- `terra/backends/hf.py`: the current Qwen `LM`, unchanged in behaviour.
- `terra/backends/mlx_backend.py`: an MLX implementation of the same interface. mlx-lm models have no forward hooks, so wrap each decoder layer in a small module that records the output at the requested positions, or overwrites it with the patch vectors during prefill only, then calls the original layer. Keep `torch` CPU tensors as the exchange format at the interface (convert MLX arrays to numpy and then to torch on return; convert patch vectors to MLX on the way in), so `terra/patch.py` and `terra/steer.py` work unchanged. Generation is greedy with a KV cache, patched during prefill only, exactly as in the HF backend.
- **Model-specific details to move out of shared code and into the backend:**
  - `USER_TURN` in `terra/patch.py` (Qwen's `"<|im_start|>user"`), used so that names are only searched for inside the user's message. Read Gemma 4's marker from its chat template.
  - `BOS` in `terra/model.py` / `terra/activations.py` (Qwen's `<|endoftext|>` stand-in). Gemma uses its real BOS token.
  - Chat formatting: Gemma 4's template may have a thinking mode. Turn it off if the template supports that, and strip any thinking channel from generated text.
- **Loading Gemma:** first check that the installed `mlx_lm` has a Gemma 4 model module and that it loads these repos (they are tagged as image-text models). If mlx-lm cannot load one, use `mlx_vlm`'s language-model submodule for text only. Check for a version mismatch between mlx-lm and the Gemma 4 12B "Unified" architecture in particular. Record what works in `DECISIONS.md`. Do not download weights to find out: read the installed package source and the repos' `config.json` through the Hub API only if a small metadata fetch is allowed; otherwise write the loader defensively for both paths.

### 3b. Registry and per-model artefacts

- `terra/registry.py`: `qwen2.5-7b` (backend `hf`, id `Qwen/Qwen2.5-7B-Instruct`), `gemma4-12b` and `gemma4-31b` (backend `mlx`, ids as above), each with a display name.
- **Artefact paths per model.** Qwen keeps its current files exactly where they are (`models/probe_final.npz`, `results/*.json`), so nothing on disk breaks. New models use `models/<key>/` and `results/models/<key>/`. Put the path lookup in one function and use it everywhere: in the server, the locator, the menus, the rigour numbers, the figures and the political map.
- **A model is "available" if its probe file exists, and "editable" if its `gate.json` exists.**

### 3c. Build pipeline for a new model (the user runs it, you do not)

`scripts/build_model.py --model <key> --stage <stage>`. It reuses the existing scripts' functions with an `lm` argument and per-model output paths; refactor those scripts to take `--model` rather than copying them. Stages, each resumable and each printing progress with an ETA:

1. `extract3k`: the 3k subset at all layers.
2. `sweep`: probe error per layer.
3. `full`: all places at the best layer and its neighbours.
4. `final`: final probe, bootstrap ensemble, out-of-fold believed positions, `places_believed.json` and `probe_oof.npz`.
5. `fictional`: the invented-names test.
6. `political`: Part 4.
7. `gate`: the band sweep and the ten-pair gate, same pass bar as Qwen.
8. `menu`: the validated pair menu.

Also `--stage all`. Also check memory before loading (refuse to start if another process holds the model, as the server does).

### 3d. Switching at run time

- `GET /api/models` lists the models with their availability, editability and which one is loaded. `POST /api/model {key}` switches models.
- **Switching** runs under the model lock:
  1. Mark the server as switching; other model routes return 503 "switching models" meanwhile.
  2. Drop every reference to the current model, call `gc.collect()`, then `torch.mps.empty_cache()` for HF or `mx.clear_cache()` for MLX.
  3. Best effort, with a short timeout: if Ollama is running on `localhost:11434`, list its loaded models via `/api/ps` and unload each with `/api/generate {"model": name, "keep_alive": 0}`, so the user's Ollama copies are not also holding memory. Ignore errors if Ollama is not running.
  4. Load the new backend, its probe and its menus, and reset every cache. All caches must be keyed per model.
- Start-up model: `TERRA_MODEL`, default `qwen2.5-7b`.
- **Frontend:** a model dropdown in the HUD. Unavailable models are shown greyed out with "not built yet: run `uv run python scripts/build_model.py --model <key> --stage all`". On switch:
  - show the loading overlay with "Unloading X, loading Y";
  - then reload places, menu, rigour, political map and custom-place markers for that model;
  - if the model is not editable, disable the Edit and Rewrite tabs with a one-line reason.
  - Show the loaded model's name and probe layer in the HUD status line.

## Part 4. The model's political map

Gurnee & Tegmark's probe errors are large (Qwen: 829 km median on places never seen). Turn that into the point: draw the world's countries **where the model believes they are**.

`terra/politics.py` and the `political` stage:

- **Input:** out-of-fold believed positions for all places (`probe_oof.npz`, random split; check its keys) joined to each place's country (pid to country via `terra.data`, ISO code via `iso_for`). Out-of-fold matters: these are readings for places the probe was not fitted on.
- **Grid:** 1-degree cells (configurable, 0.5 for a finer build). For each cell centre on the unit sphere, find the k = 12 nearest believed points (chord distance on xyz, which ranks the same as great-circle distance).
- **Land or sea in the model's world:** a cell is land if its k-th neighbour is closer than R km (start at 350 km). Report the land fraction this gives, and pick R so that it is broadly plausible; log the value chosen.
- **Country:** the distance-weighted vote of the k neighbours. Then one majority-filter pass over the grid to remove speckle. Cells weighted by cos(latitude) for every area figure.
- **Polygons:**
  - Union each country's cells with shapely, simplify lightly, and drop slivers of fewer than 3 cells.
  - Write `political_map.geojson` (per model, in its results folder). Properties: ISO_A2, name, number of places, believed centroid, true centroid, displacement in km, and `MAPCOLOR7` copied from `web/vendor/countries.geojson` for familiar colouring.
  - Cells never cross the antimeridian, so neither do polygons.
- **Stats** in `political_stats.json`:
  - per country: displacement of the centroid in km;
  - precision, the share of the model's territory for a country that lies inside the real country;
  - recall, the share of the real country the model assigns to it.
  - Compute the real-country cells by testing cell centres against `countries.geojson` with shapely.
  - Also list the most displaced countries, the countries that vanish (no cells), and the largest "imagined" territories.
- **Frontend, Map tab:**
  - A "Borders" control with three options: Real / The model's / Both. globe.gl has one polygons layer, so put real and model polygons in the same `polygonsData` with a `kind` property and style by kind: the model's are filled and translucent, coloured by MAPCOLOR7; the real ones are drawn as lines.
  - Country labels at the believed centroids for the biggest countries (labels layer).
  - Clicking a model-country polygon shows a card, e.g. "In Qwen's world, Australia sits 1,230 km north-west of where it is; 61% of its territory overlaps the real Australia", with every number read from `political_stats.json`.
  - A small "Most displaced" list.
  - A caption under the control: "As read by a linear probe from the model's hidden states (out-of-fold), not drawn by the model."
  - Switching model switches the political map.
- **Build Qwen's political map now** with `uv run python scripts/build_model.py --model qwen2.5-7b --stage political`. It needs only files already on disk, and no model is loaded. Put its headline numbers in `STATUS.md`, each with its source file.

---

## Testing (code only)

- Add `tests/` with pytest and a `FakeLM` implementing the backend interface on tiny random tensors. `TERRA_FAKE_LM=1` makes the server use it, so it starts without loading a real model. Test:
  - patch planning with invented names and mismatched token lengths;
  - unscored edits and rewrites when there is no answer key;
  - every API route via FastAPI's `TestClient`, including `/api/models` and `/api/model`. Check that a switch unloads before loading: the FakeLM counts its loads and unloads, and memory is never held by two models at once;
  - the Ollama unload helper with the HTTP calls mocked;
  - `politics.py` on a synthetic world of two or three fake countries, with known displacement, precision and recall.
- `node --check web/app.js` for syntax.
- Run the tests and make them pass. Nothing else.

## Hand-over

Update `STATUS.md` with exact commands for the user, in order:
1. downloading the two MLX models;
2. building each Gemma model's artefacts, with `--stage all` and per stage;
3. starting the server for each start-up model;
4. what to click to check each of Parts 1 to 4.

Also list anything you could not verify without running a real model (for example the exact Gemma 4 chat-turn marker, or whether mlx-lm loads the 12B), so the user knows what to watch for on the first run. Update `DEMO.md` and `DEVPOST.md` for the model switcher and the political map, keeping every number traceable to a results file.