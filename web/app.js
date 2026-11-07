/* TERRA INCOGNITA frontend.
 *
 * Layers on the globe (globe.gl):
 *   points   : ~10k places at their *believed* positions (merged for speed)
 *   arcs     : error lines from true to believed for large cities, plus the
 *              arc drawn when a city is moved
 *   html     : draggable city markers, the located pin and its cloud,
 *              and the drag ghost
 *   rings    : pulse at a located point
 *   polygons : country borders
 */

// Continent colours: a categorical palette validated for colour-blind
// separation on the dark globe (fixed order, see DECISIONS.md).
const CONT = {
  EU: ["Europe", "#3987e5"], AS: ["Asia", "#d95926"], AF: ["Africa", "#199e70"],
  NA: ["North America", "#c98500"], SA: ["South America", "#d55181"], OC: ["Oceania", "#9085e9"],
  AN: ["Antarctica", "#9aa3b2"], "?": ["Unknown", "#6b7280"],
};
const QLABEL = { country: "Country", currency: "Currency", language: "Main language",
  hemisphere: "Hemisphere", season: "July is..." };
const QKEYS = Object.keys(QLABEL);

const el = (id) => document.getElementById(id);
const state = {
  places: [], arcsErr: [], arcsEdit: [], menu: null, rigour: null,
  html: { cities: [], cloud: [], locate: [], ghost: [] },
  moves: [],          // "Rewrite the world": [{source, target}], at most menu.max_moves
  drag: null, busy: false, showArcs: true, showPoints: true, lastEdit: null, tab: "map",
};

// ------------------------------------------------------------ helpers

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.detail || j.error || `HTTP ${r.status}`);
  return j;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 0) => Number(x).toLocaleString("en-GB", { maximumFractionDigits: d, minimumFractionDigits: d });

// Unit-sphere maths, used to animate along the great circle.
const rad = (d) => (d * Math.PI) / 180, deg = (r) => (r * 180) / Math.PI;
function toXYZ(lat, lng) { const a = rad(lat), b = rad(lng); return [Math.cos(a) * Math.cos(b), Math.cos(a) * Math.sin(b), Math.sin(a)]; }
function toLatLng(v) { const n = Math.hypot(...v); return [deg(Math.asin(v[2] / n)), deg(Math.atan2(v[1], v[0]))]; }
function slerp(a, b, t) {
  const pa = toXYZ(a[0], a[1]), pb = toXYZ(b[0], b[1]);
  const dot = Math.min(1, Math.max(-1, pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2]));
  const om = Math.acos(dot);
  if (om < 1e-6) return b;
  const s = Math.sin(om), wa = Math.sin((1 - t) * om) / s, wb = Math.sin(t * om) / s;
  return toLatLng([wa * pa[0] + wb * pb[0], wa * pa[1] + wb * pb[1], wa * pa[2] + wb * pb[2]]);
}
function kmBetween(a, b) {
  const pa = toXYZ(a[0], a[1]), pb = toXYZ(b[0], b[1]);
  const dot = Math.min(1, Math.max(-1, pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2]));
  return 6371 * Math.acos(dot);
}

// ------------------------------------------------------------ globe

const container = el("globe");
const globe = Globe()(container)
  .globeImageUrl("vendor/earth-dark.jpg")
  .bumpImageUrl("vendor/earth-topology.png")
  .backgroundImageUrl("vendor/night-sky.png")
  .showAtmosphere(true).atmosphereColor("#5cc8ff").atmosphereAltitude(0.14)
  .pointLat("blat").pointLng("blon").pointAltitude(0.004).pointRadius(0.14)
  .pointColor((d) => (CONT[d.cont] || CONT["?"])[1]).pointsMerge(true)
  .arcStartLat((d) => d.slat).arcStartLng((d) => d.slng).arcEndLat((d) => d.elat).arcEndLng((d) => d.elng)
  .arcColor((d) => d.color).arcStroke((d) => d.stroke).arcAltitudeAutoScale((d) => d.alt ?? 0.3)
  .arcDashLength((d) => d.dash ?? 1).arcDashGap((d) => d.gap ?? 0).arcDashAnimateTime((d) => d.anim ?? 0)
  .htmlElement((d) => d.el).htmlLat((d) => d.lat).htmlLng((d) => d.lng).htmlAltitude((d) => d.alt ?? 0.012)
  .htmlTransitionDuration(0)
  .ringLat("lat").ringLng("lng").ringColor(() => (t) => `rgba(255,181,71,${1 - t})`)
  .ringMaxRadius(4).ringPropagationSpeed(2.5).ringRepeatPeriod(1100)
  .polygonCapColor(() => "rgba(0,0,0,0)").polygonSideColor(() => "rgba(0,0,0,0)")
  .polygonStrokeColor(() => "rgba(150,180,220,0.22)").polygonAltitude(0.0015)
  .onGlobeClick(({ lat, lng }) => pickPlace(lat, lng));

function resize() {
  const panel = el("panel").getBoundingClientRect();
  const w = window.innerWidth > 800 ? window.innerWidth - panel.width - 24 : window.innerWidth;
  globe.width(w).height(window.innerHeight);
}
window.addEventListener("resize", resize);
resize();
globe.pointOfView({ lat: 25, lng: 10, altitude: 2.3 }, 0);

function refreshHtml() {
  const h = state.html;
  globe.htmlElementsData([...h.cities, ...h.cloud, ...h.locate, ...h.ghost]);
}
function refreshArcs() { globe.arcsData([...(state.showArcs ? state.arcsErr : []), ...state.arcsEdit]); }

fetch("vendor/countries.geojson").then((r) => r.json()).then((g) => globe.polygonsData(g.features)).catch(() => {});

// ------------------------------------------------------------ tabs

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${b.dataset.tab}`));
  state.tab = b.dataset.tab;
  refreshHtml();
}));

// ------------------------------------------------------------ map layer

function legend() {
  el("legend").innerHTML = Object.entries(CONT).filter(([k]) => !["AN", "?"].includes(k))
    .map(([, [n, c]]) => `<span><i style="background:${c}"></i>${n}</span>`).join("");
}

async function loadPlaces() {
  const j = await api("/api/places");
  state.places = j.points;
  globe.pointsData(state.places);
  // Error lines for large, well-known cities only (the full set is a blur).
  state.arcsErr = state.places.filter((p) => p.big).slice(0, 400).map((p) => ({
    slat: p.blat, slng: p.blon, elat: p.lat, elng: p.lon,
    color: ["rgba(255,255,255,0.65)", "rgba(255,255,255,0.05)"], stroke: 0.25, alt: 0.15,
  }));
  refreshArcs();
  el("hud-status").textContent = `${fmt(state.places.length)} places at their believed positions (probe on layer ${j.layer})`;
}

function pickPlace(lat, lng) {
  if (!state.showPoints || !state.places.length) return;
  let best = null, bd = 1e9;
  for (const p of state.places) {
    const d = kmBetween([lat, lng], [p.blat, p.blon]);
    if (d < bd) { bd = d; best = p; }
  }
  if (!best || bd > 300) return;
  const c = el("map-picked");
  c.classList.remove("hidden");
  c.innerHTML = `<b>${esc(best.name)}</b> (${esc(best.type)}, ${esc(best.country)})<br>
    True position ${fmt(best.lat, 1)}, ${fmt(best.lon, 1)}<br>
    Believed position ${fmt(best.blat, 1)}, ${fmt(best.blon, 1)}<br>
    Error <b>${fmt(best.err_km)} km</b>`;
  state.arcsEdit = state.arcsEdit.filter((a) => !a.pick);
  state.arcsEdit.push({ pick: true, slat: best.blat, slng: best.blon, elat: best.lat, elng: best.lon,
    color: ["#ffb547", "#ffffff"], stroke: 0.6, alt: 0.2 });
  refreshArcs();
}

el("toggle-arcs").addEventListener("change", (e) => { state.showArcs = e.target.checked; refreshArcs(); });
el("toggle-points").addEventListener("change", (e) => { state.showPoints = e.target.checked; globe.pointsData(e.target.checked ? state.places : []); });

// ------------------------------------------------------------ new worlds

function clearLocate() { state.html.cloud = []; state.html.locate = []; globe.ringsData([]); refreshHtml(); }

async function locate(name) {
  name = name.trim();
  if (!name) return;
  el("locate-input").value = name;
  const card = el("locate-result");
  card.classList.remove("hidden");
  card.innerHTML = `Reading the model's hidden state for <b>${esc(name)}</b>...`;
  try {
    const r = await api(`/api/locate?name=${encodeURIComponent(name)}`);
    clearLocate();
    state.html.cloud = r.cloud.map(([lat, lng]) => {
      const e = document.createElement("div"); e.className = "cloud-dot";
      return { el: e, lat, lng, alt: 0.01 };
    });
    const pin = document.createElement("div"); pin.className = "locate-marker";
    pin.innerHTML = `<div class="lbl">${esc(name)}</div><div class="pin"></div>`;
    state.html.locate = [{ el: pin, lat: r.lat, lng: r.lon, alt: 0.02 }];
    globe.ringsData([{ lat: r.lat, lng: r.lon }]);
    refreshHtml();
    globe.pointOfView({ lat: r.lat, lng: r.lon, altitude: 1.6 }, 1400);
    const n = r.nearest;
    card.innerHTML = `The model places <b>${esc(name)}</b> at ${fmt(r.lat, 1)}, ${fmt(r.lon, 1)},
      nearest real city <b>${esc(n.name)}, ${esc(n.country)}</b> (${fmt(n.km)} km away).<br>
      Cloud spread (median distance of the 20 bootstrap probes from the main reading): ${fmt(r.cloud_median_km)} km.`;
  } catch (e) {
    card.innerHTML = `Could not locate: ${esc(e.message)}`;
  }
}

el("locate-form").addEventListener("submit", (e) => { e.preventDefault(); locate(el("locate-input").value); });

// ------------------------------------------------------------ edit: menus and markers

function targetsFor(src) {
  return state.menu.pairs.filter((p) => p.source === src).map((p) => p.target);
}

function modeLabels() {
  const src = el("edit-source").value, tgt = el("edit-target").value;
  el("lbl-swap").textContent = `swap: ${src} becomes ${tgt}`;
  el("lbl-steer").textContent = `move: ${src} stays ${src}`;
}

function fillTargets() {
  const src = el("edit-source").value;
  const ts = targetsFor(src);
  el("edit-target").innerHTML = ts.map((t) => `<option>${esc(t)}</option>`).join("");
  if (ts.includes("Sydney")) el("edit-target").value = "Sydney";
  el("ask-input").value = `I'm in ${src} for Christmas, what should I pack? Answer in two sentences.`;
  modeLabels();
}

function cityByName(n) { return state.menu.cities.find((c) => c.name === n); }

function buildCityMarkers() {
  const sources = new Set(state.menu.pairs.map((p) => p.source));
  state.html.cities = state.menu.cities.map((c) => {
    const e = document.createElement("div");
    e.className = "city-marker" + (sources.has(c.name) ? "" : " target");
    e.innerHTML = `<div class="dot"></div><div class="lbl">${esc(c.name)}</div>`;
    const d = { el: e, lat: c.lat, lng: c.lon, home: [c.lat, c.lon], name: c.name, alt: 0.015 };
    if (sources.has(c.name)) e.addEventListener("pointerdown", (ev) => startDrag(ev, d));
    return d;
  });
  refreshHtml();
}

function startDrag(ev, d) {
  if (state.busy) return;
  ev.preventDefault(); ev.stopPropagation();
  state.drag = d;
  d.el.classList.add("dragging");
  globe.controls().enabled = false;
  const g = document.createElement("div"); g.className = "ghost";
  state.html.ghost = [{ el: g, lat: d.lat, lng: d.lng, alt: 0.02 }];
  refreshHtml();
  window.addEventListener("pointermove", onDrag);
  window.addEventListener("pointerup", endDrag, { once: true });
}

function screenToGlobe(ev) {
  const r = container.getBoundingClientRect();
  return globe.toGlobeCoords(ev.clientX - r.left, ev.clientY - r.top);
}

function onDrag(ev) {
  const c = screenToGlobe(ev);
  if (!c || !state.html.ghost.length) return;
  state.html.ghost[0].lat = c.lat; state.html.ghost[0].lng = c.lng;
  refreshHtml();
}

function endDrag(ev) {
  window.removeEventListener("pointermove", onDrag);
  globe.controls().enabled = true;
  const d = state.drag; state.drag = null;
  if (!d) return;
  d.el.classList.remove("dragging");
  const drop = screenToGlobe(ev) || (state.html.ghost[0] && { lat: state.html.ghost[0].lat, lng: state.html.ghost[0].lng });
  state.html.ghost = []; refreshHtml();
  if (!drop) return;
  // Snap to the nearest validated target for this source.
  let best = null, bd = 1e9;
  for (const t of targetsFor(d.name)) {
    const c = cityByName(t); if (!c) continue;
    const km = kmBetween([drop.lat, drop.lng], [c.lat, c.lon]);
    if (km < bd) { bd = km; best = t; }
  }
  if (kmBetween([drop.lat, drop.lng], d.home) < 300) return;  // a click, not a drag
  if (state.tab === "rewrite") {
    // In Rewrite mode a drag adds a move (any demo city); press Rewrite to apply.
    let near = null, nd = 1e9;
    for (const c of state.menu.cities) {
      if (c.name === d.name) continue;
      const km = kmBetween([drop.lat, drop.lng], [c.lat, c.lon]);
      if (km < nd) { nd = km; near = c.name; }
    }
    if (near) addMove(d.name, near);
    return;
  }
  if (!best) return;
  el("edit-source").value = d.name; fillTargets(); el("edit-target").value = best;
  runEdit(d.name, best);
}

// ------------------------------------------------------------ edit: run and show

function animateMove(d, to, ms = 1800) {
  const from = [d.lat, d.lng], t0 = performance.now();
  return new Promise((resolve) => {
    function step(now) {
      const t = Math.min(1, (now - t0) / ms), e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      const [lat, lng] = slerp(from, to, e);
      d.lat = lat; d.lng = lng; refreshHtml();
      if (t < 1) requestAnimationFrame(step); else resolve();
    }
    requestAnimationFrame(step);
  });
}

function resetMarkers() {
  for (const d of state.html.cities) { d.lat = d.home[0]; d.lng = d.home[1]; d.el.classList.remove("moved");
    d.el.querySelector(".lbl").textContent = d.name; }
  state.arcsEdit = state.arcsEdit.filter((a) => a.pick); refreshArcs(); refreshHtml();
}

function cellClass(sc, q) {
  if (!sc) return "kept";
  const v = sc.per_question[q];
  if (!v.countable) return "kept";
  if (v.flipped) return "flip";
  if (v.matches_source && !v.matches_target) return "kept";
  return "other";
}

function renderRipple(r) {
  const tb = document.querySelector("#ripple tbody");
  const tick = (c) => (c === "flip" ? '<span class="tick">&#10003; flipped</span>' : "");
  let rows = QKEYS.map((q) => {
    const ca = cellClass(r.score_after, q), cc = cellClass(r.score_control, q);
    return `<tr><td class="q">${QLABEL[q]}</td><td>${esc(r.before[q])}</td>
      <td class="${ca}">${esc(r.after[q])}${tick(ca)}</td><td class="${cc}">${esc(r.control[q])}${tick(cc)}</td></tr>`;
  }).join("");
  if (r.landmark) {
    // Not scored: shows whether the city kept its identity.
    rows += `<tr><td class="q">Famous landmark<br><span class="small">(identity, not scored)</span></td>
      <td>${esc(r.landmark.before)}</td><td>${esc(r.landmark.after)}</td><td>${esc(r.landmark.control)}</td></tr>`;
  }
  tb.innerHTML = rows;
  el("ripple").classList.remove("hidden");
  el("meters").classList.remove("hidden");
  const label = (id, t) => (el(id).closest(".meter").querySelector("span").textContent = t);
  if (r.mode === "steer" && r.score_after) {
    // Tier 2 aims to move the location answers and keep the identity answers.
    const pq = r.score_after.per_question;
    const loc = ["hemisphere", "season"].filter((q) => pq[q].countable);
    const moved = loc.filter((q) => pq[q].flipped).length;
    const idq = ["country", "currency", "language"];
    const kept = idq.filter((q) => pq[q].matches_source && !pq[q].matches_target).length;
    label("meter-edit", "Location moved"); label("meter-ctrl", "Identity kept");
    el("meter-ctrl").classList.remove("ctrl");
    el("meter-edit").style.width = `${(moved / Math.max(1, loc.length)) * 100}%`;
    el("meter-ctrl").style.width = `${(kept / idq.length) * 100}%`;
    el("meter-edit-val").textContent = `${moved}/${loc.length}`;
    el("meter-ctrl-val").textContent = `${kept}/${idq.length}`;
    return;
  }
  label("meter-edit", "Edit consistency"); label("meter-ctrl", "Control");
  el("meter-ctrl").classList.add("ctrl");
  const fa = r.score_after ? r.score_after.n_flipped / Math.max(1, r.score_after.n_countable) : 0;
  const fc = r.score_control ? r.score_control.n_flipped / Math.max(1, r.score_control.n_countable) : 0;
  el("meter-edit").style.width = `${fa * 100}%`;
  el("meter-ctrl").style.width = `${fc * 100}%`;
  el("meter-edit-val").textContent = r.score_after ? `${r.score_after.n_flipped}/${r.score_after.n_countable}` : "-";
  el("meter-ctrl-val").textContent = r.score_control ? `${r.score_control.n_flipped}/${r.score_control.n_countable}` : "-";
}

function pendingRipple(src) {
  const tb = document.querySelector("#ripple tbody");
  tb.innerHTML = [...QKEYS.map((q) => QLABEL[q]), "Famous landmark"].map((l) => `<tr><td class="q">${l}</td><td class="pending">...</td><td class="pending">...</td><td class="pending">...</td></tr>`).join("");
  el("ripple").classList.remove("hidden");
}

function currentMode() {
  const m = document.querySelector('input[name="mode"]:checked');
  return m ? m.value : "swap";
}

async function runEdit(src, tgt) {
  if (state.busy) return;
  state.busy = true; el("edit-go").disabled = true;
  document.querySelector('#tabs button[data-tab="edit"]').click();
  resetMarkers();
  clearLocate();
  modeLabels();
  const mode = currentMode();
  el("edit-status").textContent = `Patching the model's hidden state for "${src}" with "${tgt}" (${mode === "steer" ? "probe-direction steering" : "layers " + state.menu.band.join(" to ")})...`;
  pendingRipple(src);
  const d = state.html.cities.find((c) => c.name === src);
  const tc = cityByName(tgt);
  try {
    const req = post("/api/edit", { source: src, target: tgt, mode });
    if (d && tc) {
      state.arcsEdit.push({ slat: d.lat, slng: d.lng, elat: tc.lat, elng: tc.lon, color: ["#ffb547", "#ff5c8a"],
        stroke: 0.9, alt: 0.35, dash: 0.4, gap: 0.2, anim: 1500 });
      refreshArcs();
      globe.pointOfView({ lat: (d.lat + tc.lat) / 2, lng: d.lng + ((((tc.lon - d.lng) % 360) + 540) % 360 - 180) / 2, altitude: 2.6 }, 1200);
    }
    const r = await req;
    state.lastEdit = r;
    if (d && tc) {
      d.el.classList.add("moved");
      d.el.querySelector(".lbl").textContent = mode === "steer" ? `${src} (moved, still ${src})` : `${src} (edited)`;
      // Fly to where the location probe now reads the edited city (measured
      // on the patched hidden state), falling back to the target's pin.
      const el2 = r.edited_location && r.edited_location.lat !== undefined ? r.edited_location : null;
      await animateMove(d, el2 ? [el2.lat, el2.lon] : [tc.lat, tc.lon]);
    }
    renderRipple(r);
    if (r.edited_location && r.edited_location.nearest) {
      const n = r.edited_location.nearest;
      el("edit-status").dataset.extra = ` The location probe now reads "${src}" near ${n.name}, ${n.country}.`;
    } else { el("edit-status").dataset.extra = ""; }
    el("edit-status").textContent = (r.seconds ? `Done in ${fmt(r.seconds, 1)} s (computed live).` : "Done (precomputed for this validated pair).")
      + (el("edit-status").dataset.extra || "");
    el("free").classList.remove("hidden");
    el("ask-result").innerHTML = "";
  } catch (e) {
    el("edit-status").textContent = `Edit failed: ${e.message}`;
  } finally {
    state.busy = false; el("edit-go").disabled = false;
  }
}

el("edit-source").addEventListener("change", fillTargets);
el("edit-target").addEventListener("change", modeLabels);
el("edit-reset").addEventListener("click", () => {
  if (state.busy) return;
  resetMarkers();
  ["ripple", "meters", "free"].forEach((id) => el(id).classList.add("hidden"));
  el("edit-status").textContent = "";
  globe.pointOfView({ lat: 25, lng: 10, altitude: 2.3 }, 1000);
});
el("edit-go").addEventListener("click", () => runEdit(el("edit-source").value, el("edit-target").value));

el("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = el("ask-input").value.trim();
  if (!q || !state.lastEdit) return;
  const btn = e.target.querySelector("button"); btn.disabled = true;
  const out = el("ask-result");
  out.innerHTML = `<div class="small">Asking the unedited and the edited model...</div>`;
  try {
    const r = await post("/api/ask", { question: q, source: state.lastEdit.source, target: state.lastEdit.target, mode: currentMode() });
    out.innerHTML = (r.note ? `<div class="small">${esc(r.note)}</div>` : "") +
      `<div class="qa"><div class="who">Unedited model</div>${esc(r.before)}</div>
       <div class="qa after"><div class="who">Edited: ${esc(r.source)} read as ${esc(r.target)}</div>${esc(r.after)}</div>`;
  } catch (err) {
    out.innerHTML = `<div class="small">Failed: ${esc(err.message)}</div>`;
  } finally { btn.disabled = false; }
});

// ------------------------------------------------------------ rewrite the world

const cityNames = () => state.menu.cities.map((c) => c.name);

function renderMoves() {
  const opts = (sel) => cityNames().map((n) => `<option${n === sel ? " selected" : ""}>${esc(n)}</option>`).join("");
  el("moves").innerHTML = state.moves.map((m, i) => `
    <div class="move-row" data-i="${i}">
      <select class="mv-src">${opts(m.source)}</select>
      <span class="arrow">&rarr;</span>
      <select class="mv-tgt">${opts(m.target)}</select>
      <button class="x" title="Remove this move">&times;</button>
    </div>`).join("");
  el("moves").querySelectorAll(".move-row").forEach((row) => {
    const i = +row.dataset.i;
    row.querySelector(".mv-src").addEventListener("change", (e) => { state.moves[i].source = e.target.value; rwFreeDefault(); });
    row.querySelector(".mv-tgt").addEventListener("change", (e) => { state.moves[i].target = e.target.value; });
    row.querySelector(".x").addEventListener("click", () => { state.moves.splice(i, 1); renderMoves(); rwFreeDefault(); });
  });
  el("move-add").disabled = state.moves.length >= (state.menu.max_moves || 3);
}

function rwFreeDefault() {
  const s = state.moves.map((m) => m.source);
  const q = s.length === 1 ? `I'm visiting ${s[0]} this July. What should I pack? Answer in two sentences.`
    : s.length === 2 ? `I'm visiting ${s[0]} and then ${s[1]} this July. What should I pack for each city? Answer in two short sentences.`
    : s.length === 3 ? `I'm visiting ${s[0]}, ${s[1]} and ${s[2]} this July. What should I pack for each city? Answer in three short sentences.` : "";
  el("rw-ask-input").value = q;
}

function addMove(source, target) {
  const max = state.menu.max_moves || 3;
  const existing = state.moves.find((m) => m.source === source);
  if (existing) existing.target = target;
  else if (state.moves.length < max) state.moves.push({ source, target });
  else state.moves[max - 1] = { source, target };
  renderMoves(); rwFreeDefault();
}

function rwMode() { const m = document.querySelector('input[name="rwmode"]:checked'); return m ? m.value : "swap"; }

function rwCell(sc, q) { return cellClass(sc, q); }

function renderRewrite(r) {
  const tick = (c) => (c === "flip" ? '<span class="tick">&#10003;</span>' : "");
  let rows = "";
  for (const c of r.cities) {
    rows += `<tr class="city-head"><td colspan="4">${esc(c.source)} &rarr; ${esc(c.target)}${c.passes === undefined ? "" :
      c.passes ? ' <span class="tick">(all answers moved)</span>' : ""}</td></tr>`;
    for (const q of QKEYS) {
      const ca = rwCell(c.score_after, q), cc = rwCell(c.score_control, q);
      rows += `<tr><td class="q">${QLABEL[q]}</td><td>${esc(c.before[q])}</td><td class="${ca}">${esc(c.after[q])}${tick(ca)}</td><td class="${cc}">${esc(c.control[q])}${tick(cc)}</td></tr>`;
    }
    if (c.landmark) rows += `<tr><td class="q">Landmark</td><td>${esc(c.landmark.before)}</td><td>${esc(c.landmark.after)}</td><td>${esc(c.landmark.control)}</td></tr>`;
  }
  el("rw-cities").innerHTML = `<table class="rw-table"><thead><tr><th>Question</th><th>Before</th><th>After rewrite</th><th>Random control</th></tr></thead><tbody>${rows}</tbody></table>`;
  const pr = r.pairs || [];
  if (pr.length) {
    const fmtKey = (p, k) => k === null ? "" : k === "a" ? `${p.a} (now at ${p.a_to})` : k === "b" ? `${p.b} (now at ${p.b_to})` : k;
    let prow = "";
    for (const p of pr) {
      const scored = p.key_after !== null && p.before_correct;
      const cls = !scored ? "" : p.after_correct ? "flip" : "wrong";
      prow += `<tr><td class="q">${esc(p.question.replace(/ Answer.*$/, ""))}</td><td>${esc(p.before)}</td>
        <td class="${cls}">${esc(p.after)}${scored ? `<div class="small">rewritten world: ${esc(fmtKey(p, p.key_after))}</div>` : ""}</td>
        <td>${esc(p.control)}</td></tr>`;
    }
    const scored = pr.filter((p) => p.key_after !== null && p.before_correct);
    const ok = scored.filter((p) => p.after_correct).length;
    el("rw-pairs").innerHTML = `<h3>Questions about two moved cities</h3>
      <div class="rw-summary">${ok} of ${scored.length} scoreable answers agree with the rewritten world.</div>
      <table class="rw-table"><thead><tr><th>Question</th><th>Before</th><th>After rewrite</th><th>Random control</th></tr></thead><tbody>${prow}</tbody></table>`;
  } else el("rw-pairs").innerHTML = "";
}

async function runRewrite() {
  if (state.busy || !state.moves.length) return;
  const moves = state.moves.map((m) => ({ source: m.source, target: m.target }));
  if (new Set(moves.map((m) => m.source)).size !== moves.length || moves.some((m) => m.source === m.target)) {
    el("rw-status").textContent = "Each city can be moved once, onto a different city.";
    return;
  }
  state.busy = true; el("rewrite-go").disabled = true;
  resetMarkers(); clearLocate();
  el("rw-status").textContent = `Applying ${moves.length} edit${moves.length > 1 ? "s" : ""} together and asking the questions (about 10 to 15 s live)...`;
  el("rw-cities").innerHTML = ""; el("rw-pairs").innerHTML = "";
  const anims = [];
  for (const m of moves) {
    const d = state.html.cities.find((c) => c.name === m.source), tc = cityByName(m.target);
    if (d && tc) state.arcsEdit.push({ slat: d.lat, slng: d.lng, elat: tc.lat, elng: tc.lon, color: ["#ffb547", "#ff5c8a"],
      stroke: 0.9, alt: 0.35, dash: 0.4, gap: 0.2, anim: 1500 });
  }
  refreshArcs();
  globe.pointOfView({ lat: 10, lng: 20, altitude: 2.8 }, 1200);
  try {
    const r = await post("/api/rewrite", { moves, mode: rwMode() });
    for (const m of moves) {
      const d = state.html.cities.find((c) => c.name === m.source), tc = cityByName(m.target);
      if (d && tc) {
        d.el.classList.add("moved");
        d.el.querySelector(".lbl").textContent = rwMode() === "steer" ? `${m.source} (moved)` : `${m.source} (edited)`;
        anims.push(animateMove(d, [tc.lat, tc.lon]));
      }
    }
    await Promise.all(anims);
    renderRewrite(r);
    el("rw-status").textContent = r.seconds ? `Done in ${fmt(r.seconds, 1)} s (computed live).` : "Done (precomputed suggested world).";
    el("rw-free").classList.remove("hidden");
    el("rw-ask-result").innerHTML = "";
  } catch (e) {
    el("rw-status").textContent = `Rewrite failed: ${e.message}`;
  } finally { state.busy = false; el("rewrite-go").disabled = false; }
}

el("move-add").addEventListener("click", () => {
  const used = new Set(state.moves.map((m) => m.source));
  const names = cityNames();
  const src = names.find((n) => !used.has(n)) || names[0];
  addMove(src, names.find((n) => n !== src));
});
el("rewrite-go").addEventListener("click", runRewrite);
el("rewrite-reset").addEventListener("click", () => {
  if (state.busy) return;
  state.moves = []; renderMoves(); resetMarkers();
  el("rw-cities").innerHTML = ""; el("rw-pairs").innerHTML = ""; el("rw-status").textContent = "";
  el("rw-free").classList.add("hidden");
});
el("rw-ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = el("rw-ask-input").value.trim();
  if (!q || !state.moves.length) return;
  const btn = e.target.querySelector("button"); btn.disabled = true;
  const out = el("rw-ask-result");
  out.innerHTML = `<div class="small">Asking the unedited model and the rewritten one...</div>`;
  try {
    const r = await post("/api/ask", { question: q, moves: state.moves, mode: rwMode(), max_new_tokens: 120 });
    const label = state.moves.map((m) => `${m.source} read as ${m.target}`).join("; ");
    out.innerHTML = (r.note ? `<div class="small">${esc(r.note)}</div>` : "") +
      `<div class="qa"><div class="who">Unedited model</div>${esc(r.before)}</div>
       <div class="qa after"><div class="who">Rewritten: ${esc(label)}</div>${esc(r.after)}</div>`;
  } catch (err) { out.innerHTML = `<div class="small">Failed: ${esc(err.message)}</div>`; }
  finally { btn.disabled = false; }
});

function buildRewrite() {
  el("world-chips").innerHTML = (state.menu.worlds || []).map((w, i) =>
    `<button class="chip" data-i="${i}">${esc(w.name)}</button>`).join("");
  el("world-chips").querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
    const w = state.menu.worlds[+c.dataset.i];
    state.moves = w.moves.map((m) => ({ source: m.source, target: m.target }));
    renderMoves(); rwFreeDefault(); runRewrite();
  }));
  if (state.menu.steer_available) el("rw-mode-row").classList.remove("hidden");
  renderMoves();
}

// ------------------------------------------------------------ rigour

function stat(v, l, src) { return `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div><div class="src">${esc(src)}</div></div>`; }

function renderRigour(R) {
  const m = [];
  if (R.probe) {
    m.push(stat(`${fmt(R.probe.random_median_km)} km`, "median error, places the probe never saw (random 5-fold)", "results/probe_final.json"));
    m.push(stat(`${fmt(R.probe.country_median_km)} km`, "median error, whole countries held out", "results/probe_final.json"));
  }
  el("map-stats").innerHTML = m.join("") + (R.probe ? stat(`${fmt(R.probe.baseline_median_km)} km`, "guessing the average position (baseline)", "results/probe_layers.json") +
    stat(`layer ${R.probe.layer}`, `of 28, chosen by held-out error (${fmt(R.probe.n_places)} places)`, "results/probe_layers.json") : "");
  const e = [];
  if (R.edit) {
    const src = R.edit_source || "results/edit_eval.json";
    e.push(stat(`${fmt(R.edit.cross_pass * 100)}%`, `swap edit success, ${R.edit.cross_n} random cross-hemisphere pairs`, src));
    e.push(stat(`${fmt(R.edit.cross_ctrl_pass * 100)}%`, "random-direction control of equal size, same pairs", src));
    e.push(stat(`${fmt(R.edit.any_pass * 100)}%`, `swap edit success, ${R.edit.any_n} random pairs (any hemispheres)`, src));
    e.push(stat(`${fmt(R.edit.any_ctrl_pass * 100)}%`, "control, same pairs", src));
  }
  if (R.steer) {
    e.push(stat(`${fmt(R.steer.both * 100)}%`, `"move, keep identity": location moved and identity kept, ${R.steer.n} fresh pairs (swap keeps identity ${fmt(R.steer.swap_kept * 100)}%)`, "results/steer_heldout.json"));
    e.push(stat(`${fmt(R.steer.ctrl_moved * 100)}%`, "same-size random control moved the location", "results/steer_heldout.json"));
  }
  if (R.hemisphere) e.push(stat(`${fmt(R.hemisphere.spoken * 100)}% vs ${fmt(R.hemisphere.probe * 100)}%`, `model says the right hemisphere vs its hidden map puts the city there (${R.hemisphere.n} cities)`, "results/hemisphere_check.json"));
  if (R.fiction) e.push(stat(`${R.fiction.n_coherent}/${R.fiction.n_scored}`, `invented names landing in the region their sound suggests (chance ${fmt(R.fiction.chance * 100)}%)`, "results/fictional.json"));
  if (R.rewrite) {
    e.push(stat(`${fmt(R.rewrite.city_pass * 100)}%`, `rewrite the world: moved cities whose answers all move, ${R.rewrite.n_worlds} random 3-move worlds (control ${fmt(R.rewrite.city_ctrl_pass * 100)}%)`, "results/rewrite_eval.json"));
    e.push(stat(`${fmt(R.rewrite.pair_consistent * 100)}%`, `answers about two moved cities consistent with the rewritten world (control ${fmt(R.rewrite.pair_ctrl * 100)}%, n=${R.rewrite.pair_n})`, "results/rewrite_eval.json"));
  }
  if (R.hemisphere && R.hemisphere.examples && R.hemisphere.examples.length) {
    const ex = R.hemisphere.examples.map((x) => `${esc(x.name)} (says "${esc(x.said)}", map reads ${fmt(Math.abs(x.probe_lat), 0)}&deg;${x.probe_lat >= 0 ? "N" : "S"})`).join(", ");
    const c = el("map-mouth");
    c.classList.remove("hidden");
    c.innerHTML = `<b>The map knows more than the mouth.</b> Asked which hemisphere a city is in, the model is right ${fmt(R.hemisphere.spoken * 100)}% of the time; its hidden map puts the city in the right hemisphere ${fmt(R.hemisphere.probe * 100)}% of the time (${R.hemisphere.n} cities). Examples: ${ex}.<div class="src small">results/hemisphere_check.json</div>`;
  }
  const figs = (R.figures || []).map((f) => `<img class="figure" src="figures/${esc(f)}" alt="${esc(f)}">`).join("");
  el("rigour").innerHTML = `<div class="stats">${m.join("")}${e.join("")}</div>
    <p class="small">Success = at least 80% of the scoreable ripple answers switch to the target city. The control adds a random vector with the same size as the real patch, at the same layers and token. Every number here is read from the results file shown under it.</p>
    ${figs}
    <h3>Prior work</h3>
    <p class="small">Linear probes for space and time: Gurnee and Tegmark 2023 (arXiv 2310.02207), whose place list we use. Causal interventions on spatial representations: Chen et al. 2023 (arXiv 2312.16257). Our additions: invented places, live edits on a 7B chat model with measured ripple consistency and a matched random control, several edits applied together, and a move-without-swap edit.</p>`;
}

// ------------------------------------------------------------ start-up

function hideLoading() { el("loading").classList.add("done"); }

async function init() {
  legend();
  try {
    const h = await api("/api/health");
    document.querySelectorAll(".band").forEach((b) => (b.textContent = `${h.band[0]} to ${h.band[1]}`));
  } catch (e) {
    el("loading-text").textContent = "Server not reachable. Start it with: uv run uvicorn terra.server:app --port 8000";
    el("hud-status").textContent = "Server not reachable.";
    return;
  }
  await loadPlaces().catch((e) => (el("hud-status").textContent = `Could not load places: ${e.message}`));
  hideLoading();
  state.menu = await api("/api/menu");
  const sources = [...new Set(state.menu.pairs.map((p) => p.source))];
  el("edit-source").innerHTML = sources.map((s) => `<option>${esc(s)}</option>`).join("");
  if (sources.includes("Paris")) el("edit-source").value = "Paris";
  fillTargets();
  buildCityMarkers();
  buildRewrite();
  if (state.menu.steer_available) el("mode-row").classList.remove("hidden");
  if ((state.menu.fictional || []).length >= 3)
    el("locate-input").placeholder = `e.g. ${state.menu.fictional.slice(0, 3).join(", ")}`;
  el("locate-chips").innerHTML = (state.menu.fictional || []).slice(0, 14)
    .map((n) => `<button class="chip" data-n="${esc(n)}">${esc(n)}</button>`).join("");
  el("locate-chips").querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => locate(c.dataset.n)));
  state.rigour = await api("/api/rigour").catch(() => null);
  if (state.rigour) renderRigour(state.rigour);
}
init();
