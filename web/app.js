/* TERRA INCOGNITA frontend.
 *
 * Layers on the globe (globe.gl):
 *   points   : ~10k places at their *believed* positions (merged into one
 *              mesh for speed, so clicks are resolved by screen distance)
 *   arcs     : error lines from believed to true for large cities, plus the
 *              arcs drawn when places are moved
 *   html     : markers (menu cities, the user's own places, a selected
 *              point), the located pin and its cloud, and the drag ghost
 *   rings    : pulse at a located or selected point
 *   polygons : real borders (lines) and the model's borders (filled), in one
 *              layer told apart by `kind`
 *   labels   : country names at the model's believed centroids
 */

// Continent colours: a categorical palette validated for colour-blind
// separation on the dark globe (fixed order, see DECISIONS.md).
const CONT = {
  EU: ["Europe", "#3987e5"], AS: ["Asia", "#d95926"], AF: ["Africa", "#199e70"],
  NA: ["North America", "#c98500"], SA: ["South America", "#d55181"], OC: ["Oceania", "#9085e9"],
  AN: ["Antarctica", "#9aa3b2"], "?": ["Unknown", "#6b7280"],
};
// Natural Earth's MAPCOLOR7 (1..7) mapped onto the same palette family, so
// neighbouring countries never share a colour.
const MAPCOL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"];
const QLABEL = { country: "Country", currency: "Currency", language: "Main language",
  hemisphere: "Hemisphere", season: "July is..." };
const QKEYS = Object.keys(QLABEL);
const CLICK_PX = 6;      // a press that moves less than this is a click, not a drag
const PICK_PX = 14;      // a click selects the nearest believed point within this distance
const SNAP_PX = 40;      // a drop snaps to the nearest marker within this distance

const el = (id) => document.getElementById(id);
const state = {
  places: [], bigPlaces: [], arcsErr: [], arcsEdit: [], menu: null, rigour: null,
  html: { cities: [], custom: [], picked: [], cloud: [], locate: [], ghost: [] },
  press: null, busy: false, showArcs: true, showPoints: true, lastEdit: null, tab: "map",
  moves: [], model: null, models: [], custom: [], customByModel: {}, selected: null,
  realFeatures: [], political: null, borders: "real",
};

// ------------------------------------------------------------ helpers

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) {
    const e = new Error(j.detail || j.error || `HTTP ${r.status}`);
    e.status = r.status;
    throw e;
  }
  return j;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 0) => Number(x).toLocaleString("en-GB", { maximumFractionDigits: d, minimumFractionDigits: d });
const ns = (v) => `${fmt(Math.abs(v), 1)}°${v >= 0 ? "N" : "S"}`;
const ew = (v) => `${fmt(Math.abs(v), 1)}°${v >= 0 ? "E" : "W"}`;
const modelName = () => (state.model ? state.model.display : "the model");

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
  .polygonGeoJsonGeometry((d) => d.geometry)
  .polygonCapColor((d) => (d.kind === "model" ? hexA(MAPCOL[((d.properties.MAPCOLOR7 || 1) - 1) % 7], 0.32) : "rgba(0,0,0,0)"))
  .polygonSideColor(() => "rgba(0,0,0,0)")
  .polygonStrokeColor((d) => (d.kind === "model" ? hexA(MAPCOL[((d.properties.MAPCOLOR7 || 1) - 1) % 7], 0.9) : "rgba(150,180,220,0.28)"))
  .polygonAltitude((d) => (d.kind === "model" ? 0.003 : 0.0015))
  .polygonLabel((d) => (d.kind === "model" ? `<b>${esc(d.properties.name)}</b> in ${esc(modelName())}'s world` : ""))
  .labelLat((d) => d.lat).labelLng((d) => d.lng).labelText((d) => d.text).labelSize(0.75)
  .labelDotRadius(0).labelColor(() => "rgba(255,255,255,0.9)").labelResolution(2).labelAltitude(0.007)
  // Every click goes to one handler with the clicked globe coordinates
  // (globe.gl passes {lat, lng} first for the globe, and (object, event,
  // {lat, lng, altitude}) for points, polygons and labels).
  .onGlobeClick((c, ev) => onMapClick(c.lat, c.lng, ev))
  .onPointClick((pt, ev, c) => onMapClick(c.lat, c.lng, ev))
  .onPolygonClick((poly, ev, c) => { if (poly && poly.kind === "model") showCountry(poly); onMapClick(c.lat, c.lng, ev); })
  .onLabelClick((lab, ev, c) => { if (lab && lab.feature) showCountry(lab.feature); onMapClick(c.lat, c.lng, ev); });

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

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
  globe.htmlElementsData([...h.cities, ...h.custom, ...h.picked, ...h.cloud, ...h.locate, ...h.ghost]);
}
function refreshArcs() { globe.arcsData([...(state.showArcs ? state.arcsErr : []), ...state.arcsEdit]); }
function allMarkers() { return [...state.html.cities, ...state.html.custom, ...state.html.picked]; }
function markerByName(n) { return allMarkers().find((m) => m.name === n); }

// ---- screen geometry

function screenFromEvent(ev) {
  if (!ev || ev.clientX == null) return null;
  const r = container.getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
}
function screenToGlobe(ev) {
  const s = screenFromEvent(ev);
  return s ? globe.toGlobeCoords(s.x, s.y) : null;
}
// A point on the sphere is visible if it faces the camera: (camera - p) . p > 0.
function isVisible(lat, lng, alt = 0) {
  const p = globe.getCoords(lat, lng, alt), c = globe.camera().position;
  return (c.x - p.x) * p.x + (c.y - p.y) * p.y + (c.z - p.z) * p.z > 0;
}
function nearestOnScreen(items, sx, sy, maxPx, latOf, lngOf, altOf = () => 0.004) {
  let best = null, bd = maxPx * maxPx;
  for (const it of items) {
    const la = latOf(it), lo = lngOf(it), al = altOf(it);
    if (!isVisible(la, lo, al)) continue;
    const s = globe.getScreenCoords(la, lo, al);
    const d2 = (s.x - sx) ** 2 + (s.y - sy) ** 2;
    if (d2 < bd) { bd = d2; best = it; }
  }
  return best;
}

// ------------------------------------------------------------ tabs

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  if (b.disabled) return;
  setTab(b.dataset.tab);
}));
function setTab(tab) {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x.dataset.tab === tab));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${tab}`));
  state.tab = tab;
  refreshHtml();
}

// ------------------------------------------------------------ map layer

function legend() {
  el("legend").innerHTML = Object.entries(CONT).filter(([k]) => !["AN", "?"].includes(k))
    .map(([, [n, c]]) => `<span><i style="background:${c}"></i>${n}</span>`).join("");
}

async function loadPlaces() {
  const j = await api("/api/places");
  state.places = j.points;
  state.bigPlaces = state.places.filter((p) => p.big);
  globe.pointsData(state.showPoints ? state.places : []);
  // Error lines for large, well-known cities only (the full set is a blur).
  state.arcsErr = state.bigPlaces.slice(0, 400).map((p) => ({
    slat: p.blat, slng: p.blon, elat: p.lat, elng: p.lon,
    color: ["rgba(255,255,255,0.65)", "rgba(255,255,255,0.05)"], stroke: 0.25, alt: 0.15,
  }));
  refreshArcs();
  return j;
}

el("toggle-arcs").addEventListener("change", (e) => { state.showArcs = e.target.checked; refreshArcs(); });
el("toggle-points").addEventListener("change", (e) => { state.showPoints = e.target.checked; globe.pointsData(e.target.checked ? state.places : []); });

// ------------------------------------------------------------ selection (Part 1)

function onMapClick(lat, lng, ev) {
  if (state.press) return;
  const s = screenFromEvent(ev) || globe.getScreenCoords(lat, lng, 0);
  const p = state.showPoints ? nearestOnScreen(state.places, s.x, s.y, PICK_PX, (q) => q.blat, (q) => q.blon) : null;
  if (p) selectPlace(p);
  else clearSelection();
}

function placeByName(n) { return state.places.find((p) => p.name === n); }

function selectPlace(p) {
  // A point from the 10k: give it a draggable marker at its believed position.
  const e = makeMarker({ name: p.name, lat: p.blat, lng: p.blon, kind: "place" }, "picked");
  state.html.picked = [e];
  select({ kind: "place", name: p.name, place: p, lat: p.blat, lng: p.blon, marker: e });
}

function select(sel) {
  allMarkers().forEach((m) => m.el.classList.remove("selected"));
  state.selected = sel;
  if (sel.marker) sel.marker.el.classList.add("selected");
  if (sel.kind !== "place") state.html.picked = [];
  refreshHtml();
  globe.ringsData([{ lat: sel.lat, lng: sel.lng }]);
  const p = sel.place || placeByName(sel.name);
  const lines = [`<b>${esc(sel.name)}</b>`];
  if (sel.kind === "custom") {
    const c = state.custom.find((x) => x.name === sel.name);
    lines.push(`<span class="small">your place</span>`);
    lines.push(`Believed position ${ns(sel.lat)}, ${ew(sel.lng)}`);
    if (c && c.nearest) lines.push(`Nearest real city ${esc(c.nearest.name)}, ${esc(c.nearest.country)}`);
  } else if (p) {
    lines.push(`<span class="small">${esc(p.type || "")}${p.country ? ", " + esc(p.country) : ""}</span>`);
    lines.push(`True position ${ns(p.lat)}, ${ew(p.lon)}`);
    lines.push(`Believed position ${ns(p.blat)}, ${ew(p.blon)}`);
    lines.push(`Error <b>${fmt(p.err_km)} km</b>`);
    state.arcsEdit = state.arcsEdit.filter((a) => !a.pick);
    state.arcsEdit.push({ pick: true, slat: p.blat, slng: p.blon, elat: p.lat, elng: p.lon,
      color: ["#ffb547", "#ffffff"], stroke: 0.6, alt: 0.2 });
    refreshArcs();
  } else {
    lines.push(`True position ${ns(sel.lat)}, ${ew(sel.lng)}`);
  }
  el("sel-body").innerHTML = lines.join("<br>");
  el("selection").classList.remove("hidden");
  const canEdit = state.model && state.model.editable;
  el("sel-source").disabled = el("sel-target").disabled = !canEdit;
}

function clearSelection() {
  state.selected = null;
  allMarkers().forEach((m) => m.el.classList.remove("selected"));
  state.html.picked = [];
  refreshHtml();
  globe.ringsData([]);
  el("selection").classList.add("hidden");
  state.arcsEdit = state.arcsEdit.filter((a) => !a.pick);
  refreshArcs();
}

el("sel-close").addEventListener("click", clearSelection);
el("sel-source").addEventListener("click", () => state.selected && useAs("source", state.selected.name));
el("sel-target").addEventListener("click", () => state.selected && useAs("target", state.selected.name));

function useAs(role, name) {
  if (state.tab === "rewrite") {
    if (role === "source") addMove(name, "");
    else {
      const open = state.moves.find((m) => !m.target) || state.moves[state.moves.length - 1];
      if (open) open.target = name; else state.moves.push({ source: "", target: name });
      renderMoves(); rwFreeDefault();
    }
    return;
  }
  setTab("edit");
  el(role === "source" ? "edit-source" : "edit-target").value = name;
  if (role === "source") onSourceChange();
  modeLabels();
}

// ------------------------------------------------------------ markers and dragging

function makeMarker(d, cls) {
  const e = document.createElement("div");
  e.className = `city-marker ${cls}`;
  e.innerHTML = `<div class="dot"></div><div class="lbl">${esc(d.name)}</div>`;
  const m = { ...d, el: e, home: [d.lat, d.lng], alt: d.alt ?? 0.015 };
  e.addEventListener("pointerdown", (ev) => onMarkerDown(ev, m));
  return m;
}

function buildCityMarkers() {
  state.html.cities = (state.menu.cities || []).map((c) => makeMarker({ name: c.name, lat: c.lat, lng: c.lon, kind: "city" }, ""));
  refreshHtml();
}

function onMarkerDown(ev, m) {
  if (state.busy) return;
  ev.preventDefault(); ev.stopPropagation();
  state.press = { m, x: ev.clientX, y: ev.clientY, dragging: false };
  window.addEventListener("pointermove", onPressMove);
  window.addEventListener("pointerup", onPressUp, { once: true });
}

function onPressMove(ev) {
  const p = state.press;
  if (!p) return;
  if (!p.dragging && Math.hypot(ev.clientX - p.x, ev.clientY - p.y) >= CLICK_PX) {
    p.dragging = true;
    p.m.el.classList.add("dragging");
    globe.controls().enabled = false;
    const g = document.createElement("div"); g.className = "ghost";
    state.html.ghost = [{ el: g, lat: p.m.lat, lng: p.m.lng, alt: 0.02 }];
    refreshHtml();
  }
  if (p.dragging) {
    const c = screenToGlobe(ev);
    if (c && state.html.ghost.length) { state.html.ghost[0].lat = c.lat; state.html.ghost[0].lng = c.lng; refreshHtml(); }
  }
}

function onPressUp(ev) {
  window.removeEventListener("pointermove", onPressMove);
  const p = state.press;
  state.press = null;
  globe.controls().enabled = true;
  if (!p) return;
  p.m.el.classList.remove("dragging");
  if (!p.dragging) {                                 // a click: select this marker
    select({ kind: p.m.kind, name: p.m.name, lat: p.m.lat, lng: p.m.lng, marker: p.m });
    return;
  }
  const g = state.html.ghost[0];
  state.html.ghost = []; refreshHtml();
  const drop = screenToGlobe(ev) || (g && { lat: g.lat, lng: g.lng });
  if (!drop) return;
  const s = screenFromEvent(ev);
  // Snap to the nearest marker of any kind within SNAP_PX, else to the
  // nearest big city in the places list.
  const others = allMarkers().filter((x) => x !== p.m && x.name !== p.m.name);
  const target = s ? nearestOnScreen(others, s.x, s.y, SNAP_PX, (x) => x.lat, (x) => x.lng, (x) => x.alt) : null;
  let tname = target ? target.name : null;
  if (!tname) {
    let bd = 1e9;
    for (const c of state.bigPlaces) {
      if (c.name === p.m.name) continue;
      const km = kmBetween([drop.lat, drop.lng], [c.lat, c.lon]);
      if (km < bd) { bd = km; tname = c.name; }
    }
  }
  if (!tname) return;
  if (state.tab === "rewrite") { addMove(p.m.name, tname); return; }
  setTab("edit");
  el("edit-source").value = p.m.name; onSourceChange(); el("edit-target").value = tname; modeLabels();
  runEdit(p.m.name, tname);
}

function coordsFor(name) {
  const c = ((state.menu && state.menu.cities) || []).find((x) => x.name === name);
  if (c) return [c.lat, c.lon];
  const u = state.custom.find((x) => x.name === name);
  if (u) return [u.blat, u.blon];
  const p = placeByName(name);
  if (p) return p.big ? [p.lat, p.lon] : [p.blat, p.blon];
  return null;
}

// ------------------------------------------------------------ new worlds and custom places (Part 2)

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
    globe.ringsData([{ lat: r.lat, lng: r.lon }]);
    addCustom({ name, blat: r.lat, blon: r.lon, nearest: r.nearest, cloud_km: r.cloud_median_km });
    globe.pointOfView({ lat: r.lat, lng: r.lon, altitude: 1.6 }, 1400);
    const n = r.nearest;
    card.innerHTML = `${esc(modelName())} places <b>${esc(name)}</b> at ${ns(r.lat)}, ${ew(r.lon)},
      nearest real city <b>${esc(n.name)}, ${esc(n.country)}</b> (${fmt(n.km)} km away).<br>
      Cloud spread (median distance of the bootstrap probes from the main reading): ${fmt(r.cloud_median_km)} km.<br>
      <span class="small">Added to "Your places" below and to the globe.</span>`;
  } catch (e) {
    card.innerHTML = `Could not locate: ${esc(e.message)}`;
  }
}
el("locate-form").addEventListener("submit", (e) => { e.preventDefault(); locate(el("locate-input").value); });

function storageKey(key) { return `terra.custom.v1.${key}`; }
function loadCustomFor(key) {
  if (state.customByModel[key]) return state.customByModel[key];
  let list = [];
  try { list = JSON.parse(localStorage.getItem(storageKey(key)) || "[]"); } catch (e) { list = []; }
  return Array.isArray(list) ? list : [];
}
function saveCustom() {
  if (!state.model) return;
  state.customByModel[state.model.key] = state.custom;
  try { localStorage.setItem(storageKey(state.model.key), JSON.stringify(state.custom)); } catch (e) { /* storage blocked */ }
}
function addCustom(c) {
  state.custom = state.custom.filter((x) => x.name !== c.name);
  state.custom.push(c);
  saveCustom(); buildCustomMarkers(); renderCustomList(); refreshDatalists();
}
function removeCustom(name) {
  state.custom = state.custom.filter((x) => x.name !== name);
  if (state.selected && state.selected.kind === "custom" && state.selected.name === name) clearSelection();
  saveCustom(); buildCustomMarkers(); renderCustomList(); refreshDatalists();
}
function buildCustomMarkers() {
  state.html.custom = state.custom.map((c) => makeMarker({ name: c.name, lat: c.blat, lng: c.blon, kind: "custom" }, "custom"));
  refreshHtml();
}
function renderCustomList() {
  const ul = el("custom-list");
  if (!state.custom.length) { ul.innerHTML = `<li class="empty">None yet: locate a name above.</li>`; return; }
  ul.innerHTML = state.custom.map((c, i) => `<li><span><b>${esc(c.name)}</b> <span class="small">near ${esc(c.nearest ? c.nearest.name : "?")}</span></span>
    <button class="use" data-i="${i}" title="Select it on the globe">select</button>
    <button class="x" data-i="${i}" title="Remove">&times;</button></li>`).join("");
  ul.querySelectorAll(".x").forEach((b) => b.addEventListener("click", () => removeCustom(state.custom[+b.dataset.i].name)));
  ul.querySelectorAll(".use").forEach((b) => b.addEventListener("click", () => {
    const c = state.custom[+b.dataset.i], m = markerByName(c.name);
    globe.pointOfView({ lat: c.blat, lng: c.blon, altitude: 1.8 }, 900);
    select({ kind: "custom", name: c.name, lat: c.blat, lng: c.blon, marker: m });
  }));
}

// ---- free-text names with suggestions

function nameOptions() {
  const out = [], seen = new Set();
  const add = (n) => { if (n && !seen.has(n)) { seen.add(n); out.push(n); } };
  ((state.menu && state.menu.cities) || []).forEach((c) => add(c.name));
  state.custom.forEach((c) => add(c.name));
  state.bigPlaces.forEach((p) => add(p.name));
  return out;
}
function validatedTargets(src) {
  return ((state.menu && state.menu.pairs) || []).filter((p) => p.source === src).map((p) => p.target);
}
function refreshDatalists() {
  const opts = nameOptions();
  el("dl-names").innerHTML = opts.map((n) => `<option value="${esc(n)}">`).join("");
  const src = el("edit-source").value.trim();
  const vt = validatedTargets(src);
  const rest = opts.filter((n) => !vt.includes(n) && n !== src);
  el("dl-targets").innerHTML = [...vt.map((n) => `<option value="${esc(n)}" label="validated pair">`),
    ...rest.map((n) => `<option value="${esc(n)}">`)].join("");
}

// ------------------------------------------------------------ edit

function modeLabels() {
  const src = el("edit-source").value.trim() || "Paris", tgt = el("edit-target").value.trim() || "Sydney";
  el("lbl-swap").textContent = `swap: ${src} becomes ${tgt}`;
  el("lbl-steer").textContent = `move: ${src} stays ${src}`;
}

function onSourceChange() {
  const src = el("edit-source").value.trim();
  refreshDatalists();
  if (src) el("ask-input").value = `I'm in ${src} for Christmas, what should I pack? Answer in two sentences.`;
  modeLabels();
}
el("edit-source").addEventListener("input", onSourceChange);
el("edit-target").addEventListener("input", modeLabels);

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
  for (const d of allMarkers()) { d.lat = d.home[0]; d.lng = d.home[1]; d.el.classList.remove("moved");
    d.el.querySelector(".lbl").textContent = d.name; }
  state.arcsEdit = state.arcsEdit.filter((a) => a.pick); refreshArcs(); refreshHtml();
}

function cellClass(sc, q) {
  if (!sc) return "unscored";
  const v = sc.per_question[q];
  if (!v.countable) return "kept";
  if (v.flipped) return "flip";
  if (v.matches_source && !v.matches_target) return "kept";
  return "other";
}

function setMeter(id, frac, text, ctrl = false) {
  el(id).style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
  el(id).classList.toggle("ctrl", ctrl);
  el(`${id}-val`).textContent = text;
}
const meterLabel = (id, t) => (el(id).closest(".meter").querySelector("span").textContent = t);

function renderRipple(r) {
  const tb = document.querySelector("#ripple tbody");
  const tick = (c) => (c === "flip" ? '<span class="tick">&#10003; flipped</span>' : "");
  let rows = QKEYS.map((q) => {
    const ca = cellClass(r.score_after, q), cc = cellClass(r.score_control, q);
    return `<tr><td class="q">${QLABEL[q]}</td><td>${esc(r.before[q])}</td>
      <td class="${ca}">${esc(r.after[q])}${tick(ca)}</td><td class="${cc}">${esc(r.control[q])}${tick(cc)}</td></tr>`;
  }).join("");
  if (r.landmark) {
    rows += `<tr><td class="q">Famous landmark<br><span class="small">(identity, not scored)</span></td>
      <td>${esc(r.landmark.before)}</td><td>${esc(r.landmark.after)}</td><td>${esc(r.landmark.control)}</td></tr>`;
  }
  tb.innerHTML = rows;
  el("ripple").classList.remove("hidden");
  el("meters").classList.remove("hidden");
  const note = el("unscored-note");
  if (!r.score_after) {
    // Invented places (and landmarks) have no GeoNames answer key.
    note.textContent = r.unscored_reason || "no answer key for invented places: answers shown unscored";
    note.classList.remove("hidden");
    meterLabel("meter-edit", "Edit consistency"); meterLabel("meter-ctrl", "Control");
    setMeter("meter-edit", 0, "unscored"); setMeter("meter-ctrl", 0, "unscored", true);
    return;
  }
  note.classList.add("hidden");
  if (r.mode === "steer") {
    // Tier 2 aims to move the location answers and keep the identity answers.
    const pq = r.score_after.per_question;
    const loc = ["hemisphere", "season"].filter((q) => pq[q].countable);
    const moved = loc.filter((q) => pq[q].flipped).length;
    const idq = ["country", "currency", "language"];
    const kept = idq.filter((q) => pq[q].matches_source && !pq[q].matches_target).length;
    meterLabel("meter-edit", "Location moved"); meterLabel("meter-ctrl", "Identity kept");
    setMeter("meter-edit", moved / Math.max(1, loc.length), `${moved}/${loc.length}`);
    setMeter("meter-ctrl", kept / idq.length, `${kept}/${idq.length}`);
    return;
  }
  meterLabel("meter-edit", "Edit consistency"); meterLabel("meter-ctrl", "Control");
  const sa = r.score_after, sc = r.score_control;
  setMeter("meter-edit", sa.n_flipped / Math.max(1, sa.n_countable), `${sa.n_flipped}/${sa.n_countable}`);
  setMeter("meter-ctrl", sc.n_flipped / Math.max(1, sc.n_countable), `${sc.n_flipped}/${sc.n_countable}`, true);
}

function pendingRipple() {
  const tb = document.querySelector("#ripple tbody");
  tb.innerHTML = [...QKEYS.map((q) => QLABEL[q]), "Famous landmark"].map((l) => `<tr><td class="q">${l}</td><td class="pending">...</td><td class="pending">...</td><td class="pending">...</td></tr>`).join("");
  el("ripple").classList.remove("hidden");
  el("unscored-note").classList.add("hidden");
}

function currentMode() {
  const m = document.querySelector('input[name="mode"]:checked');
  return m ? m.value : "swap";
}

async function runEdit(src, tgt) {
  src = (src || "").trim(); tgt = (tgt || "").trim();
  if (state.busy) return;
  if (!src || !tgt) { el("edit-status").textContent = "Type (or pick) both a source and a target."; return; }
  if (src === tgt) { el("edit-status").textContent = "Source and target are the same name."; return; }
  state.busy = true; el("edit-go").disabled = true;
  setTab("edit");
  resetMarkers();
  clearLocate();
  modeLabels();
  const mode = currentMode();
  const validated = mode === "swap" && validatedTargets(src).includes(tgt);
  el("edit-status").textContent = validated
    ? "Using the precomputed answers for this validated pair..."
    : `Computing live: patching "${src}" with "${tgt}" (a few seconds; longer for the larger models)...`;
  pendingRipple();
  const d = markerByName(src);
  const to = coordsFor(tgt);
  try {
    const req = post("/api/edit", { source: src, target: tgt, mode });
    if (d && to) {
      state.arcsEdit.push({ slat: d.lat, slng: d.lng, elat: to[0], elng: to[1], color: ["#ffb547", "#ff5c8a"],
        stroke: 0.9, alt: 0.35, dash: 0.4, gap: 0.2, anim: 1500 });
      refreshArcs();
      globe.pointOfView({ lat: (d.lat + to[0]) / 2, lng: d.lng + ((((to[1] - d.lng) % 360) + 540) % 360 - 180) / 2, altitude: 2.6 }, 1200);
    }
    const r = await req;
    state.lastEdit = r;
    if (d) {
      d.el.classList.add("moved");
      d.el.querySelector(".lbl").textContent = mode === "steer" ? `${src} (moved, still ${src})` : `${src} (edited)`;
      // Fly to where the location probe now reads the edited place (measured
      // on the patched hidden state), falling back to the target's position.
      const m = r.edited_location && r.edited_location.lat !== undefined ? [r.edited_location.lat, r.edited_location.lon] : to;
      if (m) await animateMove(d, m);
    }
    renderRipple(r);
    let msg = r.cached ? "Done (precomputed for this validated pair)." : `Done in ${fmt(r.seconds || 0, 1)} s (computed live).`;
    if (r.edited_location && r.edited_location.nearest) {
      const n = r.edited_location.nearest;
      msg += ` The location probe now reads "${src}" near ${n.name}, ${n.country}.`;
    }
    el("edit-status").textContent = msg;
    el("free").classList.remove("hidden");
    el("ask-result").innerHTML = "";
  } catch (e) {
    el("edit-status").textContent = `Edit failed: ${e.message}`;
    el("ripple").classList.add("hidden");
  } finally {
    state.busy = false; el("edit-go").disabled = !(state.model && state.model.editable);
  }
}

el("edit-go").addEventListener("click", () => runEdit(el("edit-source").value, el("edit-target").value));
el("edit-reset").addEventListener("click", () => {
  if (state.busy) return;
  resetMarkers();
  ["ripple", "meters", "free", "unscored-note"].forEach((id) => el(id).classList.add("hidden"));
  el("edit-status").textContent = "";
  globe.pointOfView({ lat: 25, lng: 10, altitude: 2.3 }, 1000);
});

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

function renderMoves() {
  el("moves").innerHTML = state.moves.map((m, i) => `
    <div class="move-row" data-i="${i}">
      <input class="mv-src" list="dl-names" value="${esc(m.source)}" placeholder="place" maxlength="80">
      <span class="arrow">&rarr;</span>
      <input class="mv-tgt" list="dl-names" value="${esc(m.target)}" placeholder="moved onto" maxlength="80">
      <button class="x" title="Remove this move">&times;</button>
    </div>`).join("");
  el("moves").querySelectorAll(".move-row").forEach((row) => {
    const i = +row.dataset.i;
    row.querySelector(".mv-src").addEventListener("input", (e) => { state.moves[i].source = e.target.value; rwFreeDefault(); });
    row.querySelector(".mv-tgt").addEventListener("input", (e) => { state.moves[i].target = e.target.value; });
    row.querySelector(".x").addEventListener("click", () => { state.moves.splice(i, 1); renderMoves(); rwFreeDefault(); });
  });
  el("move-add").disabled = state.moves.length >= ((state.menu && state.menu.max_moves) || 3);
}

function rwFreeDefault() {
  const s = state.moves.map((m) => m.source.trim()).filter(Boolean);
  const q = s.length === 1 ? `I'm visiting ${s[0]} this July. What should I pack? Answer in two sentences.`
    : s.length === 2 ? `I'm visiting ${s[0]} and then ${s[1]} this July. What should I pack for each city? Answer in two short sentences.`
    : s.length >= 3 ? `I'm visiting ${s[0]}, ${s[1]} and ${s[2]} this July. What should I pack for each city? Answer in three short sentences.` : "";
  el("rw-ask-input").value = q;
}

function addMove(source, target) {
  const max = (state.menu && state.menu.max_moves) || 3;
  const existing = source && state.moves.find((m) => m.source === source);
  if (existing) existing.target = target || existing.target;
  else if (state.moves.length < max) state.moves.push({ source, target });
  else state.moves[max - 1] = { source, target };
  renderMoves(); rwFreeDefault();
}

function rwMode() { const m = document.querySelector('input[name="rwmode"]:checked'); return m ? m.value : "swap"; }

function renderRewrite(r) {
  const tick = (c) => (c === "flip" ? '<span class="tick">&#10003;</span>' : "");
  let rows = "";
  for (const c of r.cities) {
    const tag = c.scored === false ? ` <span class="small unscored">(${esc(c.unscored_reason)})</span>`
      : c.passes ? ' <span class="tick">(all answers moved)</span>' : "";
    rows += `<tr class="city-head"><td colspan="4">${esc(c.source)} &rarr; ${esc(c.target)}${tag}</td></tr>`;
    for (const q of QKEYS) {
      const ca = cellClass(c.score_after, q), cc = cellClass(c.score_control, q);
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
      const cls = p.key_after === null ? "unscored" : !scored ? "" : p.after_correct ? "flip" : "wrong";
      const sub = p.key_after === null ? `<div class="small">unscored: no answer key</div>`
        : scored ? `<div class="small">rewritten world: ${esc(fmtKey(p, p.key_after))}</div>` : "";
      prow += `<tr><td class="q">${esc(p.question.replace(/ Answer.*$/, ""))}</td><td>${esc(p.before)}</td>
        <td class="${cls}">${esc(p.after)}${sub}</td><td>${esc(p.control)}</td></tr>`;
    }
    const scoredRows = pr.filter((p) => p.key_after !== null && p.before_correct);
    const ok = scoredRows.filter((p) => p.after_correct).length;
    el("rw-pairs").innerHTML = `<h3>Questions about two moved places</h3>
      <div class="rw-summary">${scoredRows.length ? `${ok} of ${scoredRows.length} scoreable answers agree with the rewritten world.` : "No scoreable pair answers (they involve places without an answer key)."}</div>
      <table class="rw-table"><thead><tr><th>Question</th><th>Before</th><th>After rewrite</th><th>Random control</th></tr></thead><tbody>${prow}</tbody></table>`;
  } else el("rw-pairs").innerHTML = "";
}

async function runRewrite() {
  if (state.busy || !state.moves.length) return;
  const moves = state.moves.map((m) => ({ source: m.source.trim(), target: m.target.trim() }));
  if (moves.some((m) => !m.source || !m.target)) { el("rw-status").textContent = "Every move needs a place and a target."; return; }
  if (moves.some((m) => m.source === m.target)) { el("rw-status").textContent = "A move has the same place as its target."; return; }
  if (new Set(moves.map((m) => m.source)).size !== moves.length) { el("rw-status").textContent = "Each place can only be moved once."; return; }
  state.busy = true; el("rewrite-go").disabled = true;
  resetMarkers(); clearLocate();
  const precomputed = rwMode() === "swap" && (state.menu.worlds || []).some((w) => JSON.stringify(w.moves) === JSON.stringify(moves));
  el("rw-status").textContent = precomputed ? "Using the precomputed answers for this suggested world..."
    : `Computing live: applying ${moves.length} edit${moves.length > 1 ? "s" : ""} together (about 10 to 15 s; longer for the larger models)...`;
  el("rw-cities").innerHTML = ""; el("rw-pairs").innerHTML = "";
  for (const m of moves) {
    const d = markerByName(m.source), to = coordsFor(m.target);
    if (d && to) state.arcsEdit.push({ slat: d.lat, slng: d.lng, elat: to[0], elng: to[1], color: ["#ffb547", "#ff5c8a"],
      stroke: 0.9, alt: 0.35, dash: 0.4, gap: 0.2, anim: 1500 });
  }
  refreshArcs();
  globe.pointOfView({ lat: 10, lng: 20, altitude: 2.8 }, 1200);
  try {
    const r = await post("/api/rewrite", { moves, mode: rwMode() });
    const anims = [];
    for (const m of moves) {
      const d = markerByName(m.source), to = coordsFor(m.target);
      if (d && to) {
        d.el.classList.add("moved");
        d.el.querySelector(".lbl").textContent = rwMode() === "steer" ? `${m.source} (moved)` : `${m.source} (edited)`;
        anims.push(animateMove(d, to));
      }
    }
    await Promise.all(anims);
    renderRewrite(r);
    el("rw-status").textContent = r.cached ? "Done (precomputed suggested world)." : `Done in ${fmt(r.seconds || 0, 1)} s (computed live).`;
    el("rw-free").classList.remove("hidden");
    el("rw-ask-result").innerHTML = "";
  } catch (e) {
    el("rw-status").textContent = `Rewrite failed: ${e.message}`;
  } finally { state.busy = false; el("rewrite-go").disabled = !(state.model && state.model.editable); }
}

el("move-add").addEventListener("click", () => {
  const used = new Set(state.moves.map((m) => m.source));
  const names = nameOptions();
  const src = names.find((n) => !used.has(n)) || "";
  addMove(src, names.find((n) => n !== src) || "");
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
  const moves = state.moves.map((m) => ({ source: m.source.trim(), target: m.target.trim() })).filter((m) => m.source && m.target);
  if (!q || !moves.length) return;
  const btn = e.target.querySelector("button"); btn.disabled = true;
  const out = el("rw-ask-result");
  out.innerHTML = `<div class="small">Asking the unedited model and the rewritten one...</div>`;
  try {
    const r = await post("/api/ask", { question: q, moves, mode: rwMode(), max_new_tokens: 120 });
    const label = moves.map((m) => `${m.source} read as ${m.target}`).join("; ");
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
  el("rw-mode-row").classList.toggle("hidden", !state.menu.steer_available);
  renderMoves();
}

// ------------------------------------------------------------ political map (Part 4)

async function loadRealBorders() {
  try {
    const g = await (await fetch("vendor/countries.geojson")).json();
    state.realFeatures = g.features.map((f) => ({ ...f, kind: "real" }));
  } catch (e) { state.realFeatures = []; }
}

async function loadPolitical() {
  state.political = null;
  el("country-card").classList.add("hidden");
  try {
    const j = await api("/api/political");
    j.map.features = j.map.features.map((f) => ({ ...f, kind: "model" }));
    state.political = j;
    const s = j.stats;
    el("political-note").innerHTML = s.share_of_model_land_in_right_country == null ? "" :
      `${esc(j.display)}'s map: ${fmt(s.n_countries_drawn)} countries drawn; ${fmt(s.share_of_model_land_in_right_country * 100)}% of its land lies in the right country. <span class="src">${esc(state.rigour ? state.rigour.results_dir : "results")}/political_stats.json</span>`;
  } catch (e) {
    el("political-note").textContent = e.status === 404 ? `No political map built for ${modelName()} yet.` : `Political map unavailable: ${e.message}`;
  }
  applyBorders();
  renderDisplaced();
}

function applyBorders() {
  const b = state.borders, model = state.political ? state.political.map.features : [];
  const data = b === "real" ? state.realFeatures : b === "model" ? model : [...model, ...state.realFeatures];
  globe.polygonsData(data);
  if (b !== "real" && model.length) {
    const top = [...model].sort((x, y) => (y.properties.model_area_km2 || 0) - (x.properties.model_area_km2 || 0)).slice(0, 24);
    globe.labelsData(top.map((f) => ({ lat: f.properties.believed_centroid[0], lng: f.properties.believed_centroid[1],
      text: f.properties.name, feature: f })));
  } else globe.labelsData([]);
}
document.querySelectorAll('input[name="borders"]').forEach((r) => r.addEventListener("change", (e) => {
  state.borders = e.target.value; applyBorders();
}));

function statsFor(iso) {
  return state.political ? state.political.stats.countries.find((c) => c.iso === iso) : null;
}

function showCountry(feature) {
  const p = feature.properties, s = statsFor(p.ISO_A2) || p;
  const card = el("country-card");
  const where = s.displacement_km < 100 ? "sits close to where it is (within 100 km)"
    : `sits ${fmt(s.displacement_km)} km ${esc(s.direction)} of where it is`;
  const prec = s.precision == null ? "" : `; ${fmt(s.precision * 100)}% of its territory overlaps the real ${esc(s.name)}`;
  const rec = s.recall == null ? "" : `, and it covers ${fmt(s.recall * 100)}% of the real country`;
  card.innerHTML = `In ${esc(modelName())}'s world, <b>${esc(s.name)}</b> ${where}${prec}${rec}.
    <div class="small">${fmt(s.n_places || 0)} places; displacement = distance between the centroids of their true and believed positions.</div>
    <div class="src small">${esc(state.rigour ? state.rigour.results_dir : "results")}/political_stats.json</div>`;
  card.classList.remove("hidden");
  setTab("map");
}

function renderDisplaced() {
  const box = el("displaced");
  const md = state.political ? state.political.stats.most_displaced || [] : [];
  if (!md.length) { box.classList.add("hidden"); return; }
  box.innerHTML = `<h3>Most displaced (20+ places)</h3><ol>${md.slice(0, 8).map((c) =>
    `<li data-iso="${esc(c.iso)}">${esc(c.name)}: ${fmt(c.displacement_km)} km ${esc(c.direction)}</li>`).join("")}</ol>`;
  box.querySelectorAll("li").forEach((li) => li.addEventListener("click", () => {
    const c = statsFor(li.dataset.iso);
    if (!c) return;
    globe.pointOfView({ lat: c.believed_centroid[0], lng: c.believed_centroid[1], altitude: 1.9 }, 1000);
    const f = state.political.map.features.find((x) => x.properties.ISO_A2 === c.iso);
    showCountry(f || { properties: { ...c, ISO_A2: c.iso } });
  }));
  box.classList.remove("hidden");
}

// ------------------------------------------------------------ rigour

function stat(v, l, src) { return `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div><div class="src">${esc(src)}</div></div>`; }

function renderRigour(R) {
  const dir = R.results_dir || "results";
  const m = [];
  if (R.probe) {
    m.push(stat(`${fmt(R.probe.random_median_km)} km`, "median error, places the probe never saw (random 5-fold)", `${dir}/probe_final.json`));
    m.push(stat(`${fmt(R.probe.country_median_km)} km`, "median error, whole countries held out", `${dir}/probe_final.json`));
  }
  el("map-stats").innerHTML = m.join("") + (R.probe ? (R.probe.baseline_median_km != null ? stat(`${fmt(R.probe.baseline_median_km)} km`, "guessing the average position (baseline)", `${dir}/probe_final.json`) : "") +
    stat(`layer ${R.probe.layer}`, `chosen by held-out error (${fmt(R.probe.n_places)} places)`, `${dir}/probe_final.json`) : "");
  const e = [];
  if (R.gate) e.push(stat(`${R.gate.pass}/${R.gate.n}`, `gate pairs flipping 4+ of 5 answers (control ${R.gate.control_pass}/${R.gate.n}): ${esc(R.gate.decision)}`, `${dir}/gate.json`));
  if (R.edit) {
    const src = R.edit_source || `${dir}/edit_eval.json`;
    e.push(stat(`${fmt(R.edit.cross_pass * 100)}%`, `swap edit success, ${R.edit.cross_n} random cross-hemisphere pairs`, src));
    e.push(stat(`${fmt(R.edit.cross_ctrl_pass * 100)}%`, "random-direction control of equal size, same pairs", src));
    e.push(stat(`${fmt(R.edit.any_pass * 100)}%`, `swap edit success, ${R.edit.any_n} random pairs (any hemispheres)`, src));
    e.push(stat(`${fmt(R.edit.any_ctrl_pass * 100)}%`, "control, same pairs", src));
  }
  if (R.steer) {
    e.push(stat(`${fmt(R.steer.both * 100)}%`, `"move, keep identity": location moved and identity kept, ${R.steer.n} fresh pairs (swap keeps identity ${fmt(R.steer.swap_kept * 100)}%)`, `${dir}/steer_heldout.json`));
    e.push(stat(`${fmt(R.steer.ctrl_moved * 100)}%`, "same-size random control moved the location", `${dir}/steer_heldout.json`));
  }
  if (R.rewrite) {
    e.push(stat(`${fmt(R.rewrite.city_pass * 100)}%`, `rewrite the world: moved cities whose answers all move, ${R.rewrite.n_worlds} random 3-move worlds (control ${fmt(R.rewrite.city_ctrl_pass * 100)}%)`, `${dir}/rewrite_eval.json`));
    e.push(stat(`${fmt(R.rewrite.pair_consistent * 100)}%`, `answers about two moved cities consistent with the rewritten world (control ${fmt(R.rewrite.pair_ctrl * 100)}%, n=${R.rewrite.pair_n})`, `${dir}/rewrite_eval.json`));
  }
  if (R.political && R.political.right_country != null) e.push(stat(`${fmt(R.political.right_country * 100)}%`, `of the model's land lies in the right country (political map, ${R.political.n_drawn} countries drawn)`, `${dir}/political_stats.json`));
  if (R.hemisphere) e.push(stat(`${fmt(R.hemisphere.spoken * 100)}% vs ${fmt(R.hemisphere.probe * 100)}%`, `model says the right hemisphere vs its hidden map puts the city there (${R.hemisphere.n} cities)`, `${dir}/hemisphere_check.json`));
  if (R.fiction) e.push(stat(`${R.fiction.n_coherent}/${R.fiction.n_scored}`, `invented names landing in the region their sound suggests (chance ${fmt((R.fiction.chance || 0) * 100)}%)`, `${dir}/fictional.json`));
  if (R.hemisphere && R.hemisphere.examples && R.hemisphere.examples.length) {
    const ex = R.hemisphere.examples.map((x) => `${esc(x.name)} (says "${esc(x.said)}", map reads ${fmt(Math.abs(x.probe_lat), 0)}&deg;${x.probe_lat >= 0 ? "N" : "S"})`).join(", ");
    const c = el("map-mouth");
    c.classList.remove("hidden");
    c.innerHTML = `<b>The map knows more than the mouth.</b> Asked which hemisphere a city is in, the model is right ${fmt(R.hemisphere.spoken * 100)}% of the time; its hidden map puts the city in the right hemisphere ${fmt(R.hemisphere.probe * 100)}% of the time (${R.hemisphere.n} cities). Examples: ${ex}.<div class="src small">${esc(dir)}/hemisphere_check.json</div>`;
  } else el("map-mouth").classList.add("hidden");
  const figs = (R.figures || []).map((f) => `<img class="figure" src="figures/${encodeURIComponent(f)}?m=${encodeURIComponent(R.model || "")}" alt="${esc(f)}">`).join("");
  el("rigour").innerHTML = `<div class="small">Numbers for ${esc(R.display || modelName())}.</div><div class="stats">${m.join("")}${e.join("")}</div>
    <p class="small">Success = at least 80% of the scoreable ripple answers switch to the target city. The control adds a random vector with the same size as the real patch, at the same layers and tokens. Every number here is read from the results file shown under it.</p>
    ${figs}
    <h3>Prior work</h3>
    <p class="small">Linear probes for space and time: Gurnee and Tegmark 2023 (arXiv 2310.02207), whose place list we use. Causal interventions on spatial representations: Chen et al. 2023 (arXiv 2312.16257). Our additions: invented places, live edits on chat models with measured ripple consistency and a matched random control, several edits applied together, a move-without-swap edit, and the model's political map.</p>`;
}

// ------------------------------------------------------------ models (Part 3)

function showLoading(text) { el("loading-text").textContent = text; el("loading").classList.remove("done"); }
function hideLoading() { el("loading").classList.add("done"); }

async function loadModels() {
  const j = await api("/api/models");
  state.models = j.models;
  const sel = el("model-select");
  sel.innerHTML = j.models.map((m) => `<option value="${esc(m.key)}" ${m.available ? "" : "disabled"} ${m.loaded ? "selected" : ""}
    title="${esc(m.available ? (m.editable ? "map and edits" : "map only: no edit gate yet") : "not built yet: run " + m.build_command)}">${esc(m.display)}${m.available ? (m.editable ? "" : " (map only)") : " (not built yet)"}</option>`).join("");
  const unbuilt = j.models.filter((m) => !m.available);
  const notes = unbuilt.map((m) => `${esc(m.display)} is not built yet: run <code>${esc(m.build_command)}</code>`);
  const cur = j.models.find((m) => m.loaded);
  if (cur && cur.available && !cur.editable) notes.unshift(`Edit and Rewrite are off for ${esc(cur.display)}: no edit gate yet (run <code>uv run python scripts/build_model.py --model ${esc(cur.key)} --stage gate</code>, then <code>--stage menu</code>).`);
  el("model-note").innerHTML = notes.join("<br>");
  el("model-note").classList.toggle("hidden", !notes.length);
}

el("model-select").addEventListener("change", async (e) => {
  const key = e.target.value;
  if (!state.model || key === state.model.key) return;
  const to = (state.models.find((m) => m.key === key) || {}).display || key;
  showLoading(`Unloading ${state.model.display}, loading ${to}... (this can take a minute for the larger models)`);
  saveCustom();
  try {
    await post("/api/model", { key });
  } catch (err) {
    el("loading-text").textContent = `Could not switch: ${err.message}`;
    setTimeout(hideLoading, 4000);
    await loadModels().catch(() => {});
    return;
  }
  await loadAll();
});

function setEditable(ok, reason) {
  for (const t of ["edit", "rewrite"]) {
    const b = document.querySelector(`#tabs button[data-tab="${t}"]`);
    b.disabled = !ok; b.classList.toggle("disabled", !ok); b.title = ok ? "" : reason;
  }
  for (const id of ["edit-disabled", "rw-disabled"]) {
    el(id).textContent = ok ? "" : reason;
    el(id).classList.toggle("hidden", ok);
  }
  el("edit-go").disabled = !ok; el("rewrite-go").disabled = !ok;
  if (!ok && (state.tab === "edit" || state.tab === "rewrite")) setTab("map");
}

async function loadAll() {
  showLoading(state.model ? "Loading the model's map..." : "Loading...");
  let h;
  try {
    h = await api("/api/health");
    if (!h.ok) throw new Error(h.error || (h.switching ? "switching models" : "no model loaded"));
  } catch (e) {
    el("loading-text").textContent = `Server not ready: ${e.message}. Start it with: uv run uvicorn terra.server:app --port 8000`;
    el("hud-status").textContent = "Server not ready.";
    return;
  }
  state.model = { key: h.key, display: h.display, layer: h.probe_layer, editable: h.editable, band: h.band };
  el("model-name").textContent = h.display;
  document.querySelectorAll(".band").forEach((b) => (b.textContent = h.band ? `${h.band[0]} to ${h.band[1]}` : "(no gate yet)"));
  clearSelection(); clearLocate();
  state.arcsEdit = []; state.moves = []; state.lastEdit = null;
  ["ripple", "meters", "free", "unscored-note", "rw-free"].forEach((id) => el(id).classList.add("hidden"));
  el("edit-status").textContent = ""; el("rw-status").textContent = ""; el("rw-cities").innerHTML = ""; el("rw-pairs").innerHTML = "";
  await loadModels().catch(() => {});
  try { await loadPlaces(); } catch (e) { el("hud-status").textContent = `Could not load places: ${e.message}`; }
  state.menu = await api("/api/menu");
  state.custom = loadCustomFor(h.key);
  buildCityMarkers(); buildCustomMarkers(); renderCustomList(); refreshDatalists();
  setEditable(state.menu.editable, state.menu.not_editable_reason || "");
  if (state.menu.editable && !el("edit-source").value) {
    const p = state.menu.pairs.find((x) => x.source === "Paris") || state.menu.pairs[0];
    if (p) { el("edit-source").value = p.source; el("edit-target").value = p.target; }
  }
  onSourceChange();
  el("mode-row").classList.toggle("hidden", !state.menu.steer_available);
  if ((state.menu.fictional || []).length >= 3)
    el("locate-input").placeholder = `e.g. ${state.menu.fictional.slice(0, 3).join(", ")}`;
  el("locate-chips").innerHTML = (state.menu.fictional || []).slice(0, 14)
    .map((n) => `<button class="chip" data-n="${esc(n)}">${esc(n)}</button>`).join("");
  el("locate-chips").querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => locate(c.dataset.n)));
  buildRewrite();
  refreshArcs(); refreshHtml();
  state.rigour = await api("/api/rigour").catch(() => null);
  if (state.rigour) renderRigour(state.rigour);
  await loadPolitical();
  el("hud-status").textContent = `${h.display} · probe layer ${h.probe_layer} · ${fmt(state.places.length)} places at their believed positions`;
  hideLoading();
}

async function init() {
  legend();
  await loadRealBorders();
  applyBorders();
  await loadAll();
}
init();
