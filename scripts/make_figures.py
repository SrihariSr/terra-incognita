"""Figures for the rigour slide and Devpost, drawn only from results/ files.

    uv run python scripts/make_figures.py
Writes PNGs to results/figures/:
    probe_by_layer.png    held-out probe error per layer (3k sweep)
    error_map.png         country-blocked error at each place's true position
    band_sweep.png        flip rate per ripple question per patched layer band
    edit_vs_control.png   edit success against the equal-norm random control
    believed_map.png      true vs believed positions for large cities
    swap_vs_move.png      tier 1 swap vs tier 2 steering vs control on fresh pairs
    rewrite_world.png     several moves applied together (results/rewrite_eval.json)
Each figure is skipped if its results file is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terra import RESULTS  # noqa: E402
from terra.probe import great_circle_km, xyz_to_latlon  # noqa: E402

FIG = RESULTS / "figures"
# Reference palette (light surface): slot 1 blue, slot 2 orange; grey for
# reference marks such as the control and baselines.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
})


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def probe_by_layer():
    d = load("probe_layers.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = d["modes"][d["chosen"]["mode"]]
    L = [r["layer"] for r in rows]
    for key, col, lab in (("random", BLUE, "random split"), ("blocked_country", ORANGE, "whole countries held out")):
        y = [r[key]["median_km"] for r in rows]
        ax.plot(L, y, color=col, lw=2)
        ax.annotate(lab, (L[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=10)
    base = d["baseline_mean_direction"]["random"]["median_km"]
    ax.axhline(base, color=GREY, lw=1.2, ls="--")
    ax.annotate("guess the average position", (0, base), xytext=(0, 5), textcoords="offset points", color=INK2, fontsize=10)
    full = load("probe_layers_full.json")
    if full:
        fr = full["modes"][full["chosen"]["mode"]]
        for key, col in (("random", BLUE), ("blocked_country", ORANGE)):
            ax.scatter([r["layer"] for r in fr], [r[key]["median_km"] for r in fr], s=34, color=col,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3, marker="D")
        ax.annotate(f"diamonds: all {full['n_places']:,} places", (fr[0]["layer"], fr[0]["random"]["median_km"]),
                    xytext=(-10, -22), textcoords="offset points", color=INK2, fontsize=9, ha="right")
        ch = full["chosen"]
        ax.scatter([ch["layer"]], [ch["blocked_country"]["median_km"]], s=110, facecolor="none", edgecolor=INK,
                   linewidth=1.5, zorder=4)
        ax.annotate(f"chosen: layer {ch['layer']}", (ch["layer"], ch["blocked_country"]["median_km"]),
                    xytext=(0, 14), textcoords="offset points", color=INK, fontsize=9, ha="center")
    ax.set_xlabel("layer of Qwen2.5-7B-Instruct")
    ax.set_ylabel("median held-out error (km)")
    ax.set_title(f"Probe error by layer (lines: {d['n_places']:,}-place sweep, prompt '{d['chosen']['mode']}')", loc="left", fontsize=12)
    ax.set_xlim(0, max(L) + 6)
    ax.set_ylim(0, base * 1.1)
    fig.tight_layout()
    fig.savefig(FIG / "probe_by_layer.png", dpi=160)
    plt.close(fig)


def error_map():
    p = RESULTS / "probe_oof.npz"
    if not p.exists():
        return
    z = np.load(p)
    km = great_circle_km(z["blocked_country"], z["true"])
    lat, lon = xyz_to_latlon(z["true"])
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = [0, 250, 500, 1000, 2000, 4000, 8000, 20100]
    idx = np.clip(np.digitize(km, bins) - 1, 0, len(SEQ) - 1)
    order = np.argsort(km)
    ax.scatter(lon[order], lat[order], s=1.2, c=[SEQ[i] for i in idx[order]], linewidths=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Probe error at each place's true position, whole countries held out", loc="left", fontsize=12)
    handles = [plt.Line2D([], [], marker="o", ls="", color=SEQ[i], markersize=6) for i in range(len(SEQ))]
    labels = [f"{bins[i]:,}-{bins[i + 1]:,} km" if i < len(SEQ) - 1 else f"> {bins[i]:,} km" for i in range(len(SEQ))]
    ax.legend(handles, labels, loc="lower left", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "error_map.png", dpi=160)
    plt.close(fig)


def band_sweep():
    g = load("gate.json")
    if not g:
        return
    qs = ["country", "currency", "language", "hemisphere", "season"]
    rows = sorted(g["sweep"], key=lambda r: (r["band"][0], r["band"][1]))
    M = np.array([[np.mean([p["answers"] is not None and _flipped(p, q, g) for p in r["pairs"]]) for q in qs] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.6))
    from matplotlib.colors import ListedColormap
    ax.imshow(M, cmap=ListedColormap(SEQ), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(qs)), qs)
    ax.set_yticks(range(len(rows)), [f"layers {r['band'][0]}-{r['band'][1]}" for r in rows])
    for i in range(len(rows)):
        for j in range(len(qs)):
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color=SURFACE if M[i, j] > 0.55 else INK)
    ax.grid(False)
    ax.set_title("Share of the 10 gate pairs whose answer flips, by patched band", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "band_sweep.png", dpi=160)
    plt.close(fig)


def _flipped(pair, q, g):
    # Recompute per-question flips from stored answers with the same scorer.
    from terra.data import city_info
    from terra.ripple import make_key, score_edit
    s, t = pair["source"], pair["target"]
    si, ti = city_info(s), city_info(t)
    sc = score_edit(pair["answers"], make_key(s, si["iso"], si["latitude"]), make_key(t, ti["iso"], ti["latitude"]))
    return sc["per_question"][q]["flipped"]


def edit_vs_control():
    fa, fl = load("edit_eval_fresh_alltoken.json"), load("edit_eval_fresh_lasttoken.json")
    if fa and fl:
        return edit_vs_control_fresh(fa, fl)
    e = load("edit_eval.json")
    if not e:
        return
    sets = [("cross", "opposite hemispheres"), ("any", "any two countries")]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sets))
    w = 0.36
    for k, (kind, col, lab) in enumerate((("edit", BLUE, "activation patch"), ("control", GREY, "random direction, same size"))):
        vals = [e["sets"][s][kind]["pass_rate"] for s, _ in sets]
        bars = ax.bar(x + (k - 0.5) * w, vals, w * 0.94, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x, [f"{lab}\n({e['sets'][s]['edit']['n_pairs']} random pairs)" for s, lab in sets])
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("pairs passing (>= 80% of answers flip)")
    ax.set_title(f"Edit success against the control (layers {e['band'][0]}-{e['band'][1]})", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "edit_vs_control.png", dpi=160)
    plt.close(fig)


def edit_vs_control_fresh(fa, fl):
    """Same fresh pairs: last-token swap, all-token swap, and the all-token control."""
    sets = [("cross", "opposite hemispheres"), ("any", "any two countries")]
    series = [(fl, "edit", BLUE, "swap, last name token"), (fa, "edit", ORANGE, "swap, every name token"),
              (fa, "control", GREY, "random direction, same size")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(sets))
    w = 0.26
    for k, (d, kind, col, lab) in enumerate(series):
        vals = [d["sets"][s][kind]["pass_rate"] for s, _ in sets]
        bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{lab}\n({fa['sets'][s]['edit']['n_pairs']} fresh random pairs)" for s, lab in sets])
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("pairs passing (>= 80% of answers flip)")
    ax.set_title(f"Edit success against the control (layers {fa['band'][0]}-{fa['band'][1]}, seed {fa['seed']})",
                 loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "edit_vs_control.png", dpi=160)
    plt.close(fig)


def believed_map():
    d = load("places_believed.json")
    if not d:
        return
    pts = [p for p in d["points"] if p["big"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in pts:
        ax.plot([p["lon"], p["blon"]], [p["lat"], p["blat"]], color=GRID, lw=0.6, zorder=1)
    ax.scatter([p["lon"] for p in pts], [p["lat"] for p in pts], s=6, color=GREY, label="true position", zorder=2)
    ax.scatter([p["blon"] for p in pts], [p["blat"] for p in pts], s=6, color=BLUE, label="believed position", zorder=3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_title(f"{len(pts)} large cities: true and believed positions (probe never saw the city)", loc="left", fontsize=12)
    ax.legend(loc="lower left", frameon=True, facecolor=SURFACE, edgecolor=GRID, markerscale=2)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "believed_map.png", dpi=160)
    plt.close(fig)


def swap_vs_move():
    d = load("steer_heldout.json")
    if not d:
        return
    kinds = [("swap", "swap (tier 1)"), ("steer", "move (tier 2)"), ("control", "random control")]
    metrics = [("moved", "location answer moved"), ("identity_kept", "identity kept"),
               ("moved_and_kept", "moved and kept")]
    cols = [BLUE, ORANGE, GREY]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    w = 0.26
    for k, ((kind, lab), col) in enumerate(zip(kinds, cols)):
        vals = [d["rates"][kind][m] for m, _ in metrics]
        bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(f"Swap vs move on {d['n_pairs']} fresh cross-hemisphere pairs", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "swap_vs_move.png", dpi=160)
    plt.close(fig)


def rewrite_world():
    d = load("rewrite_eval.json")
    if not d:
        return
    sw, st = d["summary"]["swap"], d["summary"]["steer"]
    groups = ["moved cities whose\nanswers all moved", "answers about two moved\ncities that fit the new world",
              "of those, answers the\nedits should change"]
    series = [
        ("swap", BLUE, [sw["city_pass_rate"], sw["pair_questions"]["all"]["after_consistent_rate"],
                        sw["pair_questions"]["all"]["countable_flip_rate"]]),
        ("move (tier 2)", ORANGE, [None, st["pair_questions"]["all"]["after_consistent_rate"],
                                   st["pair_questions"]["all"]["countable_flip_rate"]]),
        ("random control (swap size)", GREY, [sw["city_control_pass_rate"], sw["pair_questions"]["all"]["control_matches_after_rate"],
                                             sw["pair_questions"]["all"]["countable_control_flip_rate"]]),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(groups))
    w = 0.26
    for k, (lab, col, vals) in enumerate(series):
        raw = vals
        vals = [0 if v is None else v for v in vals]
        bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, label=lab)
        for b, v, r in zip(bars, vals, raw):
            # Move mode keeps country, currency and language by design, so the
            # "all answers moved" test does not apply to it.
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, "n/a" if r is None else f"{v:.0%}",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, groups, fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(f"Rewrite the world: three moves applied together ({sw['n_worlds']} random worlds)", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "rewrite_world.png", dpi=160)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (probe_by_layer, error_map, band_sweep, edit_vs_control, believed_map, swap_vs_move, rewrite_world):
        f()
        print(f"[figures] {f.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
