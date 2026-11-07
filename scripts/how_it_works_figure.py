"""One-page "how it works" figure: reading the map (probe) and editing it (patching).

Plain language, one slide. Every number and example answer on it is read
from a results file:
    results/places_believed.json   Paris's believed position (out-of-fold)
    results/probe_final.json       held-out errors, layer
    results/gate.json              patched layer band
    results/pair_menu_alltoken.json (or pair_menu.json)  Paris -> Sydney answers
    results/edit_eval_fresh_alltoken.json (or edit_eval.json)  success rates
    results/fictional.json         invented-name examples

    uv run python scripts/how_it_works_figure.py
Writes results/figures/how_it_works.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terra import RESULTS  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_L = "#2a78d6", "#e3eefb"
ORANGE, ORANGE_L = "#eb6834", "#fdebe3"
GREY, GREY_L = "#8a8984", "#efeeeb"
FONT = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def load(*names):
    for n in names:
        p = RESULTS / n
        if p.exists():
            return json.loads(p.read_text()), n
    raise FileNotFoundError(names)


def box(ax, x, y, w, h, text, fc, ec, size=12, weight="normal", color=INK, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color=color,
            family="monospace" if mono else FONT, linespacing=1.35, wrap=True)


def arrow(ax, x0, y0, x1, y1, color=INK2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16, lw=1.6,
                                 color=color, linestyle=ls, shrinkA=2, shrinkB=2))


def main() -> int:
    pb, _ = load("places_believed.json")
    fi, _ = load("fictional.json")
    pf, _ = load("probe_final.json")
    gate, _ = load("gate.json")
    menu, menu_name = load("pair_menu_alltoken.json", "pair_menu.json")
    ev, ev_name = load("edit_eval_fresh_alltoken.json", "edit_eval.json")
    paris = next(p for p in pb["points"] if p["name"] == "Paris")
    ps = next(p for p in menu["all_pairs"] if p["source"] == "Paris" and p["target"] == "Sydney")
    b0, b1 = gate["best_band"]
    layer = pf["layer"]
    med_r = pf["heldout"]["random"]["median_km"]
    med_c = pf["heldout"]["blocked_country"]["median_km"]
    e = ev["sets"]["cross"]

    def ns(v):
        return f"{abs(v):.1f}°{'N' if v >= 0 else 'S'}"

    def ew(v):
        return f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"

    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.95, "How TERRA INCOGNITA works", fontsize=24, weight="bold", color=INK, family=FONT, va="top")
    ax.text(0.04, 0.905, "Qwen2.5-7B-Instruct keeps a map of the Earth inside it. We read the map with a simple formula, "
            "and edit it by copying numbers from one word to another.", fontsize=13, color=INK2, family=FONT, va="top")

    # ---------------------------------------------------------------- 1. read
    ax.text(0.04, 0.84, "1  Reading the map", fontsize=17, weight="bold", color=BLUE, family=FONT, va="top")
    y, h = 0.6, 0.16
    xs = [0.04, 0.21, 0.42, 0.64, 0.83]
    ws = [0.13, 0.17, 0.18, 0.15, 0.13]
    texts = [
        '"What are the lat/lon\ncoordinates of Paris"',
        f"The model reads the\nprompt through its\n28 layers",
        f"At layer {layer}, the word\n\"Paris\" is a list of\n3,584 numbers",
        "A probe (a weighted\nsum of those numbers)\ngives a point on a globe",
        f"Believed position\n{ns(paris['blat'])}, {ew(paris['blon'])}\n(true: {ns(paris['lat'])}, {ew(paris['lon'])})",
    ]
    for i, (x, w, t) in enumerate(zip(xs, ws, texts)):
        box(ax, x, y, w, h, t, BLUE_L if i in (2, 3) else "white", BLUE, size=12.5,
            weight="bold" if i == 4 else "normal")
        if i:
            arrow(ax, xs[i - 1] + ws[i - 1], y + h / 2, x, y + h / 2, color=BLUE)
    ax.text(0.04, 0.54,
            f"The probe is fitted on {pf['n_places']:,} places with known coordinates. On places it never saw, its median miss is "
            f"{med_r:,.0f} km ({med_c:,.0f} km when whole countries are hidden from it).",
            fontsize=12.5, color=INK2, family=FONT, va="top")
    ex, groups = [], set()
    for r in fi["results"]:                      # first coherent name from two different regions
        if r["coherent"] and r["expected_group"] not in groups and len(ex) < 2:
            ex.append(r)
            groups.add(r["expected_group"])
    ax.text(0.04, 0.505, "Invented names work too: the same probe places " +
            " and ".join(f"\"{r['name']}\" near {r['nearest']['name']}, {r['nearest']['country']}" for r in ex) +
            f" ({fi['n_coherent']} of {fi['n_scored']} invented names land where they sound like they belong).",
            fontsize=12.5, color=INK2, family=FONT, va="top")

    # ---------------------------------------------------------------- 2. edit
    ax.text(0.04, 0.44, "2  Editing the map", fontsize=17, weight="bold", color=ORANGE, family=FONT, va="top")
    # donor lane
    yd, yr, h2 = 0.29, 0.1, 0.105
    box(ax, 0.04, yd, 0.2, h2, '"Which country is\nSydney in?"', "white", ORANGE, size=12.5)
    box(ax, 0.30, yd, 0.2, h2, f"Copy the numbers for\n\"Sydney\" at layers {b0} to {b1}", ORANGE_L, ORANGE, size=12.5)
    arrow(ax, 0.24, yd + h2 / 2, 0.30, yd + h2 / 2, color=ORANGE)
    # receiving lane
    box(ax, 0.04, yr, 0.2, h2, '"Which country is\nParis in?"\n(text unchanged)', "white", ORANGE, size=12.5)
    box(ax, 0.30, yr, 0.2, h2, f"Paste them over \"Paris\"\nat layers {b0} to {b1}", ORANGE_L, ORANGE, size=12.5)
    arrow(ax, 0.24, yr + h2 / 2, 0.30, yr + h2 / 2, color=ORANGE)
    arrow(ax, 0.40, yd, 0.40, yr + h2, color=ORANGE)
    box(ax, 0.56, yr, 0.14, h2, "The model\ncarries on\nas normal", "white", ORANGE, size=12.5)
    arrow(ax, 0.50, yr + h2 / 2, 0.56, yr + h2 / 2, color=ORANGE)
    after = ps["after"]
    box(ax, 0.76, yr + 0.035, 0.2, 0.09,
        f"{after['country']}, {after['currency']},\n{after['hemisphere']} Hemisphere, July: {after['season']}",
        ORANGE_L, ORANGE, size=12, weight="bold")
    arrow(ax, 0.70, yr + h2 / 2, 0.76, yr + 0.035 + 0.045, color=ORANGE)
    # control
    ctrl = ps["control"]
    box(ax, 0.56, yd, 0.14, h2, "Control: instead,\npush \"Paris\" the same\ndistance in a\nrandom direction", GREY_L, GREY, size=11.5)
    box(ax, 0.76, yd + 0.01, 0.2, 0.085, f"{ctrl['country']}, {ctrl['currency']},\n{ctrl['hemisphere']} Hemisphere, July: {ctrl['season']}",
        GREY_L, GREY, size=12)
    arrow(ax, 0.70, yd + h2 / 2, 0.76, yd + 0.01 + 0.0425, color=GREY)
    ax.text(0.76, yr + 0.035 + 0.09 + 0.008, "Answers about Paris after the edit", fontsize=11, color=ORANGE,
            family=FONT, va="bottom")
    ax.text(0.76, yd + 0.01 + 0.085 + 0.008, "Answers about Paris after the random push", fontsize=11, color=GREY,
            family=FONT, va="bottom")
    ax.text(0.04, 0.055,
            f"On {e['edit']['n_pairs']} fresh random pairs of cities in opposite hemispheres, {e['edit']['pass_rate']:.0%} of edits move at least 80% "
            f"of the answers; random pushes of the same size: {e['control']['pass_rate']:.0%}. Up to three cities can be moved at once.",
            fontsize=12.5, color=INK2, family=FONT, va="top")
    ax.text(0.96, 0.012, f"Numbers from results/probe_final.json, places_believed.json, fictional.json, {ev_name}, {menu_name}",
            fontsize=8.5, color=GREY, family=FONT, ha="right")
    out = RESULTS / "figures" / "how_it_works.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"[figure] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
