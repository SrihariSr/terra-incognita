"""Drive the real web page in a headless browser and screenshot each step.

Needs the server running (uv run uvicorn terra.server:app --port 8000).
Uses Playwright's Chromium (dev dependency). Checks that the page loads
without JavaScript errors, the globe draws, a name can be located, an edit
fills the ripple table, a free question is answered, a rewritten world with
several moves runs, and the rigour tab renders. Screenshots go to results/figures/screenshots/.

    uv run python scripts/ui_check.py [--url http://127.0.0.1:8000]
Writes results/ui_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "screenshots"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--name", default="Krasnovik")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []

    def check(name, ok, detail=None):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""), flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                          "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        t0 = time.time()
        page.goto(args.url, wait_until="networkidle", timeout=120_000)
        try:
            page.wait_for_function("document.getElementById('hud-status').textContent.includes('places at')",
                                   timeout=60_000)
            check("globe loads believed places", True)
        except Exception as exc:
            check("globe loads believed places", False, str(exc)[:200])
        page.wait_for_timeout(4000)  # let textures and points render
        page.screenshot(path=str(OUT / "1_map.png"))
        check("map stats rendered", page.locator("#map-stats .stat").count() >= 2,
              page.locator("#map-stats").inner_text()[:200])

        # New worlds
        page.click('#tabs button[data-tab="worlds"]')
        page.fill("#locate-input", args.name)
        page.click("#locate-form button")
        try:
            page.wait_for_function("document.getElementById('locate-result').textContent.includes('nearest real city')",
                                   timeout=60_000)
            check(f"locate {args.name}", True, page.locator("#locate-result").inner_text())
        except Exception as exc:
            check(f"locate {args.name}", False, str(exc)[:200])
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "2_new_worlds.png"))
        check("suggestion chips shown", page.locator("#locate-chips .chip").count() > 0)

        # Edit
        page.click('#tabs button[data-tab="edit"]')
        page.select_option("#edit-source", "Paris")
        page.select_option("#edit-target", "Sydney")
        page.click("#edit-go")
        try:
            page.wait_for_function("document.getElementById('edit-status').textContent.startsWith('Done')", timeout=120_000)
            txt = page.locator("#ripple tbody").inner_text()
            check("edit fills ripple table", "Australia" in txt, txt)
        except Exception as exc:
            check("edit fills ripple table", False, str(exc)[:200])
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "3_edit.png"))
        page.click("#ask-form button")
        try:
            page.wait_for_function("document.querySelectorAll('#ask-result .qa').length >= 2", timeout=120_000)
            check("free question answered", True, page.locator("#ask-result").inner_text()[:300])
        except Exception as exc:
            check("free question answered", False, str(exc)[:200])
        page.screenshot(path=str(OUT / "4_free_question.png"))

        # Drag and drop: Paris marker onto Cape Town (both visible from this view).
        page.click("#edit-reset")
        # Arrow function: pointOfView returns the (callable) globe object, which
        # Playwright would otherwise try to call.
        page.evaluate("() => { globe.pointOfView({lat: 10, lng: 10, altitude: 2.2}, 0); }")
        page.wait_for_timeout(1500)

        def marker_centre(name):
            box = page.locator(".city-marker", has=page.locator(".lbl", has_text=name)).first.locator(".dot").bounding_box()
            return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2) if box else None

        a, b = marker_centre("Paris"), marker_centre("Cape Town")
        if a and b:
            page.mouse.move(*a)
            page.mouse.down()
            for k in range(1, 21):
                page.mouse.move(a[0] + (b[0] - a[0]) * k / 20, a[1] + (b[1] - a[1]) * k / 20)
                page.wait_for_timeout(30)
            page.mouse.up()
            try:
                page.wait_for_function("document.getElementById('edit-status').textContent.startsWith('Done')",
                                       timeout=120_000)
                tgt = page.locator("#edit-target").input_value()
                check("drag Paris onto Cape Town runs an edit", tgt == "Cape Town", tgt)
            except Exception as exc:
                check("drag Paris onto Cape Town runs an edit", False, str(exc)[:200])
            page.wait_for_timeout(2500)
            page.screenshot(path=str(OUT / "3b_drag_edit.png"))
        else:
            check("drag markers visible", False, {"Paris": a, "Cape Town": b})

        # Tier 2 mode, if available
        if page.locator("#mode-row:not(.hidden)").count():
            page.check('input[name="mode"][value="steer"]')
            page.select_option("#edit-source", "Paris")
            page.select_option("#edit-target", "Sydney")
            page.click("#edit-go")
            try:
                page.wait_for_function("document.getElementById('edit-status').textContent.startsWith('Done')",
                                       timeout=120_000)
                after_country = page.evaluate("document.querySelector('#ripple tbody tr td:nth-child(3)').textContent")
                after_season = page.evaluate("document.querySelector('#ripple tbody tr:nth-child(5) td:nth-child(3)').textContent")
                check("steer mode: Paris still in France after the move", "France" in after_country,
                      {"country": after_country, "season": after_season})
            except Exception as exc:
                check("steer mode edit", False, str(exc)[:200])
            page.wait_for_timeout(2000)
            page.screenshot(path=str(OUT / "3c_move_keep_identity.png"))
            page.check('input[name="mode"][value="swap"]')

        # Rewrite the world: run the first suggested world, then the free question.
        page.click('#tabs button[data-tab="rewrite"]')
        if page.locator("#world-chips .chip").count() > 0:
            page.locator("#world-chips .chip").first.click()
            try:
                page.wait_for_function("document.getElementById('rw-status').textContent.startsWith('Done')",
                                       timeout=180_000)
                n_heads = page.locator("#rw-cities tr.city-head").count()
                check("rewrite world fills a table per moved city", n_heads >= 2, n_heads)
                check("rewrite world shows pair questions", page.locator("#rw-pairs tr").count() >= 3)
            except Exception as exc:
                check("rewrite world runs", False, str(exc)[:200])
            page.wait_for_timeout(2500)
            page.screenshot(path=str(OUT / "5_rewrite_world.png"))
            page.click("#rw-ask-form button")
            try:
                page.wait_for_function("document.querySelectorAll('#rw-ask-result .qa').length >= 2", timeout=180_000)
                check("rewrite world free question answered", True, page.locator("#rw-ask-result").inner_text()[:300])
            except Exception as exc:
                check("rewrite world free question answered", False, str(exc)[:200])
            page.screenshot(path=str(OUT / "5b_rewrite_free_question.png"))
        else:
            check("suggested worlds shown", False, "no chips")

        # Rigour
        page.click('#tabs button[data-tab="rigour"]')
        page.wait_for_timeout(1500)
        check("rigour stats rendered", page.locator("#rigour .stat").count() >= 4, page.locator("#rigour").inner_text()[:200])
        page.screenshot(path=str(OUT / "6_rigour.png"))
        browser.close()
    check("no JavaScript errors", not errors, errors[:5])
    n_ok = sum(c["ok"] for c in checks)
    out = {"time": time.strftime("%Y-%m-%d %H:%M"), "n_checks": len(checks), "n_pass": n_ok,
           "seconds": time.time() - t0, "checks": checks, "console_errors": errors}
    (ROOT / "results" / "ui_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[ui_check] {n_ok}/{len(checks)} passed; screenshots in {OUT}")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
