"""End-to-end check of the demo path.

Starts the server in a subprocess (unless --url points at one already
running), then calls every endpoint the demo uses and checks the answers:

* static page and every vendored file load, and the page references no
  external URL (so it works with Wi-Fi off);
* /api/places returns the believed map;
* /api/locate works for a real city and for invented names;
* /api/edit for the headline pair flips >= 4 of 5 answers and the control
  flips <= 1; a pair not in the cache is computed live;
* /api/ask answers the free question before and after the edit;
* /api/rewrite applies three moves together, and /api/ask answers a free
  question with all three applied;
* /api/rigour returns numbers.

    uv run python scripts/demo_check.py [--url http://127.0.0.1:8000]
Writes results/demo_check.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from terra import RESULTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    proc = None
    base = args.url
    if base is None:
        base = f"http://127.0.0.1:{args.port}"
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "terra.server:app", "--port", str(args.port)],
                                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    checks = []

    def check(name, ok, detail=None, seconds=None):
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "seconds": seconds})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" ({seconds:.1f}s)" if seconds is not None else "")
              + (f": {detail}" if detail and not ok else ""), flush=True)

    def timed(method, path, **kw):
        t0 = time.time()
        r = requests.request(method, base + path, timeout=300, **kw)
        return r, time.time() - t0

    try:
        t0 = time.time()
        while True:
            try:
                if requests.get(base + "/api/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            if time.time() - t0 > 240:
                raise RuntimeError("server did not start within 240 s")
            time.sleep(2)
        check("server starts", True, seconds=time.time() - t0)
        h = requests.get(base + "/api/health").json()
        check("health", h["ok"], h)

        # Static files and offline safety.
        html = requests.get(base + "/").text
        js = requests.get(base + "/app.js").text
        css = requests.get(base + "/style.css").text
        ext = re.findall(r"https?://[^\s\"')]+", html + js + css)
        ext = [u for u in ext if "localhost" not in u and "127.0.0.1" not in u]
        check("page has no external URLs", not ext, ext)
        for f in ["vendor/globe.gl.min.js", "vendor/earth-dark.jpg", "vendor/earth-topology.png",
                  "vendor/night-sky.png", "vendor/countries.geojson"]:
            r = requests.get(f"{base}/{f}")
            check(f"static {f}", r.ok and len(r.content) > 1000, r.status_code)

        r, s = timed("GET", "/api/places")
        pts = r.json()["points"]
        check("places", len(pts) > 5000, len(pts), s)

        menu = requests.get(base + "/api/menu").json()
        check("menu has validated pairs", len(menu["pairs"]) > 10, len(menu["pairs"]))
        check("menu has invented names", len(menu["fictional"]) > 3, menu["fictional"])

        for name in ["Paris", "Brindlemoor", "Takeshiro", "Hogwarts"]:
            r, s = timed("GET", f"/api/locate?name={name}")
            j = r.json()
            ok = r.ok and -90 <= j["lat"] <= 90 and len(j["cloud"]) == 20
            check(f"locate {name}", ok, {"lat": j.get("lat"), "lon": j.get("lon"), "nearest": j.get("nearest")}, s)
            if name == "Paris":
                check("Paris located in Europe", j["nearest"]["continent"] == "EU", j["nearest"])

        r, s = timed("POST", "/api/edit", json={"source": "Paris", "target": "Sydney"})
        j = r.json()
        check("edit Paris -> Sydney flips >= 4/5", r.ok and j["score_after"]["n_flipped"] >= 4, j.get("after"), s)
        check("control Paris -> Sydney flips <= 1", r.ok and j["score_control"]["n_flipped"] <= 1, j.get("control"))
        r, s = timed("POST", "/api/edit", json={"source": "Tokyo", "target": "Lima"})
        j = r.json()
        check("edit Tokyo -> Lima (live or cached) returns answers", r.ok and "after" in j, j.get("after"), s)
        r, s = timed("POST", "/api/edit", json={"source": "Madrid", "target": "Perth"})
        check("edit of a pair outside the menu runs live", r.ok and "after" in r.json(), r.json().get("after"), s)

        q = "I'm in Paris for Christmas, what should I pack? Answer in two sentences."
        r, s = timed("POST", "/api/ask", json={"question": q, "source": "Paris", "target": "Sydney"})
        j = r.json()
        check("free question before/after", r.ok and j["before"] and j["after"] and j["before"] != j["after"],
              {"before": j.get("before"), "after": j.get("after")}, s)
        r, s = timed("POST", "/api/ask", json={"question": "What is the capital of Japan?", "source": "Paris",
                                                "target": "Sydney"})
        check("question without the source city is handled", r.ok and "note" in r.json(), r.json(), s)

        # Rewrite the world: three moves applied together.
        moves = [{"source": "Paris", "target": "Sydney"}, {"source": "Tokyo", "target": "London"},
                 {"source": "Rio de Janeiro", "target": "Moscow"}]
        r, s = timed("POST", "/api/rewrite", json={"moves": moves})
        j = r.json()
        ok = r.ok and len(j.get("cities", [])) == 3 and sum(bool(c.get("passes")) for c in j["cities"]) >= 2
        check("rewrite world: 3 moves, at least 2 cities fully moved", ok,
              [c.get("after") for c in j.get("cities", [])], s)
        check("rewrite world: pair questions returned", r.ok and len(j.get("pairs", [])) == 6, len(j.get("pairs", [])))
        q = "I'm visiting Paris, Tokyo and Rio de Janeiro this July. What should I pack for each city? Answer in three short sentences."
        r, s = timed("POST", "/api/ask", json={"question": q, "moves": moves, "max_new_tokens": 120})
        j = r.json()
        check("rewrite world: free question before/after", r.ok and j.get("before") and j.get("after")
              and j["before"] != j["after"], {"after": j.get("after")}, s)
        r, s = timed("POST", "/api/rewrite", json={"moves": moves + [{"source": "Berlin", "target": "Lima"}]})
        check("rewrite world: more than 3 moves rejected cleanly", r.status_code == 400, r.status_code, s)
        r, s = timed("POST", "/api/rewrite", json={"moves": [{"source": "Paris", "target": "Sydney"},
                                                            {"source": "Paris", "target": "Lima"}]})
        check("rewrite world: moving one city twice rejected cleanly", r.status_code == 400, r.status_code, s)
        check("menu has suggested worlds", len(menu.get("worlds", [])) >= 1, menu.get("worlds"))

        if menu.get("steer_available"):
            r, s = timed("POST", "/api/edit", json={"source": "Paris", "target": "Sydney", "mode": "steer"})
            check("steer mode edit", r.ok and "after" in r.json(), r.json().get("after"), s)

        r, s = timed("GET", "/api/rigour")
        j = r.json()
        check("rigour numbers", r.ok and "probe" in j and "edit" in j, list(j), s)

        r, s = timed("GET", "/api/locate?name=" + "x" * 200)
        check("over-long name rejected cleanly", r.status_code == 400, r.status_code, s)
        r, s = timed("POST", "/api/edit", json={"source": "Paris", "target": "Paris"})
        check("same source and target rejected cleanly", r.status_code == 400, r.status_code, s)
    except Exception as exc:  # record the failure instead of crashing without a file
        check("demo check ran to completion", False, f"{type(exc).__name__}: {exc}")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    n_ok = sum(c["ok"] for c in checks)
    out = {"time": time.strftime("%Y-%m-%d %H:%M"), "n_checks": len(checks), "n_pass": n_ok,
           "all_pass": n_ok == len(checks), "checks": checks}
    (RESULTS / "demo_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[demo_check] {n_ok}/{len(checks)} passed -> results/demo_check.json")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
