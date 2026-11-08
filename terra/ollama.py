"""Ask a local Ollama server to unload its models (best effort).

The user also has Gemma in Ollama. Ollama keeps a model in memory for a while
after it was last used, which would compete with ours for the 36 GB. Before
loading a model we ask Ollama which models it holds (GET /api/ps) and unload
each one (POST /api/generate with keep_alive 0, Ollama's documented way to
unload). If Ollama is not running, nothing happens.
"""

from __future__ import annotations

import requests

OLLAMA = "http://localhost:11434"


def unload_ollama_models(host: str = OLLAMA, timeout: float = 2.0) -> list[str]:
    """Return the names of the models Ollama was asked to unload. Never raises."""
    try:
        r = requests.get(f"{host}/api/ps", timeout=timeout)
        r.raise_for_status()
        names = [m.get("name") or m.get("model") for m in r.json().get("models", [])]
    except Exception:
        return []
    done = []
    for name in names:
        if not name:
            continue
        try:
            requests.post(f"{host}/api/generate", json={"model": name, "keep_alive": 0}, timeout=timeout)
            done.append(name)
        except Exception:
            continue
    return done
