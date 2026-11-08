"""The Ollama unload helper, with the HTTP calls mocked."""

from __future__ import annotations

import requests

from terra import ollama


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_unloads_every_loaded_model(monkeypatch):
    posted = []
    monkeypatch.setattr(ollama.requests, "get",
                        lambda url, timeout: _Resp({"models": [{"name": "gemma4:31b"}, {"model": "gemma4:12b"}]}))
    monkeypatch.setattr(ollama.requests, "post", lambda url, json, timeout: posted.append((url, json)))
    assert ollama.unload_ollama_models() == ["gemma4:31b", "gemma4:12b"]
    assert all(u.endswith("/api/generate") and j["keep_alive"] == 0 for u, j in posted)
    assert [j["model"] for _, j in posted] == ["gemma4:31b", "gemma4:12b"]


def test_quiet_when_ollama_is_not_running(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(ollama.requests, "get", down)
    assert ollama.unload_ollama_models() == []


def test_one_failed_unload_does_not_stop_the_others(monkeypatch):
    monkeypatch.setattr(ollama.requests, "get",
                        lambda url, timeout: _Resp({"models": [{"name": "a"}, {"name": "b"}]}))

    def post(url, json, timeout):
        if json["model"] == "a":
            raise requests.Timeout("slow")

    monkeypatch.setattr(ollama.requests, "post", post)
    assert ollama.unload_ollama_models() == ["b"]
