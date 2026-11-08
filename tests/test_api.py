"""Every API route, through FastAPI's TestClient, with the fake model."""

from __future__ import annotations

from terra.backends.fake import FakeLM


def test_health_and_models(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["key"] == "qwen2.5-7b" and h["editable"] is True
    assert h["probe_layer"] == 5
    m = client.get("/api/models").json()
    by = {x["key"]: x for x in m["models"]}
    assert set(by) == {"qwen2.5-7b", "gemma4-12b", "gemma4-31b"}
    assert by["qwen2.5-7b"]["loaded"] and by["qwen2.5-7b"]["available"]
    assert by["gemma4-12b"]["available"] and not by["gemma4-12b"]["loaded"]
    assert not by["gemma4-31b"]["available"]
    assert "build_model.py --model gemma4-31b --stage all" in by["gemma4-31b"]["build_command"]


def test_places_menu_locate(client):
    pts = client.get("/api/places").json()["points"]
    assert any(p["name"] == "Paris" for p in pts)
    menu = client.get("/api/menu").json()
    assert menu["pairs"] == [{"source": "Paris", "target": "Sydney"}]
    assert menu["editable"] is True and menu["fictional"]
    r = client.get("/api/locate", params={"name": "Brindlemoor"})
    assert r.status_code == 200
    j = r.json()
    assert -90 <= j["lat"] <= 90 and len(j["cloud"]) == 4
    assert client.get("/api/locate", params={"name": "x" * 200}).status_code == 400


def test_edit_cached_pair_is_scored(client):
    r = client.post("/api/edit", json={"source": "Paris", "target": "Sydney"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["cached"] is True and j["scored"] is True
    assert "score_after" in j and "landmark" in j and "edited_location" in j


def test_edit_invented_names_runs_live_and_unscored(client):
    r = client.post("/api/edit", json={"source": "Brindlemoor", "target": "Krasnovik"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["cached"] is False and j["scored"] is False
    assert j["unscored_reason"].startswith("no answer key for invented places")
    assert "score_after" not in j
    assert j["after"] != j["before"]
    assert j["target_info"]["known"] is False


def test_edit_guards(client, monkeypatch):
    assert client.post("/api/edit", json={"source": "Paris", "target": "Paris"}).status_code == 400
    assert client.post("/api/edit", json={"source": " ", "target": "Paris"}).status_code == 400
    # A name that cannot be aligned in a prompt gives a clear 400.
    import terra.server as srv
    from terra.patch import AlignmentError

    def boom(*a, **k):
        raise AlignmentError("could not find or align 'Paris' and 'Lima' in 'x'")

    monkeypatch.setattr(srv.S.current, "planner", boom)
    r = client.post("/api/edit", json={"source": "Paris", "target": "Lima"})
    assert r.status_code == 400 and "could not find or align" in r.json()["detail"]
    # Steer mode is not available for the fake model (no steer_eval.json).
    assert client.post("/api/edit", json={"source": "Paris", "target": "Lima", "mode": "steer"}).status_code == 409


def test_ask_single_multi_and_not_mentioned(client):
    j = client.post("/api/ask", json={"question": "I'm in Paris for Christmas, what should I pack?",
                                      "source": "Paris", "target": "Brindlemoor"}).json()
    assert j["before"] != j["after"]
    j = client.post("/api/ask", json={"question": "What is the capital of Japan?",
                                      "source": "Paris", "target": "Sydney"}).json()
    assert "note" in j
    moves = [{"source": "Paris", "target": "Sydney"}, {"source": "Brindlemoor", "target": "Lima"}]
    j = client.post("/api/ask", json={"question": "I'm visiting Paris and then Brindlemoor this July.",
                                      "moves": moves}).json()
    assert j["before"] != j["after"]
    assert client.post("/api/ask", json={"question": ""}).status_code == 400


def test_rewrite_routes(client):
    moves = [{"source": "Paris", "target": "Sydney"}, {"source": "Brindlemoor", "target": "Krasnovik"}]
    r = client.post("/api/rewrite", json={"moves": moves})
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["cities"]) == 2 and len(j["pairs"]) == 2
    assert j["cities"][1]["scored"] is False and j["scored"] is False
    bad = [{"source": "Paris", "target": "Sydney"}] * 2
    assert client.post("/api/rewrite", json={"moves": bad}).status_code == 400
    four = [{"source": f"A{i}", "target": f"B{i}"} for i in range(4)]
    assert client.post("/api/rewrite", json={"moves": four}).status_code == 400
    same = [{"source": "Paris", "target": "Paris"}]
    assert client.post("/api/rewrite", json={"moves": same}).status_code == 400


def test_political_rigour_figures(client):
    p = client.get("/api/political")
    assert p.status_code == 200
    assert p.json()["map"]["features"][0]["properties"]["ISO_A2"] == "FR"
    R = client.get("/api/rigour").json()
    assert R["probe"]["random_median_km"] == 800.0 and R["model"] == "qwen2.5-7b"
    assert client.get("/figures/none.png").status_code == 404
    assert client.get("/figures/bad name.png").status_code == 400
    assert client.get("/api/free_question", params={"city": "Lima"}).json()["question"].startswith("I'm in Lima")


def test_switch_unloads_before_loading(client):
    assert FakeLM.events == [("load", "qwen2.5-7b")]
    r = client.post("/api/model", json={"key": "gemma4-12b"})
    assert r.status_code == 200, r.text
    assert FakeLM.events == [("load", "qwen2.5-7b"), ("unload", "qwen2.5-7b"), ("load", "gemma4-12b")]
    assert FakeLM.max_live == 1 and FakeLM.live == 1
    assert len(client.ollama_calls) == 1
    h = client.get("/api/health").json()
    assert h["key"] == "gemma4-12b"
    # Caches and menus are the new model's: no political map was built for it.
    assert client.get("/api/political").status_code == 404
    assert client.get("/api/menu").json()["model"] == "gemma4-12b"
    # Switching to the loaded model is a no-op.
    assert client.post("/api/model", json={"key": "gemma4-12b"}).json().get("unchanged") is True
    # And back.
    assert client.post("/api/model", json={"key": "qwen2.5-7b"}).status_code == 200
    assert FakeLM.loads == 3 and FakeLM.unloads == 2 and FakeLM.max_live == 1


def test_switch_guards(client):
    assert client.post("/api/model", json={"key": "nope"}).status_code == 404
    r = client.post("/api/model", json={"key": "gemma4-31b"})
    assert r.status_code == 409 and "not built yet" in r.json()["detail"]
    assert FakeLM.loads == 1                       # nothing was unloaded or loaded


def test_routes_answer_503_while_switching(client):
    import terra.server as srv
    srv.S.switching = True
    srv.S.switch_msg = "Unloading A, loading B"
    try:
        r = client.get("/api/menu")
        assert r.status_code == 503 and "switching models" in r.json()["detail"]
        assert client.post("/api/edit", json={"source": "Paris", "target": "Sydney"}).status_code == 503
        assert client.get("/api/health").json()["switching"] is True
    finally:
        srv.S.switching = False


def test_available_but_not_editable_model(client):
    from terra.backends.fake import make_fake_artefacts
    make_fake_artefacts(keys_built=["gemma4-31b"], keys_editable=[])
    r = client.post("/api/model", json={"key": "gemma4-31b"})
    assert r.status_code == 200
    menu = client.get("/api/menu").json()
    assert menu["editable"] is False and "stage gate" in menu["not_editable_reason"]
    assert client.post("/api/edit", json={"source": "Paris", "target": "Sydney"}).status_code == 409
    assert client.post("/api/rewrite", json={"moves": [{"source": "Paris", "target": "Lima"}]}).status_code == 409
    # Reading still works.
    assert client.get("/api/locate", params={"name": "Paris"}).status_code == 200
