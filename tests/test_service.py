from fastapi.testclient import TestClient
from test_inference import FakeNER

from ru_ner.service import MAX_BATCH_TEXTS, MAX_TEXT_CHARS, create_app


def client():
    return TestClient(create_app(ner=FakeNER()))


def test_demo_page_is_served():
    with client() as c:
        response = c.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'fetch("extract"' in response.text


def test_health():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_extract_returns_entities_with_offsets():
    text = "Иван Петров приехал в Москве"
    with client() as c:
        response = c.post("/extract", json={"text": text})

    assert response.status_code == 200
    entities = response.json()["entities"]
    assert [(e["text"], e["type"]) for e in entities] == [("Иван Петров", "PER"), ("Москве", "LOC")]
    assert all(text[e["start"] : e["end"]] == e["text"] for e in entities)


def test_extract_empty_text():
    with client() as c:
        assert c.post("/extract", json={"text": ""}).json() == {"entities": []}


def test_extract_batch_keeps_order():
    with client() as c:
        response = c.post("/extract/batch", json={"texts": ["Госдума", "ничего", "Иван"]})

    results = response.json()["results"]
    assert [[e["type"] for e in r] for r in results] == [["ORG"], [], ["PER"]]


def test_validation_errors():
    with client() as c:
        assert c.post("/extract", json={}).status_code == 422
        assert c.post("/extract", json={"text": "а" * (MAX_TEXT_CHARS + 1)}).status_code == 422
        assert c.post("/extract/batch", json={"texts": []}).status_code == 422
        too_many = {"texts": ["а"] * (MAX_BATCH_TEXTS + 1)}
        assert c.post("/extract/batch", json=too_many).status_code == 422
        too_long = {"texts": ["ok", "а" * (MAX_TEXT_CHARS + 1)]}
        assert c.post("/extract/batch", json=too_long).status_code == 422
