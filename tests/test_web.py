from pathlib import Path
import pytest
from workbuddy.web import create_app


@pytest.fixture
def client():
    root = Path(__file__).resolve().parents[1]
    app = create_app(root)
    app.testing = True
    return app.test_client()


def test_shipped_demo_loads_without_raw_dataset(client):
    assert client.get("/").status_code == 200
    assert client.get("/health").json["model_ready"]
    assert (
        client.get("/api/overview").json["metrics"]["cleaning"]["retained_rows"] > 100
    )


def test_recommendation_only_contains_observed_supported_codes(client):
    item = client.get("/api/catalog").json["items"][0]
    response = client.post(
        "/api/recommend",
        json={
            "item_id": item["item_id"],
            "height_cm": 168,
            "weight_kg": 62,
            "body_type": "unknown",
        },
    )
    assert response.status_code == 200
    result = response.json
    assert result["recommended_size_code"] in {
        c["size_code"] for c in result["candidates"]
    }
    for candidate in result["candidates"]:
        assert item["sizes"][str(candidate["size_code"])] >= 3
        assert sum(candidate["probabilities"].values()) == pytest.approx(1, abs=0.0002)


@pytest.mark.parametrize("value", [None, True, 0, -1, "nan", "Infinity", 300])
def test_invalid_profile_is_rejected(client, value):
    response = client.post(
        "/api/recommend", json={"item_id": "1", "height_cm": value, "weight_kg": 62}
    )
    assert response.status_code == 400


def test_unknown_item_and_nonobject_requests_are_rejected(client):
    assert client.post("/api/recommend", json=[]).status_code == 400
    assert (
        client.post(
            "/api/recommend",
            json={"item_id": "missing", "height_cm": 168, "weight_kg": 62},
        ).status_code
        == 400
    )


def test_extract_does_not_return_html_or_invented_quotes(client):
    text = "<script>alert(1)</script> The material is soft."
    response = client.post("/api/extract", json={"review": text, "engine": "rules"})
    assert response.status_code == 200
    assert response.content_type.startswith("application/json")
    assert all(f["evidence"] in text for f in response.json["findings"])


def test_missing_llm_key_is_explicit_not_silent_fallback(client, monkeypatch):
    monkeypatch.delenv("WORKBUDDY_LLM_API_KEY", raising=False)
    assert (
        client.post(
            "/api/extract", json={"review": "soft fabric", "engine": "llm"}
        ).status_code
        == 400
    )


def test_empty_and_oversize_reviews_are_rejected(client):
    assert client.post("/api/extract", json={"review": ""}).status_code == 400
    assert client.post("/api/extract", json={"review": "x" * 8001}).status_code == 400
    assert (
        client.post(
            "/api/extract", data="x" * 40000, content_type="application/json"
        ).status_code
        == 413
    )
