from copy import deepcopy
from datetime import date, timedelta
import numpy as np
import pytest
from workbuddy.data import LABELS, height_cm, normalize, chronological_split
from workbuddy.model import features, training_features
from workbuddy.aspects import extract_rules, validate_findings, extract_llm


def row(day="2016-01-01", label="small", review="Runs small."):
    return {
        "item_id": "item-1",
        "user_id": "user-1",
        "date": day,
        "label": label,
        "review": review,
        "height_cm": 168.0,
        "weight_kg": 62.0,
        "size": 8.0,
        "category": "dress",
        "body_type": "unknown",
    }


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("5' 8\"", 172.72),
        ("5' 0\"", 152.4),
        ("5' 14\"", None),
        ("unknown", None),
        (None, None),
    ],
)
def test_height_units_and_invalid_inches(raw, expected):
    actual = height_cm(raw)
    assert actual == pytest.approx(expected) if expected is not None else actual is None


def test_normalization_keeps_missingness_without_invalid_numbers():
    result = normalize(
        {
            "fit": "fit",
            "user_id": "1",
            "item_id": "2",
            "review_date": "April 20, 2016",
            "height": "5' 8\"",
            "weight": "137lbs",
            "size": 14,
        }
    )
    assert result["height_cm"] == pytest.approx(172.72)
    assert result["weight_kg"] == pytest.approx(62.142)
    assert normalize({"fit": "invalid"}) is None


def test_split_never_divides_a_date_and_preserves_class_coverage():
    rows = [
        row(str(date(2016, 1, 1) + timedelta(days=i)), label)
        for i in range(120)
        for label in LABELS
    ]
    train, val, test = chronological_split(rows)
    assert max(r["date"] for r in train) < min(r["date"] for r in val)
    assert max(r["date"] for r in val) < min(r["date"] for r in test)
    assert len(train) + len(val) + len(test) == len(rows)


def test_training_history_excludes_self_same_day_and_future_reviews():
    rows = [row(), row(label="large"), row("2016-01-02", "fit")]
    _, encoded, history = training_features(rows)
    assert encoded[0]["history_log_count"] == encoded[1]["history_log_count"] == 0
    assert encoded[2]["history_log_count"] == pytest.approx(np.log(3))
    assert encoded[2]["history_small"] == pytest.approx(4 / 11)
    assert history["item-1"]["count"] == 3


def test_current_review_label_and_rating_never_change_prediction_features():
    _, _, history = training_features([row()])
    candidate = row("2016-02-01", "fit", "Perfect fit.")
    changed = {
        **candidate,
        "review": "This is too small and scratchy.",
        "label": "large",
        "rating": 1,
    }
    assert features(candidate, history) == features(changed, history)
    before = deepcopy(history)
    features(changed, history)
    assert history == before


def test_rule_findings_keep_exact_evidence():
    text = "This dress runs small. The fabric is soft. The hem is too long."
    findings = extract_rules(text)["findings"]
    assert {(f["aspect"], f["sentiment"]) for f in findings} == {
        ("sizing", "negative"),
        ("material", "positive"),
        ("length", "negative"),
    }
    assert all(f["evidence"] in text for f in findings)


@pytest.mark.parametrize(
    "finding",
    [
        {"aspect": "material", "sentiment": "positive", "evidence": "invented quote"},
        {"aspect": "diagnosis", "sentiment": "positive", "evidence": "soft"},
        {"aspect": "material", "sentiment": "positive", "evidence": ""},
    ],
)
def test_llm_cannot_add_unsupported_evidence_or_aspects(finding):
    with pytest.raises(ValueError):
        validate_findings({"findings": [finding]}, "soft fabric")


def test_llm_adapter_uses_configured_model_and_validates_response(monkeypatch):
    monkeypatch.setenv("WORKBUDDY_LLM_API_KEY", "test-placeholder")
    monkeypatch.setenv("WORKBUDDY_LLM_MODEL", "test-model")
    monkeypatch.setenv("WORKBUDDY_LLM_URL", "https://example.invalid/chat/completions")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"findings":[{"aspect":"material","sentiment":"positive","evidence":"soft fabric"}]}'
                        }
                    }
                ]
            }

    def fake_post(url, **kwargs):
        assert url == "https://example.invalid/chat/completions"
        assert kwargs["json"]["model"] == "test-model"
        return Response()

    monkeypatch.setattr("workbuddy.aspects.requests.post", fake_post)
    assert extract_llm("soft fabric")["findings"][0]["evidence"] == "soft fabric"
