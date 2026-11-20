"""Pre-purchase features, strictly historical aggregates and portable JSON inference."""

from itertools import groupby
import math
import numpy as np
from .aspects import ASPECTS, extract_rules
from .data import LABELS


def new_history(row):
    return {
        "item_id": row["item_id"],
        "category": row["category"],
        "count": 0,
        "labels": {label: 0 for label in LABELS},
        "negative_aspects": {a: 0 for a in ASPECTS},
        "sizes": {},
    }


def update_history(history, row):
    item = history.setdefault(row["item_id"], new_history(row))
    item["count"] += 1
    item["labels"][row["label"]] += 1
    code = str(int(row["size"]))
    item["sizes"][code] = item["sizes"].get(code, 0) + 1
    negatives = {
        f["aspect"]
        for f in extract_rules(row["review"])["findings"]
        if f["sentiment"] == "negative"
    }
    for aspect in negatives:
        item["negative_aspects"][aspect] += 1


def features(profile, history, enhanced=True):
    # Deliberately ignore review, label, rating, user_id and future transactions.
    values = {
        "size": float(profile["size"]),
        "height_cm": float(profile.get("height_cm") or 170),
        "weight_kg": float(profile.get("weight_kg") or 70),
        "height_missing": int(profile.get("height_cm") is None),
        "weight_missing": int(profile.get("weight_kg") is None),
        "category": profile.get("category") or "unknown",
        "body_type": profile.get("body_type") or "unknown",
    }
    if enhanced:
        item = history.get(str(profile["item_id"]), {})
        n = item.get("count", 0)
        values["history_log_count"] = math.log1p(n)
        for label in LABELS:
            values["history_" + label] = (item.get("labels", {}).get(label, 0) + 3) / (
                n + 9
            )
        for aspect in ASPECTS:
            values["negative_" + aspect] = item.get("negative_aspects", {}).get(
                aspect, 0
            ) / max(n, 1)
    return values


def training_features(rows):
    history, baseline, enhanced = {}, [], []
    # Updates occur only after every record on the same date has been featurized.
    # Thus neither a row's own label/review nor any same-day review is its input.
    for _, same_day in groupby(rows, key=lambda r: r["date"]):
        batch = list(same_day)
        for row in batch:
            baseline.append(features(row, history, enhanced=False))
            enhanced.append(features(row, history, enhanced=True))
        for row in batch:
            update_history(history, row)
    return baseline, enhanced, history


def serialize_estimator(estimator, vectorizer, scaler, name, enhanced):
    model = {
        "name": name,
        "enhanced": enhanced,
        "labels": LABELS,
        "feature_names": vectorizer.feature_names_,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    if hasattr(estimator, "coefs_"):
        model.update(
            kind="mlp",
            weights=[w.tolist() for w in estimator.coefs_],
            biases=[b.tolist() for b in estimator.intercepts_],
        )
    else:
        model.update(
            kind="logistic",
            weights=[estimator.coef_.T.tolist()],
            biases=[estimator.intercept_.tolist()],
        )
    return model


def predict_proba(model, dictionaries):
    positions = {name: i for i, name in enumerate(model["feature_names"])}
    x = np.zeros((len(dictionaries), len(positions)))
    for i, dictionary in enumerate(dictionaries):
        for key, value in dictionary.items():
            name = f"{key}={value}" if isinstance(value, str) else key
            if name in positions:
                x[i, positions[name]] = 1 if isinstance(value, str) else value
    x = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    for i, (weight, bias) in enumerate(zip(model["weights"], model["biases"])):
        x = x @ np.asarray(weight) + np.asarray(bias)
        if i < len(model["weights"]) - 1:
            x = np.maximum(x, 0)
    x = x - x.max(axis=1, keepdims=True)
    probability = np.exp(x)
    return probability / probability.sum(axis=1, keepdims=True)


def recommend(bundle, profile):
    item = bundle["catalog"].get(str(profile["item_id"]))
    if not item:
        raise ValueError("Choose an item available in the demo catalog.")
    # Only codes actually seen at least three times for this item are candidates.
    codes = sorted(int(s) for s, n in item["sizes"].items() if n >= 3)
    if not codes:
        raise ValueError("Insufficient size observations for this item.")
    model = bundle["model"]
    dictionaries = [
        features(
            {**profile, "category": item["category"], "size": code},
            bundle["catalog"],
            model["enhanced"],
        )
        for code in codes
    ]
    probabilities = predict_proba(model, dictionaries)
    results = [
        {
            "size_code": code,
            "observations": item["sizes"][str(code)],
            "probabilities": {
                label: round(float(p), 4) for label, p in zip(LABELS, row)
            },
        }
        for code, row in zip(codes, probabilities)
    ]
    selected = max(results, key=lambda r: r["probabilities"]["fit"])
    return {
        "item_id": item["item_id"],
        "recommended_size_code": selected["size_code"],
        "candidates": results,
        "model": model["name"],
        "history_count": item["count"],
        "history_cutoff": bundle["history_cutoff"],
        "notice": "Dataset-specific size codes, not universal brand sizes. Model probabilities are uncalibrated.",
    }
