"""Reproducible chronological experiment; test labels never select the model."""

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import sklearn
from .data import LABELS, SOURCE, read_rows, chronological_split
from .model import features, training_features, serialize_estimator, predict_proba


def scores(y, prediction):
    return {
        "macro_f1": float(
            f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
        "per_class": classification_report(
            y,
            prediction,
            labels=[0, 1, 2],
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def run_experiment(path, output, limit=None, max_iter=35):
    start = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, cleaning = read_rows(path, limit)
    train, validation, test = chronological_split(rows)
    print("Split:", len(train), len(validation), len(test), flush=True)
    baseline, enriched, history = training_features(train)
    y_train = np.array([LABELS.index(r["label"]) for r in train])
    y_val = np.array([LABELS.index(r["label"]) for r in validation])
    y_test = np.array([LABELS.index(r["label"]) for r in test])
    most_common = Counter(y_train).most_common(1)[0][0]
    results = {
        "majority": {
            "validation": scores(y_val, np.full(len(y_val), most_common)),
            "test": scores(y_test, np.full(len(y_test), most_common)),
        }
    }
    fitted = {}
    for enhanced in (False, True):
        vectorizer = DictVectorizer(sparse=False, dtype=np.float32)
        x_train = vectorizer.fit_transform(enriched if enhanced else baseline)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(
            vectorizer.transform([features(r, history, enhanced) for r in validation])
        )
        x_test = scaler.transform(
            vectorizer.transform([features(r, history, enhanced) for r in test])
        )
        candidates = [
            (
                "history_logistic" if enhanced else "metadata_logistic",
                LogisticRegression(
                    max_iter=500, class_weight="balanced", C=1, random_state=42
                ),
            )
        ]
        if enhanced:
            candidates.append(
                (
                    "history_mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        max_iter=max_iter,
                        batch_size=512,
                        alpha=0.001,
                        learning_rate_init=0.001,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=5,
                        random_state=42,
                    ),
                )
            )
        for name, estimator in candidates:
            print("Training", name, flush=True)
            if name.endswith("mlp"):
                estimator.fit(
                    x_train,
                    y_train,
                    sample_weight=compute_sample_weight("balanced", y_train),
                )
            else:
                estimator.fit(x_train, y_train)
            results[name] = {
                "validation": scores(y_val, estimator.predict(x_val)),
                "test": scores(y_test, estimator.predict(x_test)),
                "iterations": np.asarray(estimator.n_iter_).tolist(),
            }
            portable = serialize_estimator(
                estimator, vectorizer, scaler, name, enhanced
            )
            dictionaries = [features(r, history, enhanced) for r in test[:100]]
            np.testing.assert_allclose(
                predict_proba(portable, dictionaries),
                estimator.predict_proba(x_test[:100]),
                atol=2e-5,
            )
            fitted[name] = portable
            print(
                name,
                "validation Macro-F1:",
                round(results[name]["validation"]["macro_f1"], 4),
                flush=True,
            )
    selected = max(fitted, key=lambda name: results[name]["validation"]["macro_f1"])
    splits = {
        name: {
            "rows": len(part),
            "start": part[0]["date"],
            "end": part[-1]["date"],
            "labels": dict(Counter(r["label"] for r in part)),
        }
        for name, part in [("train", train), ("validation", validation), ("test", test)]
    }
    report = {
        "dataset": "RentTheRunway Clothing Fit",
        "source": SOURCE,
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "run_date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "seed": 42,
        "scikit_learn": sklearn.__version__,
        "sample_limit": limit,
        "cleaning": cleaning,
        "splits": splits,
        "labels": LABELS,
        "models": results,
        "selected_model": selected,
        "selection_rule": "Highest validation Macro-F1; test scores not used for selection.",
        "history_policy": "Training rows see only earlier dates. Validation and test use frozen training history.",
        "aspect_engine": "rules; no LLM-generated features or human extraction benchmark claimed",
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
    # Public demo contains only model coefficients and aggregate item counts; no raw reviews/users.
    catalog = dict(
        sorted(
            ((k, v) for k, v in history.items() if v["count"] >= 20),
            key=lambda pair: pair[1]["count"],
            reverse=True,
        )[:250]
    )
    bundle = {
        "format_version": 1,
        "model": fitted[selected],
        "catalog": catalog,
        "history_cutoff": train[-1]["date"],
        "dataset": report["dataset"],
        "size_semantics": "Original dataset size codes; no brand-size decoding available.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "demo-model.json").write_text(
        json.dumps(bundle, separators=(",", ":")) + "\n"
    )
    print(
        "Selected",
        selected,
        "test Macro-F1:",
        results[selected]["test"]["macro_f1"],
        flush=True,
    )
    return report
