"""Local demonstration API. Profiles and submitted reviews are not persisted."""

import json
import math
import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request
import requests
from .aspects import extract_llm, extract_rules
from .model import recommend


def create_app(root=None):
    root = Path(root or Path.cwd())
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32768
    bundle_path, metrics_path = (
        root / "reports/demo-model.json",
        root / "reports/metrics.json",
    )
    bundle = json.loads(bundle_path.read_text()) if bundle_path.exists() else None
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_ready=bundle is not None)

    @app.get("/api/overview")
    def overview():
        if not bundle or not metrics:
            return jsonify(
                error="No trained report found. Run workbuddy train first."
            ), 503
        return jsonify(
            metrics=metrics,
            catalog_size=len(bundle["catalog"]),
            llm_available=bool(
                os.getenv("WORKBUDDY_LLM_API_KEY") and os.getenv("WORKBUDDY_LLM_MODEL")
            ),
        )

    @app.get("/api/catalog")
    def catalog():
        if not bundle:
            return jsonify(error="No model loaded."), 503
        return jsonify(items=list(bundle["catalog"].values()))

    @app.post("/api/recommend")
    def prediction():
        if not bundle:
            return jsonify(error="No model loaded."), 503
        profile = request.get_json(silent=True)
        if not isinstance(profile, dict):
            return jsonify(error="Provide a JSON object."), 400
        for key, low, high in [("height_cm", 120, 220), ("weight_kg", 30, 250)]:
            value = profile.get(key)
            if isinstance(value, bool):
                return jsonify(error="Invalid " + key), 400
            try:
                value = float(value)
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError
            except (ValueError, TypeError):
                return jsonify(error=f"{key} must be between {low} and {high}."), 400
            profile[key] = value
        body_type = profile.get("body_type", "unknown")
        if body_type not in (
            "unknown",
            "hourglass",
            "straight & narrow",
            "athletic",
            "pear",
            "apple",
            "full bust",
            "petite",
        ):
            return jsonify(error="Unsupported body type."), 400
        try:
            return jsonify(recommend(bundle, profile))
        except (ValueError, KeyError, TypeError):
            return jsonify(
                error="Choose a supported item and complete the profile."
            ), 400

    @app.post("/api/extract")
    def extraction():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Provide a JSON object."), 400
        text = payload.get("review")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 8000:
            return jsonify(error="Review must contain 1–8000 characters."), 400
        engine = payload.get("engine", "rules")
        try:
            if engine == "rules":
                return jsonify(extract_rules(text))
            if engine == "llm":
                return jsonify(extract_llm(text))
            return jsonify(error="Engine must be rules or llm."), 400
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except (requests.RequestException, KeyError, TypeError, IndexError):
            return jsonify(
                error="LLM request failed or returned an unsupported response."
            ), 502

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Request too large."), 413

    return app
