"""Download and normalize the official RentTheRunway JSONL dataset."""

from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import requests

SOURCE = "https://mcauleylab.ucsd.edu/public_datasets/data/renttherunway/renttherunway_final_data.json.gz"
LABELS = ["small", "fit", "large"]


def download(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        with requests.get(SOURCE, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1 << 20):
                    output.write(chunk)
        with gzip.open(temporary, "rt") as test:
            json.loads(next(test))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def number(value, low, high):
    try:
        value = float(value)
        return value if math.isfinite(value) and low <= value <= high else None
    except (ValueError, TypeError):
        return None


def height_cm(value):
    match = re.fullmatch(r"\s*(\d)'\s*(\d{1,2})\"?\s*", str(value))
    if not match or int(match[2]) >= 12:
        return None
    return number((int(match[1]) * 12 + int(match[2])) * 2.54, 120, 220)


def normalize(raw):
    label = str(raw.get("fit", "")).lower()
    if label not in LABELS or not raw.get("item_id") or not raw.get("user_id"):
        return None
    try:
        day = datetime.strptime(raw["review_date"], "%B %d, %Y").date().isoformat()
    except (ValueError, KeyError, TypeError):
        return None
    weight = str(raw.get("weight", "")).lower().replace("lbs", "").strip()
    pounds = number(weight, 60, 550)
    size = number(raw.get("size"), 0, 100)
    if size is None:
        return None
    return {
        "item_id": str(raw["item_id"]),
        "user_id": str(raw["user_id"]),
        "date": day,
        "label": label,
        "size": size,
        "height_cm": height_cm(raw.get("height")),
        "weight_kg": round(pounds * 0.45359237, 3) if pounds is not None else None,
        "category": str(raw.get("category") or "unknown").strip().lower(),
        "body_type": str(raw.get("body type") or "unknown").strip().lower(),
        "review": str(raw.get("review_text") or "")[:12000],
    }


def read_rows(path, limit=None):
    rows, seen = [], set()
    counts = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            counts["source_rows"] += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue
            digest = hashlib.sha256(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest()
            if digest in seen:
                counts["exact_duplicates"] += 1
                continue
            seen.add(digest)
            row = normalize(raw)
            if row is None:
                counts["invalid_records"] += 1
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    rows.sort(key=lambda r: r["date"])
    counts["retained_rows"] = len(rows)
    counts["missing_height"] = sum(r["height_cm"] is None for r in rows)
    counts["missing_weight"] = sum(r["weight_kg"] is None for r in rows)
    return rows, dict(counts)


def chronological_split(rows):
    if len(rows) < 100:
        raise ValueError("At least 100 valid records are required for evaluation.")
    first, second = (
        rows[int(len(rows) * 0.70)]["date"],
        rows[int(len(rows) * 0.85)]["date"],
    )
    train = [r for r in rows if r["date"] < first]
    validation = [r for r in rows if first <= r["date"] < second]
    test = [r for r in rows if r["date"] >= second]
    if any(
        len(part) < 20 or {r["label"] for r in part} != set(LABELS)
        for part in (train, validation, test)
    ):
        raise ValueError(
            "Each chronological split needs at least 20 rows and all three classes."
        )
    return train, validation, test
