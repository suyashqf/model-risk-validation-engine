from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


REGULATORY_REF = "SR 26-2"

REQUIRED_CONFIG_FIELDS = [
    "model_name",
    "model_type",
    "intended_use",
    "features",
    "development_data",
    "validation_data",
    "analyst_name",
    "review_date",
]


def load_validation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"Validation config file is required and was not found: {config_path}")

    content = config_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError("Validation config file is empty.")

    if config_path.suffix.lower() == ".json":
        config = json.loads(content)
    else:
        config = yaml.safe_load(content)

    if not config or not isinstance(config, dict):
        raise ValueError("Validation config must contain a mapping of model metadata.")

    # Align validation_date and review_date for backwards compatibility
    if "validation_date" in config and "review_date" not in config:
        config["review_date"] = config["validation_date"]
    elif "review_date" in config and "validation_date" not in config:
        config["validation_date"] = config["review_date"]

    config.setdefault("target_variable", "Binary default outcome (1 = default, 0 = performing)")
    config.setdefault("train_test_split", "70/30 random split")

    missing = [field for field in REQUIRED_CONFIG_FIELDS if not config.get(field)]
    if missing:
        raise ValueError(f"Validation config is missing required fields: {', '.join(missing)}")
    if not isinstance(config["features"], list) or not config["features"]:
        raise ValueError("Validation config field 'features' must be a non-empty list.")

    config.setdefault("data_type", infer_data_type(str(config.get("validation_data", ""))))
    return config


def infer_data_type(validation_data: str) -> str:
    return "synthetic" if "synthetic" in validation_data.lower() else "production"

