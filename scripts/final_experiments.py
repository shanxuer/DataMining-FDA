"""Feature token policies for final-project ablation experiments."""

from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import run_faers_pipeline as pipeline


ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "all_tokens": {
        "description": "Structured, binned, and all text token families",
        "include_structured": True,
        "include_prefixes": None,
        "exclude_prefixes": (),
    },
    "without_reaction_pt": {
        "description": "All features except reaction PT tokens",
        "include_structured": True,
        "include_prefixes": None,
        "exclude_prefixes": ("reac:",),
    },
    "without_reaction_outcome": {
        "description": "All features except reaction outcome tokens",
        "include_structured": True,
        "include_prefixes": None,
        "exclude_prefixes": ("reactionoutcome:",),
    },
    "drug_indication_only": {
        "description": "Drug and indication text tokens only",
        "include_structured": False,
        "include_prefixes": ("drug:", "indi:"),
        "exclude_prefixes": (),
    },
    "structured_only": {
        "description": "Demographic, source, quarter, and count-bin tokens only",
        "include_structured": True,
        "include_prefixes": (),
        "exclude_prefixes": (),
    },
}


def _matches_prefix(token: str, prefixes: tuple[str, ...]) -> bool:
    return any(token.startswith(prefix) for prefix in prefixes)


def build_ablation_text(row: dict[str, str], experiment_name: str) -> str:
    config = ABLATION_CONFIGS.get(experiment_name)
    if config is None:
        raise ValueError(f"Unknown ablation experiment: {experiment_name}")

    tokens: list[str] = []
    if config["include_structured"]:
        structured_row = dict(row)
        structured_row["text_tokens"] = ""
        tokens.extend(pipeline.build_hash_text(structured_row).split())

    include_prefixes = config["include_prefixes"]
    exclude_prefixes = tuple(config["exclude_prefixes"])
    for token in row.get("text_tokens", "").split():
        if include_prefixes is not None and not _matches_prefix(token, tuple(include_prefixes)):
            continue
        if _matches_prefix(token, exclude_prefixes):
            continue
        tokens.append(token)

    return " ".join(tokens)


def iter_ablation_chunks(
    paths: list[Path],
    split: str,
    chunk_size: int,
) -> Iterable[tuple[dict[str, list[str]], np.ndarray, list[dict[str, str]]]]:
    texts = {name: [] for name in ABLATION_CONFIGS}
    labels: list[int] = []
    metadata: list[dict[str, str]] = []
    for row in pipeline.iter_feature_rows(paths, split):
        for name in ABLATION_CONFIGS:
            texts[name].append(build_ablation_text(row, name))
        labels.append(int(row["label_serious"]))
        metadata.append(pipeline.metadata_from_row(row))
        if len(labels) >= chunk_size:
            yield texts, np.asarray(labels, dtype=np.int8), metadata
            texts = {name: [] for name in ABLATION_CONFIGS}
            labels = []
            metadata = []
    if labels:
        yield texts, np.asarray(labels, dtype=np.int8), metadata


def _new_model(name: str, n_features: int, random_state: int) -> dict[str, Any]:
    return {
        "weights": np.zeros(n_features, dtype=np.float64),
        "bias": 0.0,
        "n_features": int(n_features),
        "type": "hash_logistic_numpy",
        "experiment": name,
        "random_state": random_state,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def train_ablation_models(
    paths: list[Path],
    model_dir: Path,
    chunk_size: int,
    n_features: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    random_state: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    train_counts = pipeline.split_label_counts(paths)["train"]
    if train_counts[0] == 0 or train_counts[1] == 0:
        raise ValueError(
            f"Training split must contain both classes, got {dict(train_counts)}"
        )
    class_weights = pipeline.class_weights_from_counts(train_counts)
    models = {
        name: _new_model(name, n_features, random_state)
        for name in ABLATION_CONFIGS
    }

    seen = 0
    for epoch in range(epochs):
        epoch_seen = 0
        epoch_learning_rate = learning_rate / math.sqrt(epoch + 1)
        for texts, labels, _ in iter_ablation_chunks(paths, "train", chunk_size):
            for name, model in models.items():
                pipeline.update_hash_logistic_model(
                    model,
                    texts[name],
                    labels,
                    epoch_learning_rate,
                    l2,
                    class_weights,
                )
            epoch_seen += len(labels)
        seen = max(seen, epoch_seen)

    for name, model in models.items():
        _atomic_write_pickle(model_dir / f"{name}.pkl", model)
    return models, seen


def predict_ablation_models(
    paths: list[Path],
    split: str,
    models: dict[str, dict[str, Any]],
    chunk_size: int,
    collect_rows: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, str]]]:
    labels: list[int] = []
    probabilities: dict[str, list[float]] = {name: [] for name in models}
    rows: list[dict[str, str]] = []
    for texts, chunk_labels, metadata in iter_ablation_chunks(
        paths,
        split,
        chunk_size,
    ):
        labels.extend(int(value) for value in chunk_labels)
        if collect_rows:
            rows.extend(metadata)
        for name, model in models.items():
            chunk_probabilities = pipeline.predict_hash_logistic_examples(
                model,
                texts[name],
            )
            probabilities[name].extend(float(value) for value in chunk_probabilities)
    return (
        np.asarray(labels, dtype=np.int8),
        {
            name: np.asarray(values, dtype=np.float64)
            for name, values in probabilities.items()
        },
        rows,
    )


def run_ablation_experiments(
    paths: list[Path],
    ablation_dir: Path,
    numeric_baseline: dict[str, Any],
    *,
    chunk_size: int,
    n_features: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    random_state: int,
) -> dict[str, Any]:
    models, train_rows = train_ablation_models(
        paths,
        ablation_dir / "models",
        chunk_size,
        n_features,
        epochs,
        learning_rate,
        l2,
        random_state,
    )
    predictions = {}
    for split in ("train", "valid", "test"):
        predictions[split] = predict_ablation_models(
            paths,
            split,
            models,
            chunk_size,
            collect_rows=split in {"valid", "test"},
        )

    valid_labels, valid_probabilities, _ = predictions["valid"]
    thresholds = {
        name: pipeline.best_threshold(valid_labels, valid_probabilities[name])
        for name in models
    }
    result: dict[str, Any] = {
        "metadata": {
            "split_policy": pipeline.SPLIT_POLICY,
            "n_features": n_features,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "random_state": random_state,
            "experiments": {
                name: config["description"]
                for name, config in ABLATION_CONFIGS.items()
            },
        },
        "experiments": {},
    }
    for name in models:
        item: dict[str, Any] = {
            "description": ABLATION_CONFIGS[name]["description"],
            "model_path": str(ablation_dir / "models" / f"{name}.pkl"),
            "train_rows": train_rows,
            "threshold_from_valid": thresholds[name],
            "split_metrics": {},
            "error_cases": {},
            "strata": {},
        }
        for split in ("train", "valid", "test"):
            labels, split_probabilities, rows = predictions[split]
            probabilities = split_probabilities[name]
            item["split_metrics"][split] = pipeline.classification_metrics(
                labels,
                probabilities,
                thresholds[name],
            )
            if split in {"valid", "test"}:
                item["error_cases"][split] = pipeline.select_error_cases(
                    rows,
                    labels,
                    probabilities,
                    thresholds[name],
                )
                item["strata"][split] = pipeline.stratified_metrics(
                    rows,
                    labels,
                    probabilities,
                    thresholds[name],
                )
        result["experiments"][name] = item

    numeric_item = dict(numeric_baseline)
    numeric_item["description"] = "Existing numeric logistic baseline"
    result["experiments"]["numeric_logistic"] = numeric_item
    _atomic_write_json(ablation_dir / "ablation_metrics.json", result)
    return result
