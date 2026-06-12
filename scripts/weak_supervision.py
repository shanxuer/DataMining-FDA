"""Weak-supervision rules and majority-vote audit for final experiments."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import run_faers_pipeline as pipeline


DEATH_TERMS = {
    "reac:DEATH",
    "reac:FATAL",
    "reac:CARDIAC_ARREST",
}

HIGH_RISK_TERMS = {
    "reac:SEPSIS",
    "reac:SEPTIC_SHOCK",
    "reac:ACUTE_RESPIRATORY_FAILURE",
    "reac:RESPIRATORY_FAILURE",
    "reac:SHOCK",
}

DEVICE_TERMS = {
    "reac:DEVICE_DEFECTIVE",
    "reac:DEVICE_ISSUE",
    "reac:DEVICE_MALFUNCTION",
    "reac:PRODUCT_QUALITY_ISSUE",
    "reac:PRODUCT_COMMUNICATION_ISSUE",
    "reac:NO_ADVERSE_EVENT",
}

SERIOUS_OUTCOME_TOKENS = {
    "reactionoutcome:3",
    "reactionoutcome:4",
    "reactionoutcome:5",
}

RULE_NAMES = (
    "death_or_fatal_reaction_term",
    "high_polypharmacy_10plus",
    "senior_with_multiple_suspect_drugs",
    "low_complexity_younger_case",
    "high_risk_reaction",
    "serious_reaction_outcome",
    "extreme_polypharmacy_30plus",
    "device_product_issue",
)


def rule_votes(row: dict[str, Any]) -> dict[str, int | None]:
    tokens = set(str(row.get("text_tokens") or "").split())
    age = pipeline.parse_float(row.get("age_years"))
    drug_count = pipeline.parse_float(row.get("drug_count"))
    suspect_count = pipeline.parse_float(row.get("suspect_drug_count"))
    reaction_count = pipeline.parse_float(row.get("reaction_count"))

    has_death_term = bool(tokens & DEATH_TERMS)
    has_high_risk_term = bool(tokens & HIGH_RISK_TERMS)
    has_device_term = bool(tokens & DEVICE_TERMS)

    return {
        "death_or_fatal_reaction_term": 1 if has_death_term else None,
        "high_polypharmacy_10plus": (
            1 if drug_count is not None and drug_count >= 10 else None
        ),
        "senior_with_multiple_suspect_drugs": (
            1
            if (
                age is not None
                and age >= 65
                and suspect_count is not None
                and suspect_count >= 2
            )
            else None
        ),
        "low_complexity_younger_case": (
            0
            if (
                age is not None
                and age < 50
                and drug_count is not None
                and drug_count <= 2
                and reaction_count is not None
                and reaction_count <= 1
            )
            else None
        ),
        "high_risk_reaction": 1 if has_high_risk_term else None,
        "serious_reaction_outcome": (
            1 if tokens & SERIOUS_OUTCOME_TOKENS else None
        ),
        "extreme_polypharmacy_30plus": (
            1 if drug_count is not None and drug_count >= 30 else None
        ),
        "device_product_issue": (
            0
            if has_device_term and not has_death_term and not has_high_risk_term
            else None
        ),
    }


def majority_vote(votes: Iterable[int | None]) -> tuple[int | None, bool]:
    positive = 0
    negative = 0
    for vote in votes:
        if vote is None:
            continue
        if vote == 1:
            positive += 1
        elif vote == 0:
            negative += 1
        else:
            raise ValueError(f"Votes must be 0, 1, or None, got {vote!r}")

    if positive == 0 and negative == 0:
        return None, False
    if positive == negative:
        return None, True
    return (1 if positive > negative else 0), False


def _parse_label(row: dict[str, Any]) -> int:
    value = row.get("label_serious")
    if isinstance(value, bool):
        raise ValueError(
            f"label_serious must be 0 or 1, got boolean {value!r}"
        )
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    raise ValueError(f"label_serious must be 0 or 1, got {value!r}")


def _new_rule_counts() -> dict[str, int]:
    return {
        "fires": 0,
        "positive_votes": 0,
        "negative_votes": 0,
        "positive_labels": 0,
        "correct": 0,
    }


def _new_bucket() -> dict[str, Any]:
    return {
        "total": 0,
        "covered": 0,
        "conflicts": 0,
        "voted": 0,
        "_correct": 0,
        "_true_positive": 0,
        "_false_positive": 0,
        "_false_negative": 0,
        "rules": {
            name: _new_rule_counts()
            for name in RULE_NAMES
        },
    }


def _update_bucket(
    bucket: dict[str, Any],
    label: int,
    votes: dict[str, int | None],
) -> None:
    bucket["total"] += 1
    for name in RULE_NAMES:
        vote = votes[name]
        if vote is None:
            continue
        rule = bucket["rules"][name]
        rule["fires"] += 1
        rule["positive_votes"] += int(vote == 1)
        rule["negative_votes"] += int(vote == 0)
        rule["positive_labels"] += label
        rule["correct"] += int(vote == label)

    active_votes = [vote for vote in votes.values() if vote is not None]
    if not active_votes:
        return
    bucket["covered"] += 1

    prediction, conflict = majority_vote(active_votes)
    if conflict:
        bucket["conflicts"] += 1
        return
    if prediction is None:
        return

    bucket["voted"] += 1
    bucket["_correct"] += int(prediction == label)
    bucket["_true_positive"] += int(prediction == 1 and label == 1)
    bucket["_false_positive"] += int(prediction == 1 and label == 0)
    bucket["_false_negative"] += int(prediction == 0 and label == 1)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    total = bucket["total"]
    covered = bucket["covered"]
    voted = bucket["voted"]
    true_positive = bucket["_true_positive"]
    false_positive = bucket["_false_positive"]
    false_negative = bucket["_false_negative"]
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)

    rules: dict[str, dict[str, int | float]] = {}
    for name in RULE_NAMES:
        counts = bucket["rules"][name]
        fires = counts["fires"]
        rules[name] = {
            **counts,
            "coverage_rate": _ratio(fires, total),
            "positive_label_rate": _ratio(
                counts["positive_labels"],
                fires,
            ),
            "accuracy": _ratio(counts["correct"], fires),
        }

    return {
        "total": total,
        "covered": covered,
        "coverage_rate": _ratio(covered, total),
        "conflicts": bucket["conflicts"],
        "conflict_rate": _ratio(bucket["conflicts"], covered),
        "voted": voted,
        "accuracy": _ratio(bucket["_correct"], voted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rules": rules,
    }


def summarize_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    overall = _new_bucket()
    splits: dict[str, dict[str, Any]] = {}

    for row in rows:
        label = _parse_label(row)
        votes = rule_votes(row)
        split = str(row.get("split") or "")
        split_bucket = splits.setdefault(split, _new_bucket())
        _update_bucket(overall, label, votes)
        _update_bucket(split_bucket, label, votes)

    return {
        "overall": _finalize_bucket(overall),
        "splits": {
            split: _finalize_bucket(bucket)
            for split, bucket in sorted(splits.items())
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
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


def run_weak_supervision(
    paths: Iterable[Path],
    output_dir: Path,
) -> dict[str, Any]:
    result = summarize_rows(pipeline.iter_feature_rows(paths))
    result["metadata"] = {
        "rules": list(RULE_NAMES),
        "note": (
            "Weak supervision votes are audit only and are not training "
            "labels."
        ),
    }
    _atomic_write_json(
        output_dir / "weak_supervision_metrics.json",
        result,
    )
    return result
