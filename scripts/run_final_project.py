#!/usr/bin/env python3
"""Validate base outputs and run all final-project deliverables."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import dashboard
import final_experiments
import final_reporting
import run_faers_pipeline as pipeline
import weak_supervision


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_QUARTERS = tuple(f"2025Q{quarter}" for quarter in range(1, 5))
CASE_FILENAME_PATTERN = re.compile(
    r"cases_(2025Q[1-4])\.csv",
    flags=re.IGNORECASE,
)
REQUIRED_CASE_COLUMNS = {
    "quarter",
    "split",
    "label_serious",
    "text_tokens",
    "safetyreportid",
    "receivedate",
    "primarysourcecountry",
    "occurcountry",
    "reporttype",
    "fulfillexpeditecriteria",
    "duplicate",
    "reportercountry",
    "qualification",
    "sendertype",
    "patientsex",
    "patientagegroup",
    "age_years",
    "drug_count",
    "suspect_drug_count",
    "reaction_count",
    "indication_count",
}
VALID_SPLITS = ("train", "valid", "test")
FEATURE_AUDIT_MISSING_FIELDS = frozenset(
    pipeline.FEATURE_AUDIT_MISSING_FIELDS
)


def _load_strict_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    text = path.read_text(encoding="utf-8")

    def reject_constant(value: str) -> None:
        raise ValueError(
            f"{path}: non-standard JSON constant is not allowed: {value}"
        )

    try:
        return json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{context}: expected mapping, got {type(value).__name__}"
        )
    return value


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{context}: required value is missing")
    return mapping[key]


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: expected a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{context}: expected a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite number")
    return number


def _unit_interval(value: Any, context: str) -> float:
    number = _finite_number(value, context)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{context}: expected a value between 0 and 1")
    return number


def _nonnegative_integer(value: Any, context: str) -> int:
    number = _finite_number(value, context)
    if not number.is_integer() or number < 0:
        raise ValueError(f"{context}: expected a non-negative integer")
    return int(number)


def _discover_case_paths(interim_dir: Path) -> list[Path]:
    paths = (
        sorted(
            path
            for path in interim_dir.iterdir()
            if path.is_file() and path.name.lower().startswith("cases_")
        )
        if interim_dir.is_dir()
        else []
    )
    by_quarter: dict[str, list[Path]] = {
        quarter: []
        for quarter in EXPECTED_QUARTERS
    }
    invalid: list[Path] = []
    for path in paths:
        match = CASE_FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            invalid.append(path)
            continue
        quarter = match.group(1).upper()
        by_quarter[quarter].append(path)

    missing = [
        quarter
        for quarter, quarter_paths in by_quarter.items()
        if not quarter_paths
    ]
    duplicates = {
        quarter: quarter_paths
        for quarter, quarter_paths in by_quarter.items()
        if len(quarter_paths) > 1
    }
    if (
        missing
        or duplicates
        or invalid
        or len(paths) != len(EXPECTED_QUARTERS)
    ):
        details = [
            "Case CSV files must exactly cover 2025Q1, 2025Q2, 2025Q3, "
            f"and 2025Q4 in {interim_dir}"
        ]
        if invalid:
            details.append(
                "invalid filename candidates: "
                + ", ".join(path.name for path in invalid)
            )
        if duplicates:
            duplicate_text = ", ".join(
                f"{quarter} ({', '.join(path.name for path in quarter_paths)})"
                for quarter, quarter_paths in duplicates.items()
            )
            details.append(f"duplicate quarters: {duplicate_text}")
        if missing:
            details.append(f"missing quarters: {', '.join(missing)}")
        details.append(f"found {len(paths)} case CSV files")
        raise ValueError("; ".join(details))

    return [by_quarter[quarter][0] for quarter in EXPECTED_QUARTERS]


def _validate_numeric_baseline(metrics_path: Path) -> dict[str, Any]:
    metrics = _mapping(
        _load_strict_json(metrics_path),
        str(metrics_path),
    )
    models = _mapping(
        _required(metrics, "models", "model_metrics.models"),
        "model_metrics.models",
    )
    baseline = _mapping(
        _required(
            models,
            "numeric_logistic",
            "model_metrics.models.numeric_logistic",
        ),
        "model_metrics.models.numeric_logistic",
    )
    context = "numeric_logistic"

    model_path = _required(baseline, "model_path", f"{context}.model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError(f"{context}.model_path: expected a non-empty string")
    _nonnegative_integer(
        _required(baseline, "train_rows", f"{context}.train_rows"),
        f"{context}.train_rows",
    )
    _unit_interval(
        _required(
            baseline,
            "threshold_from_valid",
            f"{context}.threshold_from_valid",
        ),
        f"{context}.threshold_from_valid",
    )

    split_metrics = _mapping(
        _required(
            baseline,
            "split_metrics",
            f"{context}.split_metrics",
        ),
        f"{context}.split_metrics",
    )
    for split in VALID_SPLITS:
        split_context = f"{context}.split_metrics.{split}"
        values = _mapping(
            _required(split_metrics, split, split_context),
            split_context,
        )
        for metric_name in final_reporting.TEST_METRICS:
            metric_context = f"{split_context}.{metric_name}"
            _unit_interval(
                _required(values, metric_name, metric_context),
                metric_context,
            )
        n = _nonnegative_integer(
            _required(values, "n", f"{split_context}.n"),
            f"{split_context}.n",
        )
        positives = _nonnegative_integer(
            _required(
                values,
                "positives",
                f"{split_context}.positives",
            ),
            f"{split_context}.positives",
        )
        if positives > n:
            raise ValueError(
                f"{split_context}.positives: expected at most n={n}, "
                f"got {positives}"
            )

    error_context = f"{context}.error_cases"
    error_cases = _mapping(
        _required(baseline, "error_cases", error_context),
        error_context,
    )
    for split in ("valid", "test"):
        split_context = f"{error_context}.{split}"
        split_errors = _mapping(
            _required(error_cases, split, split_context),
            split_context,
        )
        for error_type in ("false_positive", "false_negative"):
            cases_context = f"{split_context}.{error_type}"
            cases = _required(split_errors, error_type, cases_context)
            if not isinstance(cases, list):
                raise ValueError(
                    f"{cases_context}: expected list, "
                    f"got {type(cases).__name__}"
                )

    strata_context = f"{context}.strata"
    strata = _mapping(
        _required(baseline, "strata", strata_context),
        strata_context,
    )
    for split in ("valid", "test"):
        split_context = f"{strata_context}.{split}"
        _mapping(
            _required(strata, split, split_context),
            split_context,
        )

    return copy.deepcopy(baseline)


def _scan_case_files(
    case_paths: list[Path],
    audit_missing_fields: set[str],
) -> tuple[
    int,
    dict[str, Counter],
    dict[str, Counter],
    Counter,
]:
    total = 0
    by_quarter = {
        quarter: Counter(n=0, positive=0)
        for quarter in EXPECTED_QUARTERS
    }
    by_split = {
        split: Counter(n=0, positive=0)
        for split in VALID_SPLITS
    }
    missing_counts: Counter = Counter()

    for path, expected_quarter in zip(case_paths, EXPECTED_QUARTERS):
        file_rows = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_CASE_COLUMNS - fieldnames)
            if missing:
                raise ValueError(f"{path}: missing columns: {missing}")
            unknown_audit_fields = sorted(
                audit_missing_fields - fieldnames
            )
            if unknown_audit_fields:
                field = unknown_audit_fields[0]
                raise ValueError(
                    f"feature_audit.missing.{field}: field is not in "
                    f"CSV schema for {path}"
                )
            for line_number, row in enumerate(reader, start=2):
                file_rows += 1
                split = row.get("split")
                if split not in by_split:
                    raise ValueError(
                        f"{path}:{line_number}: unknown split {split!r}; "
                        f"expected train, valid, or test"
                    )
                label_text = row.get("label_serious")
                if label_text not in ("0", "1"):
                    raise ValueError(
                        f"{path}:{line_number}: label_serious must be 0 or 1, "
                        f"got {label_text!r}"
                    )
                row_quarter = row.get("quarter")
                if row_quarter != expected_quarter:
                    raise ValueError(
                        f"{path}:{line_number}: row quarter {row_quarter!r} "
                        f"does not match file quarter {expected_quarter}"
                    )
                if expected_quarter != "2025Q4" and split != "train":
                    raise ValueError(
                        f"{path}:{line_number}: {expected_quarter} may only "
                        f"contain split 'train', got {split!r}"
                    )
                if expected_quarter == "2025Q4" and split not in (
                    "valid",
                    "test",
                ):
                    raise ValueError(
                        f"{path}:{line_number}: 2025Q4 may only contain "
                        f"splits 'valid' or 'test', got {split!r}"
                    )

                label = int(label_text)
                total += 1
                by_quarter[expected_quarter]["n"] += 1
                by_quarter[expected_quarter]["positive"] += label
                by_split[split]["n"] += 1
                by_split[split]["positive"] += label
                for field in audit_missing_fields:
                    if row.get(field) == "":
                        missing_counts[field] += 1

        if file_rows == 0:
            raise ValueError(f"{path}: CSV must contain at least one data row")

    for split, counts in by_split.items():
        negatives = counts["n"] - counts["positive"]
        if counts["positive"] == 0 or negatives == 0:
            raise ValueError(
                f"Split {split} must contain both classes, got "
                f"0={negatives}, 1={counts['positive']}"
            )
    return total, by_quarter, by_split, missing_counts


def _validate_numeric_split_counts(
    baseline: dict[str, Any],
    by_split: dict[str, Counter],
) -> None:
    split_metrics = baseline["split_metrics"]
    for split in VALID_SPLITS:
        metrics = split_metrics[split]
        expected = by_split[split]
        for metric_name, count_name in (
            ("n", "n"),
            ("positives", "positive"),
        ):
            context = (
                f"numeric_logistic.split_metrics.{split}.{metric_name}"
            )
            actual = _nonnegative_integer(metrics[metric_name], context)
            if actual != expected[count_name]:
                raise ValueError(
                    f"{context}: expected {expected[count_name]} from "
                    f"case CSVs, got {actual}"
                )


def _validate_audit_bucket(
    actual: dict[str, Any],
    expected: Counter,
    context: str,
) -> None:
    for field in ("n", "positive"):
        value_context = f"{context}.{field}"
        value = _nonnegative_integer(
            _required(actual, field, value_context),
            value_context,
        )
        if value != expected[field]:
            raise ValueError(
                f"{value_context}: expected {expected[field]} from case CSVs, "
                f"got {value}"
            )


def _validate_feature_audit(
    audit: dict[str, Any],
    total: int,
    by_quarter: dict[str, Counter],
    by_split: dict[str, Counter],
    declared_missing: dict[str, int],
    actual_missing: Counter,
) -> None:
    audit_total = _nonnegative_integer(
        _required(audit, "total", "feature_audit.total"),
        "feature_audit.total",
    )
    if audit_total != total:
        raise ValueError(
            f"feature_audit.total: expected {total} from case CSVs, "
            f"got {audit_total}"
        )

    audit_quarters = _mapping(
        _required(audit, "by_quarter", "feature_audit.by_quarter"),
        "feature_audit.by_quarter",
    )
    if set(audit_quarters) != set(EXPECTED_QUARTERS):
        raise ValueError(
            "feature_audit.by_quarter: expected exactly "
            + ", ".join(EXPECTED_QUARTERS)
        )
    for quarter in EXPECTED_QUARTERS:
        context = f"feature_audit.by_quarter.{quarter}"
        _validate_audit_bucket(
            _mapping(audit_quarters[quarter], context),
            by_quarter[quarter],
            context,
        )

    if "by_split" in audit:
        audit_splits = _mapping(audit["by_split"], "feature_audit.by_split")
        if set(audit_splits) != set(VALID_SPLITS):
            raise ValueError(
                "feature_audit.by_split: expected exactly train, valid, test"
            )
        for split in VALID_SPLITS:
            context = f"feature_audit.by_split.{split}"
            _validate_audit_bucket(
                _mapping(audit_splits[split], context),
                by_split[split],
                context,
            )

    for field in FEATURE_AUDIT_MISSING_FIELDS:
        context = f"feature_audit.missing.{field}"
        declared_count = declared_missing.get(field, 0)
        actual_count = actual_missing[field]
        if declared_count != actual_count:
            raise ValueError(
                f"{context}: expected {actual_count} empty strings from "
                f"case CSVs, got {declared_count}"
            )


def _load_feature_audit(
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    audit = _mapping(_load_strict_json(audit_path), "feature_audit")
    total = _nonnegative_integer(
        _required(audit, "total", "feature_audit.total"),
        "feature_audit.total",
    )
    missing = _mapping(
        _required(audit, "missing", "feature_audit.missing"),
        "feature_audit.missing",
    )
    declared_missing: dict[str, int] = {}
    for field, value in missing.items():
        context = f"feature_audit.missing.{field}"
        count = _nonnegative_integer(value, context)
        if count > total:
            raise ValueError(
                f"{context}: count {count} exceeds feature_audit.total "
                f"{total}"
            )
        declared_missing[field] = count
    return audit, declared_missing


def validate_inputs(output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    output_dir = Path(output_dir)
    case_paths = _discover_case_paths(output_dir / "interim")
    numeric_baseline = _validate_numeric_baseline(
        output_dir / "reports" / "model_metrics.json"
    )
    audit, declared_missing = _load_feature_audit(
        output_dir / "reports" / "feature_audit.json"
    )
    total, by_quarter, by_split, actual_missing = _scan_case_files(
        case_paths,
        FEATURE_AUDIT_MISSING_FIELDS | set(declared_missing),
    )
    _validate_numeric_split_counts(numeric_baseline, by_split)
    _validate_feature_audit(
        audit,
        total,
        by_quarter,
        by_split,
        declared_missing,
        actual_missing,
    )
    return case_paths, numeric_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run final FAERS project experiments and deliverables"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--report", type=Path, default=ROOT / "最终报告.md")
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=ROOT / "demo" / "index.html",
    )
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument(
        "--epochs",
        type=int,
        default=pipeline.HASH_LOGISTIC_EPOCHS,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=pipeline.HASH_LOGISTIC_LEARNING_RATE,
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=pipeline.HASH_LOGISTIC_L2,
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    case_paths, numeric_baseline = validate_inputs(args.out)
    ablation_path = args.out / "ablations" / "ablation_metrics.json"
    weak_path = (
        args.out
        / "weak_supervision"
        / "weak_supervision_metrics.json"
    )

    final_experiments.run_ablation_experiments(
        case_paths,
        args.out / "ablations",
        numeric_baseline,
        chunk_size=args.chunk_size,
        n_features=args.n_features,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        random_state=args.random_state,
    )
    weak_supervision.run_weak_supervision(
        case_paths,
        args.out / "weak_supervision",
    )
    final_reporting.generate_final_report(args.out, args.report)
    dashboard.generate_dashboard(args.out, args.dashboard)

    print(f"[done] ablation metrics: {ablation_path}")
    print(f"[done] weak-supervision metrics: {weak_path}")
    print(f"[done] final report: {args.report}")
    print(f"[done] offline dashboard: {args.dashboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
