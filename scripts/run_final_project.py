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
FINAL_CASE_COLUMNS = {
    "split",
    "label_serious",
    "text_tokens",
    "safetyreportid",
    "receivedate",
    "quarter",
    "primarysourcecountry",
    "patientsex",
    "age_years",
    "drug_count",
    "suspect_drug_count",
    "reaction_count",
    "indication_count",
}
VALID_SPLITS = ("train", "valid", "test")


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
    paths = sorted(interim_dir.glob("cases_*.csv"))
    by_quarter: dict[str, list[Path]] = {
        quarter: []
        for quarter in EXPECTED_QUARTERS
    }
    extra: list[Path] = []
    for path in paths:
        matches = re.findall(r"2025q([1-4])", path.stem, flags=re.IGNORECASE)
        if len(matches) != 1:
            extra.append(path)
            continue
        quarter = f"2025Q{matches[0]}"
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
    if missing or duplicates or extra or len(paths) != len(EXPECTED_QUARTERS):
        details = [
            "Case CSV files must exactly cover 2025Q1, 2025Q2, 2025Q3, "
            f"and 2025Q4 in {interim_dir}"
        ]
        if duplicates:
            duplicate_text = ", ".join(
                f"{quarter} ({', '.join(path.name for path in quarter_paths)})"
                for quarter, quarter_paths in duplicates.items()
            )
            details.append(f"duplicate quarters: {duplicate_text}")
        if missing:
            details.append(f"missing quarters: {', '.join(missing)}")
        if extra:
            details.append(
                "extra case CSV files: "
                + ", ".join(path.name for path in extra)
            )
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
) -> tuple[int, dict[str, Counter], dict[str, Counter]]:
    total = 0
    by_quarter = {
        quarter: Counter(n=0, positive=0)
        for quarter in EXPECTED_QUARTERS
    }
    by_split = {
        split: Counter(n=0, positive=0)
        for split in VALID_SPLITS
    }

    for path, expected_quarter in zip(case_paths, EXPECTED_QUARTERS):
        file_rows = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(FINAL_CASE_COLUMNS - fieldnames)
            if missing:
                raise ValueError(f"{path}: missing columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                file_rows += 1
                split = (row.get("split") or "").strip()
                if split not in by_split:
                    raise ValueError(
                        f"{path}:{line_number}: unknown split {split!r}; "
                        f"expected train, valid, or test"
                    )
                label_text = (row.get("label_serious") or "").strip()
                if label_text not in ("0", "1"):
                    raise ValueError(
                        f"{path}:{line_number}: label_serious must be 0 or 1, "
                        f"got {label_text!r}"
                    )
                row_quarter = (row.get("quarter") or "").strip().upper()
                if row_quarter != expected_quarter:
                    raise ValueError(
                        f"{path}:{line_number}: row quarter {row_quarter!r} "
                        f"does not match file quarter {expected_quarter}"
                    )

                label = int(label_text)
                total += 1
                by_quarter[expected_quarter]["n"] += 1
                by_quarter[expected_quarter]["positive"] += label
                by_split[split]["n"] += 1
                by_split[split]["positive"] += label

        if file_rows == 0:
            raise ValueError(f"{path}: CSV must contain at least one data row")

    for split, counts in by_split.items():
        negatives = counts["n"] - counts["positive"]
        if counts["positive"] == 0 or negatives == 0:
            raise ValueError(
                f"Split {split} must contain both classes, got "
                f"0={negatives}, 1={counts['positive']}"
            )
    return total, by_quarter, by_split


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
    audit_path: Path,
    total: int,
    by_quarter: dict[str, Counter],
    by_split: dict[str, Counter],
) -> None:
    audit = _mapping(_load_strict_json(audit_path), "feature_audit")
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


def validate_inputs(output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    output_dir = Path(output_dir)
    case_paths = _discover_case_paths(output_dir / "interim")
    numeric_baseline = _validate_numeric_baseline(
        output_dir / "reports" / "model_metrics.json"
    )
    total, by_quarter, by_split = _scan_case_files(case_paths)
    _validate_feature_audit(
        output_dir / "reports" / "feature_audit.json",
        total,
        by_quarter,
        by_split,
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
