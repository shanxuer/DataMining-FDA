# DataMining-FDA Final Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible final-project pipeline that runs feature ablations and expanded weak supervision on existing FAERS case CSVs, then generates a data-driven Markdown report and offline HTML Dashboard.

**Architecture:** Keep the existing XML/ETL pipeline unchanged and add a thin final-project entry point over four focused modules. Reuse `run_faers_pipeline.py` for CSV iteration, hashing, logistic updates, threshold selection, metrics, metadata, and error-case selection; store real experiment artifacts under ignored `outputs*` directories and generate the tracked report and Demo from those artifacts.

**Tech Stack:** Python 3.9+, standard library, NumPy, `unittest`, native HTML/CSS/JavaScript.

---

## File Map

- Create `scripts/final_experiments.py`: ablation token policies, joint model training/evaluation, model persistence, atomic metrics JSON.
- Create `scripts/weak_supervision.py`: rule votes, majority aggregation, rule/split metrics, atomic metrics JSON.
- Create `scripts/final_reporting.py`: strict input validation and Markdown rendering from experiment JSON.
- Create `scripts/dashboard.py`: self-contained offline Dashboard rendering from experiment JSON.
- Create `scripts/run_final_project.py`: CLI validation and orchestration.
- Create `tests/test_final_experiments.py`: ablation policy and tiny end-to-end experiment tests.
- Create `tests/test_weak_supervision.py`: rule, voting, and covered-subset metric tests.
- Create `tests/test_final_outputs.py`: Markdown, Dashboard, and CLI input validation tests.
- Modify `readme.md`: final-project commands and artifacts.
- Generate `最终报告.md`: full-data final report.
- Generate `demo/index.html`: full-data offline Dashboard.

## Task 1: Implement Ablation Token Policies

**Files:**
- Create: `scripts/final_experiments.py`
- Create: `tests/test_final_experiments.py`

- [ ] **Step 1: Write failing policy tests**

Create `tests/test_final_experiments.py`:

```python
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import final_experiments as experiments


class AblationTokenPolicyTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "quarter": "2025Q1",
            "primarysourcecountry": "US",
            "patientsex": "2",
            "age_years": "70",
            "drug_count": "3",
            "reaction_count": "2",
            "indication_count": "1",
            "serious": "1",
            "seriousnessdeath": "1",
            "label_serious": "1",
            "text_tokens": (
                "drug:ABC indi:PAIN reac:SEPSIS "
                "reactionoutcome:5 route:048 actiondrug:1 drugchar:1"
            ),
        }

    def test_all_tokens_keeps_structured_and_text_families(self):
        text = experiments.build_ablation_text(self.row, "all_tokens")
        self.assertIn("quarter:2025Q1", text)
        self.assertIn("drug:ABC", text)
        self.assertIn("reac:SEPSIS", text)
        self.assertIn("reactionoutcome:5", text)

    def test_reaction_ablations_remove_only_requested_family(self):
        no_pt = experiments.build_ablation_text(self.row, "without_reaction_pt")
        self.assertNotIn("reac:SEPSIS", no_pt)
        self.assertIn("reactionoutcome:5", no_pt)
        self.assertIn("drug:ABC", no_pt)

        no_outcome = experiments.build_ablation_text(self.row, "without_reaction_outcome")
        self.assertIn("reac:SEPSIS", no_outcome)
        self.assertNotIn("reactionoutcome:5", no_outcome)
        self.assertIn("drug:ABC", no_outcome)

    def test_restricted_policies_keep_exact_feature_families(self):
        drug_indication = set(experiments.build_ablation_text(self.row, "drug_indication_only").split())
        self.assertEqual(drug_indication, {"drug:ABC", "indi:PAIN"})

        structured = experiments.build_ablation_text(self.row, "structured_only")
        self.assertIn("quarter:2025Q1", structured)
        self.assertIn("age_bin:senior", structured)
        self.assertNotIn("drug:ABC", structured)
        self.assertNotIn("reac:SEPSIS", structured)

    def test_no_policy_includes_target_fields(self):
        for name in experiments.ABLATION_CONFIGS:
            text = experiments.build_ablation_text(self.row, name).lower()
            self.assertNotIn("serious", text)
            self.assertNotIn("label_serious", text)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown ablation"):
            experiments.build_ablation_text(self.row, "missing")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the policy tests and verify RED**

Run:

```bash
python3 tests/test_final_experiments.py -v
```

Expected: import failure for missing `final_experiments`.

- [ ] **Step 3: Implement the policy layer**

Create `scripts/final_experiments.py` with these definitions:

```python
from __future__ import annotations

from typing import Any

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
```

- [ ] **Step 4: Run policy tests and verify GREEN**

Run:

```bash
python3 tests/test_final_experiments.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Run the existing leakage regression suite**

Run:

```bash
python3 -m unittest discover -s tests -p "test_faers_pipeline.py" -v
```

Expected: existing 11 tests pass.

- [ ] **Step 6: Commit the policy layer**

```bash
git add scripts/final_experiments.py tests/test_final_experiments.py
git commit -m "feat: define final feature ablations"
```

## Task 2: Train and Evaluate All Ablation Models

**Files:**
- Modify: `scripts/final_experiments.py`
- Modify: `tests/test_final_experiments.py`

- [ ] **Step 1: Add a failing tiny experiment test**

Append to `tests/test_final_experiments.py`:

```python
import csv
import json
import tempfile


class AblationExperimentTests(unittest.TestCase):
    def test_tiny_experiment_writes_models_metrics_and_error_cases(self):
        fieldnames = [
            "quarter",
            "split",
            "safetyreportid",
            "receivedate",
            "primarysourcecountry",
            "patientsex",
            "age_years",
            "drug_count",
            "reaction_count",
            "indication_count",
            "label_serious",
            "text_tokens",
        ]
        rows = []
        for split in ("train", "valid", "test"):
            rows.extend(
                [
                    {
                        "quarter": "2025Q1" if split == "train" else "2025Q4",
                        "split": split,
                        "safetyreportid": f"{split}-0",
                        "receivedate": "20251001",
                        "primarysourcecountry": "US",
                        "patientsex": "1",
                        "age_years": "30",
                        "drug_count": "1",
                        "reaction_count": "1",
                        "indication_count": "1",
                        "label_serious": "0",
                        "text_tokens": "drug:SAFE indi:PAIN reac:MILD reactionoutcome:1",
                    },
                    {
                        "quarter": "2025Q1" if split == "train" else "2025Q4",
                        "split": split,
                        "safetyreportid": f"{split}-1",
                        "receivedate": "20251002",
                        "primarysourcecountry": "US",
                        "patientsex": "2",
                        "age_years": "70",
                        "drug_count": "5",
                        "reaction_count": "2",
                        "indication_count": "1",
                        "label_serious": "1",
                        "text_tokens": "drug:RISK indi:CANCER reac:SEPSIS reactionoutcome:5",
                    },
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "cases.csv"
            with case_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            baseline = {
                "model_path": "numeric_logistic.pkl",
                "train_rows": 2,
                "threshold_from_valid": 0.5,
                "split_metrics": {
                    split: {"n": 2, "auroc": 0.5, "auprc": 0.5, "f1": 0.5}
                    for split in ("train", "valid", "test")
                },
                "error_cases": {"test": {"false_positive": [], "false_negative": []}},
            }
            result = experiments.run_ablation_experiments(
                [case_path],
                root / "ablations",
                baseline,
                chunk_size=2,
                n_features=64,
                epochs=8,
                learning_rate=0.3,
                l2=0.0,
                random_state=42,
            )

            self.assertEqual(
                set(result["experiments"]),
                set(experiments.ABLATION_CONFIGS) | {"numeric_logistic"},
            )
            self.assertIn("test", result["experiments"]["all_tokens"]["split_metrics"])
            self.assertIn("error_cases", result["experiments"]["all_tokens"])
            self.assertTrue((root / "ablations" / "models" / "all_tokens.pkl").exists())
            metrics_path = root / "ablations" / "ablation_metrics.json"
            self.assertEqual(json.loads(metrics_path.read_text())["metadata"]["n_features"], 64)
```

- [ ] **Step 2: Run the tiny experiment test and verify RED**

Run:

```bash
python3 tests/test_final_experiments.py -v
```

Expected: failure because `run_ablation_experiments` does not exist.

- [ ] **Step 3: Add chunk construction and joint training**

Extend `scripts/final_experiments.py` with imports and helpers:

```python
import json
import math
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


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
    counts = pipeline.split_label_counts(paths)["train"]
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"Training split must contain both classes, got {dict(counts)}")
    class_weights = pipeline.class_weights_from_counts(counts)
    models = {
        name: _new_model(name, n_features, random_state)
        for name in ABLATION_CONFIGS
    }
    seen = 0
    for epoch in range(epochs):
        epoch_seen = 0
        step = learning_rate / math.sqrt(epoch + 1)
        for texts, labels, _ in iter_ablation_chunks(paths, "train", chunk_size):
            for name, model in models.items():
                pipeline.update_hash_logistic_model(
                    model,
                    texts[name],
                    labels,
                    step,
                    l2,
                    class_weights,
                )
            epoch_seen += len(labels)
        seen = max(seen, epoch_seen)

    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        _atomic_write_pickle(model_dir / f"{name}.pkl", model)
    return models, seen
```

- [ ] **Step 4: Add joint prediction and result assembly**

Extend `scripts/final_experiments.py`:

```python
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
    for texts, chunk_labels, metadata in iter_ablation_chunks(paths, split, chunk_size):
        labels.extend(int(value) for value in chunk_labels)
        if collect_rows:
            rows.extend(metadata)
        for name, model in models.items():
            chunk_probabilities = pipeline.predict_hash_logistic_examples(model, texts[name])
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
    predictions: dict[str, tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, str]]]] = {}
    for split in ("train", "valid", "test"):
        predictions[split] = predict_ablation_models(
            paths,
            split,
            models,
            chunk_size,
            collect_rows=split in {"valid", "test"},
        )

    valid_labels, valid_probs, _ = predictions["valid"]
    thresholds = {
        name: pipeline.best_threshold(valid_labels, valid_probs[name])
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
            labels, split_probs, rows = predictions[split]
            probabilities = split_probs[name]
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
```

- [ ] **Step 5: Run experiment tests and verify GREEN**

Run:

```bash
python3 tests/test_final_experiments.py -v
```

Expected: all policy and tiny experiment tests pass.

- [ ] **Step 6: Run all existing tests**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit ablation training**

```bash
git add scripts/final_experiments.py tests/test_final_experiments.py
git commit -m "feat: run reproducible ablation experiments"
```

## Task 3: Expand Weak Supervision and Majority Voting

**Files:**
- Create: `scripts/weak_supervision.py`
- Create: `tests/test_weak_supervision.py`

- [ ] **Step 1: Write failing rule and voting tests**

Create `tests/test_weak_supervision.py`:

```python
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import weak_supervision as weak


class WeakSupervisionTests(unittest.TestCase):
    def test_majority_vote_handles_abstain_and_tie(self):
        self.assertEqual(weak.majority_vote([1, 1, 0, None]), (1, False))
        self.assertEqual(weak.majority_vote([0, 0, 1, None]), (0, False))
        self.assertEqual(weak.majority_vote([1, 0, None]), (None, True))
        self.assertEqual(weak.majority_vote([None, None]), (None, False))

    def test_device_negative_rule_abstains_when_high_risk_reaction_exists(self):
        device_only = {
            "age_years": "35",
            "drug_count": "1",
            "suspect_drug_count": "1",
            "reaction_count": "1",
            "text_tokens": "reac:DEVICE_MALFUNCTION reac:NO_ADVERSE_EVENT",
        }
        with_sepsis = dict(device_only)
        with_sepsis["text_tokens"] += " reac:SEPSIS"

        self.assertEqual(weak.rule_votes(device_only)["device_product_issue"], 0)
        self.assertIsNone(weak.rule_votes(with_sepsis)["device_product_issue"])
        self.assertEqual(weak.rule_votes(with_sepsis)["high_risk_reaction"], 1)

    def test_rule_votes_cover_outcome_and_extreme_polypharmacy(self):
        row = {
            "age_years": "72",
            "drug_count": "35",
            "suspect_drug_count": "3",
            "reaction_count": "2",
            "text_tokens": "reac:SHOCK reactionoutcome:5",
        }
        votes = weak.rule_votes(row)
        self.assertEqual(votes["serious_reaction_outcome"], 1)
        self.assertEqual(votes["extreme_polypharmacy_30plus"], 1)
        self.assertEqual(votes["senior_with_multiple_suspect_drugs"], 1)

    def test_summary_metrics_use_only_non_abstain_non_conflict_rows(self):
        rows = [
            {
                "split": "test",
                "label_serious": "1",
                "age_years": "70",
                "drug_count": "1",
                "suspect_drug_count": "1",
                "reaction_count": "1",
                "text_tokens": "reac:SEPSIS",
            },
            {
                "split": "test",
                "label_serious": "0",
                "age_years": "30",
                "drug_count": "1",
                "suspect_drug_count": "1",
                "reaction_count": "1",
                "text_tokens": "reac:DEVICE_MALFUNCTION reac:NO_ADVERSE_EVENT",
            },
            {
                "split": "test",
                "label_serious": "1",
                "age_years": "",
                "drug_count": "3",
                "suspect_drug_count": "1",
                "reaction_count": "2",
                "text_tokens": "drug:UNKNOWN",
            },
        ]
        result = weak.summarize_rows(rows)
        test = result["splits"]["test"]
        self.assertEqual(test["total"], 3)
        self.assertEqual(test["covered"], 2)
        self.assertEqual(test["voted"], 2)
        self.assertEqual(test["accuracy"], 1.0)
        self.assertEqual(test["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run weak-supervision tests and verify RED**

Run:

```bash
python3 tests/test_weak_supervision.py -v
```

Expected: import failure for missing `weak_supervision`.

- [ ] **Step 3: Implement rule votes and majority aggregation**

Create `scripts/weak_supervision.py`:

```python
from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import run_faers_pipeline as pipeline


DEATH_TERMS = ("reac:DEATH", "reac:FATAL", "reac:CARDIAC_ARREST")
HIGH_RISK_TERMS = (
    "reac:SEPSIS",
    "reac:SEPTIC_SHOCK",
    "reac:ACUTE_RESPIRATORY_FAILURE",
    "reac:RESPIRATORY_FAILURE",
    "reac:SHOCK",
)
DEVICE_TERMS = (
    "reac:DEVICE_DEFECTIVE",
    "reac:DEVICE_ISSUE",
    "reac:DEVICE_MALFUNCTION",
    "reac:PRODUCT_QUALITY_ISSUE",
    "reac:PRODUCT_COMMUNICATION_ISSUE",
    "reac:NO_ADVERSE_EVENT",
)
SERIOUS_OUTCOME_TOKENS = ("reactionoutcome:3", "reactionoutcome:4", "reactionoutcome:5")


def _contains_any(tokens: set[str], candidates: tuple[str, ...]) -> bool:
    return any(candidate in tokens for candidate in candidates)


def rule_votes(row: dict[str, str]) -> dict[str, int | None]:
    tokens = set(row.get("text_tokens", "").split())
    age = pipeline.parse_float(row.get("age_years"))
    drug_count = pipeline.parse_float(row.get("drug_count")) or 0.0
    suspect_count = pipeline.parse_float(row.get("suspect_drug_count")) or 0.0
    reaction_count = pipeline.parse_float(row.get("reaction_count")) or 0.0
    high_risk = _contains_any(tokens, DEATH_TERMS + HIGH_RISK_TERMS)
    device_issue = _contains_any(tokens, DEVICE_TERMS)
    return {
        "death_or_fatal_reaction_term": 1 if _contains_any(tokens, DEATH_TERMS) else None,
        "high_polypharmacy_10plus": 1 if drug_count >= 10 else None,
        "senior_with_multiple_suspect_drugs": (
            1 if age is not None and age >= 65 and suspect_count >= 2 else None
        ),
        "low_complexity_younger_case": (
            0
            if age is not None and age < 50 and drug_count <= 2 and reaction_count <= 1
            else None
        ),
        "high_risk_reaction": 1 if _contains_any(tokens, HIGH_RISK_TERMS) else None,
        "serious_reaction_outcome": (
            1 if _contains_any(tokens, SERIOUS_OUTCOME_TOKENS) else None
        ),
        "extreme_polypharmacy_30plus": 1 if drug_count >= 30 else None,
        "device_product_issue": 0 if device_issue and not high_risk else None,
    }


def majority_vote(votes: Iterable[int | None]) -> tuple[int | None, bool]:
    active = [int(vote) for vote in votes if vote is not None]
    if not active:
        return None, False
    positives = sum(active)
    negatives = len(active) - positives
    if positives == negatives:
        return None, True
    return (1 if positives > negatives else 0), False


def _new_bucket() -> dict[str, Any]:
    return {
        "total": 0,
        "covered": 0,
        "conflicts": 0,
        "voted": 0,
        "correct": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "rules": defaultdict(
            lambda: {
                "fires": 0,
                "positive_votes": 0,
                "negative_votes": 0,
                "positive_labels": 0,
                "correct": 0,
            }
        ),
    }


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    voted = bucket["voted"]
    tp = bucket["tp"]
    fp = bucket["fp"]
    fn = bucket["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {
        "total": bucket["total"],
        "covered": bucket["covered"],
        "coverage_rate": bucket["covered"] / bucket["total"] if bucket["total"] else 0.0,
        "conflicts": bucket["conflicts"],
        "conflict_rate": (
            bucket["conflicts"] / bucket["covered"] if bucket["covered"] else 0.0
        ),
        "voted": voted,
        "accuracy": bucket["correct"] / voted if voted else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "rules": {},
    }
    for name, item in sorted(bucket["rules"].items()):
        fires = item["fires"]
        result["rules"][name] = {
            **dict(item),
            "coverage_rate": fires / bucket["total"] if bucket["total"] else 0.0,
            "positive_label_rate": item["positive_labels"] / fires if fires else 0.0,
            "accuracy": item["correct"] / fires if fires else 0.0,
        }
    return result


def summarize_rows(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    buckets = {"overall": _new_bucket()}
    split_buckets: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    for row in rows:
        label = int(row["label_serious"])
        votes = rule_votes(row)
        split = row.get("split") or "unknown"
        targets = (buckets["overall"], split_buckets[split])
        for bucket in targets:
            bucket["total"] += 1
            for name, vote in votes.items():
                if vote is None:
                    continue
                rule = bucket["rules"][name]
                rule["fires"] += 1
                rule["positive_votes"] += int(vote == 1)
                rule["negative_votes"] += int(vote == 0)
                rule["positive_labels"] += label
                rule["correct"] += int(vote == label)
            active = [vote for vote in votes.values() if vote is not None]
            if active:
                bucket["covered"] += 1
            prediction, conflict = majority_vote(active)
            bucket["conflicts"] += int(conflict)
            if prediction is None:
                continue
            bucket["voted"] += 1
            bucket["correct"] += int(prediction == label)
            bucket["tp"] += int(prediction == 1 and label == 1)
            bucket["fp"] += int(prediction == 1 and label == 0)
            bucket["fn"] += int(prediction == 0 and label == 1)
    return {
        "overall": _finalize_bucket(buckets["overall"]),
        "splits": {
            split: _finalize_bucket(bucket)
            for split, bucket in sorted(split_buckets.items())
        },
    }
```

- [ ] **Step 4: Add atomic output entry point**

Append to `scripts/weak_supervision.py`:

```python
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


def run_weak_supervision(paths: list[Path], output_dir: Path) -> dict[str, Any]:
    result = summarize_rows(pipeline.iter_feature_rows(paths))
    result["metadata"] = {
        "rules": list(rule_votes({}).keys()),
        "note": "Majority-vote weak labels are for audit only and are not training labels.",
    }
    _atomic_write_json(output_dir / "weak_supervision_metrics.json", result)
    return result
```

- [ ] **Step 5: Run weak-supervision tests and verify GREEN**

Run:

```bash
python3 tests/test_weak_supervision.py -v
```

Expected: 4 tests pass.

- [ ] **Step 6: Run all tests**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit weak supervision**

```bash
git add scripts/weak_supervision.py tests/test_weak_supervision.py
git commit -m "feat: expand weak supervision audit"
```

## Task 4: Generate the Data-Driven Markdown Final Report

**Files:**
- Create: `scripts/final_reporting.py`
- Create: `tests/test_final_outputs.py`

- [ ] **Step 1: Write failing report tests**

Create `tests/test_final_outputs.py`:

```python
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import final_reporting


def sample_ablation():
    payload = {
        "metadata": {"split_policy": "time split"},
        "experiments": {
            "all_tokens": {
                "description": "all",
                "split_metrics": {
                    "test": {
                        "n": 200,
                        "auroc": 0.91234,
                        "auprc": 0.90123,
                        "precision": 0.88,
                        "recall": 0.87,
                        "f1": 0.875,
                        "recall_at_top_5pct": 0.08,
                        "hit_rate_top_5pct": 0.98,
                    }
                },
                "error_cases": {
                    "test": {
                        "false_positive": [
                            {
                                "safetyreportid": "FP-1",
                                "predicted_probability": 0.99,
                                "true_label": 0,
                                "tokens": "reac:SEPSIS",
                            }
                        ],
                        "false_negative": [],
                    }
                },
                "strata": {
                    "test": {
                        "sex": {
                            "1": {
                                "n": 100,
                                "auroc": 0.9,
                                "auprc": 0.89,
                                "f1": 0.86,
                            }
                        }
                    }
                },
            },
            "without_reaction_pt": {
                "description": "no reaction",
                "split_metrics": {
                    "test": {
                        "n": 200,
                        "auroc": 0.71234,
                        "auprc": 0.70123,
                        "precision": 0.68,
                        "recall": 0.67,
                        "f1": 0.675,
                        "recall_at_top_5pct": 0.06,
                        "hit_rate_top_5pct": 0.78,
                    }
                },
                "error_cases": {"test": {"false_positive": [], "false_negative": []}},
            },
            "numeric_logistic": {
                "description": "numeric",
                "split_metrics": {
                    "test": {
                        "n": 200,
                        "auroc": 0.61234,
                        "auprc": 0.60123,
                        "precision": 0.58,
                        "recall": 0.57,
                        "f1": 0.575,
                        "recall_at_top_5pct": 0.05,
                        "hit_rate_top_5pct": 0.68,
                    }
                },
                "error_cases": {"test": {"false_positive": [], "false_negative": []}},
            },
        },
    }
    source = payload["experiments"]["without_reaction_pt"]
    for name, auroc in (
        ("without_reaction_outcome", 0.81234),
        ("drug_indication_only", 0.76234),
        ("structured_only", 0.66234),
    ):
        metrics = dict(source["split_metrics"]["test"])
        metrics["auroc"] = auroc
        payload["experiments"][name] = {
            "description": name,
            "split_metrics": {"test": metrics},
            "error_cases": {"test": {"false_positive": [], "false_negative": []}},
            "strata": {"test": {}},
        }
    return payload


def sample_weak():
    return {
        "overall": {
            "total": 400,
            "covered": 120,
            "coverage_rate": 0.3,
            "conflicts": 10,
            "conflict_rate": 0.0833,
            "voted": 110,
            "accuracy": 0.8,
            "precision": 0.82,
            "recall": 0.78,
            "f1": 0.7995,
            "rules": {
                "high_risk_reaction": {
                    "fires": 50,
                    "coverage_rate": 0.125,
                    "positive_label_rate": 0.9,
                    "accuracy": 0.9,
                }
            },
        },
        "splits": {
            "test": {
                "total": 100,
                "covered": 30,
                "coverage_rate": 0.3,
                "conflicts": 2,
                "conflict_rate": 0.0667,
                "voted": 28,
                "accuracy": 0.82,
                "precision": 0.84,
                "recall": 0.8,
                "f1": 0.8195,
                "rules": {},
            }
        },
    }


class FinalReportTests(unittest.TestCase):
    def test_report_uses_input_metrics_and_required_sections(self):
        audit = {
            "total": 400,
            "by_quarter": {"2025Q1": {"n": 100, "positive": 60}},
            "missing": {"age_years": 40},
        }
        text = final_reporting.render_final_report(sample_ablation(), sample_weak(), audit)
        self.assertIn("# 数据挖掘课程项目最终报告", text)
        self.assertIn("0.9123", text)
        self.assertIn("0.7123", text)
        self.assertIn("30.00%", text)
        self.assertIn("FP-1", text)
        self.assertIn("## 完整复现方式", text)

    def test_report_rejects_missing_required_experiment(self):
        ablation = sample_ablation()
        del ablation["experiments"]["all_tokens"]
        with self.assertRaisesRegex(ValueError, "all_tokens"):
            final_reporting.render_final_report(ablation, sample_weak(), {"total": 1})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run report tests and verify RED**

Run:

```bash
python3 tests/test_final_outputs.py -v
```

Expected: import failure for missing `final_reporting`.

- [ ] **Step 3: Implement strict report rendering**

Create `scripts/final_reporting.py` with:

```python
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


EXPERIMENT_ORDER = [
    "all_tokens",
    "without_reaction_pt",
    "without_reaction_outcome",
    "drug_indication_only",
    "structured_only",
    "numeric_logistic",
]


def _metric(value: Any) -> str:
    if value is None:
        raise ValueError("Required metric is missing")
    return f"{float(value):.4f}"


def _percent(value: Any) -> str:
    if value is None:
        raise ValueError("Required percentage is missing")
    return f"{100.0 * float(value):.2f}%"


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required {context}: {key}")
    return mapping[key]


def render_final_report(
    ablation: dict[str, Any],
    weak: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    experiments = _required(ablation, "experiments", "ablation field")
    for name in EXPERIMENT_ORDER:
        _required(experiments, name, "experiment")
    total = int(_required(audit, "total", "audit field"))
    all_test = _required(
        _required(experiments["all_tokens"], "split_metrics", "all_tokens field"),
        "test",
        "all_tokens split",
    )
    no_reaction_test = _required(
        experiments["without_reaction_pt"]["split_metrics"],
        "test",
        "without_reaction_pt split",
    )
    no_outcome_test = _required(
        experiments["without_reaction_outcome"]["split_metrics"],
        "test",
        "without_reaction_outcome split",
    )
    weak_overall = _required(weak, "overall", "weak supervision field")
    weak_test = _required(
        _required(weak, "splits", "weak supervision field"),
        "test",
        "weak supervision split",
    )

    lines = [
        "# 数据挖掘课程项目最终报告",
        "",
        "## 项目摘要",
        "",
        f"本项目基于 2025 年 FAERS XML 构建了 {total:,} 条病例级样本，完成资料治理、"
        "时间切分、数值逻辑回归、稳定哈希 token 逻辑回归、特征族消融和弱监督审计。",
        "",
        "## 数据来源与治理",
        "",
        "- 数据范围：2025Q1-Q4 FDA FAERS XML。",
        "- 训练集：2025Q1-Q3；验证集和测试集：2025Q4 按接收日期前后 50% 划分。",
        "- 标签字段仅用于生成 `label_serious`，不进入模型输入。",
        "",
        "| 季度 | 样本数 | 重症数 | 重症率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for quarter, item in sorted(audit.get("by_quarter", {}).items()):
        rate = item["positive"] / item["n"] if item["n"] else 0.0
        lines.append(
            f"| {quarter} | {item['n']:,} | {item['positive']:,} | {_percent(rate)} |"
        )
    lines.extend(
        [
        "",
        "### 关键字段缺失",
        "",
        "| 字段 | 缺失数 | 缺失率 |",
        "| --- | ---: | ---: |",
        ]
    )
    for field, count in sorted(audit.get("missing", {}).items()):
        lines.append(f"| `{field}` | {count:,} | {_percent(count / total)} |")
    lines.extend(
        [
        "",
        "## 方法",
        "",
        "- 数值基线：标准化数值聚合特征的 NumPy 逻辑回归。",
        "- 主模型：固定维度稳定哈希 token 的在线 NumPy 逻辑回归。",
        "- 阈值：仅在验证集选择，测试集只用于最终评估。",
        "",
        "## 消融实验结果",
        "",
        "| 实验 | AUROC | AUPRC | Precision | Recall | F1 | Recall@Top5% | Hit Rate@Top5% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in EXPERIMENT_ORDER:
        test = experiments[name]["split_metrics"]["test"]
        lines.append(
            f"| `{name}` | {_metric(test['auroc'])} | {_metric(test['auprc'])} | "
            f"{_metric(test['precision'])} | {_metric(test['recall'])} | "
            f"{_metric(test['f1'])} | {_metric(test['recall_at_top_5pct'])} | "
            f"{_percent(test['hit_rate_top_5pct'])} |"
        )
    reaction_drop = float(all_test["auroc"]) - float(no_reaction_test["auroc"])
    outcome_drop = float(all_test["auroc"]) - float(no_outcome_test["auroc"])
    lines.extend(
        [
            "",
            "## 语义捷径诊断",
            "",
            f"移除反应 PT 后测试集 AUROC 从 {_metric(all_test['auroc'])} 降至 "
            f"{_metric(no_reaction_test['auroc'])}，下降 {reaction_drop:.4f}；移除反应结果后"
            f" AUROC 为 {_metric(no_outcome_test['auroc'])}，下降 {outcome_drop:.4f}。这些变化"
            "用于量化反应语义对高指标的贡献，并提示模型可能部分依赖与重症标签高度同源的输入。",
            "",
            "## 弱监督扩展",
            "",
            f"- 总体规则覆盖率：{_percent(weak_overall['coverage_rate'])}",
            f"- 总体冲突率：{_percent(weak_overall['conflict_rate'])}",
            f"- 测试集非弃权弱标签准确率：{_percent(weak_test['accuracy'])}",
            f"- 测试集弱标签 F1：{_metric(weak_test['f1'])}",
            "",
            "| 规则 | 触发数 | 覆盖率 | 命中样本重症率 | 规则准确率 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, item in sorted(weak_overall["rules"].items()):
        lines.append(
            f"| `{name}` | {item['fires']:,} | {_percent(item['coverage_rate'])} | "
            f"{_percent(item['positive_label_rate'])} | {_percent(item['accuracy'])} |"
        )

    lines.extend(["", "## 失败案例", ""])
    error_cases = experiments["all_tokens"].get("error_cases", {}).get("test", {})
    lines.extend(
        [
            "| 类型 | safetyreportid | 概率 | 真实标签 | token 摘要 |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for error_type in ("false_positive", "false_negative"):
        for case in error_cases.get(error_type, []):
            tokens = str(case.get("tokens", "")).replace("|", "\\|")
            lines.append(
                f"| {error_type} | {case.get('safetyreportid', '')} | "
                f"{float(case.get('predicted_probability', 0.0)):.4f} | "
                f"{case.get('true_label', '')} | {tokens} |"
            )

    lines.extend(
        [
            "",
            "## 分层误差分析",
            "",
            "| 分层字段 | 分组 | 样本数 | AUROC | AUPRC | F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    strata = experiments["all_tokens"].get("strata", {}).get("test", {})
    for group_name, groups in sorted(strata.items()):
        for value, metrics in sorted(groups.items()):
            lines.append(
                f"| {group_name} | {value} | {metrics['n']:,} | "
                f"{_metric(metrics['auroc'])} | {_metric(metrics['auprc'])} | "
                f"{_metric(metrics['f1'])} |"
            )

    lines.extend(
        [
            "",
            "## 局限性与结论",
            "",
            "- 反应术语可能构成目标语义捷径，完整模型指标不能直接解释为因果药物风险。",
            "- 高缺失人口学字段限制剂量、体重和年龄相关解释。",
            "- 多数投票只覆盖规则触发子集，不与全测试集模型指标作无条件横向比较。",
            "- 当前药名正规化未接入 RxNorm，商品名、错拼和复方映射仍可改进。",
            "",
            "## 完整复现方式",
            "",
            "从原始 XML 完整运行：",
            "",
            "```bash",
            "python3 scripts/run_faers_pipeline.py --data data --out outputs --mode full",
            "python3 scripts/run_final_project.py --out outputs",
            "```",
            "",
            "已有病例级 CSV 时只运行终期实验：",
            "",
            "```bash",
            "python3 scripts/run_final_project.py --out outputs",
            "```",
            "",
            "快速验证：",
            "",
            "```bash",
            "python3 scripts/run_final_project.py --out outputs_sample",
            "```",
            "",
            "## AI 工具辅助使用声明",
            "",
            "本项目使用 ChatGPT/Codex 辅助代码实现、测试设计、实验汇总和报告排版；"
            "所有结果均来自本地流水线产物，并通过单元测试和端到端运行核验。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def generate_final_report(output_dir: Path, destination: Path) -> Path:
    ablation = json.loads(
        (output_dir / "ablations" / "ablation_metrics.json").read_text(encoding="utf-8")
    )
    weak = json.loads(
        (output_dir / "weak_supervision" / "weak_supervision_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (output_dir / "reports" / "feature_audit.json").read_text(encoding="utf-8")
    )
    _atomic_write_text(destination, render_final_report(ablation, weak, audit))
    return destination
```

- [ ] **Step 4: Run report tests and verify GREEN**

Run:

```bash
python3 tests/test_final_outputs.py -v
```

Expected: 2 report tests pass.

- [ ] **Step 5: Commit report generation**

```bash
git add scripts/final_reporting.py tests/test_final_outputs.py
git commit -m "feat: generate final markdown report"
```

## Task 5: Generate the Offline HTML Dashboard

**Files:**
- Create: `scripts/dashboard.py`
- Modify: `tests/test_final_outputs.py`

- [ ] **Step 1: Add failing Dashboard tests**

Append imports and tests to `tests/test_final_outputs.py`:

```python
import dashboard


class DashboardTests(unittest.TestCase):
    def test_dashboard_is_self_contained_and_has_filters(self):
        audit = {
            "total": 400,
            "by_quarter": {"2025Q1": {"n": 100, "positive": 60}},
            "missing": {"age_years": 40},
        }
        html = dashboard.render_dashboard(sample_ablation(), sample_weak(), audit)
        self.assertIn("<!doctype html>", html.lower())
        self.assertIn('id="model-filter"', html)
        self.assertIn('id="error-filter"', html)
        self.assertIn("FP-1", html)
        self.assertIn("0.9123", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)

    def test_dashboard_escapes_embedded_script_closing_tag(self):
        ablation = sample_ablation()
        ablation["experiments"]["all_tokens"]["error_cases"]["test"]["false_positive"][0][
            "tokens"
        ] = "</script><script>alert(1)</script>"
        html = dashboard.render_dashboard(ablation, sample_weak(), {"total": 400})
        self.assertNotIn("</script><script>alert(1)</script>", html)
        self.assertIn("<\\/script>", html)
```

- [ ] **Step 2: Run Dashboard tests and verify RED**

Run:

```bash
python3 tests/test_final_outputs.py -v
```

Expected: import failure for missing `dashboard`.

- [ ] **Step 3: Implement self-contained Dashboard data shaping**

Create `scripts/dashboard.py` with:

```python
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


EXPERIMENT_ORDER = [
    "all_tokens",
    "without_reaction_pt",
    "without_reaction_outcome",
    "drug_indication_only",
    "structured_only",
    "numeric_logistic",
]


def _dashboard_data(
    ablation: dict[str, Any],
    weak: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    experiments = ablation["experiments"]
    quarter_rows = audit.get("by_quarter", {}).values()
    total_positive = sum(int(item["positive"]) for item in quarter_rows)
    serious_rate = total_positive / audit["total"] if audit["total"] else 0.0
    models = []
    errors = []
    for name in EXPERIMENT_ORDER:
        item = experiments[name]
        test = item["split_metrics"]["test"]
        models.append(
            {
                "name": name,
                "description": item.get("description", ""),
                "auroc": test["auroc"],
                "auprc": test["auprc"],
                "f1": test["f1"],
            }
        )
        for error_type in ("false_positive", "false_negative"):
            for case in item.get("error_cases", {}).get("test", {}).get(error_type, []):
                errors.append({"model": name, "type": error_type, **case})
    weak_overall = weak["overall"]
    return {
        "total": audit["total"],
        "serious_rate": serious_rate,
        "models": models,
        "best": max(models, key=lambda item: item["auroc"]),
        "weak": {
            "coverage_rate": weak_overall["coverage_rate"],
            "conflict_rate": weak_overall["conflict_rate"],
            "accuracy": weak_overall["accuracy"],
            "rules": [
                {"name": name, **item}
                for name, item in sorted(weak_overall["rules"].items())
            ],
        },
        "errors": errors,
    }


def _safe_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
```

- [ ] **Step 4: Implement the complete HTML renderer**

Append `render_dashboard` to `scripts/dashboard.py`. Use this page structure and IDs exactly so the tests and browser verification have stable selectors:

```python
def render_dashboard(
    ablation: dict[str, Any],
    weak: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    data = _safe_json(_dashboard_data(ablation, weak, audit))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FAERS 药物风险终期实验 Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5d6975;
      --line: #d9dee3;
      --paper: #ffffff;
      --band: #f3f5f6;
      --green: #147d64;
      --red: #b5483d;
      --blue: #2e6f9e;
      --gold: #b17814;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.5;
    }}
    header {{
      padding: 28px max(20px, calc((100vw - 1180px) / 2));
      border-bottom: 1px solid var(--line);
      background: #16252d;
      color: #fff;
    }}
    h1 {{ margin: 0; font-size: 28px; letter-spacing: 0; }}
    header p {{ max-width: 760px; margin: 8px 0 0; color: #d7e0e4; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px 20px 48px; }}
    h2 {{ margin: 30px 0 12px; font-size: 20px; letter-spacing: 0; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      min-width: 0;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--paper);
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 24px; }}
    .panel {{ border-top: 1px solid var(--line); padding-top: 16px; }}
    .model-row {{
      display: grid;
      grid-template-columns: minmax(170px, 1.4fr) repeat(3, minmax(100px, 1fr));
      gap: 12px;
      align-items: center;
      min-height: 54px;
      border-bottom: 1px solid var(--line);
    }}
    .bar-track {{ height: 10px; background: #e8ecef; overflow: hidden; }}
    .bar {{ height: 100%; background: var(--blue); }}
    .rule-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 20px;
    }}
    .rule {{
      display: grid;
      grid-template-columns: minmax(170px, 1fr) 80px 80px;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px solid var(--line);
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 12px;
    }}
    label {{ color: var(--muted); font-size: 13px; }}
    select {{
      min-height: 36px;
      margin-left: 6px;
      padding: 0 30px 0 10px;
      border: 1px solid #aeb7bf;
      border-radius: 4px;
      background: #fff;
      color: var(--ink);
    }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); }}
    table {{ width: 100%; min-width: 850px; border-collapse: collapse; }}
    th, td {{ padding: 10px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ background: var(--band); font-size: 13px; }}
    td.tokens {{ max-width: 390px; overflow-wrap: anywhere; }}
    .note {{
      padding: 14px 16px;
      border-left: 4px solid var(--gold);
      background: #fff8e8;
    }}
    @media (max-width: 760px) {{
      header {{ padding: 22px 16px; }}
      h1 {{ font-size: 23px; }}
      main {{ padding: 18px 14px 36px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .metric strong {{ font-size: 20px; }}
      .model-row {{ grid-template-columns: minmax(140px, 1.3fr) repeat(3, 80px); overflow-x: auto; }}
      .rule-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>FAERS 药物风险终期实验</h1>
    <p>2025Q1-Q4 病例级资料治理、稳定哈希模型消融与弱监督规则审计。</p>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <h2>模型与消融对比</h2>
    <section class="panel" id="models"></section>
    <h2>弱监督审计</h2>
    <section class="summary" id="weak-summary"></section>
    <section class="rule-grid" id="rules"></section>
    <h2>失败案例</h2>
    <div class="controls">
      <label>模型<select id="model-filter"></select></label>
      <label>错误类型<select id="error-filter">
        <option value="all">全部</option>
        <option value="false_positive">假阳性</option>
        <option value="false_negative">假阴性</option>
      </select></label>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>模型</th><th>类型</th><th>病例 ID</th><th>概率</th><th>真实标签</th><th>Token 摘要</th></tr></thead>
        <tbody id="errors"></tbody>
      </table>
    </div>
    <h2>结果解释边界</h2>
    <div class="note">反应术语可能与重症标签存在语义同源性；弱监督指标只针对规则覆盖且非冲突的子集；高缺失人口学字段不支持过度医学解释。</div>
  </main>
  <script id="dashboard-data" type="application/json">{data}</script>
  <script>
    const data = JSON.parse(document.getElementById("dashboard-data").textContent);
    const pct = value => `${{(100 * Number(value)).toFixed(2)}}%`;
    const score = value => Number(value).toFixed(4);
    document.getElementById("summary").innerHTML = [
      ["病例数", Number(data.total).toLocaleString()],
      ["重症率", pct(data.serious_rate)],
      ["最佳模型", data.best.name],
      ["测试 AUROC", score(data.best.auroc)],
      ["测试 AUPRC", score(data.best.auprc)]
    ].map(([label, value]) => `<div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>`).join("");
    document.getElementById("models").innerHTML =
      `<div class="model-row"><strong>实验</strong><strong>AUROC</strong><strong>AUPRC</strong><strong>F1</strong></div>` +
      data.models.map(model => `<div class="model-row"><div><strong>${{model.name}}</strong><div class="bar-track"><div class="bar" style="width:${{100 * model.auroc}}%"></div></div></div><span>${{score(model.auroc)}}</span><span>${{score(model.auprc)}}</span><span>${{score(model.f1)}}</span></div>`).join("");
    document.getElementById("weak-summary").innerHTML = [
      ["规则覆盖率", pct(data.weak.coverage_rate)],
      ["冲突率", pct(data.weak.conflict_rate)],
      ["非弃权准确率", pct(data.weak.accuracy)]
    ].map(([label, value]) => `<div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>`).join("");
    document.getElementById("rules").innerHTML = data.weak.rules.map(rule =>
      `<div class="rule"><strong>${{rule.name}}</strong><span>${{pct(rule.coverage_rate)}}</span><span>${{pct(rule.accuracy)}}</span></div>`
    ).join("");
    const modelFilter = document.getElementById("model-filter");
    modelFilter.innerHTML = `<option value="all">全部模型</option>` +
      data.models.map(model => `<option value="${{model.name}}">${{model.name}}</option>`).join("");
    const errorFilter = document.getElementById("error-filter");
    const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[char]));
    function renderErrors() {{
      const rows = data.errors.filter(item =>
        (modelFilter.value === "all" || item.model === modelFilter.value) &&
        (errorFilter.value === "all" || item.type === errorFilter.value)
      );
      document.getElementById("errors").innerHTML = rows.length
        ? rows.map(item => `<tr><td>${{escapeHtml(item.model)}}</td><td>${{escapeHtml(item.type)}}</td><td>${{escapeHtml(item.safetyreportid)}}</td><td>${{score(item.predicted_probability)}}</td><td>${{escapeHtml(item.true_label)}}</td><td class="tokens">${{escapeHtml(item.tokens)}}</td></tr>`).join("")
        : `<tr><td colspan="6">当前筛选条件下没有失败案例。</td></tr>`;
    }}
    modelFilter.addEventListener("change", renderErrors);
    errorFilter.addEventListener("change", renderErrors);
    renderErrors();
  </script>
</body>
</html>
"""
```

- [ ] **Step 5: Add atomic Dashboard generation**

Append to `scripts/dashboard.py`:

```python
def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def generate_dashboard(output_dir: Path, destination: Path) -> Path:
    ablation = json.loads(
        (output_dir / "ablations" / "ablation_metrics.json").read_text(encoding="utf-8")
    )
    weak = json.loads(
        (output_dir / "weak_supervision" / "weak_supervision_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (output_dir / "reports" / "feature_audit.json").read_text(encoding="utf-8")
    )
    _atomic_write_text(destination, render_dashboard(ablation, weak, audit))
    return destination
```

- [ ] **Step 6: Run output tests and verify GREEN**

Run:

```bash
python3 tests/test_final_outputs.py -v
```

Expected: all report and Dashboard tests pass.

- [ ] **Step 7: Commit the Dashboard**

```bash
git add scripts/dashboard.py tests/test_final_outputs.py
git commit -m "feat: generate offline experiment dashboard"
```

## Task 6: Add the Unified Final-Project CLI

**Files:**
- Create: `scripts/run_final_project.py`
- Modify: `tests/test_final_outputs.py`
- Modify: `readme.md`

- [ ] **Step 1: Add failing input validation tests**

Append to `tests/test_final_outputs.py`:

```python
import csv
import json
import run_final_project


class FinalProjectCliTests(unittest.TestCase):
    def test_validation_requires_four_case_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "interim").mkdir()
            (output_dir / "reports").mkdir()
            (output_dir / "reports" / "model_metrics.json").write_text(
                json.dumps({"models": {"numeric_logistic": {}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "four quarterly"):
                run_final_project.validate_inputs(output_dir)

    def test_validation_requires_numeric_baseline_and_all_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            interim = output_dir / "interim"
            reports = output_dir / "reports"
            interim.mkdir()
            reports.mkdir()
            for quarter in range(1, 5):
                with (interim / f"cases_2025Q{quarter}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["split", "label_serious"],
                    )
                    writer.writeheader()
                    writer.writerow({"split": "train", "label_serious": "1"})
            (reports / "model_metrics.json").write_text(
                json.dumps({"models": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "numeric_logistic"):
                run_final_project.validate_inputs(output_dir)

    def test_validation_rejects_missing_validation_and_test_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            interim = output_dir / "interim"
            reports = output_dir / "reports"
            interim.mkdir()
            reports.mkdir()
            for quarter in range(1, 5):
                with (interim / f"cases_2025Q{quarter}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["split", "label_serious", "text_tokens"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {"split": "train", "label_serious": "0", "text_tokens": "drug:A"}
                    )
                    writer.writerow(
                        {"split": "train", "label_serious": "1", "text_tokens": "drug:B"}
                    )
            (reports / "model_metrics.json").write_text(
                json.dumps({"models": {"numeric_logistic": {}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Split valid"):
                run_final_project.validate_inputs(output_dir)
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
python3 tests/test_final_outputs.py -v
```

Expected: import failure for missing `run_final_project`.

- [ ] **Step 3: Implement validation and orchestration**

Create `scripts/run_final_project.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import dashboard
import final_experiments
import final_reporting
import run_faers_pipeline as pipeline
import weak_supervision


ROOT = Path(__file__).resolve().parents[1]


def validate_inputs(output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    case_paths = sorted((output_dir / "interim").glob("cases_*.csv"))
    if len(case_paths) < 4:
        raise ValueError(
            f"Expected four quarterly case CSV files in {output_dir / 'interim'}, "
            f"found {len(case_paths)}"
        )
    metrics_path = output_dir / "reports" / "model_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing baseline metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = metrics.get("models", {})
    if "numeric_logistic" not in models:
        raise ValueError("model_metrics.json must contain numeric_logistic")

    split_counts: dict[str, Counter] = {
        "train": Counter(),
        "valid": Counter(),
        "test": Counter(),
    }
    for path in case_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"split", "label_serious", "text_tokens"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for row in reader:
                split = row.get("split", "")
                if split in split_counts:
                    split_counts[split][int(row["label_serious"])] += 1
    for split, counts in split_counts.items():
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(
                f"Split {split} must contain both classes, got {dict(counts)}"
            )
    return case_paths, models["numeric_logistic"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run final FAERS project experiments")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--report", type=Path, default=ROOT / "最终报告.md")
    parser.add_argument("--dashboard", type=Path, default=ROOT / "demo" / "index.html")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument("--epochs", type=int, default=pipeline.HASH_LOGISTIC_EPOCHS)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=pipeline.HASH_LOGISTIC_LEARNING_RATE,
    )
    parser.add_argument("--l2", type=float, default=pipeline.HASH_LOGISTIC_L2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    case_paths, numeric_baseline = validate_inputs(args.out)
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
    print(f"[done] final report: {args.report}")
    print(f"[done] dashboard: {args.dashboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run:

```bash
python3 tests/test_final_outputs.py -v
```

Expected: all output and CLI tests pass.

- [ ] **Step 5: Update README**

Append this section to `readme.md`:

````markdown
## 终期实验、最终报告与 Demo

完成基础流水线后，运行终期消融、弱监督扩展、最终 Markdown 报告和离线 HTML Dashboard：

```bash
python3 scripts/run_final_project.py --out outputs
```

快速验证已有样本产物：

```bash
python3 scripts/run_final_project.py --out outputs_sample
```

终期产物：

- `outputs/ablations/ablation_metrics.json`：五组哈希特征实验与数值基线对照。
- `outputs/weak_supervision/weak_supervision_metrics.json`：扩展规则与多数投票评测。
- `最终报告.md`：由真实 JSON 结果自动生成的最终报告。
- `demo/index.html`：无需服务器或网络依赖，可直接打开的结果 Dashboard。
````

- [ ] **Step 6: Run the complete unit suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass with no warnings or errors.

- [ ] **Step 7: Commit CLI and documentation**

```bash
git add scripts/run_final_project.py tests/test_final_outputs.py readme.md
git commit -m "feat: add final project runner"
```

## Task 7: Run Real Experiments and Verify Deliverables

**Files:**
- Generate: `outputs_sample/ablations/ablation_metrics.json`
- Generate: `outputs_sample/weak_supervision/weak_supervision_metrics.json`
- Generate: `outputs/ablations/ablation_metrics.json`
- Generate: `outputs/weak_supervision/weak_supervision_metrics.json`
- Generate: `最终报告.md`
- Generate: `demo/index.html`

- [ ] **Step 1: Run static checks**

Run:

```bash
python3 -m py_compile \
  scripts/final_experiments.py \
  scripts/weak_supervision.py \
  scripts/final_reporting.py \
  scripts/dashboard.py \
  scripts/run_final_project.py
git diff --check
```

Expected: both commands exit 0 with no output from `git diff --check`.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Run the sample end-to-end pipeline**

Run:

```bash
python3 scripts/run_final_project.py \
  --out outputs_sample \
  --report /tmp/DataMining-FDA-sample-report.md \
  --dashboard /tmp/DataMining-FDA-sample-dashboard.html \
  --n-features 4096
```

Expected:

- Command exits 0.
- `outputs_sample/ablations/ablation_metrics.json` contains six experiments.
- `outputs_sample/weak_supervision/weak_supervision_metrics.json` contains `train`, `valid`, and `test`.
- Temporary sample report and Dashboard are non-empty.

- [ ] **Step 4: Inspect sample JSON consistency**

Run:

```bash
python3 -c 'import json; from pathlib import Path; a=json.loads(Path("outputs_sample/ablations/ablation_metrics.json").read_text()); w=json.loads(Path("outputs_sample/weak_supervision/weak_supervision_metrics.json").read_text()); assert len(a["experiments"]) == 6; assert {"train","valid","test"} <= set(w["splits"]); print("sample artifacts valid")'
```

Expected: `sample artifacts valid`.

- [ ] **Step 5: Run the full-data final pipeline**

Run:

```bash
python3 scripts/run_final_project.py --out outputs
```

Expected:

- Command exits 0 after training five full-data ablation models.
- `最终报告.md` is regenerated from full-data JSON.
- `demo/index.html` is regenerated from full-data JSON.

- [ ] **Step 6: Verify report and Dashboard numbers against JSON**

Run:

```bash
python3 -c 'import json; from pathlib import Path; a=json.loads(Path("outputs/ablations/ablation_metrics.json").read_text()); w=json.loads(Path("outputs/weak_supervision/weak_supervision_metrics.json").read_text()); report=Path("最终报告.md").read_text(); html=Path("demo/index.html").read_text(); auc=f"{a[\"experiments\"][\"all_tokens\"][\"split_metrics\"][\"test\"][\"auroc\"]:.4f}"; coverage=f"{100*w[\"overall\"][\"coverage_rate\"]:.2f}%"; assert auc in report and auc in html; assert coverage in report; assert "model-filter" in html and "error-filter" in html; print("full artifacts consistent")'
```

Expected: `full artifacts consistent`.

- [ ] **Step 7: Verify the Dashboard in the in-app browser**

Load the Browser skill, open the absolute `file://` URL for `demo/index.html`, and verify:

- Desktop viewport around 1440x900: summary metrics, model rows, weak rules, and failure table are visible without overlap.
- Mobile viewport around 390x844: summary becomes two columns, tables scroll horizontally, and text remains inside containers.
- Change `model-filter` and `error-filter`; the table updates.
- Browser console contains no errors.
- Screenshot evidence shows a nonblank page with full-data metrics.

- [ ] **Step 8: Run final repository checks**

Run:

```bash
git diff --check
git status --short --branch
git diff --stat
```

Expected:

- No whitespace errors.
- `data/`, `outputs/`, and `outputs_sample/` remain ignored.
- Existing untracked `.obsidian/` and `中期进展报告.pdf` remain untouched.
- Tracked changes are limited to final-project source, tests, README, `最终报告.md`, and `demo/index.html`.

- [ ] **Step 9: Commit generated full-data deliverables**

```bash
git add 最终报告.md demo/index.html
git commit -m "docs: add final report and experiment dashboard"
```

- [ ] **Step 10: Record final evidence**

Run:

```bash
git log --oneline --decorate -8
git status --short --branch
```

Expected: the final-project commits are present and only the pre-existing untracked `.obsidian/` and `中期进展报告.pdf` remain.
