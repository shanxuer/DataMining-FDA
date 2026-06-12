import csv
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import final_experiments
import run_faers_pipeline as pipeline


FIELDNAMES = [
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


def write_tiny_cases(path):
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def numeric_baseline():
    metric = {
        "n": 2,
        "positives": 1,
        "positive_rate": 0.5,
        "threshold": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "auroc": 0.5,
        "auprc": 0.5,
        "recall_at_top_1pct": 0.5,
        "recall_at_top_5pct": 0.5,
        "recall_at_top_10pct": 0.5,
        "hit_rate_top_1pct": 0.5,
        "hit_rate_top_5pct": 0.5,
        "hit_rate_top_10pct": 0.5,
    }
    return {
        "model_path": "numeric_logistic.pkl",
        "train_rows": 2,
        "threshold_from_valid": 0.5,
        "split_metrics": {
            split: dict(metric)
            for split in ("train", "valid", "test")
        },
        "error_cases": {
            split: {"false_positive": [], "false_negative": []}
            for split in ("valid", "test")
        },
        "strata": {
            split: {}
            for split in ("valid", "test")
        },
    }


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
                "drug:ABC indi:PAIN reac:SEPSIS reactionoutcome:5 "
                "route:048 actiondrug:1 drugchar:1"
            ),
        }
        self.all_tokens = [
            "quarter:2025Q1",
            "primarysourcecountry:US",
            "patientsex:2",
            "age_bin:senior",
            "drug_count_bin:2_4",
            "reaction_count_bin:2_4",
            "indication_count_bin:1",
            "drug:ABC",
            "indi:PAIN",
            "reac:SEPSIS",
            "reactionoutcome:5",
            "route:048",
            "actiondrug:1",
            "drugchar:1",
        ]

    def tokens_for(self, experiment_name):
        return final_experiments.build_ablation_text(self.row, experiment_name).split()

    def assert_exact_tokens(self, experiment_name, expected):
        actual = self.tokens_for(experiment_name)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(set(actual)))

    def test_all_tokens_has_exact_order_without_duplicates(self):
        self.assert_exact_tokens("all_tokens", self.all_tokens)

    def test_without_reaction_pt_removes_only_reaction_pt(self):
        expected = [token for token in self.all_tokens if token != "reac:SEPSIS"]
        self.assert_exact_tokens("without_reaction_pt", expected)

    def test_without_reaction_outcome_removes_only_reaction_outcome(self):
        expected = [token for token in self.all_tokens if token != "reactionoutcome:5"]
        self.assert_exact_tokens("without_reaction_outcome", expected)

    def test_drug_indication_only_has_exact_order_without_duplicates(self):
        self.assert_exact_tokens(
            "drug_indication_only",
            ["drug:ABC", "indi:PAIN"],
        )

    def test_structured_only_matches_pipeline_without_text_tokens(self):
        structured_row = dict(self.row)
        structured_row["text_tokens"] = ""
        expected = pipeline.build_hash_text(structured_row).split()
        self.assert_exact_tokens("structured_only", expected)

    def test_strategies_exclude_labels_and_unknown_strategy_fails(self):
        for experiment_name in final_experiments.ABLATION_CONFIGS:
            text = final_experiments.build_ablation_text(self.row, experiment_name)
            self.assertNotIn("serious", text.lower())
            self.assertNotIn("label_serious", text)

        with self.assertRaisesRegex(ValueError, "Unknown ablation"):
            final_experiments.build_ablation_text(self.row, "not_a_strategy")


class AblationExperimentTests(unittest.TestCase):
    def test_tiny_experiment_writes_models_metrics_and_error_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "cases.csv"
            write_tiny_cases(case_path)
            baseline = numeric_baseline()

            result = final_experiments.run_ablation_experiments(
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

            expected_experiments = set(final_experiments.ABLATION_CONFIGS) | {
                "numeric_logistic"
            }
            self.assertEqual(set(result["experiments"]), expected_experiments)
            all_tokens = result["experiments"]["all_tokens"]
            self.assertIn("test", all_tokens["split_metrics"])
            self.assertEqual(
                set(all_tokens["error_cases"]),
                {"valid", "test"},
            )
            self.assertEqual(set(all_tokens["strata"]), {"valid", "test"})
            for name in final_experiments.ABLATION_CONFIGS:
                model_path = root / "ablations" / "models" / f"{name}.pkl"
                self.assertTrue(model_path.exists())
                with model_path.open("rb") as handle:
                    model = pickle.load(handle)
                self.assertTrue(np.any(model["weights"] != 0), name)

            metrics_path = root / "ablations" / "ablation_metrics.json"
            self.assertTrue(metrics_path.exists())
            written = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(written["metadata"]["n_features"], 64)
            self.assertEqual(written["metadata"]["epochs"], 8)
            self.assertEqual(written["metadata"]["random_state"], 42)
            numeric = written["experiments"]["numeric_logistic"]
            self.assertEqual(numeric["model_path"], baseline["model_path"])
            self.assertEqual(numeric["split_metrics"], baseline["split_metrics"])
            self.assertEqual(
                numeric["description"],
                "Existing numeric logistic baseline",
            )
            self.assertNotIn("description", baseline)
            result_numeric = result["experiments"]["numeric_logistic"]
            result_numeric["split_metrics"]["test"]["n"] = 999
            result_numeric["error_cases"]["test"]["false_positive"].append(
                {"id": "changed"}
            )
            self.assertEqual(baseline["split_metrics"]["test"]["n"], 2)
            self.assertEqual(
                baseline["error_cases"]["test"]["false_positive"],
                [],
            )

    def test_joint_training_scans_once_per_epoch_and_updates_every_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "cases.csv"
            write_tiny_cases(case_path)
            original_chunks = final_experiments.iter_ablation_chunks
            original_update = pipeline.update_hash_logistic_model
            epochs = 3

            with mock.patch.object(
                final_experiments,
                "iter_ablation_chunks",
                wraps=original_chunks,
            ) as chunk_spy, mock.patch.object(
                pipeline,
                "update_hash_logistic_model",
                wraps=original_update,
            ) as update_spy:
                final_experiments.train_ablation_models(
                    [case_path],
                    root / "models",
                    chunk_size=2,
                    n_features=64,
                    epochs=epochs,
                    learning_rate=0.3,
                    l2=0.0,
                    random_state=42,
                )

            self.assertEqual(chunk_spy.call_count, epochs)
            self.assertTrue(
                all(
                    call.kwargs.get("collect_rows") is False
                    for call in chunk_spy.call_args_list
                )
            )
            self.assertEqual(
                update_spy.call_count,
                epochs * len(final_experiments.ABLATION_CONFIGS),
            )

    def test_validation_thresholds_are_reused_for_every_split(self):
        names = list(final_experiments.ABLATION_CONFIGS)
        models = {
            name: {"experiment": name}
            for name in names
        }
        thresholds = [0.11, 0.22, 0.33, 0.44, 0.55]
        labels = np.asarray([0, 1], dtype=np.int8)
        prediction_sets = {}
        expected_threshold_by_probabilities = {}
        for split_index, split in enumerate(("valid", "train", "test"), start=1):
            probabilities = {}
            for model_index, name in enumerate(names, start=1):
                values = np.asarray(
                    [split_index / 10, model_index / 10],
                    dtype=np.float64,
                )
                probabilities[name] = values
                expected_threshold_by_probabilities[tuple(values)] = thresholds[
                    model_index - 1
                ]
            rows = [
                {"safetyreportid": f"{split}-0"},
                {"safetyreportid": f"{split}-1"},
            ] if split != "train" else []
            prediction_sets[split] = (labels, probabilities, rows)

        metric_calls = []

        def record_metrics(y_true, y_prob, threshold):
            metric_calls.append((tuple(y_prob), threshold))
            return {"n": len(y_true), "threshold": threshold}

        def check_error_lengths(rows, y_true, y_prob, threshold):
            self.assertEqual(len(rows), len(y_true))
            self.assertEqual(len(rows), len(y_prob))
            return {"false_positive": [], "false_negative": []}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            final_experiments,
            "train_ablation_models",
            return_value=(models, 2),
        ), mock.patch.object(
            final_experiments,
            "predict_ablation_models",
            side_effect=lambda paths, split, models, chunk_size, collect_rows: (
                prediction_sets[split]
            ),
        ) as predict_spy, mock.patch.object(
            pipeline,
            "best_threshold",
            side_effect=thresholds,
        ) as threshold_spy, mock.patch.object(
            pipeline,
            "classification_metrics",
            side_effect=record_metrics,
        ), mock.patch.object(
            pipeline,
            "select_error_cases",
            side_effect=check_error_lengths,
        ), mock.patch.object(
            pipeline,
            "stratified_metrics",
            return_value={},
        ):
            final_experiments.run_ablation_experiments(
                [Path("unused.csv")],
                Path(tmp),
                numeric_baseline(),
                chunk_size=2,
                n_features=64,
                epochs=2,
                learning_rate=0.3,
                l2=0.0,
                random_state=42,
            )

        self.assertEqual(
            [call.args[1] for call in predict_spy.call_args_list],
            ["valid", "train", "test"],
        )
        self.assertEqual(threshold_spy.call_count, len(names))
        for call, name in zip(threshold_spy.call_args_list, names):
            self.assertTrue(np.array_equal(call.args[0], labels))
            self.assertTrue(
                np.array_equal(
                    call.args[1],
                    prediction_sets["valid"][1][name],
                )
            )
        self.assertEqual(len(metric_calls), len(names) * 3)
        for probabilities, threshold in metric_calls:
            self.assertEqual(
                threshold,
                expected_threshold_by_probabilities[probabilities],
            )

    def test_prediction_metadata_matches_label_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "cases.csv"
            write_tiny_cases(case_path)
            models, _ = final_experiments.train_ablation_models(
                [case_path],
                root / "models",
                chunk_size=2,
                n_features=64,
                epochs=2,
                learning_rate=0.3,
                l2=0.0,
                random_state=42,
            )

            labels, probabilities, rows = final_experiments.predict_ablation_models(
                [case_path],
                "test",
                models,
                chunk_size=1,
                collect_rows=True,
            )

            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(
                [row["safetyreportid"] for row in rows],
                ["test-0", "test-1"],
            )
            for values in probabilities.values():
                self.assertEqual(len(values), len(labels))
            self.assertEqual(len(rows), len(labels))

    def test_chunk_iteration_skips_metadata_when_rows_are_not_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_path = Path(tmp) / "cases.csv"
            write_tiny_cases(case_path)
            with mock.patch.object(
                pipeline,
                "metadata_from_row",
                wraps=pipeline.metadata_from_row,
            ) as metadata_spy:
                chunks = list(
                    final_experiments.iter_ablation_chunks(
                        [case_path],
                        "train",
                        chunk_size=2,
                        collect_rows=False,
                    )
                )

            self.assertEqual(metadata_spy.call_count, 0)
            self.assertEqual(chunks[0][2], [])

    def test_invalid_training_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "cases.csv"
            write_tiny_cases(case_path)
            defaults = {
                "chunk_size": 2,
                "n_features": 64,
                "epochs": 2,
                "learning_rate": 0.3,
                "l2": 0.0,
                "random_state": 42,
            }
            invalid_values = [
                ("chunk_size", 0),
                ("n_features", 0),
                ("epochs", 0),
                ("learning_rate", 0.0),
                ("learning_rate", float("inf")),
                ("learning_rate", float("nan")),
                ("l2", -0.1),
                ("l2", float("inf")),
                ("l2", float("nan")),
            ]
            for parameter, value in invalid_values:
                with self.subTest(parameter=parameter, value=value):
                    arguments = dict(defaults)
                    arguments[parameter] = value
                    with self.assertRaisesRegex(ValueError, parameter):
                        final_experiments.train_ablation_models(
                            [case_path],
                            root / "models",
                            **arguments,
                        )


if __name__ == "__main__":
    unittest.main()
