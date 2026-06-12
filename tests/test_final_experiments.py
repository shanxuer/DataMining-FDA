import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import final_experiments
import run_faers_pipeline as pipeline


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
        baseline = {
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

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "cases.csv"
            with case_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

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
                self.assertTrue(
                    (root / "ablations" / "models" / f"{name}.pkl").exists()
                )

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


if __name__ == "__main__":
    unittest.main()
