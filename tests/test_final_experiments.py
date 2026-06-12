import sys
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


if __name__ == "__main__":
    unittest.main()
