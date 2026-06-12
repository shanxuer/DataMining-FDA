import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import final_experiments


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

    def tokens_for(self, experiment_name):
        return set(final_experiments.build_ablation_text(self.row, experiment_name).split())

    def test_all_tokens_keeps_structured_and_text_features(self):
        tokens = self.tokens_for("all_tokens")

        self.assertIn("quarter:2025Q1", tokens)
        self.assertIn("drug:ABC", tokens)
        self.assertIn("reac:SEPSIS", tokens)
        self.assertIn("reactionoutcome:5", tokens)

    def test_reaction_ablations_remove_only_the_selected_prefix(self):
        without_pt = self.tokens_for("without_reaction_pt")
        self.assertNotIn("reac:SEPSIS", without_pt)
        self.assertIn("reactionoutcome:5", without_pt)
        self.assertIn("drug:ABC", without_pt)

        without_outcome = self.tokens_for("without_reaction_outcome")
        self.assertNotIn("reactionoutcome:5", without_outcome)
        self.assertIn("reac:SEPSIS", without_outcome)
        self.assertIn("drug:ABC", without_outcome)

    def test_drug_indication_only_has_exact_token_set(self):
        self.assertEqual(
            self.tokens_for("drug_indication_only"),
            {"drug:ABC", "indi:PAIN"},
        )

    def test_structured_only_excludes_text_features(self):
        tokens = self.tokens_for("structured_only")

        self.assertIn("quarter:2025Q1", tokens)
        self.assertIn("age_bin:senior", tokens)
        self.assertNotIn("drug:ABC", tokens)
        self.assertNotIn("reac:SEPSIS", tokens)

    def test_strategies_exclude_labels_and_unknown_strategy_fails(self):
        for experiment_name in final_experiments.ABLATION_CONFIGS:
            text = final_experiments.build_ablation_text(self.row, experiment_name)
            self.assertNotIn("serious", text.lower())
            self.assertNotIn("label_serious", text)

        with self.assertRaisesRegex(ValueError, "Unknown ablation"):
            final_experiments.build_ablation_text(self.row, "not_a_strategy")


if __name__ == "__main__":
    unittest.main()
