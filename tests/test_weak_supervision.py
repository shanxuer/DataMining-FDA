import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import weak_supervision


RULE_NAMES = [
    "death_or_fatal_reaction_term",
    "high_polypharmacy_10plus",
    "senior_with_multiple_suspect_drugs",
    "low_complexity_younger_case",
    "high_risk_reaction",
    "serious_reaction_outcome",
    "extreme_polypharmacy_30plus",
    "device_product_issue",
]

FIELDNAMES = [
    "split",
    "label_serious",
    "age_years",
    "drug_count",
    "suspect_drug_count",
    "reaction_count",
    "text_tokens",
]


def make_row(
    *,
    split="test",
    label="0",
    age="",
    drug_count="0",
    suspect_count="0",
    reaction_count="0",
    tokens="",
):
    return {
        "split": split,
        "label_serious": label,
        "age_years": age,
        "drug_count": drug_count,
        "suspect_drug_count": suspect_count,
        "reaction_count": reaction_count,
        "text_tokens": tokens,
    }


class MajorityVoteTests(unittest.TestCase):
    def test_majority_vote_handles_positive_negative_tie_and_abstention(self):
        self.assertEqual(
            weak_supervision.majority_vote([1, 1, 0, None]),
            (1, False),
        )
        self.assertEqual(
            weak_supervision.majority_vote([0, None, 0, 1]),
            (0, False),
        )
        self.assertEqual(
            weak_supervision.majority_vote([1, 0, None]),
            (None, True),
        )
        self.assertEqual(
            weak_supervision.majority_vote([None, None]),
            (None, False),
        )


class RuleVoteTests(unittest.TestCase):
    def test_rule_constants_and_rule_order_are_exact(self):
        self.assertEqual(
            weak_supervision.DEATH_TERMS,
            {"reac:DEATH", "reac:FATAL", "reac:CARDIAC_ARREST"},
        )
        self.assertEqual(
            weak_supervision.HIGH_RISK_TERMS,
            {
                "reac:SEPSIS",
                "reac:SEPTIC_SHOCK",
                "reac:ACUTE_RESPIRATORY_FAILURE",
                "reac:RESPIRATORY_FAILURE",
                "reac:SHOCK",
            },
        )
        self.assertEqual(
            weak_supervision.DEVICE_TERMS,
            {
                "reac:DEVICE_DEFECTIVE",
                "reac:DEVICE_ISSUE",
                "reac:DEVICE_MALFUNCTION",
                "reac:PRODUCT_QUALITY_ISSUE",
                "reac:PRODUCT_COMMUNICATION_ISSUE",
                "reac:NO_ADVERSE_EVENT",
            },
        )
        self.assertEqual(
            weak_supervision.SERIOUS_OUTCOME_TOKENS,
            {"reactionoutcome:3", "reactionoutcome:4", "reactionoutcome:5"},
        )
        self.assertEqual(
            list(weak_supervision.rule_votes(make_row()).keys()),
            RULE_NAMES,
        )

    def test_device_only_votes_negative_but_abstains_with_high_risk_term(self):
        device_row = make_row(
            age="60",
            drug_count="3",
            reaction_count="2",
            tokens="reac:DEVICE_MALFUNCTION",
        )
        votes = weak_supervision.rule_votes(device_row)
        self.assertEqual(votes["device_product_issue"], 0)
        self.assertIsNone(votes["high_risk_reaction"])

        device_row["text_tokens"] += " reac:SEPSIS"
        votes = weak_supervision.rule_votes(device_row)
        self.assertIsNone(votes["device_product_issue"])
        self.assertEqual(votes["high_risk_reaction"], 1)

    def test_device_rule_abstains_when_any_other_reaction_is_present(self):
        mixed_row = make_row(
            age="60",
            drug_count="3",
            reaction_count="2",
            tokens="reac:DEVICE_MALFUNCTION reac:DIZZINESS",
        )

        votes = weak_supervision.rule_votes(mixed_row)

        self.assertIsNone(votes["device_product_issue"])

    def test_outcome_extreme_polypharmacy_and_senior_rules_vote_positive(self):
        row = make_row(
            age="70",
            drug_count="35",
            suspect_count="2",
            reaction_count="3",
            tokens="reactionoutcome:5",
        )
        votes = weak_supervision.rule_votes(row)
        self.assertEqual(votes["serious_reaction_outcome"], 1)
        self.assertEqual(votes["extreme_polypharmacy_30plus"], 1)
        self.assertEqual(votes["high_polypharmacy_10plus"], 1)
        self.assertEqual(votes["senior_with_multiple_suspect_drugs"], 1)


class WeakSupervisionSummaryTests(unittest.TestCase):
    def test_summary_reports_perfect_votes_and_uncovered_positive(self):
        rows = [
            make_row(
                label="1",
                age="60",
                drug_count="3",
                reaction_count="2",
                tokens="reac:SEPSIS",
            ),
            make_row(
                label="0",
                age="60",
                drug_count="3",
                reaction_count="2",
                tokens="reac:DEVICE_DEFECTIVE",
            ),
            make_row(
                label="1",
                age="60",
                drug_count="3",
                reaction_count="2",
            ),
        ]

        summary = weak_supervision.summarize_rows(rows)

        self.assertEqual(set(summary), {"overall", "splits"})
        self.assertEqual(set(summary["splits"]), {"test"})
        for bucket in (summary["overall"], summary["splits"]["test"]):
            self.assertEqual(bucket["total"], 3)
            self.assertEqual(bucket["covered"], 2)
            self.assertAlmostEqual(bucket["coverage_rate"], 2 / 3)
            self.assertEqual(bucket["conflicts"], 0)
            self.assertEqual(bucket["conflict_rate"], 0.0)
            self.assertEqual(bucket["voted"], 2)
            self.assertEqual(bucket["accuracy"], 1.0)
            self.assertEqual(bucket["precision"], 1.0)
            self.assertEqual(bucket["recall"], 1.0)
            self.assertEqual(bucket["f1"], 1.0)

    def test_tied_positive_and_negative_rules_are_covered_conflicts(self):
        summary = weak_supervision.summarize_rows(
            [
                make_row(
                    label="1",
                    age="40",
                    drug_count="1",
                    reaction_count="1",
                    tokens="reac:SEPSIS",
                )
            ]
        )

        overall = summary["overall"]
        self.assertEqual(overall["covered"], 1)
        self.assertEqual(overall["conflicts"], 1)
        self.assertEqual(overall["conflict_rate"], 1.0)
        self.assertEqual(overall["voted"], 0)
        self.assertEqual(overall["accuracy"], 0.0)
        self.assertEqual(overall["f1"], 0.0)

    def test_rule_statistics_track_vote_direction_labels_and_accuracy(self):
        rows = [
            make_row(
                label="0",
                age="30",
                drug_count="1",
                reaction_count="1",
            ),
            make_row(
                label="1",
                age="30",
                drug_count="1",
                reaction_count="1",
            ),
            make_row(
                label="1",
                age="60",
                drug_count="3",
                reaction_count="2",
                tokens="reac:SHOCK",
            ),
        ]

        rules = weak_supervision.summarize_rows(rows)["overall"]["rules"]

        low_complexity = rules["low_complexity_younger_case"]
        self.assertEqual(low_complexity["fires"], 2)
        self.assertEqual(low_complexity["positive_votes"], 0)
        self.assertEqual(low_complexity["negative_votes"], 2)
        self.assertEqual(low_complexity["positive_labels"], 1)
        self.assertEqual(low_complexity["correct"], 1)
        self.assertAlmostEqual(low_complexity["coverage_rate"], 2 / 3)
        self.assertEqual(low_complexity["positive_label_rate"], 0.5)
        self.assertEqual(low_complexity["accuracy"], 0.5)

        high_risk = rules["high_risk_reaction"]
        self.assertEqual(high_risk["fires"], 1)
        self.assertEqual(high_risk["positive_votes"], 1)
        self.assertEqual(high_risk["negative_votes"], 0)
        self.assertEqual(high_risk["positive_labels"], 1)
        self.assertEqual(high_risk["correct"], 1)
        self.assertEqual(high_risk["accuracy"], 1.0)

    def test_invalid_labels_are_rejected_and_empty_input_is_zeroed(self):
        for invalid_label in ("2", "-1", "", True, False):
            with self.subTest(label=invalid_label):
                with self.assertRaisesRegex(ValueError, "label_serious"):
                    weak_supervision.summarize_rows(
                        [make_row(label=invalid_label)]
                    )

        summary = weak_supervision.summarize_rows([])
        self.assertEqual(summary["splits"], {})
        overall = summary["overall"]
        self.assertEqual(overall["total"], 0)
        self.assertEqual(overall["covered"], 0)
        self.assertEqual(overall["coverage_rate"], 0.0)
        self.assertEqual(overall["conflicts"], 0)
        self.assertEqual(overall["conflict_rate"], 0.0)
        self.assertEqual(overall["voted"], 0)
        self.assertEqual(overall["accuracy"], 0.0)
        self.assertEqual(overall["precision"], 0.0)
        self.assertEqual(overall["recall"], 0.0)
        self.assertEqual(overall["f1"], 0.0)
        self.assertEqual(set(overall["rules"]), set(RULE_NAMES))
        self.assertTrue(
            all(rule["fires"] == 0 for rule in overall["rules"].values())
        )


class WeakSupervisionOutputTests(unittest.TestCase):
    def test_run_weak_supervision_streams_csv_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_path = root / "cases.csv"
            with case_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(
                    [
                        make_row(
                            split="train",
                            label="1",
                            age="60",
                            drug_count="3",
                            reaction_count="2",
                            tokens="reactionoutcome:4",
                        ),
                        make_row(
                            split="test",
                            label="0",
                            age="60",
                            drug_count="3",
                            reaction_count="2",
                            tokens="reac:DEVICE_ISSUE",
                        ),
                    ]
                )

            result = weak_supervision.run_weak_supervision(
                [case_path],
                root / "weak_supervision",
            )

            output_path = (
                root
                / "weak_supervision"
                / "weak_supervision_metrics.json"
            )
            self.assertTrue(output_path.exists())
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written, result)
            self.assertEqual(written["metadata"]["rules"], RULE_NAMES)
            note = written["metadata"]["note"].lower()
            self.assertIn("audit only", note)
            self.assertIn("not training labels", note)
            self.assertNotIn("summary", written)
            self.assertEqual(set(written["splits"]), {"train", "test"})
            self.assertEqual(written["splits"]["test"]["total"], 1)

    def test_atomic_json_failure_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "metrics.json"
            with mock.patch.object(
                weak_supervision.os,
                "replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    weak_supervision._atomic_write_json(
                        output_path,
                        {"ok": True},
                    )

            self.assertFalse(output_path.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
