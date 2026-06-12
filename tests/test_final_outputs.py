import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import final_reporting


EXPERIMENT_ORDER = [
    "all_tokens",
    "without_reaction_pt",
    "without_reaction_outcome",
    "drug_indication_only",
    "structured_only",
    "numeric_logistic",
]


def test_metrics(auroc, auprc, precision, recall, f1, recall_top5, hit_top5):
    return {
        "n": 200,
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_at_top_5pct": recall_top5,
        "hit_rate_top_5pct": hit_top5,
    }


def sample_ablation():
    metric_rows = {
        "all_tokens": (0.91234, 0.90123, 0.88, 0.87, 0.875, 0.08, 0.98),
        "without_reaction_pt": (
            0.71234,
            0.70123,
            0.68,
            0.67,
            0.675,
            0.06,
            0.78,
        ),
        "without_reaction_outcome": (
            0.81234,
            0.80123,
            0.78,
            0.77,
            0.775,
            0.07,
            0.88,
        ),
        "drug_indication_only": (
            0.76234,
            0.75123,
            0.73,
            0.72,
            0.725,
            0.065,
            0.83,
        ),
        "structured_only": (
            0.66234,
            0.65123,
            0.63,
            0.62,
            0.625,
            0.055,
            0.73,
        ),
        "numeric_logistic": (
            0.61234,
            0.60123,
            0.58,
            0.57,
            0.575,
            0.05,
            0.68,
        ),
    }
    experiments = {}
    for name in EXPERIMENT_ORDER:
        experiments[name] = {
            "description": name,
            "split_metrics": {"test": test_metrics(*metric_rows[name])},
            "error_cases": {
                "test": {
                    "false_positive": [],
                    "false_negative": [],
                }
            },
            "strata": {"test": {}},
        }

    experiments["all_tokens"]["error_cases"]["test"]["false_positive"] = [
        {
            "safetyreportid": "FP-1",
            "predicted_probability": 0.99123,
            "true_label": 0,
            "tokens": "reac:SEPSIS|reactionoutcome:5",
        }
    ]
    experiments["all_tokens"]["strata"]["test"] = {
        "sex": {
            "1": {
                "n": 100,
                "auroc": 0.90123,
                "auprc": 0.89123,
                "f1": 0.86432,
            }
        },
        "age_bin": {
            "senior": {
                "n": 40,
                "auroc": 0.80111,
                "auprc": 0.79111,
                "f1": 0.77111,
            }
        },
        "quarter": {
            "2025Q4": {
                "n": 200,
                "auroc": 0.91234,
                "auprc": 0.90123,
                "f1": 0.875,
            }
        },
    }
    return {
        "metadata": {"split_policy": "time split"},
        "experiments": experiments,
    }


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
                },
                "device_product_issue": {
                    "fires": 20,
                    "coverage_rate": 0.05,
                    "positive_label_rate": 0.1,
                    "accuracy": 0.85,
                },
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


def sample_audit():
    return {
        "total": 400,
        "by_quarter": {
            "2025Q1": {"n": 100, "positive": 60},
            "2025Q4": {"n": 300, "positive": 180},
        },
        "missing": {
            "age_years": 40,
            "patientsex": 20,
        },
    }


def write_inputs(output_dir, ablation=None, weak=None, audit=None):
    paths = {
        "ablation": output_dir / "ablations" / "ablation_metrics.json",
        "weak": (
            output_dir
            / "weak_supervision"
            / "weak_supervision_metrics.json"
        ),
        "audit": output_dir / "reports" / "feature_audit.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["ablation"].write_text(
        json.dumps(ablation or sample_ablation()),
        encoding="utf-8",
    )
    paths["weak"].write_text(
        json.dumps(weak or sample_weak()),
        encoding="utf-8",
    )
    paths["audit"].write_text(
        json.dumps(audit or sample_audit()),
        encoding="utf-8",
    )
    return paths


class FinalReportTests(unittest.TestCase):
    def test_experiment_order_is_fixed(self):
        self.assertEqual(final_reporting.EXPERIMENT_ORDER, EXPERIMENT_ORDER)

    def test_report_uses_input_values_and_contains_all_required_sections(self):
        text = final_reporting.render_final_report(
            sample_ablation(),
            sample_weak(),
            sample_audit(),
        )

        self.assertTrue(text.startswith("# 数据挖掘课程项目最终报告"))
        for section in (
            "## 项目摘要与研究问题",
            "## 数据来源与治理",
            "### 季度样本与重症率",
            "### 关键字段缺失",
            "## 方法",
            "## 消融实验结果",
            "## 语义捷径诊断",
            "## 弱监督扩展",
            "## 失败案例",
            "## 分层误差分析",
            "## 局限性与结论",
            "## 完整复现方式",
            "## AI 工具辅助使用声明",
        ):
            self.assertIn(section, text)

        self.assertIn("0.9123", text)
        self.assertIn("0.7123", text)
        self.assertIn("0.8123", text)
        self.assertIn("下降 0.2000", text)
        self.assertIn("下降 0.1000", text)
        self.assertIn("30.00%", text)
        self.assertIn("82.00%", text)
        self.assertIn("0.8195", text)
        self.assertIn("FP-1", text)
        self.assertIn("| `all_tokens` | false_positive | FP-1 |", text)
        self.assertIn("| sex | 1 | 100 | 0.9012 | 0.8912 | 0.8643 |", text)
        self.assertIn("| 2025Q1 | 100 | 60 | 60.00% |", text)
        self.assertIn("| `age_years` | 40 | 10.00% |", text)
        self.assertIn(
            "python3 scripts/run_faers_pipeline.py --data data "
            "--out outputs --mode full",
            text,
        )
        self.assertIn(
            "python3 scripts/run_final_project.py --out outputs",
            text,
        )
        self.assertIn(
            "python3 scripts/run_final_project.py --out outputs_sample",
            text,
        )

    def test_report_changes_when_input_metric_changes(self):
        ablation = sample_ablation()
        original = final_reporting.render_final_report(
            ablation,
            sample_weak(),
            sample_audit(),
        )
        ablation["experiments"]["all_tokens"]["split_metrics"]["test"][
            "auroc"
        ] = 0.93456
        changed = final_reporting.render_final_report(
            ablation,
            sample_weak(),
            sample_audit(),
        )

        self.assertIn("0.9123", original)
        self.assertIn("0.9346", changed)
        self.assertNotEqual(original, changed)

    def test_missing_required_inputs_and_none_top_metric_are_rejected(self):
        cases = []

        ablation = sample_ablation()
        del ablation["experiments"]["all_tokens"]
        cases.append(("all_tokens", ablation, sample_weak(), sample_audit()))

        weak = sample_weak()
        del weak["splits"]["test"]
        cases.append(("test", sample_ablation(), weak, sample_audit()))

        audit = sample_audit()
        del audit["total"]
        cases.append(("total", sample_ablation(), sample_weak(), audit))

        ablation = sample_ablation()
        ablation["experiments"]["structured_only"]["split_metrics"]["test"][
            "auroc"
        ] = None
        cases.append(("auroc", ablation, sample_weak(), sample_audit()))

        for expected, ablation, weak, audit in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    final_reporting.render_final_report(
                        ablation,
                        weak,
                        audit,
                    )

    def test_stratum_none_auc_metrics_render_as_na(self):
        ablation = sample_ablation()
        senior = ablation["experiments"]["all_tokens"]["strata"]["test"][
            "age_bin"
        ]["senior"]
        senior["auroc"] = None
        senior["auprc"] = None

        text = final_reporting.render_final_report(
            ablation,
            sample_weak(),
            sample_audit(),
        )

        self.assertIn(
            "| age_bin | senior | 40 | NA | NA | 0.7711 |",
            text,
        )

    def test_failure_case_tokens_escape_markdown_pipe(self):
        text = final_reporting.render_final_report(
            sample_ablation(),
            sample_weak(),
            sample_audit(),
        )

        self.assertIn(r"reac:SEPSIS\|reactionoutcome:5", text)
        self.assertNotIn(
            "| reac:SEPSIS|reactionoutcome:5 |",
            text,
        )

    def test_helper_formatters_reject_none(self):
        with self.assertRaisesRegex(ValueError, "metric"):
            final_reporting._metric(None)
        with self.assertRaisesRegex(ValueError, "percentage"):
            final_reporting._percent(None)
        with self.assertRaisesRegex(ValueError, "field"):
            final_reporting._required({"field": None}, "field", "test field")


class FinalReportFileTests(unittest.TestCase):
    def test_generate_final_report_reads_json_and_writes_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_inputs(root)
            destination = root / "deliverables" / "report.md"

            result = final_reporting.generate_final_report(root, destination)

            self.assertEqual(result, destination)
            self.assertTrue(destination.exists())
            text = destination.read_text(encoding="utf-8")
            self.assertIn("0.9123", text)
            self.assertIn("30.00%", text)

    def test_atomic_write_failure_cleans_temp_file_and_preserves_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_inputs(root)
            destination = root / "report.md"
            destination.write_text("original", encoding="utf-8")

            with mock.patch.object(
                final_reporting.os,
                "replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    final_reporting.generate_final_report(root, destination)

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "original",
            )
            leftovers = [
                path
                for path in root.iterdir()
                if path.name.startswith(".report.md.")
            ]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
