import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import final_reporting
import dashboard
import run_final_project


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
            "receivedate": "20251231",
            "predicted_probability": 0.99123,
            "predicted_label": 1,
            "true_label": 0,
            "patientsex": "1",
            "age_years": "72",
            "drug_count": "8",
            "reaction_count": "3",
            "indication_count": "2",
            "tokens": "reac:SEPSIS|reactionoutcome:5",
        }
    ]
    experiments["all_tokens"]["strata"]["test"] = {
        "sex": {
            "1": {
                "n": 100,
                "auroc": 0.90123,
                "auprc": 0.89123,
                "precision": 0.88,
                "recall": 0.85,
                "f1": 0.86432,
            }
        },
        "age_bin": {
            "senior": {
                "n": 40,
                "auroc": 0.80111,
                "auprc": 0.79111,
                "precision": 0.78,
                "recall": 0.76,
                "f1": 0.77111,
            }
        },
        "quarter": {
            "2025Q4": {
                "n": 200,
                "auroc": 0.91234,
                "auprc": 0.90123,
                "precision": 0.88,
                "recall": 0.87,
                "f1": 0.875,
            }
        },
    }
    return {
        "metadata": {"split_policy": "time split"},
        "experiments": experiments,
    }


def sample_weak():
    rules = {
        "death_or_fatal_reaction_term": {
            "fires": 8,
            "coverage_rate": 0.02,
            "positive_label_rate": 1.0,
            "accuracy": 1.0,
        },
        "high_polypharmacy_10plus": {
            "fires": 36,
            "coverage_rate": 0.09,
            "positive_label_rate": 0.75,
            "accuracy": 0.75,
        },
        "senior_with_multiple_suspect_drugs": {
            "fires": 28,
            "coverage_rate": 0.07,
            "positive_label_rate": 0.71,
            "accuracy": 0.71,
        },
        "low_complexity_younger_case": {
            "fires": 24,
            "coverage_rate": 0.06,
            "positive_label_rate": 0.12,
            "accuracy": 0.88,
        },
        "high_risk_reaction": {
            "fires": 50,
            "coverage_rate": 0.125,
            "positive_label_rate": 0.9,
            "accuracy": 0.9,
        },
        "serious_reaction_outcome": {
            "fires": 44,
            "coverage_rate": 0.11,
            "positive_label_rate": 0.86,
            "accuracy": 0.86,
        },
        "extreme_polypharmacy_30plus": {
            "fires": 12,
            "coverage_rate": 0.03,
            "positive_label_rate": 0.83,
            "accuracy": 0.83,
        },
        "device_product_issue": {
            "fires": 20,
            "coverage_rate": 0.05,
            "positive_label_rate": 0.1,
            "accuracy": 0.85,
        },
    }
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
            "rules": rules,
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
        json.dumps(sample_ablation() if ablation is None else ablation),
        encoding="utf-8",
    )
    paths["weak"].write_text(
        json.dumps(sample_weak() if weak is None else weak),
        encoding="utf-8",
    )
    paths["audit"].write_text(
        json.dumps(sample_audit() if audit is None else audit),
        encoding="utf-8",
    )
    return paths


def dashboard_payload(html):
    marker = '<script id="dashboard-data" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return html[start:end]


def reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


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
        self.assertIn(
            "| sex | 1 | 100 | 0.9012 | 0.8912 | 0.8800 | 0.8500 | "
            "0.8643 |",
            text,
        )
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

    def test_stratum_none_classification_metrics_render_as_na(self):
        ablation = sample_ablation()
        senior = ablation["experiments"]["all_tokens"]["strata"]["test"][
            "age_bin"
        ]["senior"]
        for metric_name in (
            "auroc",
            "auprc",
            "precision",
            "recall",
            "f1",
        ):
            senior[metric_name] = None

        text = final_reporting.render_final_report(
            ablation,
            sample_weak(),
            sample_audit(),
        )

        self.assertIn(
            "| age_bin | senior | 40 | NA | NA | NA | NA | NA |",
            text,
        )

    def test_weak_supervision_outputs_complete_overall_and_test_metrics(self):
        text = final_reporting.render_final_report(
            sample_ablation(),
            sample_weak(),
            sample_audit(),
        )

        self.assertIn(
            "| overall | 400 | 120 | 30.00% | 10 | 8.33% | 110 | "
            "80.00% | 82.00% | 78.00% | 0.7995 |",
            text,
        )
        self.assertIn(
            "| test | 100 | 30 | 30.00% | 2 | 6.67% | 28 | "
            "82.00% | 84.00% | 80.00% | 0.8195 |",
            text,
        )

    def test_weak_supervision_metrics_are_dynamic(self):
        weak = sample_weak()
        original = final_reporting.render_final_report(
            sample_ablation(),
            weak,
            sample_audit(),
        )
        weak["splits"]["test"]["voted"] = 27
        weak["splits"]["test"]["precision"] = 0.91
        weak["splits"]["test"]["recall"] = 0.73
        changed = final_reporting.render_final_report(
            sample_ablation(),
            weak,
            sample_audit(),
        )

        self.assertIn("| test | 100 | 30 | 30.00% | 2 | 6.67% | 28 |", original)
        self.assertIn(
            "| test | 100 | 30 | 30.00% | 2 | 6.67% | 27 | "
            "82.00% | 91.00% | 73.00% | 0.8195 |",
            changed,
        )
        self.assertNotEqual(original, changed)

    def test_missing_any_required_weak_supervision_metric_is_rejected(self):
        required_metrics = (
            "total",
            "covered",
            "coverage_rate",
            "conflicts",
            "conflict_rate",
            "voted",
            "accuracy",
            "precision",
            "recall",
            "f1",
        )
        for bucket_name in ("overall", "test"):
            for metric_name in required_metrics:
                with self.subTest(
                    bucket=bucket_name,
                    metric=metric_name,
                ):
                    weak = sample_weak()
                    bucket = (
                        weak["overall"]
                        if bucket_name == "overall"
                        else weak["splits"]["test"]
                    )
                    del bucket[metric_name]
                    with self.assertRaisesRegex(ValueError, metric_name):
                        final_reporting.render_final_report(
                            sample_ablation(),
                            weak,
                            sample_audit(),
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

    def test_invalid_metric_errors_include_complete_context(self):
        context = "ablation.experiments.all_tokens.split_metrics.test.auroc"
        for invalid in (None, True, "not-a-number", math.nan, math.inf):
            with self.subTest(value=invalid):
                ablation = sample_ablation()
                ablation["experiments"]["all_tokens"]["split_metrics"]["test"][
                    "auroc"
                ] = invalid
                with self.assertRaisesRegex(ValueError, context):
                    final_reporting.render_final_report(
                        ablation,
                        sample_weak(),
                        sample_audit(),
                    )

    def test_invalid_nullable_metric_includes_complete_context(self):
        ablation = sample_ablation()
        ablation["experiments"]["all_tokens"]["strata"]["test"]["sex"]["1"][
            "f1"
        ] = "not-a-number"

        with self.assertRaisesRegex(
            ValueError,
            r"ablation\.experiments\.all_tokens\.strata\.test\.sex\.1\.f1",
        ):
            final_reporting.render_final_report(
                ablation,
                sample_weak(),
                sample_audit(),
            )

    def test_invalid_percent_error_includes_complete_context(self):
        weak = sample_weak()
        weak["overall"]["coverage_rate"] = math.inf

        with self.assertRaisesRegex(
            ValueError,
            r"weak\.overall\.coverage_rate",
        ):
            final_reporting.render_final_report(
                sample_ablation(),
                weak,
                sample_audit(),
            )

    def test_invalid_containers_raise_contextual_value_errors(self):
        cases = []

        ablation = sample_ablation()
        ablation["experiments"] = []
        cases.append(
            (
                r"ablation\.experiments",
                ablation,
                sample_weak(),
                sample_audit(),
            )
        )

        audit = sample_audit()
        audit["by_quarter"] = []
        cases.append(
            (
                r"audit\.by_quarter",
                sample_ablation(),
                sample_weak(),
                audit,
            )
        )

        audit = sample_audit()
        audit["missing"] = []
        cases.append(
            (
                r"audit\.missing",
                sample_ablation(),
                sample_weak(),
                audit,
            )
        )

        weak = sample_weak()
        weak["overall"]["rules"] = []
        cases.append(
            (
                r"weak\.overall\.rules",
                sample_ablation(),
                weak,
                sample_audit(),
            )
        )

        ablation = sample_ablation()
        ablation["experiments"]["all_tokens"]["error_cases"] = []
        cases.append(
            (
                r"ablation\.experiments\.all_tokens\.error_cases",
                ablation,
                sample_weak(),
                sample_audit(),
            )
        )

        ablation = sample_ablation()
        ablation["experiments"]["all_tokens"]["error_cases"]["test"][
            "false_positive"
        ] = {}
        cases.append(
            (
                r"ablation\.experiments\.all_tokens\.error_cases\.test"
                r"\.false_positive",
                ablation,
                sample_weak(),
                sample_audit(),
            )
        )

        for context, ablation, weak, audit in cases:
            with self.subTest(context=context):
                with self.assertRaisesRegex(ValueError, context):
                    final_reporting.render_final_report(
                        ablation,
                        weak,
                        audit,
                    )

    def test_invalid_counts_are_rejected_with_complete_context(self):
        invalid_values = (True, 1.9, math.nan, math.inf, -1)
        cases = (
            (
                r"audit\.total",
                lambda ablation, weak, audit, value: audit.__setitem__(
                    "total",
                    value,
                ),
            ),
            (
                r"audit\.by_quarter\.2025Q1\.n",
                lambda ablation, weak, audit, value: audit["by_quarter"][
                    "2025Q1"
                ].__setitem__("n", value),
            ),
            (
                r"audit\.by_quarter\.2025Q1\.positive",
                lambda ablation, weak, audit, value: audit["by_quarter"][
                    "2025Q1"
                ].__setitem__("positive", value),
            ),
            (
                r"audit\.missing\.age_years",
                lambda ablation, weak, audit, value: audit["missing"].__setitem__(
                    "age_years",
                    value,
                ),
            ),
            (
                r"weak\.overall\.covered",
                lambda ablation, weak, audit, value: weak["overall"].__setitem__(
                    "covered",
                    value,
                ),
            ),
            (
                r"weak\.overall\.conflicts",
                lambda ablation, weak, audit, value: weak["overall"].__setitem__(
                    "conflicts",
                    value,
                ),
            ),
            (
                r"weak\.overall\.voted",
                lambda ablation, weak, audit, value: weak["overall"].__setitem__(
                    "voted",
                    value,
                ),
            ),
            (
                r"weak\.overall\.rules\.high_risk_reaction\.fires",
                lambda ablation, weak, audit, value: weak["overall"]["rules"][
                    "high_risk_reaction"
                ].__setitem__("fires", value),
            ),
            (
                r"ablation\.experiments\.all_tokens\.strata\.test\.sex\.1\.n",
                lambda ablation, weak, audit, value: ablation["experiments"][
                    "all_tokens"
                ]["strata"]["test"]["sex"]["1"].__setitem__("n", value),
            ),
        )
        for context, mutate in cases:
            for invalid in invalid_values:
                with self.subTest(context=context, value=invalid):
                    ablation = sample_ablation()
                    weak = sample_weak()
                    audit = sample_audit()
                    mutate(ablation, weak, audit, invalid)
                    with self.assertRaisesRegex(ValueError, context):
                        final_reporting.render_final_report(
                            ablation,
                            weak,
                            audit,
                        )

    def test_integer_counts_accept_supported_representations(self):
        audit = sample_audit()
        audit["total"] = "400"
        audit["by_quarter"]["2025Q1"]["n"] = 100.0
        audit["by_quarter"]["2025Q1"]["positive"] = "60"
        audit["missing"]["age_years"] = 40.0

        text = final_reporting.render_final_report(
            sample_ablation(),
            sample_weak(),
            audit,
        )

        self.assertIn("构建了 400 条病例级样本", text)
        self.assertIn("| 2025Q1 | 100 | 60 | 60.00% |", text)

    def test_helper_formatters_include_context(self):
        with self.assertRaisesRegex(ValueError, r"metrics\.example"):
            final_reporting._metric(None, "metrics.example")
        with self.assertRaisesRegex(ValueError, r"rates\.example"):
            final_reporting._percent(None, "rates.example")
        with self.assertRaisesRegex(ValueError, r"payload\.field"):
            final_reporting._required(
                {"field": None},
                "field",
                "payload.field",
            )


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

    def test_fdopen_failure_closes_raw_fd_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "report.md"
            destination.write_text("original", encoding="utf-8")
            real_close = final_reporting.os.close

            with mock.patch.object(
                final_reporting.os,
                "fdopen",
                side_effect=OSError("fdopen failed"),
            ), mock.patch.object(
                final_reporting.os,
                "close",
                wraps=real_close,
            ) as close_spy:
                with self.assertRaisesRegex(OSError, "fdopen failed"):
                    final_reporting._atomic_write_text(
                        destination,
                        "replacement",
                    )

            close_spy.assert_called_once()
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

    def test_missing_input_file_error_contains_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "ablations" / "ablation_metrics.json"

            with self.assertRaises(FileNotFoundError) as error:
                final_reporting.generate_final_report(
                    root,
                    root / "report.md",
                )

            self.assertIn(str(expected), str(error.exception))

    def test_write_inputs_preserves_empty_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_inputs(root, ablation={}, weak={}, audit={})

            for path in paths.values():
                self.assertEqual(
                    json.loads(path.read_text(encoding="utf-8")),
                    {},
                )


class DashboardTests(unittest.TestCase):
    def test_dashboard_is_self_contained_and_contains_results(self):
        text = dashboard.render_dashboard(
            sample_ablation(),
            sample_weak(),
            sample_audit(),
        )

        self.assertTrue(text.startswith("<!doctype html>"))
        self.assertIn('lang="zh-CN"', text)
        self.assertIn('id="dashboard-data"', text)
        self.assertIn('id="model-filter"', text)
        self.assertIn('id="error-filter"', text)
        for name in EXPERIMENT_ORDER:
            self.assertIn(name, text)
        self.assertIn("FP-1", text)
        self.assertIn("0.9123", text)
        self.assertIn("30.00%", text)
        self.assertNotIn("http://", text)
        self.assertNotIn("https://", text)

    def test_safe_json_prevents_script_breakout_and_line_separators(self):
        ablation = sample_ablation()
        unsafe = "</script><script>alert(1)</script>\u2028\u2029"
        ablation["experiments"]["all_tokens"]["error_cases"]["test"][
            "false_positive"
        ][0]["tokens"] = unsafe

        text = dashboard.render_dashboard(
            ablation,
            sample_weak(),
            sample_audit(),
        )

        self.assertNotIn(unsafe, text)
        self.assertNotIn("</script><script>alert(1)</script>", text)
        self.assertIn(r"<\/script>", text)
        self.assertNotIn("\u2028", text)
        self.assertNotIn("\u2029", text)
        self.assertIn(r"\u2028", text)
        self.assertIn(r"\u2029", text)

    def test_safe_json_rejects_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            dashboard._safe_json({"nested": {"value": math.nan}})

    def test_dashboard_data_shapes_models_weak_rules_and_errors(self):
        data = dashboard._dashboard_data(
            sample_ablation(),
            sample_weak(),
            sample_audit(),
        )

        self.assertEqual(data["total"], 400)
        self.assertEqual(data["serious_rate"], 0.6)
        self.assertEqual(len(data["models"]), 6)
        self.assertEqual(
            [model["name"] for model in data["models"]],
            EXPERIMENT_ORDER,
        )
        self.assertEqual(data["best"]["name"], "all_tokens")
        self.assertEqual(data["best"]["auroc"], 0.91234)
        self.assertEqual(data["weak"]["coverage"], 0.3)
        self.assertEqual(data["weak"]["conflict"], 0.0833)
        self.assertEqual(data["weak"]["accuracy"], 0.8)
        self.assertEqual(data["weak"]["precision"], 0.82)
        self.assertEqual(data["weak"]["recall"], 0.78)
        self.assertEqual(data["weak"]["f1"], 0.7995)
        self.assertEqual(len(data["weak"]["rules"]), 8)
        self.assertEqual(data["errors"][0]["model"], "all_tokens")
        self.assertEqual(data["errors"][0]["type"], "false_positive")
        self.assertEqual(data["errors"][0]["safetyreportid"], "FP-1")

    def test_dashboard_rejects_missing_experiment_invalid_metric_and_bad_audit(self):
        cases = []

        ablation = sample_ablation()
        del ablation["experiments"]["all_tokens"]
        cases.append(
            (
                r"ablation\.experiments\.all_tokens",
                ablation,
                sample_weak(),
                sample_audit(),
            )
        )

        ablation = sample_ablation()
        ablation["experiments"]["all_tokens"]["split_metrics"]["test"][
            "auroc"
        ] = math.inf
        cases.append(
            (
                r"ablation\.experiments\.all_tokens\.split_metrics\.test"
                r"\.auroc",
                ablation,
                sample_weak(),
                sample_audit(),
            )
        )

        audit = sample_audit()
        audit["by_quarter"] = []
        cases.append(
            (
                r"audit\.by_quarter",
                sample_ablation(),
                sample_weak(),
                audit,
            )
        )

        for context, ablation, weak, audit in cases:
            with self.subTest(context=context):
                with self.assertRaisesRegex(ValueError, context):
                    dashboard._dashboard_data(ablation, weak, audit)

    def test_dashboard_rejects_non_numeric_metric_and_non_string_description(self):
        ablation = sample_ablation()
        ablation["experiments"]["all_tokens"]["split_metrics"]["test"][
            "auroc"
        ] = "0.91234"
        with self.assertRaisesRegex(
            ValueError,
            r"ablation\.experiments\.all_tokens\.split_metrics\.test\.auroc",
        ):
            dashboard._dashboard_data(
                ablation,
                sample_weak(),
                sample_audit(),
            )

        ablation = sample_ablation()
        ablation["experiments"]["all_tokens"]["description"] = {}
        with self.assertRaisesRegex(
            ValueError,
            r"ablation\.experiments\.all_tokens\.description",
        ):
            dashboard._dashboard_data(
                ablation,
                sample_weak(),
                sample_audit(),
            )

    def test_dashboard_rejects_incomplete_weak_rule_set(self):
        weak = sample_weak()
        del weak["overall"]["rules"]["device_product_issue"]

        with self.assertRaisesRegex(
            ValueError,
            r"weak\.overall\.rules\.device_product_issue",
        ):
            dashboard._dashboard_data(
                sample_ablation(),
                weak,
                sample_audit(),
            )

    def test_dashboard_rejects_invalid_error_case_fields_with_context(self):
        cases = (
            *(("predicted_probability", value) for value in (
                True,
                None,
                "bad",
                math.nan,
                math.inf,
            )),
            ("true_label", {"value": 0}),
            ("predicted_label", 2),
            *(("tokens", value) for value in (
                {"term": "reac:SEPSIS"},
                ["reac:SEPSIS"],
                {"reac:SEPSIS"},
                ("reac:SEPSIS",),
            )),
        )
        context = (
            r"ablation\.experiments\.all_tokens\.error_cases\.test"
            r"\.false_positive\[0\]"
        )
        for field, invalid in cases:
            with self.subTest(field=field, invalid=invalid):
                ablation = sample_ablation()
                ablation["experiments"]["all_tokens"]["error_cases"]["test"][
                    "false_positive"
                ][0][field] = invalid
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{context}\.{field}",
                ):
                    dashboard._dashboard_data(
                        ablation,
                        sample_weak(),
                        sample_audit(),
                    )

    def test_dashboard_rejects_non_mapping_error_case_with_context(self):
        ablation = sample_ablation()
        ablation["experiments"]["all_tokens"]["error_cases"]["test"][
            "false_positive"
        ][0] = ["not", "a", "mapping"]

        with self.assertRaisesRegex(
            ValueError,
            r"ablation\.experiments\.all_tokens\.error_cases\.test"
            r"\.false_positive\[0\]",
        ):
            dashboard._dashboard_data(
                ablation,
                sample_weak(),
                sample_audit(),
            )

    def test_dashboard_payload_is_strict_json_and_error_fields_are_whitelisted(self):
        ablation = sample_ablation()
        error_case = ablation["experiments"]["all_tokens"]["error_cases"][
            "test"
        ]["false_positive"][0]
        error_case["unknown_nonfinite"] = math.nan

        html = dashboard.render_dashboard(
            ablation,
            sample_weak(),
            sample_audit(),
        )
        data = json.loads(
            dashboard_payload(html),
            parse_constant=reject_json_constant,
        )

        expected_fields = {
            "safetyreportid",
            "receivedate",
            "predicted_probability",
            "predicted_label",
            "true_label",
            "patientsex",
            "age_years",
            "drug_count",
            "reaction_count",
            "indication_count",
            "tokens",
            "model",
            "type",
        }
        self.assertEqual(set(data["errors"][0]), expected_fields)
        self.assertNotIn("unknown_nonfinite", dashboard_payload(html))

    def test_dashboard_rejects_nonfinite_known_error_scalars(self):
        context = (
            r"ablation\.experiments\.all_tokens\.error_cases\.test"
            r"\.false_positive\[0\]"
        )
        for field, invalid in (
            ("age_years", math.nan),
            ("tokens", math.inf),
        ):
            with self.subTest(field=field):
                ablation = sample_ablation()
                ablation["experiments"]["all_tokens"]["error_cases"]["test"][
                    "false_positive"
                ][0][field] = invalid
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{context}\.{field}",
                ):
                    dashboard._dashboard_data(
                        ablation,
                        sample_weak(),
                        sample_audit(),
                    )

    def test_dashboard_rejects_out_of_range_metrics_and_probability(self):
        cases = (
            (
                r"ablation\.experiments\.all_tokens\.split_metrics\.test"
                r"\.auroc",
                lambda ablation, weak: ablation["experiments"]["all_tokens"][
                    "split_metrics"
                ]["test"].__setitem__("auroc", -0.01),
            ),
            (
                r"weak\.overall\.coverage_rate",
                lambda ablation, weak: weak["overall"].__setitem__(
                    "coverage_rate",
                    1.01,
                ),
            ),
            (
                r"weak\.overall\.rules\.high_risk_reaction\.accuracy",
                lambda ablation, weak: weak["overall"]["rules"][
                    "high_risk_reaction"
                ].__setitem__("accuracy", -0.01),
            ),
            (
                r"ablation\.experiments\.all_tokens\.error_cases\.test"
                r"\.false_positive\[0\]\.predicted_probability",
                lambda ablation, weak: ablation["experiments"]["all_tokens"][
                    "error_cases"
                ]["test"]["false_positive"][0].__setitem__(
                    "predicted_probability",
                    1.01,
                ),
            ),
        )
        for context, mutate in cases:
            with self.subTest(context=context):
                ablation = sample_ablation()
                weak = sample_weak()
                mutate(ablation, weak)
                with self.assertRaisesRegex(ValueError, context):
                    dashboard._dashboard_data(
                        ablation,
                        weak,
                        sample_audit(),
                    )

    def test_dashboard_rejects_inconsistent_audit_quarters(self):
        cases = (
            (
                r"audit\.by_quarter\.2025Q1\.positive",
                lambda audit: audit["by_quarter"]["2025Q1"].__setitem__(
                    "positive",
                    101,
                ),
            ),
            (
                r"audit\.by_quarter.*sum.*audit\.total",
                lambda audit: audit["by_quarter"]["2025Q4"].__setitem__(
                    "n",
                    299,
                ),
            ),
        )
        for context, mutate in cases:
            with self.subTest(context=context):
                audit = sample_audit()
                mutate(audit)
                with self.assertRaisesRegex(ValueError, context):
                    dashboard._dashboard_data(
                        sample_ablation(),
                        sample_weak(),
                        audit,
                    )

    def test_dashboard_rejects_empty_quarters_when_total_is_positive(self):
        audit = {
            "total": 1,
            "by_quarter": {},
        }

        with self.assertRaisesRegex(
            ValueError,
            r"audit\.by_quarter.*sum.*audit\.total",
        ):
            dashboard._dashboard_data(
                sample_ablation(),
                sample_weak(),
                audit,
            )

    def test_dashboard_accepts_zero_total_with_zero_quarters(self):
        audit = {
            "total": 0,
            "by_quarter": {
                "2025Q1": {"n": 0, "positive": 0},
            },
        }

        data = dashboard._dashboard_data(
            sample_ablation(),
            sample_weak(),
            audit,
        )

        self.assertEqual(data["serious_rate"], 0.0)

    def test_dashboard_wraps_numeric_overflow_with_metric_context(self):
        ablation = sample_ablation()
        ablation["experiments"]["all_tokens"]["split_metrics"]["test"][
            "auroc"
        ] = 10**10000

        with self.assertRaisesRegex(
            ValueError,
            r"ablation\.experiments\.all_tokens\.split_metrics\.test\.auroc",
        ):
            dashboard._dashboard_data(
                ablation,
                sample_weak(),
                sample_audit(),
            )

    def test_dashboard_rejects_rule_fires_above_total(self):
        weak = sample_weak()
        weak["overall"]["rules"]["high_risk_reaction"]["fires"] = 401

        with self.assertRaisesRegex(
            ValueError,
            r"weak\.overall\.rules\.high_risk_reaction\.fires",
        ):
            dashboard._dashboard_data(
                sample_ablation(),
                weak,
                sample_audit(),
            )

    def test_dashboard_rejects_error_type_label_mismatch(self):
        cases = (
            (
                "false_positive",
                0,
                0,
            ),
            (
                "false_negative",
                1,
                1,
            ),
        )
        for error_type, predicted_label, true_label in cases:
            with self.subTest(error_type=error_type):
                ablation = sample_ablation()
                source = dict(
                    ablation["experiments"]["all_tokens"]["error_cases"][
                        "test"
                    ]["false_positive"][0]
                )
                source["predicted_label"] = predicted_label
                source["true_label"] = true_label
                ablation["experiments"]["all_tokens"]["error_cases"]["test"][
                    error_type
                ] = [source]
                with self.assertRaisesRegex(
                    ValueError,
                    r"ablation\.experiments\.all_tokens\.error_cases\.test"
                    rf"\.{error_type}\[0\]",
                ):
                    dashboard._dashboard_data(
                        ablation,
                        sample_weak(),
                        sample_audit(),
                    )

    def test_dashboard_without_errors_keeps_filters_and_empty_state(self):
        ablation = sample_ablation()
        for experiment in ablation["experiments"].values():
            experiment["error_cases"] = {
                "test": {
                    "false_positive": [],
                    "false_negative": [],
                }
            }

        text = dashboard.render_dashboard(
            ablation,
            sample_weak(),
            sample_audit(),
        )

        self.assertIn('id="model-filter"', text)
        self.assertIn('id="error-filter"', text)
        self.assertIn("暂无符合筛选条件的失败案例", text)
        self.assertIn("function escapeHtml", text)
        self.assertIn("textContent", text)


class DashboardFileTests(unittest.TestCase):
    def test_generate_dashboard_reads_json_and_writes_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_inputs(root)
            destination = root / "demo" / "index.html"

            result = dashboard.generate_dashboard(root, destination)

            self.assertEqual(result, destination)
            text = destination.read_text(encoding="utf-8")
            self.assertIn("0.9123", text)
            self.assertIn("30.00%", text)
            self.assertIn("FP-1", text)

    def test_atomic_replace_failure_cleans_temp_and_preserves_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_inputs(root)
            destination = root / "index.html"
            destination.write_text("original", encoding="utf-8")

            with mock.patch.object(
                dashboard.os,
                "replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    dashboard.generate_dashboard(root, destination)

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "original",
            )
            leftovers = [
                path
                for path in root.iterdir()
                if path.name.startswith(".index.html.")
            ]
            self.assertEqual(leftovers, [])

    def test_fdopen_failure_closes_raw_fd_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "index.html"
            destination.write_text("original", encoding="utf-8")
            real_close = dashboard.os.close

            with mock.patch.object(
                dashboard.os,
                "fdopen",
                side_effect=OSError("fdopen failed"),
            ), mock.patch.object(
                dashboard.os,
                "close",
                wraps=real_close,
            ) as close_spy:
                with self.assertRaisesRegex(OSError, "fdopen failed"):
                    dashboard._atomic_write_text(
                        destination,
                        "replacement",
                    )

            close_spy.assert_called_once()
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "original",
            )
            leftovers = [
                path
                for path in root.iterdir()
                if path.name.startswith(".index.html.")
            ]
            self.assertEqual(leftovers, [])

    def test_missing_input_file_error_contains_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "ablations" / "ablation_metrics.json"

            with self.assertRaises(FileNotFoundError) as error:
                dashboard.generate_dashboard(
                    root,
                    root / "index.html",
                )

            self.assertIn(str(expected), str(error.exception))

    def test_generate_dashboard_rejects_nonstandard_json_constants_with_path(self):
        for constant in ("NaN", "Infinity"):
            with self.subTest(constant=constant):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    paths = write_inputs(root)
                    paths["ablation"].write_text(
                        '{"experiments": %s}' % constant,
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        rf"{paths['ablation']}.*{constant}",
                    ):
                        dashboard.generate_dashboard(
                            root,
                            root / "index.html",
                        )


FINAL_CASE_FIELDS = [
    "quarter",
    "split",
    "safetyreportid",
    "receivedate",
    "primarysourcecountry",
    "patientsex",
    "age_years",
    "drug_count",
    "suspect_drug_count",
    "reaction_count",
    "indication_count",
    "label_serious",
    "text_tokens",
]


def complete_numeric_baseline():
    metrics = {
        name: value
        for name, value in test_metrics(
            0.75,
            0.74,
            0.7,
            0.68,
            0.69,
            0.08,
            0.8,
        ).items()
        if name in final_reporting.TEST_METRICS
    }
    return {
        "model_path": "outputs/models/numeric_logistic.pkl",
        "train_rows": 6,
        "threshold_from_valid": 0.5,
        "split_metrics": {
            split: dict(metrics)
            for split in ("train", "valid", "test")
        },
        "error_cases": {
            split: {
                "false_positive": [],
                "false_negative": [],
            }
            for split in ("valid", "test")
        },
        "strata": {
            split: {}
            for split in ("valid", "test")
        },
    }


def minimal_case_rows():
    layout = {
        "2025Q1": (("train", 0), ("train", 1)),
        "2025Q2": (("train", 0), ("train", 1)),
        "2025Q3": (("train", 0), ("train", 1)),
        "2025Q4": (
            ("valid", 0),
            ("valid", 1),
            ("test", 0),
            ("test", 1),
        ),
    }
    rows = {}
    for quarter, split_labels in layout.items():
        rows[quarter] = []
        for index, (split, label) in enumerate(split_labels):
            rows[quarter].append(
                {
                    "quarter": quarter,
                    "split": split,
                    "safetyreportid": f"{quarter}-{split}-{index}",
                    "receivedate": "20250101",
                    "primarysourcecountry": "US",
                    "patientsex": "1",
                    "age_years": "60",
                    "drug_count": "2",
                    "suspect_drug_count": "1",
                    "reaction_count": "1",
                    "indication_count": "1",
                    "label_serious": str(label),
                    "text_tokens": "drug:TEST reac:TEST",
                }
            )
    return rows


def complete_feature_audit(rows_by_quarter):
    by_quarter = {}
    by_split = {}
    total = 0
    for quarter, rows in rows_by_quarter.items():
        by_quarter[quarter] = {
            "n": len(rows),
            "positive": sum(int(row["label_serious"]) for row in rows),
        }
        total += len(rows)
        for row in rows:
            split = row["split"]
            bucket = by_split.setdefault(split, {"n": 0, "positive": 0})
            bucket["n"] += 1
            bucket["positive"] += int(row["label_serious"])
    return {
        "total": total,
        "by_quarter": by_quarter,
        "by_split": by_split,
        "missing": {},
    }


def write_final_cli_fixture(
    output_dir,
    *,
    rows_by_quarter=None,
    fieldnames=None,
    filenames=None,
    baseline=None,
    audit=None,
):
    rows_by_quarter = (
        minimal_case_rows()
        if rows_by_quarter is None
        else rows_by_quarter
    )
    fieldnames = FINAL_CASE_FIELDS if fieldnames is None else fieldnames
    filenames = (
        {
            quarter: f"cases_{quarter}.csv"
            for quarter in rows_by_quarter
        }
        if filenames is None
        else filenames
    )
    interim = output_dir / "interim"
    reports = output_dir / "reports"
    interim.mkdir(parents=True)
    reports.mkdir(parents=True)
    for quarter, rows in rows_by_quarter.items():
        path = interim / filenames[quarter]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

    numeric = (
        complete_numeric_baseline()
        if baseline is None
        else baseline
    )
    (reports / "model_metrics.json").write_text(
        json.dumps({"models": {"numeric_logistic": numeric}}),
        encoding="utf-8",
    )
    feature_audit = (
        complete_feature_audit(rows_by_quarter)
        if audit is None
        else audit
    )
    (reports / "feature_audit.json").write_text(
        json.dumps(feature_audit),
        encoding="utf-8",
    )
    return rows_by_quarter


class FinalProjectInputValidationTests(unittest.TestCase):
    def test_validation_requires_exactly_four_quarterly_case_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "interim").mkdir()

            with self.assertRaisesRegex(ValueError, r"2025Q1.*2025Q4"):
                run_final_project.validate_inputs(output_dir)

    def test_validation_rejects_duplicate_and_missing_quarter(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rows = minimal_case_rows()
            filenames = {
                "2025Q1": "cases_2025Q1.csv",
                "2025Q2": "cases_2025Q2.csv",
                "2025Q3": "cases_2025Q3.csv",
                "2025Q4": "cases_2025Q3_copy.csv",
            }
            write_final_cli_fixture(
                output_dir,
                rows_by_quarter=rows,
                filenames=filenames,
            )

            with self.assertRaisesRegex(
                ValueError,
                r"duplicate.*2025Q3.*missing.*2025Q4",
            ):
                run_final_project.validate_inputs(output_dir)

    def test_validation_rejects_extra_case_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_final_cli_fixture(output_dir)
            extra = output_dir / "interim" / "cases_2025Q5.csv"
            extra.write_text("split,label_serious\ntrain,0\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"extra.*cases_2025Q5.csv"):
                run_final_project.validate_inputs(output_dir)

    def test_numeric_baseline_is_checked_before_csv_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            interim = output_dir / "interim"
            interim.mkdir()
            for quarter in range(1, 5):
                (interim / f"cases_2025Q{quarter}.csv").write_text(
                    "split\ntrain\n",
                    encoding="utf-8",
                )

            expected = output_dir / "reports" / "model_metrics.json"
            with self.assertRaises(FileNotFoundError) as error:
                run_final_project.validate_inputs(output_dir)

            self.assertIn(str(expected), str(error.exception))

    def test_validation_rejects_missing_numeric_baseline_before_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            interim = output_dir / "interim"
            reports = output_dir / "reports"
            interim.mkdir()
            reports.mkdir()
            for quarter in range(1, 5):
                (interim / f"cases_2025Q{quarter}.csv").write_text(
                    "split\ntrain\n",
                    encoding="utf-8",
                )
            (reports / "model_metrics.json").write_text(
                json.dumps({"models": {}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "numeric_logistic"):
                run_final_project.validate_inputs(output_dir)

    def test_validation_reports_path_and_missing_schema_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rows = minimal_case_rows()
            write_final_cli_fixture(
                output_dir,
                rows_by_quarter=rows,
                fieldnames=["quarter", "split", "label_serious"],
            )
            expected_path = output_dir / "interim" / "cases_2025Q1.csv"

            with self.assertRaises(ValueError) as error:
                run_final_project.validate_inputs(output_dir)

            message = str(error.exception)
            self.assertIn(str(expected_path), message)
            self.assertIn("text_tokens", message)
            self.assertIn("safetyreportid", message)

    def test_validation_requires_all_splits_and_both_classes(self):
        cases = (
            (
                "missing valid",
                lambda rows: [
                    row.__setitem__("split", "train")
                    for row in rows["2025Q4"]
                    if row["split"] == "valid"
                ],
                r"Split valid.*both classes",
            ),
            (
                "valid single class",
                lambda rows: [
                    row.__setitem__("label_serious", "0")
                    for row in rows["2025Q4"]
                    if row["split"] == "valid"
                ],
                r"Split valid.*both classes",
            ),
            (
                "test single class",
                lambda rows: [
                    row.__setitem__("label_serious", "1")
                    for row in rows["2025Q4"]
                    if row["split"] == "test"
                ],
                r"Split test.*both classes",
            ),
        )
        for name, mutate, expected in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    output_dir = Path(tmp)
                    rows = minimal_case_rows()
                    mutate(rows)
                    write_final_cli_fixture(output_dir, rows_by_quarter=rows)

                    with self.assertRaisesRegex(ValueError, expected):
                        run_final_project.validate_inputs(output_dir)

    def test_validation_rejects_unknown_split_invalid_label_and_quarter_mismatch(self):
        cases = (
            (
                "unknown split",
                lambda rows: rows["2025Q1"][0].__setitem__(
                    "split",
                    "holdout",
                ),
                r"unknown split.*holdout",
            ),
            (
                "invalid label",
                lambda rows: rows["2025Q1"][0].__setitem__(
                    "label_serious",
                    "2",
                ),
                r"label_serious.*0 or 1",
            ),
            (
                "quarter mismatch",
                lambda rows: rows["2025Q1"][0].__setitem__(
                    "quarter",
                    "2025Q2",
                ),
                r"quarter.*2025Q2.*2025Q1",
            ),
        )
        for name, mutate, expected in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    output_dir = Path(tmp)
                    rows = minimal_case_rows()
                    mutate(rows)
                    write_final_cli_fixture(output_dir, rows_by_quarter=rows)

                    with self.assertRaisesRegex(ValueError, expected):
                        run_final_project.validate_inputs(output_dir)

    def test_validation_requires_nonempty_csv_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rows = minimal_case_rows()
            rows["2025Q2"] = []
            write_final_cli_fixture(output_dir, rows_by_quarter=rows)

            with self.assertRaisesRegex(ValueError, r"cases_2025Q2.csv.*at least one"):
                run_final_project.validate_inputs(output_dir)

    def test_validation_rejects_incomplete_numeric_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            baseline = complete_numeric_baseline()
            del baseline["split_metrics"]["test"]["auroc"]
            write_final_cli_fixture(output_dir, baseline=baseline)

            with self.assertRaisesRegex(
                ValueError,
                r"numeric_logistic\.split_metrics\.test\.auroc",
            ):
                run_final_project.validate_inputs(output_dir)

    def test_validation_rejects_incomplete_numeric_error_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            baseline = complete_numeric_baseline()
            del baseline["error_cases"]["test"]["false_positive"]
            write_final_cli_fixture(output_dir, baseline=baseline)

            with self.assertRaisesRegex(
                ValueError,
                r"numeric_logistic\.error_cases\.test\.false_positive",
            ):
                run_final_project.validate_inputs(output_dir)

    def test_validation_wraps_numeric_overflow_with_field_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            baseline = complete_numeric_baseline()
            baseline["split_metrics"]["test"]["auroc"] = 10**10000
            write_final_cli_fixture(output_dir, baseline=baseline)

            with self.assertRaisesRegex(
                ValueError,
                r"numeric_logistic\.split_metrics\.test\.auroc",
            ):
                run_final_project.validate_inputs(output_dir)

    def test_validation_returns_ordered_paths_and_deep_copied_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rows = minimal_case_rows()
            filenames = {
                "2025Q1": "cases_2025q1.csv",
                "2025Q2": "cases_2025Q2.csv",
                "2025Q3": "cases_2025q3.csv",
                "2025Q4": "cases_2025Q4.csv",
            }
            baseline = complete_numeric_baseline()
            write_final_cli_fixture(
                output_dir,
                rows_by_quarter=rows,
                filenames=filenames,
                baseline=baseline,
            )

            paths, result = run_final_project.validate_inputs(output_dir)
            result["split_metrics"]["test"]["auroc"] = 0.01
            persisted = json.loads(
                (output_dir / "reports" / "model_metrics.json").read_text(
                    encoding="utf-8",
                )
            )

            self.assertEqual(
                [path.name for path in paths],
                [
                    "cases_2025q1.csv",
                    "cases_2025Q2.csv",
                    "cases_2025q3.csv",
                    "cases_2025Q4.csv",
                ],
            )
            self.assertEqual(
                persisted["models"]["numeric_logistic"]["split_metrics"][
                    "test"
                ]["auroc"],
                baseline["split_metrics"]["test"]["auroc"],
            )

    def test_validation_rejects_nonstandard_json_constants(self):
        cases = (
            ("model_metrics.json", '{"models":{"numeric_logistic":NaN}}'),
            ("feature_audit.json", '{"total":NaN}'),
        )
        for filename, payload in cases:
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as tmp:
                    output_dir = Path(tmp)
                    write_final_cli_fixture(output_dir)
                    path = output_dir / "reports" / filename
                    path.write_text(payload, encoding="utf-8")

                    with self.assertRaisesRegex(
                        ValueError,
                        rf"{filename}.*NaN",
                    ):
                        run_final_project.validate_inputs(output_dir)

    def test_validation_rejects_audit_total_quarter_and_split_mismatch(self):
        cases = (
            (
                "total",
                lambda audit: audit.__setitem__("total", 11),
                r"feature_audit\.total.*10",
            ),
            (
                "quarter",
                lambda audit: audit["by_quarter"]["2025Q4"].__setitem__(
                    "positive",
                    1,
                ),
                r"feature_audit\.by_quarter\.2025Q4.*positive",
            ),
            (
                "split",
                lambda audit: audit["by_split"]["test"].__setitem__("n", 3),
                r"feature_audit\.by_split\.test.*n",
            ),
        )
        for name, mutate, expected in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    output_dir = Path(tmp)
                    rows = minimal_case_rows()
                    audit = complete_feature_audit(rows)
                    mutate(audit)
                    write_final_cli_fixture(
                        output_dir,
                        rows_by_quarter=rows,
                        audit=audit,
                    )

                    with self.assertRaisesRegex(ValueError, expected):
                        run_final_project.validate_inputs(output_dir)


class FinalProjectCliTests(unittest.TestCase):
    def test_main_runs_final_steps_in_order_with_custom_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "custom-output"
            report = root / "report.md"
            dashboard_path = root / "dashboard.html"
            case_paths = [root / f"cases_2025Q{quarter}.csv" for quarter in range(1, 5)]
            baseline = complete_numeric_baseline()
            calls = []

            with mock.patch.object(
                run_final_project,
                "validate_inputs",
                return_value=(case_paths, baseline),
            ) as validate, mock.patch.object(
                run_final_project.final_experiments,
                "run_ablation_experiments",
                side_effect=lambda *args, **kwargs: calls.append("ablation"),
            ) as ablation, mock.patch.object(
                run_final_project.weak_supervision,
                "run_weak_supervision",
                side_effect=lambda *args, **kwargs: calls.append("weak"),
            ) as weak, mock.patch.object(
                run_final_project.final_reporting,
                "generate_final_report",
                side_effect=lambda *args, **kwargs: calls.append("report"),
            ) as report_generator, mock.patch.object(
                run_final_project.dashboard,
                "generate_dashboard",
                side_effect=lambda *args, **kwargs: calls.append("dashboard"),
            ) as dashboard_generator, mock.patch(
                "builtins.print",
            ) as printer:
                result = run_final_project.main(
                    [
                        "--out",
                        str(output_dir),
                        "--report",
                        str(report),
                        "--dashboard",
                        str(dashboard_path),
                        "--chunk-size",
                        "12",
                        "--n-features",
                        "64",
                        "--epochs",
                        "3",
                        "--learning-rate",
                        "0.2",
                        "--l2",
                        "0.01",
                        "--random-state",
                        "7",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(calls, ["ablation", "weak", "report", "dashboard"])
            validate.assert_called_once_with(output_dir)
            ablation.assert_called_once_with(
                case_paths,
                output_dir / "ablations",
                baseline,
                chunk_size=12,
                n_features=64,
                epochs=3,
                learning_rate=0.2,
                l2=0.01,
                random_state=7,
            )
            weak.assert_called_once_with(
                case_paths,
                output_dir / "weak_supervision",
            )
            report_generator.assert_called_once_with(output_dir, report)
            dashboard_generator.assert_called_once_with(
                output_dir,
                dashboard_path,
            )
            output = "\n".join(str(call.args[0]) for call in printer.call_args_list)
            self.assertEqual(printer.call_count, 4)
            self.assertIn(
                str(output_dir / "ablations" / "ablation_metrics.json"),
                output,
            )
            self.assertIn(
                str(
                    output_dir
                    / "weak_supervision"
                    / "weak_supervision_metrics.json"
                ),
                output,
            )
            self.assertIn(str(report), output)
            self.assertIn(str(dashboard_path), output)

    def test_main_uses_root_based_defaults(self):
        case_paths = [
            run_final_project.ROOT / f"cases_2025Q{quarter}.csv"
            for quarter in range(1, 5)
        ]
        baseline = complete_numeric_baseline()
        with mock.patch.object(
            run_final_project,
            "validate_inputs",
            return_value=(case_paths, baseline),
        ) as validate, mock.patch.object(
            run_final_project.final_experiments,
            "run_ablation_experiments",
        ), mock.patch.object(
            run_final_project.weak_supervision,
            "run_weak_supervision",
        ), mock.patch.object(
            run_final_project.final_reporting,
            "generate_final_report",
        ) as report_generator, mock.patch.object(
            run_final_project.dashboard,
            "generate_dashboard",
        ) as dashboard_generator, mock.patch(
            "builtins.print",
        ):
            result = run_final_project.main([])

        self.assertEqual(result, 0)
        validate.assert_called_once_with(run_final_project.ROOT / "outputs")
        report_generator.assert_called_once_with(
            run_final_project.ROOT / "outputs",
            run_final_project.ROOT / "最终报告.md",
        )
        dashboard_generator.assert_called_once_with(
            run_final_project.ROOT / "outputs",
            run_final_project.ROOT / "demo" / "index.html",
        )

    def test_main_does_not_run_later_steps_or_print_completion_on_failure(self):
        case_paths = [Path(f"cases_2025Q{quarter}.csv") for quarter in range(1, 5)]
        with mock.patch.object(
            run_final_project,
            "validate_inputs",
            return_value=(case_paths, complete_numeric_baseline()),
        ), mock.patch.object(
            run_final_project.final_experiments,
            "run_ablation_experiments",
        ), mock.patch.object(
            run_final_project.weak_supervision,
            "run_weak_supervision",
            side_effect=RuntimeError("weak failed"),
        ), mock.patch.object(
            run_final_project.final_reporting,
            "generate_final_report",
        ) as report_generator, mock.patch.object(
            run_final_project.dashboard,
            "generate_dashboard",
        ) as dashboard_generator, mock.patch(
            "builtins.print",
        ) as printer:
            with self.assertRaisesRegex(RuntimeError, "weak failed"):
                run_final_project.main(["--out", "custom"])

        report_generator.assert_not_called()
        dashboard_generator.assert_not_called()
        printer.assert_not_called()


class FinalProjectReadmeTests(unittest.TestCase):
    def test_readme_documents_final_commands_and_artifacts(self):
        text = (ROOT / "readme.md").read_text(encoding="utf-8")

        self.assertIn("## 终期实验、最终报告与 Demo", text)
        self.assertIn(
            "python3 scripts/run_final_project.py --out outputs",
            text,
        )
        self.assertIn(
            "python3 scripts/run_final_project.py --out outputs_sample",
            text,
        )
        for artifact in (
            "outputs/ablations/ablation_metrics.json",
            "outputs/weak_supervision/weak_supervision_metrics.json",
            "最终报告.md",
            "demo/index.html",
        ):
            self.assertIn(artifact, text)
        self.assertIn("完成基础流水线后", text)
        self.assertIn("双击", text)
        self.assertIn("离线", text)
        self.assertNotIn("-  数据依据说明", text)


if __name__ == "__main__":
    unittest.main()
