"""Render the final Markdown report from structured experiment outputs."""

from __future__ import annotations

import json
import math
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

TEST_METRICS = (
    "auroc",
    "auprc",
    "precision",
    "recall",
    "f1",
    "recall_at_top_5pct",
    "hit_rate_top_5pct",
)


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping or mapping[key] is None:
        raise ValueError(f"Missing required {context}: {key}")
    return mapping[key]


def _metric(value: Any) -> str:
    if value is None:
        raise ValueError("Required metric is missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Required metric is invalid: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Required metric is invalid: {value!r}")
    return f"{number:.4f}"


def _percent(value: Any) -> str:
    if value is None:
        raise ValueError("Required percentage is missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Required percentage is invalid: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Required percentage is invalid: {value!r}")
    return f"{100.0 * number:.2f}%"


def _integer(value: Any, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Required count is invalid for {context}: {value!r}") from exc


def _escape_cell(value: Any) -> str:
    return (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("|", "\\|")
    )


def _stratum_metric(
    metrics: dict[str, Any],
    key: str,
    context: str,
) -> str:
    if not isinstance(metrics, dict) or key not in metrics:
        raise ValueError(f"Missing required {context}: {key}")
    value = metrics[key]
    if value is None:
        return "NA"
    return _metric(value)


def _test_metrics(
    experiment: dict[str, Any],
    experiment_name: str,
) -> dict[str, Any]:
    split_metrics = _required(
        experiment,
        "split_metrics",
        f"{experiment_name} field",
    )
    test = _required(
        split_metrics,
        "test",
        f"{experiment_name} split",
    )
    for metric_name in TEST_METRICS:
        value = _required(
            test,
            metric_name,
            f"{experiment_name} test metric",
        )
        _metric(value)
    return test


def _change_description(reference: Any, comparison: Any) -> str:
    reference_value = float(reference)
    comparison_value = float(comparison)
    change = reference_value - comparison_value
    if change >= 0:
        return f"下降 {_metric(change)}"
    return f"上升 {_metric(-change)}"


def render_final_report(
    ablation: dict[str, Any],
    weak: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    experiments = _required(ablation, "experiments", "ablation field")
    experiment_items: dict[str, dict[str, Any]] = {}
    test_results: dict[str, dict[str, Any]] = {}
    for name in EXPERIMENT_ORDER:
        experiment = _required(experiments, name, "experiment")
        experiment_items[name] = experiment
        test_results[name] = _test_metrics(experiment, name)

    total = _integer(_required(audit, "total", "audit field"), "audit total")
    by_quarter = _required(audit, "by_quarter", "audit field")
    missing = _required(audit, "missing", "audit field")

    weak_overall = _required(weak, "overall", "weak supervision field")
    weak_splits = _required(weak, "splits", "weak supervision field")
    weak_test = _required(
        weak_splits,
        "test",
        "weak supervision split",
    )
    overall_coverage = _required(
        weak_overall,
        "coverage_rate",
        "weak overall metric",
    )
    overall_conflict = _required(
        weak_overall,
        "conflict_rate",
        "weak overall metric",
    )
    test_accuracy = _required(
        weak_test,
        "accuracy",
        "weak test metric",
    )
    test_f1 = _required(weak_test, "f1", "weak test metric")
    rules = _required(weak_overall, "rules", "weak overall field")
    _percent(overall_coverage)
    _percent(overall_conflict)
    _percent(test_accuracy)
    _metric(test_f1)

    all_tokens = experiment_items["all_tokens"]
    error_splits = _required(
        all_tokens,
        "error_cases",
        "all_tokens field",
    )
    test_errors = _required(
        error_splits,
        "test",
        "all_tokens error split",
    )
    strata_splits = _required(
        all_tokens,
        "strata",
        "all_tokens field",
    )
    test_strata = _required(
        strata_splits,
        "test",
        "all_tokens strata split",
    )

    all_test = test_results["all_tokens"]
    no_reaction_test = test_results["without_reaction_pt"]
    no_outcome_test = test_results["without_reaction_outcome"]

    lines = [
        "# 数据挖掘课程项目最终报告",
        "",
        "## 项目摘要与研究问题",
        "",
        f"本项目基于 2025 年 FDA 不良事件报告系统（FAERS）构建了 "
        f"{total:,} 条病例级样本，完成数据治理、时间切分、风险预测、"
        "特征族消融和弱监督审计。",
        "",
        "研究问题包括：病例级结构化与文本 token 能否预测重症报告；"
        "反应术语及反应结果是否形成语义捷径；规则弱标签在何种覆盖范围内"
        "可提供可解释的风险信号。",
        "",
        "## 数据来源与治理",
        "",
        "- 数据来源：FDA FAERS 2025Q1-Q4 XML 季度数据。",
        "- 时间切分：2025Q1-Q3 用于训练，2025Q4 按接收日期划分验证集与测试集。",
        "- 标签治理：严重性字段仅用于生成 `label_serious`，不进入模型特征。",
        "- 产物治理：病例级 CSV、模型和实验 JSON 保存在忽略目录中，报告由 JSON 自动生成。",
        "",
        "### 季度样本与重症率",
        "",
        "| 季度 | 样本数 | 重症数 | 重症率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for quarter, item in sorted(by_quarter.items()):
        quarter_context = f"audit quarter {quarter}"
        quarter_n = _integer(
            _required(item, "n", quarter_context),
            f"{quarter_context} n",
        )
        positive = _integer(
            _required(item, "positive", quarter_context),
            f"{quarter_context} positive",
        )
        serious_rate = positive / quarter_n if quarter_n else 0.0
        lines.append(
            f"| {_escape_cell(quarter)} | {quarter_n:,} | {positive:,} | "
            f"{_percent(serious_rate)} |"
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
    for field, value in sorted(missing.items()):
        missing_count = _integer(value, f"audit missing {field}")
        missing_rate = missing_count / total if total else 0.0
        lines.append(
            f"| `{_escape_cell(field)}` | {missing_count:,} | "
            f"{_percent(missing_rate)} |"
        )

    lines.extend(
        [
            "",
            "## 方法",
            "",
            "- 数值基线：对数值聚合特征进行标准化后训练 NumPy 逻辑回归。",
            "- 主模型：使用固定维度稳定哈希表示结构化、分箱和文本 token，"
            "通过在线 NumPy 逻辑回归训练。",
            "- 评测：阈值仅由验证集选择，测试集报告 AUROC、AUPRC、"
            "Precision、Recall、F1 与 Top 5% 排序指标。",
            "- 消融：分别移除反应 PT、移除反应结果、仅保留药物与适应症、"
            "仅保留结构化特征，并与数值基线比较。",
            "- 弱监督：规则投票仅用于覆盖率、冲突率和一致性审计，不参与模型训练。",
            "",
            "## 消融实验结果",
            "",
            "| 实验 | AUROC | AUPRC | Precision | Recall | F1 | "
            "Recall@Top5% | Hit Rate@Top5% |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in EXPERIMENT_ORDER:
        test = test_results[name]
        lines.append(
            f"| `{name}` | {_metric(test['auroc'])} | "
            f"{_metric(test['auprc'])} | {_metric(test['precision'])} | "
            f"{_metric(test['recall'])} | {_metric(test['f1'])} | "
            f"{_metric(test['recall_at_top_5pct'])} | "
            f"{_metric(test['hit_rate_top_5pct'])} |"
        )

    reaction_change = _change_description(
        all_test["auroc"],
        no_reaction_test["auroc"],
    )
    outcome_change = _change_description(
        all_test["auroc"],
        no_outcome_test["auroc"],
    )
    lines.extend(
        [
            "",
            "## 语义捷径诊断",
            "",
            f"完整模型测试集 AUROC 为 {_metric(all_test['auroc'])}。"
            f"移除反应 PT 后 AUROC 为 {_metric(no_reaction_test['auroc'])}，"
            f"相对完整模型{reaction_change}；移除 `reactionoutcome:*` 后 AUROC 为 "
            f"{_metric(no_outcome_test['auroc'])}，相对完整模型{outcome_change}。",
            "",
            "两项变化量化了反应语义对预测性能的贡献。若指标明显下降，应将完整模型"
            "结果解释为病例重症识别能力，而不是未经约束的药物因果风险估计。",
            "",
            "## 弱监督扩展",
            "",
            f"- 总体规则覆盖率：{_percent(overall_coverage)}",
            f"- 总体冲突率：{_percent(overall_conflict)}",
            f"- 测试集非弃权弱标签准确率：{_percent(test_accuracy)}",
            f"- 测试集弱标签 F1：{_metric(test_f1)}",
            "",
            "弱监督指标只针对规则非弃权且非冲突的覆盖子集，不能与完整测试集模型"
            "指标作无条件横向比较。",
            "",
            "| 规则 | 触发数 | 覆盖率 | 命中样本重症率 | 规则准确率 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, item in sorted(rules.items()):
        rule_context = f"weak rule {name}"
        fires = _integer(
            _required(item, "fires", rule_context),
            f"{rule_context} fires",
        )
        coverage = _required(item, "coverage_rate", rule_context)
        positive_rate = _required(item, "positive_label_rate", rule_context)
        accuracy = _required(item, "accuracy", rule_context)
        lines.append(
            f"| `{_escape_cell(name)}` | {fires:,} | {_percent(coverage)} | "
            f"{_percent(positive_rate)} | {_percent(accuracy)} |"
        )

    lines.extend(
        [
            "",
            "## 失败案例",
            "",
            "| 模型 | 错误类型 | 病例 ID | 预测概率 | 真实标签 | token 摘要 |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for error_type in ("false_positive", "false_negative"):
        cases = _required(
            test_errors,
            error_type,
            "all_tokens test error type",
        )
        for case in cases:
            case_context = f"all_tokens {error_type} case"
            report_id = _required(case, "safetyreportid", case_context)
            probability = _required(
                case,
                "predicted_probability",
                case_context,
            )
            true_label = _required(case, "true_label", case_context)
            tokens = _required(case, "tokens", case_context)
            lines.append(
                f"| `all_tokens` | {error_type} | "
                f"{_escape_cell(report_id)} | {_metric(probability)} | "
                f"{_escape_cell(true_label)} | {_escape_cell(tokens)} |"
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
    for group_name, groups in sorted(test_strata.items()):
        for value, metrics in sorted(groups.items()):
            context = f"all_tokens stratum {group_name}={value}"
            count = _integer(
                _required(metrics, "n", context),
                f"{context} n",
            )
            auroc = _stratum_metric(metrics, "auroc", context)
            auprc = _stratum_metric(metrics, "auprc", context)
            f1 = _metric(_required(metrics, "f1", context))
            lines.append(
                f"| {_escape_cell(group_name)} | {_escape_cell(value)} | "
                f"{count:,} | {auroc} | {auprc} | {f1} |"
            )

    lines.extend(
        [
            "",
            "## 局限性与结论",
            "",
            "- FAERS 是自发报告系统，存在漏报、重复、选择偏倚与字段缺失，"
            "不能直接估计药物真实发生率。",
            "- 反应 PT 与反应结果可能与重症标签同源，完整模型的高指标包含语义捷径贡献。",
            "- 弱监督规则只覆盖部分病例，规则准确率受规则触发分布影响。",
            "- 当前药名规范化、剂量信息和外部医学知识仍有限，后续可接入标准术语映射"
            "并开展跨年度外部验证。",
            "",
            "结论：完整 token 模型提供了病例级重症识别能力；消融结果用于界定不同"
            "特征族的贡献，弱监督结果提供可解释但覆盖有限的独立审计信号。",
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
            "已有病例级 CSV 时运行终期实验：",
            "",
            "```bash",
            "python3 scripts/run_final_project.py --out outputs",
            "```",
            "",
            "使用 `outputs_sample` 快速验证：",
            "",
            "```bash",
            "python3 scripts/run_final_project.py --out outputs_sample",
            "```",
            "",
            "## AI 工具辅助使用声明",
            "",
            "本项目使用 ChatGPT/Codex 辅助代码实现、测试设计、实验汇总和报告排版；"
            "所有结果值均来自本地结构化产物，并通过自动化测试和端到端运行核验。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
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
    ablation_path = output_dir / "ablations" / "ablation_metrics.json"
    weak_path = (
        output_dir
        / "weak_supervision"
        / "weak_supervision_metrics.json"
    )
    audit_path = output_dir / "reports" / "feature_audit.json"

    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    weak = json.loads(weak_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    text = render_final_report(ablation, weak, audit)
    _atomic_write_text(destination, text)
    return destination
