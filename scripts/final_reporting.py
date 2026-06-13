"""Render the final Markdown report from structured experiment outputs."""

from __future__ import annotations

import json
import math
import os
import tempfile
from decimal import Decimal, InvalidOperation
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


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected mapping, got {type(value).__name__}")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected list, got {type(value).__name__}")
    return value


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if not isinstance(mapping, dict):
        raise ValueError(f"{context}: parent must be a mapping")
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{context}: required value is missing")
    return mapping[key]


def _present(mapping: dict[str, Any], key: str, context: str) -> Any:
    if not isinstance(mapping, dict):
        raise ValueError(f"{context}: parent must be a mapping")
    if key not in mapping:
        raise ValueError(f"{context}: required value is missing")
    return mapping[key]


def _number(value: Any, context: str, kind: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{context}: required {kind} is invalid: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}: required {kind} is invalid: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: required {kind} is invalid: {value!r}")
    return number


def _metric(value: Any, context: str) -> str:
    number = _number(value, context, "metric")
    return f"{number:.4f}"


def _percent(value: Any, context: str) -> str:
    number = _number(value, context, "percentage")
    return f"{100.0 * number:.2f}%"


def _integer(value: Any, context: str) -> int:
    invalid = f"{context}: required count is invalid: {value!r}"
    if value is None or isinstance(value, bool):
        raise ValueError(invalid)

    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(invalid)
        number = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(invalid)
        try:
            decimal_value = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(invalid) from exc
        if (
            not decimal_value.is_finite()
            or decimal_value != decimal_value.to_integral_value()
        ):
            raise ValueError(invalid)
        number = int(decimal_value)
    else:
        raise ValueError(invalid)

    if number < 0:
        raise ValueError(invalid)
    return number


def _escape_cell(value: Any) -> str:
    return (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("|", "\\|")
    )


def _nullable_metric(value: Any, context: str) -> str:
    if value is None:
        return "NA"
    return _metric(value, context)


def _test_metrics(
    experiment: dict[str, Any],
    experiment_name: str,
) -> dict[str, Any]:
    experiment_context = f"ablation.experiments.{experiment_name}"
    split_metrics = _mapping(
        _required(
            experiment,
            "split_metrics",
            f"{experiment_context}.split_metrics",
        ),
        f"{experiment_context}.split_metrics",
    )
    test = _mapping(
        _required(
            split_metrics,
            "test",
            f"{experiment_context}.split_metrics.test",
        ),
        f"{experiment_context}.split_metrics.test",
    )
    for metric_name in TEST_METRICS:
        metric_context = (
            f"{experiment_context}.split_metrics.test.{metric_name}"
        )
        value = _required(
            test,
            metric_name,
            metric_context,
        )
        _metric(value, metric_context)
    return test


def _change_description(
    reference: Any,
    comparison: Any,
    reference_context: str,
    comparison_context: str,
) -> str:
    reference_value = _number(reference, reference_context, "metric")
    comparison_value = _number(comparison, comparison_context, "metric")
    change = reference_value - comparison_value
    if change >= 0:
        return f"下降 {_metric(change, f'{comparison_context}.change')}"
    return f"上升 {_metric(-change, f'{comparison_context}.change')}"


def _weak_summary_row(
    label: str,
    bucket: dict[str, Any],
    context: str,
) -> str:
    total = _integer(
        _required(bucket, "total", f"{context}.total"),
        f"{context}.total",
    )
    covered = _integer(
        _required(bucket, "covered", f"{context}.covered"),
        f"{context}.covered",
    )
    coverage_rate = _required(
        bucket,
        "coverage_rate",
        f"{context}.coverage_rate",
    )
    conflicts = _integer(
        _required(bucket, "conflicts", f"{context}.conflicts"),
        f"{context}.conflicts",
    )
    conflict_rate = _required(
        bucket,
        "conflict_rate",
        f"{context}.conflict_rate",
    )
    voted = _integer(
        _required(bucket, "voted", f"{context}.voted"),
        f"{context}.voted",
    )
    accuracy = _required(bucket, "accuracy", f"{context}.accuracy")
    precision = _required(bucket, "precision", f"{context}.precision")
    recall = _required(bucket, "recall", f"{context}.recall")
    f1 = _required(bucket, "f1", f"{context}.f1")
    return (
        f"| {label} | {total:,} | {covered:,} | "
        f"{_percent(coverage_rate, f'{context}.coverage_rate')} | "
        f"{conflicts:,} | "
        f"{_percent(conflict_rate, f'{context}.conflict_rate')} | "
        f"{voted:,} | {_percent(accuracy, f'{context}.accuracy')} | "
        f"{_percent(precision, f'{context}.precision')} | "
        f"{_percent(recall, f'{context}.recall')} | "
        f"{_metric(f1, f'{context}.f1')} |"
    )


def render_final_report(
    ablation: dict[str, Any],
    weak: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    ablation = _mapping(ablation, "ablation")
    experiments = _mapping(
        _required(ablation, "experiments", "ablation.experiments"),
        "ablation.experiments",
    )
    experiment_items: dict[str, dict[str, Any]] = {}
    test_results: dict[str, dict[str, Any]] = {}
    for name in EXPERIMENT_ORDER:
        experiment_context = f"ablation.experiments.{name}"
        experiment = _mapping(
            _required(experiments, name, experiment_context),
            experiment_context,
        )
        experiment_items[name] = experiment
        test_results[name] = _test_metrics(experiment, name)

    audit = _mapping(audit, "audit")
    total = _integer(
        _required(audit, "total", "audit.total"),
        "audit.total",
    )
    by_quarter = _mapping(
        _required(audit, "by_quarter", "audit.by_quarter"),
        "audit.by_quarter",
    )
    missing = _mapping(
        _required(audit, "missing", "audit.missing"),
        "audit.missing",
    )

    weak = _mapping(weak, "weak")
    weak_overall = _mapping(
        _required(weak, "overall", "weak.overall"),
        "weak.overall",
    )
    weak_splits = _mapping(
        _required(weak, "splits", "weak.splits"),
        "weak.splits",
    )
    weak_test = _mapping(
        _required(weak_splits, "test", "weak.splits.test"),
        "weak.splits.test",
    )
    rules = _mapping(
        _required(weak_overall, "rules", "weak.overall.rules"),
        "weak.overall.rules",
    )
    weak_summary_rows = [
        _weak_summary_row("overall", weak_overall, "weak.overall"),
        _weak_summary_row("test", weak_test, "weak.splits.test"),
    ]

    all_tokens = experiment_items["all_tokens"]
    error_splits = _mapping(
        _required(
            all_tokens,
            "error_cases",
            "ablation.experiments.all_tokens.error_cases",
        ),
        "ablation.experiments.all_tokens.error_cases",
    )
    test_errors = _mapping(
        _required(
            error_splits,
            "test",
            "ablation.experiments.all_tokens.error_cases.test",
        ),
        "ablation.experiments.all_tokens.error_cases.test",
    )
    strata_splits = _mapping(
        _required(
            all_tokens,
            "strata",
            "ablation.experiments.all_tokens.strata",
        ),
        "ablation.experiments.all_tokens.strata",
    )
    test_strata = _mapping(
        _required(
            strata_splits,
            "test",
            "ablation.experiments.all_tokens.strata.test",
        ),
        "ablation.experiments.all_tokens.strata.test",
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
    for quarter, raw_item in sorted(
        by_quarter.items(),
        key=lambda pair: str(pair[0]),
    ):
        quarter_context = f"audit.by_quarter.{quarter}"
        item = _mapping(raw_item, quarter_context)
        quarter_n = _integer(
            _required(item, "n", f"{quarter_context}.n"),
            f"{quarter_context}.n",
        )
        positive = _integer(
            _required(item, "positive", f"{quarter_context}.positive"),
            f"{quarter_context}.positive",
        )
        serious_rate = positive / quarter_n if quarter_n else 0.0
        lines.append(
            f"| {_escape_cell(quarter)} | {quarter_n:,} | {positive:,} | "
            f"{_percent(serious_rate, f'{quarter_context}.positive_rate')} |"
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
    for field, value in sorted(missing.items(), key=lambda pair: str(pair[0])):
        missing_context = f"audit.missing.{field}"
        missing_count = _integer(value, missing_context)
        missing_rate = missing_count / total if total else 0.0
        lines.append(
            f"| `{_escape_cell(field)}` | {missing_count:,} | "
            f"{_percent(missing_rate, f'{missing_context}.rate')} |"
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
        test_context = f"ablation.experiments.{name}.split_metrics.test"
        lines.append(
            f"| `{name}` | {_metric(test['auroc'], f'{test_context}.auroc')} | "
            f"{_metric(test['auprc'], f'{test_context}.auprc')} | "
            f"{_metric(test['precision'], f'{test_context}.precision')} | "
            f"{_metric(test['recall'], f'{test_context}.recall')} | "
            f"{_metric(test['f1'], f'{test_context}.f1')} | "
            f"{_metric(test['recall_at_top_5pct'], f'{test_context}.recall_at_top_5pct')} | "
            f"{_metric(test['hit_rate_top_5pct'], f'{test_context}.hit_rate_top_5pct')} |"
        )

    all_auroc_context = (
        "ablation.experiments.all_tokens.split_metrics.test.auroc"
    )
    no_reaction_auroc_context = (
        "ablation.experiments.without_reaction_pt.split_metrics.test.auroc"
    )
    no_outcome_auroc_context = (
        "ablation.experiments.without_reaction_outcome.split_metrics.test.auroc"
    )
    reaction_change = _change_description(
        all_test["auroc"],
        no_reaction_test["auroc"],
        all_auroc_context,
        no_reaction_auroc_context,
    )
    outcome_change = _change_description(
        all_test["auroc"],
        no_outcome_test["auroc"],
        all_auroc_context,
        no_outcome_auroc_context,
    )
    lines.extend(
        [
            "",
            "## 语义捷径诊断",
            "",
            f"完整模型测试集 AUROC 为 "
            f"{_metric(all_test['auroc'], all_auroc_context)}。"
            f"移除反应 PT 后 AUROC 为 "
            f"{_metric(no_reaction_test['auroc'], no_reaction_auroc_context)}，"
            f"相对完整模型{reaction_change}；移除 `reactionoutcome:*` 后 AUROC 为 "
            f"{_metric(no_outcome_test['auroc'], no_outcome_auroc_context)}，"
            f"相对完整模型{outcome_change}。",
            "",
            "两项变化量化了反应语义对预测性能的贡献。若指标明显下降，应将完整模型"
            "结果解释为病例重症识别能力，而不是未经约束的药物因果风险估计。",
            "",
            "## 弱监督扩展",
            "",
            "弱监督指标只针对规则非弃权且非冲突的覆盖子集，不能与完整测试集模型"
            "指标作无条件横向比较。",
            "",
            "| 范围 | 总样本 | 已覆盖 | 覆盖率 | 冲突数 | 冲突率 | 有效投票 | "
            "Accuracy | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: |",
            *weak_summary_rows,
            "",
            "| 规则 | 触发数 | 覆盖率 | 命中样本重症率 | 规则准确率 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, raw_item in sorted(rules.items(), key=lambda pair: str(pair[0])):
        rule_context = f"weak.overall.rules.{name}"
        item = _mapping(raw_item, rule_context)
        fires = _integer(
            _required(item, "fires", f"{rule_context}.fires"),
            f"{rule_context}.fires",
        )
        coverage = _required(
            item,
            "coverage_rate",
            f"{rule_context}.coverage_rate",
        )
        positive_rate = _required(
            item,
            "positive_label_rate",
            f"{rule_context}.positive_label_rate",
        )
        accuracy = _required(
            item,
            "accuracy",
            f"{rule_context}.accuracy",
        )
        lines.append(
            f"| `{_escape_cell(name)}` | {fires:,} | "
            f"{_percent(coverage, f'{rule_context}.coverage_rate')} | "
            f"{_percent(positive_rate, f'{rule_context}.positive_label_rate')} | "
            f"{_percent(accuracy, f'{rule_context}.accuracy')} |"
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
        cases_context = (
            "ablation.experiments.all_tokens.error_cases.test."
            f"{error_type}"
        )
        cases = _list(
            _required(test_errors, error_type, cases_context),
            cases_context,
        )
        for index, raw_case in enumerate(cases):
            case_context = f"{cases_context}.{index}"
            case = _mapping(raw_case, case_context)
            report_id = _required(
                case,
                "safetyreportid",
                f"{case_context}.safetyreportid",
            )
            probability = _required(
                case,
                "predicted_probability",
                f"{case_context}.predicted_probability",
            )
            true_label = _integer(
                _required(
                    case,
                    "true_label",
                    f"{case_context}.true_label",
                ),
                f"{case_context}.true_label",
            )
            tokens = _required(
                case,
                "tokens",
                f"{case_context}.tokens",
            )
            lines.append(
                f"| `all_tokens` | {error_type} | "
                f"{_escape_cell(report_id)} | "
                f"{_metric(probability, f'{case_context}.predicted_probability')} | "
                f"{_escape_cell(true_label)} | {_escape_cell(tokens)} |"
            )

    lines.extend(
        [
            "",
            "## 分层误差分析",
            "",
            "| 分层字段 | 分组 | 样本数 | AUROC | AUPRC | Precision | Recall | F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group_name, raw_groups in sorted(
        test_strata.items(),
        key=lambda pair: str(pair[0]),
    ):
        group_context = (
            "ablation.experiments.all_tokens.strata.test."
            f"{group_name}"
        )
        groups = _mapping(raw_groups, group_context)
        for value, raw_metrics in sorted(
            groups.items(),
            key=lambda pair: str(pair[0]),
        ):
            context = f"{group_context}.{value}"
            metrics = _mapping(raw_metrics, context)
            count = _integer(
                _required(metrics, "n", f"{context}.n"),
                f"{context}.n",
            )
            auroc = _nullable_metric(
                _present(metrics, "auroc", f"{context}.auroc"),
                f"{context}.auroc",
            )
            auprc = _nullable_metric(
                _present(metrics, "auprc", f"{context}.auprc"),
                f"{context}.auprc",
            )
            precision = _nullable_metric(
                _present(metrics, "precision", f"{context}.precision"),
                f"{context}.precision",
            )
            recall = _nullable_metric(
                _present(metrics, "recall", f"{context}.recall"),
                f"{context}.recall",
            )
            f1 = _nullable_metric(
                _present(metrics, "f1", f"{context}.f1"),
                f"{context}.f1",
            )
            lines.append(
                f"| {_escape_cell(group_name)} | {_escape_cell(value)} | "
                f"{count:,} | {auroc} | {auprc} | {precision} | "
                f"{recall} | {f1} |"
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
            "python3 scripts/run_final_project.py --out outputs_sample "
            "--report /tmp/DataMining-FDA-sample-report.md "
            "--dashboard /tmp/DataMining-FDA-sample-dashboard.html",
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
    fd_owned = True
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
        fd_owned = False
        with handle:
            handle.write(text)
        os.replace(temp_name, path)
    except Exception:
        if fd_owned:
            try:
                os.close(fd)
            except OSError:
                pass
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
