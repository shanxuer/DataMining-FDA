"""Generate a self-contained offline dashboard for final experiment outputs."""

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

MODEL_METRICS = ("auroc", "auprc", "f1")
WEAK_METRICS = (
    "coverage_rate",
    "conflict_rate",
    "accuracy",
    "precision",
    "recall",
    "f1",
)
RULE_ORDER = (
    "death_or_fatal_reaction_term",
    "high_polypharmacy_10plus",
    "senior_with_multiple_suspect_drugs",
    "low_complexity_younger_case",
    "high_risk_reaction",
    "serious_reaction_outcome",
    "extreme_polypharmacy_30plus",
    "device_product_issue",
)
ERROR_SCALAR_FIELDS = (
    "safetyreportid",
    "receivedate",
    "patientsex",
    "age_years",
    "drug_count",
    "reaction_count",
    "indication_count",
    "tokens",
)
ERROR_OUTPUT_FIELDS = (
    *ERROR_SCALAR_FIELDS,
    "predicted_probability",
    "predicted_label",
    "true_label",
)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{context}: expected mapping, got {type(value).__name__}"
        )
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(
            f"{context}: expected list, got {type(value).__name__}"
        )
    return value


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{context}: required value is missing")
    return mapping[key]


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: required number is invalid: {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"{context}: required number is invalid: numeric overflow"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: required number is invalid: {value!r}")
    return number


def _integer(value: Any, context: str) -> int:
    number = _number(value, context)
    if not number.is_integer() or number < 0:
        raise ValueError(f"{context}: required count is invalid: {value!r}")
    return int(number)


def _unit_interval(value: Any, context: str) -> float:
    number = _number(value, context)
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{context}: required value must be between 0 and 1: {value!r}"
        )
    return number


def _safe_scalar(value: Any, context: str) -> Any:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(
            f"{context}: expected scalar, got {type(value).__name__}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context}: scalar is not finite: {value!r}")
    return value


def _binary_label(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: required label must be 0 or 1: {value!r}")
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    raise ValueError(f"{context}: required label must be 0 or 1: {value!r}")


def _dashboard_data(
    ablation: dict[str, Any],
    weak: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    ablation = _mapping(ablation, "ablation")
    experiments = _mapping(
        _required(ablation, "experiments", "ablation.experiments"),
        "ablation.experiments",
    )

    models = []
    errors = []
    for name in EXPERIMENT_ORDER:
        experiment_context = f"ablation.experiments.{name}"
        experiment = _mapping(
            _required(experiments, name, experiment_context),
            experiment_context,
        )
        description_context = f"{experiment_context}.description"
        description = _required(
            experiment,
            "description",
            description_context,
        )
        if not isinstance(description, str):
            raise ValueError(
                f"{description_context}: expected string, got "
                f"{type(description).__name__}"
            )
        split_metrics = _mapping(
            _required(
                experiment,
                "split_metrics",
                f"{experiment_context}.split_metrics",
            ),
            f"{experiment_context}.split_metrics",
        )
        test_metrics = _mapping(
            _required(
                split_metrics,
                "test",
                f"{experiment_context}.split_metrics.test",
            ),
            f"{experiment_context}.split_metrics.test",
        )
        model = {"name": name, "description": description}
        for metric_name in MODEL_METRICS:
            context = (
                f"{experiment_context}.split_metrics.test.{metric_name}"
            )
            model[metric_name] = _unit_interval(
                _required(test_metrics, metric_name, context),
                context,
            )
        models.append(model)

        raw_error_cases = experiment.get("error_cases")
        if raw_error_cases is None:
            continue
        error_cases = _mapping(
            raw_error_cases,
            f"{experiment_context}.error_cases",
        )
        test_errors = _mapping(
            _required(
                error_cases,
                "test",
                f"{experiment_context}.error_cases.test",
            ),
            f"{experiment_context}.error_cases.test",
        )
        for error_type in ("false_positive", "false_negative"):
            cases_context = (
                f"{experiment_context}.error_cases.test.{error_type}"
            )
            cases = _list(
                _required(test_errors, error_type, cases_context),
                cases_context,
            )
            for index, raw_case in enumerate(cases):
                case_context = f"{cases_context}[{index}]"
                case = _mapping(raw_case, case_context)
                validated_case = {}
                for field in ERROR_SCALAR_FIELDS:
                    if field in case:
                        validated_case[field] = _safe_scalar(
                            case[field],
                            f"{case_context}.{field}",
                        )
                probability_context = (
                    f"{case_context}.predicted_probability"
                )
                validated_case["predicted_probability"] = _unit_interval(
                    _required(
                        case,
                        "predicted_probability",
                        probability_context,
                    ),
                    probability_context,
                )
                for field in ("predicted_label", "true_label"):
                    field_context = f"{case_context}.{field}"
                    validated_case[field] = _binary_label(
                        _required(case, field, field_context),
                        field_context,
                    )
                expected_labels = {
                    "false_positive": (1, 0),
                    "false_negative": (0, 1),
                }[error_type]
                actual_labels = (
                    validated_case["predicted_label"],
                    validated_case["true_label"],
                )
                if actual_labels != expected_labels:
                    raise ValueError(
                        f"{case_context}: labels are inconsistent with "
                        f"{error_type}; expected predicted_label="
                        f"{expected_labels[0]} and true_label="
                        f"{expected_labels[1]}"
                    )
                errors.append(
                    {
                        **{
                            field: validated_case[field]
                            for field in ERROR_OUTPUT_FIELDS
                            if field in validated_case
                        },
                        "model": name,
                        "type": error_type,
                    }
                )

    best = max(models, key=lambda item: item["auroc"])

    audit = _mapping(audit, "audit")
    total = _integer(
        _required(audit, "total", "audit.total"),
        "audit.total",
    )
    by_quarter = _mapping(
        _required(audit, "by_quarter", "audit.by_quarter"),
        "audit.by_quarter",
    )
    quarter_total = 0
    positive_total = 0
    for quarter, raw_bucket in by_quarter.items():
        context = f"audit.by_quarter.{quarter}"
        bucket = _mapping(raw_bucket, context)
        quarter_n = _integer(
            _required(bucket, "n", f"{context}.n"),
            f"{context}.n",
        )
        positive = _integer(
            _required(bucket, "positive", f"{context}.positive"),
            f"{context}.positive",
        )
        if positive > quarter_n:
            raise ValueError(
                f"{context}.positive: count {positive} exceeds "
                f"{context}.n value {quarter_n}"
            )
        quarter_total += quarter_n
        positive_total += positive
    if quarter_total != total:
        raise ValueError(
            "audit.by_quarter sum of n "
            f"({quarter_total}) does not equal audit.total ({total})"
        )
    serious_rate = positive_total / total if total else 0.0

    weak = _mapping(weak, "weak")
    overall = _mapping(
        _required(weak, "overall", "weak.overall"),
        "weak.overall",
    )
    weak_total = _integer(
        _required(overall, "total", "weak.overall.total"),
        "weak.overall.total",
    )
    weak_summary = {}
    for metric_name in WEAK_METRICS:
        context = f"weak.overall.{metric_name}"
        weak_summary[metric_name] = _unit_interval(
            _required(overall, metric_name, context),
            context,
        )
    rules_mapping = _mapping(
        _required(overall, "rules", "weak.overall.rules"),
        "weak.overall.rules",
    )
    rules = []
    for name in RULE_ORDER:
        context = f"weak.overall.rules.{name}"
        raw_rule = _required(rules_mapping, name, context)
        rule = _mapping(raw_rule, context)
        fires = _integer(
            _required(rule, "fires", f"{context}.fires"),
            f"{context}.fires",
        )
        if fires > weak_total:
            raise ValueError(
                f"{context}.fires: count {fires} exceeds "
                f"weak.overall.total value {weak_total}"
            )
        rules.append(
            {
                "name": str(name),
                "fires": fires,
                "coverage": _unit_interval(
                    _required(
                        rule,
                        "coverage_rate",
                        f"{context}.coverage_rate",
                    ),
                    f"{context}.coverage_rate",
                ),
                "accuracy": _unit_interval(
                    _required(rule, "accuracy", f"{context}.accuracy"),
                    f"{context}.accuracy",
                ),
            }
        )

    return {
        "total": total,
        "serious_rate": serious_rate,
        "models": models,
        "best": dict(best),
        "weak": {
            "coverage": weak_summary["coverage_rate"],
            "conflict": weak_summary["conflict_rate"],
            "accuracy": weak_summary["accuracy"],
            "precision": weak_summary["precision"],
            "recall": weak_summary["recall"],
            "f1": weak_summary["f1"],
            "rules": rules,
        },
        "errors": errors,
    }


def _safe_json(value: Any) -> str:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError(
            "dashboard payload contains a non-finite numeric value"
        ) from exc
    return (
        text.replace("</", r"<\/")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )


def render_dashboard(
    ablation: dict[str, Any],
    weak: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    data = _dashboard_data(ablation, weak, audit)
    data["display"] = {
        "best_auroc": f"{data['best']['auroc']:.4f}",
        "weak_coverage": f"{100 * data['weak']['coverage']:.2f}%",
    }
    payload = _safe_json(data)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FAERS 终期实验 Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #20231f;
      --muted: #646a62;
      --line: #d8ddd5;
      --surface: #ffffff;
      --soft: #f3f5f1;
      --header: #1b211d;
      --green: #23865f;
      --red: #c64b48;
      --blue: #3976b8;
      --gold: #b48118;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--soft);
      color: var(--ink);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI",
        "PingFang SC", "Microsoft YaHei", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      min-height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 14px max(24px, calc((100% - 1180px) / 2));
      background: var(--header);
      color: #f7faf6;
      border-bottom: 3px solid var(--green);
    }}
    header h1 {{ margin: 0; font-size: 21px; font-weight: 680; }}
    header p {{ margin: 2px 0 0; color: #bfc7bd; font-size: 12px; }}
    .status {{
      flex: 0 0 auto;
      padding: 5px 8px;
      border: 1px solid #637068;
      border-radius: 4px;
      color: #dce5da;
      font-size: 12px;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 22px auto 44px;
    }}
    section {{ margin-top: 30px; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; }}
    .section-note {{ margin: -7px 0 14px; color: var(--muted); }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }}
    .summary-item {{
      min-height: 92px;
      padding: 14px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .summary-label {{ color: var(--muted); font-size: 12px; }}
    .summary-value {{
      display: block;
      margin-top: 9px;
      font-size: 21px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .chart {{
      background: var(--surface);
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }}
    .chart-row {{
      display: grid;
      grid-template-columns: minmax(180px, 1.3fr) repeat(3, minmax(130px, 1fr));
      gap: 16px;
      align-items: center;
      min-height: 76px;
      padding: 11px 14px;
      border-bottom: 1px solid var(--line);
    }}
    .chart-row:last-child {{ border-bottom: 0; }}
    .model-name {{ font-weight: 650; overflow-wrap: anywhere; }}
    .model-description {{ color: var(--muted); font-size: 12px; }}
    .metric-head {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-size: 12px;
    }}
    .bar-track {{
      width: 100%;
      height: 10px;
      margin-top: 5px;
      overflow: hidden;
      background: #e3e7e1;
      border-radius: 3px;
    }}
    .bar-fill {{ height: 100%; border-radius: 3px; }}
    .auroc {{ background: var(--green); }}
    .auprc {{ background: var(--blue); }}
    .f1 {{ background: var(--gold); }}
    .weak-summary {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 1px;
      background: var(--line);
      border: 1px solid var(--line);
    }}
    .weak-stat {{
      min-height: 78px;
      padding: 12px;
      background: var(--surface);
    }}
    .weak-stat span {{ display: block; color: var(--muted); font-size: 12px; }}
    .weak-stat strong {{ display: block; margin-top: 7px; font-size: 18px; }}
    .table-wrap {{
      width: 100%;
      margin-top: 14px;
      overflow-x: auto;
      border: 1px solid var(--line);
      background: var(--surface);
    }}
    table {{ width: 100%; min-width: 660px; border-collapse: collapse; }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #e9ede7;
      color: #3f453e;
      font-size: 12px;
      font-weight: 650;
    }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 12px;
    }}
    label {{ display: grid; gap: 4px; color: var(--muted); font-size: 12px; }}
    select {{
      min-width: 210px;
      height: 36px;
      padding: 0 30px 0 9px;
      border: 1px solid #aeb5ac;
      border-radius: 4px;
      background: var(--surface);
      color: var(--ink);
      font: inherit;
    }}
    .error-type {{
      display: inline-block;
      min-width: 48px;
      padding: 2px 5px;
      border-radius: 3px;
      color: #fff;
      text-align: center;
      font-size: 11px;
    }}
    .false-positive {{ background: var(--red); }}
    .false-negative {{ background: var(--blue); }}
    .empty {{
      padding: 28px 12px;
      color: var(--muted);
      text-align: center;
    }}
    .boundary {{
      padding: 14px 16px;
      border-left: 4px solid var(--gold);
      background: #fff9e8;
      color: #514525;
    }}
    @media (max-width: 900px) {{
      .summary-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .weak-summary {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .chart-row {{ grid-template-columns: minmax(160px, 1fr) 1fr; }}
    }}
    @media (max-width: 600px) {{
      header {{ align-items: flex-start; padding: 12px 16px; }}
      header h1 {{ font-size: 18px; }}
      .status {{ display: none; }}
      main {{ width: min(100% - 20px, 1180px); margin-top: 14px; }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .weak-summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .chart-row {{
        grid-template-columns: 1fr;
        gap: 9px;
        padding: 13px 10px;
      }}
      .controls {{ display: grid; grid-template-columns: 1fr 1fr; }}
      select {{ width: 100%; min-width: 0; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>FAERS 终期实验 Dashboard</h1>
      <p>消融实验、弱监督审计与失败案例</p>
    </div>
    <div class="status">离线结果快照</div>
  </header>
  <main>
    <section aria-labelledby="summary-title">
      <h2 id="summary-title">实验概览</h2>
      <div class="summary-grid" id="summary-grid"></div>
    </section>

    <section aria-labelledby="models-title">
      <h2 id="models-title">模型与消融对比</h2>
      <p class="section-note">测试集指标统一使用验证集选择的阈值。</p>
      <div class="chart" id="model-chart"></div>
    </section>

    <section aria-labelledby="weak-title">
      <h2 id="weak-title">弱监督审计</h2>
      <div class="weak-summary" id="weak-summary"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>规则</th><th>触发数</th><th>覆盖率</th><th>准确率</th></tr>
          </thead>
          <tbody id="rules-body"></tbody>
        </table>
      </div>
    </section>

    <section aria-labelledby="errors-title">
      <h2 id="errors-title">失败案例</h2>
      <div class="controls">
        <label>模型
          <select id="model-filter" aria-label="按模型筛选">
            <option value="all">全部模型</option>
          </select>
        </label>
        <label>错误类型
          <select id="error-filter" aria-label="按错误类型筛选">
            <option value="all">全部</option>
            <option value="false_positive">假阳性</option>
            <option value="false_negative">假阴性</option>
          </select>
        </label>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>模型</th><th>类型</th><th>病例 ID</th>
              <th>概率</th><th>真实标签</th><th>Token 摘要</th>
            </tr>
          </thead>
          <tbody id="errors-body"></tbody>
        </table>
      </div>
    </section>

    <section aria-labelledby="boundary-title">
      <h2 id="boundary-title">结果解释边界</h2>
      <div class="boundary">
        FAERS 属于自发报告数据，模型结果用于病例级重症识别和特征贡献审计，
        不能直接解释为药物因果关系或真实发生率。反应术语与结果字段可能形成语义捷径；
        弱监督指标只代表规则覆盖子集。
      </div>
    </section>
  </main>

  <script id="dashboard-data" type="application/json">{payload}</script>
  <script>
    "use strict";
    const data = JSON.parse(document.getElementById("dashboard-data").textContent);
    const formatMetric = value => Number(value).toFixed(4);
    const formatPercent = value => `${{(Number(value) * 100).toFixed(2)}}%`;
    const formatCount = value => Number(value).toLocaleString("zh-CN");
    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    const summaryItems = [
      ["病例数", formatCount(data.total)],
      ["重症率", formatPercent(data.serious_rate)],
      ["最佳模型", data.best.name],
      ["AUROC", data.display.best_auroc],
      ["AUPRC", formatMetric(data.best.auprc)]
    ];
    const summaryGrid = document.getElementById("summary-grid");
    summaryItems.forEach(([label, value]) => {{
      const item = document.createElement("div");
      item.className = "summary-item";
      const labelNode = document.createElement("span");
      labelNode.className = "summary-label";
      labelNode.textContent = label;
      const valueNode = document.createElement("strong");
      valueNode.className = "summary-value";
      valueNode.textContent = value;
      item.append(labelNode, valueNode);
      summaryGrid.appendChild(item);
    }});

    function metricBar(label, value, className) {{
      const width = Math.max(0, Math.min(100, Number(value) * 100));
      return `<div>
        <div class="metric-head"><span>${{escapeHtml(label)}}</span>
          <strong>${{escapeHtml(formatMetric(value))}}</strong></div>
        <div class="bar-track"><div class="bar-fill ${{className}}"
          style="width:${{width.toFixed(2)}}%"></div></div>
      </div>`;
    }}
    const chart = document.getElementById("model-chart");
    data.models.forEach(model => {{
      const row = document.createElement("div");
      row.className = "chart-row";
      row.innerHTML = `<div>
        <div class="model-name">${{escapeHtml(model.name)}}</div>
        <div class="model-description">${{escapeHtml(model.description)}}</div>
      </div>
      ${{metricBar("AUROC", model.auroc, "auroc")}}
      ${{metricBar("AUPRC", model.auprc, "auprc")}}
      ${{metricBar("F1", model.f1, "f1")}}`;
      chart.appendChild(row);
    }});

    const weakItems = [
      ["覆盖率", data.display.weak_coverage, false],
      ["冲突率", data.weak.conflict, true],
      ["Accuracy", data.weak.accuracy, true],
      ["Precision", data.weak.precision, true],
      ["Recall", data.weak.recall, true],
      ["F1", data.weak.f1, false]
    ];
    const weakSummary = document.getElementById("weak-summary");
    weakItems.forEach(([label, value, percent]) => {{
      const item = document.createElement("div");
      item.className = "weak-stat";
      const labelNode = document.createElement("span");
      labelNode.textContent = label;
      const valueNode = document.createElement("strong");
      valueNode.textContent = typeof value === "string"
        ? value
        : (percent ? formatPercent(value) : formatMetric(value));
      item.append(labelNode, valueNode);
      weakSummary.appendChild(item);
    }});

    document.getElementById("rules-body").innerHTML = data.weak.rules
      .map(rule => `<tr>
        <td>${{escapeHtml(rule.name)}}</td>
        <td>${{escapeHtml(formatCount(rule.fires))}}</td>
        <td>${{escapeHtml(formatPercent(rule.coverage))}}</td>
        <td>${{escapeHtml(formatPercent(rule.accuracy))}}</td>
      </tr>`).join("");

    const modelFilter = document.getElementById("model-filter");
    data.models.forEach(model => {{
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.name;
      modelFilter.appendChild(option);
    }});
    const errorFilter = document.getElementById("error-filter");
    const errorsBody = document.getElementById("errors-body");
    function renderErrors() {{
      const rows = data.errors.filter(item =>
        (modelFilter.value === "all" || item.model === modelFilter.value) &&
        (errorFilter.value === "all" || item.type === errorFilter.value)
      );
      if (rows.length === 0) {{
        errorsBody.innerHTML =
          '<tr><td class="empty" colspan="6">暂无符合筛选条件的失败案例</td></tr>';
        return;
      }}
      errorsBody.innerHTML = rows.map(item => {{
        const isPositive = item.type === "false_positive";
        const typeLabel = isPositive ? "假阳性" : "假阴性";
        const typeClass = isPositive ? "false-positive" : "false-negative";
        const probability = Number.isFinite(Number(item.predicted_probability))
          ? formatMetric(item.predicted_probability) : "—";
        return `<tr>
          <td>${{escapeHtml(item.model)}}</td>
          <td><span class="error-type ${{typeClass}}">${{typeLabel}}</span></td>
          <td>${{escapeHtml(item.safetyreportid ?? "—")}}</td>
          <td>${{escapeHtml(probability)}}</td>
          <td>${{escapeHtml(item.true_label ?? "—")}}</td>
          <td>${{escapeHtml(item.tokens ?? "—")}}</td>
        </tr>`;
      }}).join("");
    }}
    modelFilter.addEventListener("change", renderErrors);
    errorFilter.addEventListener("change", renderErrors);
    renderErrors();
  </script>
</body>
</html>
"""


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


def _load_strict_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")

    def reject_constant(value: str) -> None:
        raise ValueError(
            f"{path}: non-standard JSON constant is not allowed: {value}"
        )

    return json.loads(text, parse_constant=reject_constant)


def generate_dashboard(output_dir: Path, destination: Path) -> Path:
    ablation_path = output_dir / "ablations" / "ablation_metrics.json"
    weak_path = (
        output_dir
        / "weak_supervision"
        / "weak_supervision_metrics.json"
    )
    audit_path = output_dir / "reports" / "feature_audit.json"

    ablation = _load_strict_json(ablation_path)
    weak = _load_strict_json(weak_path)
    audit = _load_strict_json(audit_path)
    text = render_dashboard(ablation, weak, audit)
    _atomic_write_text(destination, text)
    return destination
