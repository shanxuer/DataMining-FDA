# 基于本地 FAERS XML 的药物风险预测研究包

本仓库按本地真实数据实现 `plan.md` 的研究路线。当前数据目录为 `data/faers_xml_2025q1` 至 `data/faers_xml_2025q4`，格式是 FDA FAERS XML。

## 数据口径

- 数据源：本地 `data` 目录中的 2025Q1-2025Q4 XML。
- 切分：2025Q1-Q3 训练集；2025Q4 按 `receivedate` 前后 50/50 拆为验证集和测试集，同日用 `safetyreportid` 稳定排序。
- 数据依据说明：本项目以本地 FAERS XML 数据目录、时间切分策略、重症标签规则和防泄漏约束作为实验数据依据；具体数据清点、解析日志和模型审计结果由运行流程输出至 `outputs/`。
- 标签：`serious == 1` 或任一 seriousness flag 为 `1` 时标记为重症病例。
- 防泄漏：`serious`、`seriousness*`、`label_serious` 仅用于标签和审计，不进入模型特征。

## 运行命令

建议使用 Python 3.9+。项目运行时依赖 NumPy；安装命令：

```bash
python3 -m pip install -r requirements.txt
```

快速端到端验证，每季度抽取 1000 条病例：

```bash
python3 scripts/run_faers_pipeline.py --data data --out outputs_sample --mode full --sample 1000
```

全量运行：

```bash
python3 scripts/run_faers_pipeline.py --data data --out outputs --mode full
```

也可以分阶段运行：

```bash
python3 scripts/run_faers_pipeline.py --data data --out outputs --mode inventory
python3 scripts/run_faers_pipeline.py --data data --out outputs --mode etl
python3 scripts/run_faers_pipeline.py --data data --out outputs --mode train
python3 scripts/run_faers_pipeline.py --data data --out outputs --mode report
```

## 产出目录

- `outputs/interim/`：季度病例级 CSV、`inventory.json`、`parse_log.json`。
- `outputs/models/`：`hash_logistic.pkl`、`numeric_logistic.pkl`、`feature_config.json`。
- `outputs/reports/`：`data_audit.md`、`final_summary.md`、`model_metrics.json`、`feature_audit.json`。
- `outputs/reports/figures/`：标签分布、缺失率、模型测试指标图。

## 测试

```bash
python3 -m unittest discover -s tests
```

当前实现不依赖 pandas、scikit-learn、LightGBM、Snorkel 或 RxNorm。药物名称正规化使用本地可复现规则：优先 `activesubstancename`，否则使用 `medicinalproduct`，统一大写、去标点并合并空白。

## 终期实验、最终报告与 Demo

完成基础流水线后，运行终期消融、弱监督扩展、最终 Markdown 报告和离线 HTML Dashboard：

```bash
python3 scripts/run_final_project.py --out outputs
```

快速验证已有样本产物：

```bash
python3 scripts/run_final_project.py --out outputs_sample
```

终期产物：

- `outputs/ablations/ablation_metrics.json`：五组哈希特征实验与数值基线对照。
- `outputs/weak_supervision/weak_supervision_metrics.json`：扩展规则与多数投票评测。
- `最终报告.md`：由真实 JSON 结果自动生成的最终报告。
- `demo/index.html`：无需服务器或网络依赖，双击即可离线打开的结果 Dashboard。
