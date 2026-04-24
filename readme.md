# 基于本地 FAERS XML 的药物风险预测研究包

本仓库按本地真实数据实现 `plan.md` 的研究路线。当前数据目录为 `FAERS/faers_xml_2025q1` 至 `FAERS/faers_xml_2025q4`，格式是 FDA FAERS XML。

## 数据口径

- 数据源：本地 `FAERS` 目录中的 2025Q1-2025Q4 XML。
- 切分：2025Q1-Q3 训练集；2025Q4 按 `receivedate` 前后 50/50 拆为验证集和测试集，同日用 `safetyreportid` 稳定排序。
- 标签：`serious == 1` 或任一 seriousness flag 为 `1` 时标记为重症病例。
- 防泄漏：`serious`、`seriousness*`、`label_serious` 仅用于标签和审计，不进入模型特征。

## 运行命令

快速端到端验证，每季度抽取 1000 条病例：

```powershell
python scripts/run_faers_pipeline.py --data FAERS --out outputs_sample --mode full --sample 1000
```

全量运行：

```powershell
python scripts/run_faers_pipeline.py --data FAERS --out outputs --mode full
```

也可以分阶段运行：

```powershell
python scripts/run_faers_pipeline.py --data FAERS --out outputs --mode inventory
python scripts/run_faers_pipeline.py --data FAERS --out outputs --mode etl
python scripts/run_faers_pipeline.py --data FAERS --out outputs --mode train
python scripts/run_faers_pipeline.py --data FAERS --out outputs --mode report
```

## 产出目录

- `outputs/interim/`：季度病例级 CSV、`inventory.json`、`parse_log.json`。
- `outputs/models/`：`logistic_sgd.pkl`、`numeric_hgb.pkl`、`feature_config.json`。
- `outputs/reports/`：`data_audit.md`、`final_summary.md`、`model_metrics.json`、`feature_audit.json`。
- `outputs/reports/figures/`：标签分布、缺失率、模型测试指标图。

## 测试

```powershell
python -m unittest discover -s tests
```

当前实现不依赖 pandas、LightGBM、Snorkel 或 RxNorm。药物名称正规化使用本地可复现规则：优先 `activesubstancename`，否则使用 `medicinalproduct`，统一大写、去标点并合并空白。
