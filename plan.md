# 课程项目实验报告：基于 2025 年度 FDA FAERS XML 的药物风险诊断性特征工程与资料治理研究

> 本文件已根据本地 FAERS 数据与完整实验结果修订。实际实验以 `FAERS/faers_xml_2025q1` 至 `FAERS/faers_xml_2025q4` 的 XML 数据为准，结果产出位于 `outputs/`。

## 团队成员与分工

| 成员 | 学号 | 主要分工 |
| --- | --- | --- |
| 虞姿 | 3220251479 | 负责资料整合与清洗 Pipeline，识别并修复跨季度分布漂移，设计资料品质 Dashboard。 |
| 刘贺雨 | 3120251311 | 负责分箱、交互、聚合与比值特征工程设计，将医学知识转译为可计算特征。 |
| 沉诗璇 | 3120251026 | 负责弱监督规则设计、标签冲突分析与规则覆盖率审计。 |
| 魏婷婷 | 3220251294 | 负责分层评估、敏感性分析、误差分析与防止虚假增益审计。 |

## GitHub 仓库

- 仓库连结：https://github.com/shanxuer/DataMining-FDA

## 问题定义

### 研究背景与动机

药物警戒场景中的不良事件资料具有高噪声、高缺失、多表或多层级关联与药物名称不一致等典型特征。若仅依赖原始栏位直接建模，往往难以有效表达患者背景、共用药组合、适应症、给药时序与重症风险之间的关联，导致严重不良反应预警能力不足。

本研究基于本地已下载的 FDA FAERS XML 数据，建立一套可复现的资料治理、诊断性特征工程与基线建模流程，目标是预测病例是否属于重症不良反应，并支持高风险病例排序、质量审计与后续人工复核。

### 任务形式化

设第 \(i\) 个病例的原始输入为：

- 人口学与病例资讯 \(D_i\)
- 用药资讯 \(M_i\)
- 不良反应资讯 \(R_i\)
- 适应症资讯 \(I_i\)
- 治疗时序资讯 \(T_i\)
- 报告来源资讯 \(S_i\)

则病例表示为：

\(x_i = \{D_i, M_i, R_i, I_i, T_i, S_i\}\)

定义标签 \(y_i \in \{0, 1\}\)：

- \(y_i = 1\)：病例属于重症不良反应。
- \(y_i = 0\)：病例不属于重症不良反应。

实验中重症标签由 FAERS XML 的 `serious == 1` 或任一 seriousness flag 为 `1` 生成。`serious`、`seriousnessdeath`、`seriousnesslifethreatening`、`seriousnesshospitalization`、`seriousnessdisabling`、`seriousnesscongenitalanomali`、`seriousnessother` 和 `label_serious` 仅用于标签生成与审计，不进入模型特征，避免目标泄漏。

研究目标是学习映射函数：

\(f(x_i) \rightarrow P(y_i = 1)\)

使模型能够在跨季度资料与分布漂移条件下稳定预测病例重症风险，并支持风险排序与高风险案例优先筛查。

## 资料来源与实验数据

### 资料来源

本研究使用 FDA 公开 FAERS/AEMS 季度资料，实际处理的数据已下载至本地：

- 本地数据目录：`FAERS/faers_xml_2025q1` 至 `FAERS/faers_xml_2025q4`
- 原始格式：FAERS XML，不是 ASCII 七表格式。
- 官方入口：
https://www.fda.gov/drugs/fda-adverse-event-monitoring-system-aems/fda-adverse-event-monitoring-system-aems-latest-quarterly-data-files

原计划中提到的 2023 Q1-2024 Q4 ASCII 数据和 RxNorm 正规化没有作为本次正式实验的硬依赖。本次药物名称正规化采用本地可复现规则：优先使用 `activesubstancename`，否则使用 `medicinalproduct`，统一大写、去标点并合并空白。

### 资料规模与结构

本次完整实验共读取 12 个 XML 文件，原始 `safetyreport` 总数为 1,617,444。过滤删除列表后保留 1,617,443 个病例级样本，其中 2025Q1 有 1 条删除列表记录被过滤。

| 季度 | 保留病例数 | 重症病例数 | 重症率 |
| --- | ---: | ---: | ---: |
| 2025Q1 | 400,513 | 224,700 | 56.10% |
| 2025Q2 | 393,130 | 222,287 | 56.54% |
| 2025Q3 | 438,512 | 254,386 | 58.01% |
| 2025Q4 | 385,288 | 218,342 | 56.67% |

FAERS XML 为嵌套病例结构，核心节点包括 `safetyreport`、`primarysource`、`sender`、`patient`、`reaction`、`drug` 与 `activesubstance`。实验通过流式 XML 解析生成病例级 CSV，而非一次性将原始 XML 全部载入内存。

### 资料质量发现

完整实验的主要资料质量结果如下：

- `age_years` 缺失率：39.69%
- `patientsex` 缺失率：19.84%
- `patientweight` 缺失率：83.08%
- `primarysourcecountry` 缺失率：0.03%
- `qualification` 缺失率：0.73%
- 药物名称规则正规化覆盖率：100.00%
- active substance 覆盖率：98.32%

原计划中“重症案例相对较少、类别不平衡明显”的假设与实验不符。本次标签下重症比例约 56%-58%，并非少数类问题；更关键的风险是标签定义带来的语义捷径、反应术语与重症标签之间的强关联，以及跨季度分布稳定性。

## 方法与技术栈

### 特征工程

实验构建病例级特征，覆盖以下类别：

- 人口学特征：年龄换算、年龄分箱、性别、体重。
- 报告来源特征：报告国家、发生国家、报告者资格、发送方类型、报告类型。
- 用药曝光特征：药物数、疑似用药数、合并用药数、交互用药数、剂量可用性、平均剂量、疗程天数摘要。
- 反应与适应症特征：反应数、唯一反应数、适应症数。
- 文本 token 特征：标准化药名、反应 PT、适应症、给药途径、处理动作、反应结果等。
- 治理审计特征：药物名称覆盖率、active substance 覆盖率、缺失率与删除列表过滤统计。

### 模型

本次实验根据实际环境调整了模型方案。当前环境不依赖 pandas、LightGBM、XGBoost、Snorkel 或 PyTorch，完整流程使用 Python 标准库、NumPy、scikit-learn 与 Matplotlib 完成。

- **Baseline 1：HashingVectorizer + SGD Logistic Regression**
  使用稀疏哈希文本特征与类别/分箱 token，训练可增量处理大样本的逻辑回归基线。

- **Baseline 2：sklearn HistGradientBoostingClassifier**
  使用数值聚合特征训练树模型基线。训练阶段默认从训练集抽样 200,000 条，降低全量训练的内存与时间压力。

### 弱监督规则审计

原计划中的 Snorkel 概率标签整合没有作为本次正式实验依赖。实际实现为轻量弱监督规则审计，用于观察规则覆盖率、冲突率与规则命中样本的标签一致性。规则示例包括：

- 反应术语含死亡或致命相关词。
- 多药共用数量达到 10 种以上。
- 老年患者且疑似用药数量较多。
- 年轻、低复杂度病例作为弱阴性规则。

完整实验中弱监督规则覆盖率为 26.96%，规则冲突率为 0.45%，非冲突规则与标签一致率为 75.19%。

## 实验设计与结果

### 资料切分策略

本研究采用时间导向切分，并根据实际需求将 2025Q4 内部拆分为验证集与测试集：

- 训练集：2025Q1、2025Q2、2025Q3，共 1,232,155 条。
- 验证集：2025Q4 按 `receivedate` 排序后的较早 50%，共 192,644 条。
- 测试集：2025Q4 按 `receivedate` 排序后的较晚 50%，共 192,644 条。

同一 `receivedate` 下使用 `safetyreportid` 稳定排序；缺失或非法 `receivedate` 排在 Q4 后段。模型阈值在验证集上选择，最终泛化表现以测试集为准。

### 评估指标

实验报告以下指标：

- AUROC：整体排序能力。
- AUPRC：精确率-召回率曲线面积。
- F1-score：precision 与 recall 的平衡。
- Recall@Top1%、Recall@Top5%、Recall@Top10%：高风险排序下对重症病例的召回能力。
- Hit Rate@TopK：高风险样本中的重症命中率。
- 分层指标：按性别、年龄层、季度等分组审计模型表现。

### 完整实验结果

| 模型 | 切分 | AUROC | AUPRC | F1 | Recall@Top5% |
| --- | --- | ---: | ---: | ---: | ---: |
| logistic_sgd | train | 0.9856 | 0.9886 | 0.9553 | 0.0878 |
| logistic_sgd | valid | 0.9869 | 0.9898 | 0.9628 | 0.0861 |
| logistic_sgd | test | 0.9837 | 0.9859 | 0.9552 | 0.0904 |
| numeric_hgb | train | 0.8858 | 0.9156 | 0.8286 | 0.0872 |
| numeric_hgb | valid | 0.8849 | 0.9182 | 0.8339 | 0.0857 |
| numeric_hgb | test | 0.8757 | 0.9007 | 0.8142 | 0.0893 |

测试集上，`logistic_sgd` 的 AUROC 与 AUPRC 明显高于纯数值树模型，说明标准化药名、反应 PT、适应症等稀疏 token 对重症标签具有较强解释与排序能力。与此同时，过高的指标也提示需要持续排查目标泄漏与语义捷径，尤其是反应术语中可能直接含有死亡、住院或致命相关信息。

### 产出文件

完整实验产出位于 `outputs/`：

- `outputs/interim/`：季度病例级 CSV、`inventory.json`、`parse_log.json`
- `outputs/models/`：`logistic_sgd.pkl`、`numeric_hgb.pkl`、`feature_config.json`
- `outputs/reports/`：`data_audit.md`、`final_summary.md`、`model_metrics.json`、`feature_audit.json`
- `outputs/reports/figures/`：标签率、样本数、缺失率与测试集模型指标图

可复现命令：

```powershell
python scripts/run_faers_pipeline.py --data FAERS --out outputs --mode full
```

快速验证命令：

```powershell
python scripts/run_faers_pipeline.py --data FAERS --out outputs_sample --mode full --sample 1000
```

## 风险清单与后续改进

1. **标签语义捷径风险**
反应术语、反应结果或叙述性字段可能与重症标签高度同源，导致模型表现虚高。后续应进行更严格的目标泄漏审计，例如移除死亡或住院强相关反应词后重新评估。

2. **药物名称正规化深度不足**
本次规则正规化覆盖率高，但仍未接入 RxNorm，无法系统处理商品名、错拼名与成分层级映射。后续可在不破坏主流程可复现性的前提下加入 RxNorm 字典。

3. **缺失值较高**
体重缺失率达到 83.08%，年龄缺失率达到 39.69%，限制了剂量/体重等比值特征的稳定性。后续应重点审计缺失机制，并避免对高缺失字段做过度解释。

4. **弱监督尚未进入训练**
本次仅完成规则覆盖率、冲突率与一致性审计，没有使用 Snorkel 生成概率标签。后续可在有依赖支持时加入 Snorkel LabelModel，并与当前硬标签基线对照。

5. **模型对照仍可扩展**
当前树模型为 sklearn HistGradientBoostingClassifier，未使用 LightGBM/XGBoost。后续若允许安装依赖，可加入更强树模型与消融实验。

## 里程碑与完成情况

| 阶段 | 主要任务 | 完成情况 |
| --- | --- | --- |
| 资料准备 | 下载并确认 2025Q1-Q4 FAERS XML 数据 | 已完成 |
| ETL | 流式解析 XML、过滤删除列表、生成病例级 CSV | 已完成 |
| 资料审计 | 输出缺失率、标签分布、药物名称覆盖率、弱监督规则审计 | 已完成 |
| 基线建模 | 训练 logistic_sgd 与 numeric_hgb 两个基线 | 已完成 |
| 评估报告 | 输出训练/验证/测试指标、图表与中文摘要 | 已完成 |
| 后续增强 | RxNorm、Snorkel、LightGBM/XGBoost、严格泄漏消融 | 待扩展 |

## AI 辅助工具使用声明

- 本文件修订、实验流水线实现、报告整理与结果口径校准过程中使用了 ChatGPT/Codex 作为代码实现、文字整理与规格检查辅助工具。
