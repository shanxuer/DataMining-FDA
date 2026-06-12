# DataMining-FDA 终期项目设计

## 1. 目标与范围

在现有 FAERS 病例级 CSV、模型和中期结果基础上完成四项终期交付：

1. 特征族消融实验。
2. 弱监督规则扩展与多数投票评测。
3. 自动生成的 Markdown 最终报告。
4. 可离线直接打开的静态 HTML Dashboard Demo。

本轮不包含答辩 PPT，不重新解析 8.6GB 原始 XML，不引入 Snorkel、pandas、scikit-learn 或前端构建工具。所有实验继续保持 Python 标准库与 NumPy 可复现路径。

## 2. 总体架构

新增 `scripts/run_final_project.py` 作为唯一终期入口：

```bash
python3 scripts/run_final_project.py --out outputs
```

入口直接读取 `outputs/interim/cases_*.csv`，按顺序调用四个职责独立的模块：

- `scripts/final_experiments.py`：构建消融特征、训练哈希逻辑回归并评测。
- `scripts/weak_supervision.py`：执行规则、聚合多数投票并评测弱标签。
- `scripts/final_reporting.py`：读取结构化实验产物并生成 `最终报告.md`。
- `scripts/dashboard.py`：将结构化实验数据与失败案例嵌入 `demo/index.html`。

现有 `scripts/run_faers_pipeline.py` 保持为 XML 清点、ETL、基础训练和中期报告入口。终期模块通过明确函数接口复用其 CSV 读取、哈希模型、指标和阈值选择能力，不把新行为继续堆叠进现有主脚本。

## 3. 数据流与产物

输入：

- `outputs/interim/cases_2025Q1.csv` 至 `cases_2025Q4.csv`
- `outputs/reports/model_metrics.json`
- 现有时间切分字段 `train`、`valid`、`test`

处理流程：

1. 校验病例 CSV 与既有指标文件是否存在。
2. 在相同时间切分、哈希维度和训练参数下运行五组哈希特征实验。
3. 运行扩展弱监督规则，生成规则级统计和多数投票结果。
4. 将实验 JSON 汇总为 Markdown 报告。
5. 将汇总指标和真实失败案例嵌入独立 HTML。

输出：

- `outputs/ablations/ablation_metrics.json`
- `outputs/ablations/models/*.pkl`
- `outputs/weak_supervision/weak_supervision_metrics.json`
- `最终报告.md`
- `demo/index.html`

`outputs/` 继续由 `.gitignore` 排除，保存本地真实实验产物。`最终报告.md`、`demo/index.html`、新增脚本和测试纳入版本控制。

## 4. 消融实验设计

所有哈希模型沿用现有参数、时间切分和评测逻辑。阈值只在验证集选择，最终结论以测试集为准。每组输出训练样本数、验证阈值，以及 train、valid、test 的 AUROC、AUPRC、Precision、Recall、F1、Recall@Top5% 和 Hit Rate@Top5%。

实验组：

| 实验 | 特征口径 |
| --- | --- |
| `all_tokens` | 完整结构化 token、分箱 token 和所有文本 token |
| `without_reaction_pt` | 从完整模型移除 `reac:*` |
| `without_reaction_outcome` | 从完整模型移除 `reactionoutcome:*` |
| `drug_indication_only` | 仅保留 `drug:*` 与 `indi:*` |
| `structured_only` | 仅保留人口学、报告来源、季度和计数分箱 token |
| `numeric_logistic` | 读取现有数值逻辑回归基线结果，不重复训练 |

特征选择通过命名配置完成，而不是为每组复制训练代码。每个配置必须声明允许或排除的 token 前缀，确保实验定义可测试且可写入结果文件。

若某个实验训练失败，终期入口立即返回非零状态，不生成带有部分结果的最终报告，避免将不完整结果误当成完整实验。

## 5. 弱监督扩展设计

弱监督仅用于规则审计和对照，不参与模型训练。规则返回 `1`、`0` 或弃权 `None`。

保留规则：

- 反应术语包含死亡、致命或心脏骤停相关词时投正票。
- 用药数量达到 10 种以上时投正票。
- 65 岁及以上且疑似用药不少于 2 种时投正票。
- 50 岁以下、用药不超过 2 种且反应不超过 1 种时投负票。

扩展规则：

- 高危反应术语规则：脓毒症、急性呼吸衰竭、休克等明确高危反应投正票。
- 严重反应结果规则：配置为死亡、未恢复或留有后遗症的反应结果投正票。
- 极端多药规则：用药数量达到更高阈值时投正票。
- 设备或产品质量问题规则：仅出现设备故障、产品质量、无不良事件等模式且无高危反应时投负票。

每条规则输出：

- 触发数和覆盖率。
- 正票、负票数量。
- 命中样本真实重症率。
- 非弃权票的准确率。

多数投票：

- 忽略弃权票。
- 正票多于负票时预测为 `1`。
- 负票多于正票时预测为 `0`。
- 正负票相等时记为冲突并弃权。
- 无规则触发时记为未覆盖。

总体输出覆盖率、冲突率、非弃权样本数、弱标签准确率、Precision、Recall 和 F1，并按 train、valid、test 分别统计。测试集结果与 `all_tokens` 模型并列表达，但报告必须明确两者覆盖范围不同，不能直接把弱监督覆盖子集指标解释为全测试集模型指标。

## 6. 最终报告设计

根目录 `最终报告.md` 由实验 JSON 自动生成，包含：

1. 项目摘要与研究问题。
2. 数据来源、规模、时间切分和资料治理。
3. 数值基线与稳定哈希模型方法。
4. 消融实验设计和完整结果表。
5. 语义捷径诊断，包括移除反应 PT 和反应结果后的指标变化。
6. 弱监督规则、覆盖率、冲突率和多数投票结果。
7. 真实失败案例与分组误差分析。
8. 局限性、复现约束和未来工作。
9. 从原始 XML、已有病例 CSV及终期实验三种起点执行的复现命令。
10. AI 工具辅助使用声明。

所有样本数、百分比和指标必须从 JSON 产物读取。生成器不得在模板中硬编码实验结果。缺少必需字段时应报错，而不是输出 `NA` 或未经解释的空表。

## 7. HTML Dashboard 设计

`demo/index.html` 使用原生 HTML、CSS 和 JavaScript，不依赖服务器、CDN、外部字体、外部脚本或额外 JSON 文件。实验数据以转义后的 JSON 嵌入页面，双击即可离线打开。

页面面向课程答辩展示，首屏直接呈现研究结果，不制作营销式落地页。内容包括：

- 数据规模、重症率、最佳模型及关键测试指标概览。
- 六组模型和消融实验的 AUROC、AUPRC、F1 对比图。
- 弱监督规则覆盖率、准确率和多数投票汇总。
- 可按模型和错误类型筛选的真实失败案例表。
- 关于反应语义捷径、弱监督覆盖范围和数据缺失的结论提示。

图表使用 HTML/CSS 或 Canvas 绘制，不引入图表库。筛选控件使用下拉菜单或分段选项，表格在窄屏下允许横向滚动。所有尺寸有稳定约束，桌面与移动视口不得出现文本重叠或控件溢出。

HTML 只展示真实实验数据，不提供在线单病例预测输入。

## 8. 错误处理与可复现性

终期入口执行前校验：

- `outputs/interim/cases_*.csv` 至少包含四个季度文件。
- CSV 包含训练、验证和测试切分。
- 每个训练切分同时包含正负样本。
- 现有 `model_metrics.json` 包含 `numeric_logistic` 基线。

命令支持 `--out outputs_sample` 快速验证和 `--out outputs` 全量运行。随机种子、哈希维度、训练轮数和特征配置写入结果 JSON。产物写入采用先生成完整内容再替换目标文件的方式，减少中途失败留下损坏报告的风险。

## 9. 测试与验收

新增行为按测试驱动开发实现，至少覆盖：

- 各消融配置保留和移除正确的 token 前缀。
- serious 字段和 `label_serious` 仍不会进入模型输入。
- 多数投票正确处理正票、负票、弃权和平票冲突。
- 设备问题负规则不会覆盖同时存在高危反应的病例。
- 弱监督指标只在非弃权样本上计算准确率等分类指标。
- 报告中的核心数值来自输入 JSON。
- Dashboard 输出包含内嵌数据、筛选控件且无外部网络依赖。
- 缺少输入文件或必需指标时返回清晰错误。

验收命令：

```bash
python3 -m unittest discover -s tests
python3 scripts/run_final_project.py --out outputs_sample
python3 scripts/run_final_project.py --out outputs
```

最终还需在浏览器中检查 `demo/index.html` 的桌面和移动布局、筛选交互、控制台错误及指标一致性。不得修改或提交 `data/`、`outputs/`、`outputs_sample/`、`.obsidian/` 和现有未跟踪 PDF。

