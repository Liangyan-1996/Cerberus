# Benchmark_PLAN.md —— 最终定稿方案（2026-09-11）

> **文档定位**：本文件是 `test10_eval_new/20260909.final_benchmark/` 的**唯一口径
> 来源**（口径、公式、命名、交付物）。凡与其它文档冲突，以本文件为准。
>
> 配套文档：
> `README.md`（目录导航与速查）·
> `EXECUTION_PLAN.md`（历史执行链 S1–S8 与作业号）·
> `GAP_AUDIT_20260909.md`（资产核对与决策记录 D1–D18）·
> `summary/BENCHMARK_REPORT.md`（结果报告）·
> `figs/README.md`（图集索引与绘图规范）·
> `summary/PR_AUC_mean_sem.{csv,md}`（PR-AUC mean±SEM 汇总）。
>
> 旧版方案（2026-09-09 起草、2026-09-10 修订稿）备份于
> `_archive/Benchmark_PLAN_20260909_v1.md`，仅作历史追溯，不再维护。
> 已归档的口径（full-mean / prc-mean，Exp1/Exp3 的峰级试做）见 §9.4，
> 不进入任何主表主图。

---

## 0. 速查（TL;DR）

| | Exp1 | Exp2 | Exp3 |
|---|---|---|---|
| 科学命题 | ROSE 缝合簇级 Union 二分类 | 全染色体连续目标检测与荒漠抗噪 | 谱系特异性本系对决 |
| 任务层级 | cluster | bin + region（Dual-Tier） | cluster（每谱系独立） |
| 基线训练 | 复用 20 个 Union ckpt，**零重训** | 同 Exp1 | **80 次谱系特异重训** |
| 主指标 | AUPRC（+AUROC / F1_vt） | bin PR/ROC；region PR/F1/IoU；Desert FP/Mb | AUPRC（+AUROC / F1_vt）+ 4×4 交叉热图 |
| 主图数量 | 5 | 9（bin 5 + region 4） | 8（3 bar + 5 热图） |
| Cerberus 结果 | AUPRC 0.698 | region 0.643 / bin 0.888 | AUPRC 0.544–0.703 |
| 基线结果 | 0.468–0.516 | 0.291–0.341 / 0.626–0.649 | 0.131–0.369 |

**唯一主口径：`prcmax`** —— 每个 GT 单元（簇 / 区间）的分数 = 其**构成峰分数取
最大值**（定义见 §2.3）。三个实验统一使用，保证跨实验可比。

---

## 1. 科学问题与设计准则

### 1.1 现有四篇工作的评估隐患

1. **随机 9:1 划分 → 同源污染**：转座子、旁系同源与重复元件跨染色体分布，随机
   划分让训练/测试高度同源，模型靠"记忆"即可取得虚高指标。
2. **单一染色体划分 → 偶然偏倚**：单组测试染色体（如仅 chr3/chr11）的基因密度、
   GC 含量、异染色质比例都会影响结论。
3. **输入尺度断层**：基线仅接受 ≤3 kb 预裁剪片段，无法建模数十至数百 kb 的
   长程结构域拓扑与边界绝缘。
4. **细胞特异性失明**：在泛组织并集上训练单一标量分类器，无法解析谱系特异的
   master TF 协同语法。

### 1.2 三条评估准则（所有实验共同遵守）

| 编号 | 准则 | 落地方式 |
|---|---|---|
| **R1** | 五重染色体正交隔离 | 每 split 三组互斥染色体（train / val / test）；test 染色体绝不进入训练或选参 |
| **R2** | 验证集驱动阈值 | F1 阈值仅在 val 上搜索并冻结，再用于 test；禁止在 test 上调参 |
| **R3** | 跨尺度严格对齐 | 同一 GT 单元在各自原生尺度打分（Cerberus 524 kb 真实上下文 / 基线 3 kb 窗），并保证"均值对均值、max 对 max"的数学对应 |

---

## 2. 公共基础设施

### 2.1 数据与参考资产

所有外部资产以**只读符号链接**集中在本目录 `assets/`（不复制实体）：

| 资产 | 路径 | 用途 |
|---|---|---|
| 参考基因组 | `Genome/gencode/hg38/GRCh38.primary_assembly.genome.fa` | 序列输入、N/gap 掩码生成 |
| 染色体长度 | `project/SE_model/data/genomesize/hg38.chrom.sizes` | 边界与 safety shift |
| ENCODE blacklist | `Genome/blacklist/hg38.blacklist.bed` | 全实验统一掩码 |
| ROSE 注释（90 channel） | `SE_pipeline/SuperEnhancer/<channel>/hg38_AllStitched.table.txt` | SE/TE 标签、簇定义 |
| MACS2 peaks（90 channel） | `SE_pipeline/peaks/<channel>_peaks.bed` | prcmax 的构成峰、Exp3 训练标签 |
| channel 映射 | `SE_pipeline/AllSample.hg38.SE.label_cache.metadata.pkl` | Cerberus 通道 ↔ 细胞类型 |
| Cerberus ckpt（5） | s0 `test10/Cerberus_human_best.pth`；s1–s4 `20260907.multi_chr_split__benchmark/cerberus_split{s}/finetune/Cerberus_human_best.pth` | 5-split 主模型（**finetune 终版**） |
| 基线 Union ckpt（20） | s0 `20260906.ownSE_annotation__benchmark/ckpt/`；s1–s4 `20260907.multi_chr_split__benchmark/ckpt_split{s}/` | Exp1/Exp2 基线（零重训） |
| 全染色体 GT / 轨道 | `20260907.full_chrom__benchmark/`（split0）、`multi_chr_split__benchmark/fullchrom_gt*_split{s}/`、`tracks*_split{s}/` | Exp2/Exp3 复用 |

### 2.2 5-split 染色体隔离协议

| split | val 染色体 | test 染色体 |
|---|---|---|
| split0 | chr9, chr14 | chr3, chr11 |
| split1 | chr1, chr15 | chr2, chr12 |
| split2 | chr4, chr18 | chr5, chr17 |
| split3 | chr6, chr19 | chr7, chr20 |
| split4 | chr8, chr21 | chr10, chr13 |

- 5 套划分互不重叠；所有指标报 **mean ± SEM（n = 5 splits）**。
- split0 的 Cerberus 为项目长期固定模型 `test10/Cerberus_human_best.pth`；
  s1–s4 为该 split 从 HF Borzoi + 随机 head 重训的 finetune 终版模型。

### 2.3 统一评分口径 `prcmax`（peak-restricted + max）

对任意 GT 单元 `U`（Exp1/Exp3 = 一条 ROSE 缝合簇；Exp2 region = 一个 GT 区间）：

1. **构成峰**：落在 `U` 内的 90-cell MACS2 narrowPeak（去重）。
2. **单峰分数**：
   - **基线模型**：峰中心 3 kb 窗的模型输出（与训练口径一致）；
   - **Cerberus**：该峰覆盖全部 256-bp bins 的通道预测**空间均值**
     （Exp1 在 90 通道上先取均值再跨通道取 max；Exp2 用 Union 轨道；
     Exp3 用该谱系单通道）。
3. **单元分数**：`score(U) = max(构成峰分数)`。
4. **退化情形**：无构成峰的单元（少数 desert 瓦片）回退为单元内 bin 的 max，
   回退数量落盘在指标 CSV 的 `fallback_*` 列。

**为什么用 max 而不是 mean**：ROSE 语义下 SE 由最强 constituent peak 定义；
mean 会被长单元内的弱峰/非峰窗口稀释，对"按 peak 单窗训练"的基线不公平。
prcmax 让三个实验的标量分数可比。

**prcmax 的适用边界**：只改"单元标量分数"。以下内容**不随 prcmax 改变**——
样本池定义、F1 的缝合与 IoU≥0.3 匹配流程、bin 级指标、Desert FP/Mb、
TP median IoU。

### 2.4 掩码规范

| 掩码 | 定义 | 用途 |
|---|---|---|
| blacklist | ENCODE `hg38.blacklist.bed` | 全实验剔除 |
| gap | 从 FASTA 生成的连续 N 段（GRCh38 primary assembly，无现成文件） | Exp2 全染色体扫描与 desert 定义 |
| desert | `(90-cell cluster 并集之外) ∧ ¬blacklist ∧ ¬gap` | Desert FP/Mb 的分母与 region 负样本瓦片 |

- bin 级指标只在非 blacklist / 非 gap 的 bins 上计算。
- 簇级样本池（Exp1/Exp3）沿用既往口径剔除与 blacklist 重叠的行。

---

## 3. 实验一：Union 簇级二分类（cluster-level）

### 3.1 命题与任务

在基线的原生输入约束（固定 3 kb 窗）下，以完整 ROSE 缝合簇为样本单元做
泛组织并集二分类，考察 Cerberus 的 524 kb 真实上下文先验能否在"基线舒适区"
任务上仍显著领先。

### 3.2 样本池

- **正样本** = 90-cell 并集的 SE 缝合块（`isSuper==1`，跨 channel merge gap=0）。
- **负样本** = **TE-only 并集块**（`isSuper==0` 行的并集 − SE 并集，碱基层次差集，
  保证与正样本零重叠）。
- **自然分布，不做下采样**；实测 test 池正负比约 **1 : 9–11.5**，
  中位长度：SE 块 58–67 kb、TE 块 1–2 kb。
- 与既有 `union/se_union.{chr}.npy` 做一致性交叉校验。

### 3.3 打分（prcmax 特化）

| 模型 | 单元分数的实际计算 |
|---|---|
| Cerberus | `max over peaks( max over 90 channels( mean over 峰 bins( joint 概率 ) ) )`，读 Exp2 全染色体稠密轨道（零新推理） |
| 基线 ×4 | `max over peaks( 峰中心 3 kb 窗 模型输出 )`，复用该 split 的 Union ckpt（零重训） |

### 3.4 指标与阈值

- **AUROC / AUPRC**：无阈值，直接在簇级标量上计算。
- **F1_vt**：阈值在 val 簇池上以 F1 最大为准则搜索（网格 0.01–0.99，step 0.005），
  冻结后应用于 test 簇池；簇级二分类直接阈值判定，不做缝合/IoU。
- **显著性**：Cerberus vs 各基线，5-split 双尾配对 t 检验（df = 4）。

### 3.5 交付物

`exp1/fig/EXP1_prcmax_{auprc,auroc,f1vt}_bar.pdf`、`EXP1_prcmax_{roc,pr}.pdf`；
表 `summary/EXP1_prcmax_cluster_metrics_all.csv`、
`summary/EXP1_prcmax_significance_tests.csv`。

### 3.6 结果（n = 5 splits，mean ± SEM）

| model | AUROC | AUPRC | F1_vt |
|---|---|---|---|
| **Cerberus_union** | **0.9377 ± 0.0033** | **0.6983 ± 0.0108** | **0.6140 ± 0.0122** |
| SENet | 0.8953 ± 0.0087 | 0.5157 ± 0.0184 | 0.4970 ± 0.0170 |
| MuSE_official | 0.8932 ± 0.0090 | 0.4881 ± 0.0171 | 0.4830 ± 0.0132 |
| DcSE | 0.8915 ± 0.0084 | 0.4882 ± 0.0184 | 0.4912 ± 0.0154 |
| DeepSE_official | 0.8825 ± 0.0078 | 0.4676 ± 0.0156 | 0.4710 ± 0.0114 |

12/12 项配对 t 检验 p < 0.01（最大 p = 2.6e-3，SENet AUROC）。

---

## 4. 实验二：全染色体 bin / region 目标检测（Union）

### 4.1 命题

在未经切割的整条盲测染色体上，评估模型从大片非编码区域中定位物理边界并抑制
假阳性的能力。

### 4.2 两阶段口径

1. **GT-agnostic 扫描**：整条 test 染色体先无标签地跑一遍，得到连续预测轨道。
2. **GT 定义评估**：扫描完成后才引入 GT —— bin 标签 = Union SE mask
   （排除 blacklist/gap）；region 正 = GT SE 区间，region 负 = typical ∪ desert
   瓦片；区间分数 = 区间内 bin 分数的 **max**（prcmax 下改用构成峰 max，见 §2.3）。

### 4.3 轨道与聚合（复用既有产物，不重扫）

| 模型 | 窗口 | 步长 | 聚合 |
|---|---|---|---|
| Cerberus | 524,288 bp（有效输出 1,600 × 256 bp） | test 25,600 bp / val 51,200 bp | 重叠区逐 bin **max**；Union = 90 通道逐 bin max |
| 基线 ×4 | 3,000 bp | test 500 bp / val 5,000 bp | 窗口分数写入覆盖 bins，重叠取 **max** |

### 4.4 评估单元与指标

- **Tier 1 · bin 级（无阈值）**：样本 = 每个 256 bp bin；指标 = AUROC / AUPRC。
  注意 bin 级正例比例高达 **0.34–0.39**（90-cell 并集 SE 覆盖率本就高），
  AUPRC 随机基线即等于该比例，故 bin 级数值天然高于 region 级。
- **Tier 2 · region 级（有阈值）**：
  - region 池：正 = Union SE 区间（与 Exp1 的 SE 块同源）；负 = typical
    （`cluster_union − se_union`）∪ desert 瓦片（80 bins ≈ 20.48 kb）；
    实测（split0）1,072 正 / 23,814 负 → prevalence ≈ **4.3%**。
  - 区间分数 = prcmax（构成峰 max；无峰回退 bin max）。
  - **Region F1**：bin > 阈值后缺口 ≤50 bins（12.8 kb）缝合为预测候选，与 GT SE 做
    **IoU ≥ 0.3 贪心单射匹配**；阈值由 val 上最大化 F1 选出。
  - **TP median IoU @ recall 0.8**：命中区间的中位 IoU。
  - **Region 不做 ROC**（见 §4.5）。
- **Desert FP / Mb @ recall 0.8**：`荒漠中被判阳性的 256-bp bin 总长(Mb) ÷ 该 split
  荒漠总长(Mb)`（bin 级长度口径）。

### 4.5 为什么 region 级坚决不做 ROC

真实 SE 只有数十个物理块，而其余非编码背景没有任何生物学标准能定义它是
"一个大负例"还是"十几万个小负例"。若人工切块，庞大的 TN 会把
`FPR = FP/(FP+TN)` 压到接近 0，所有模型的 region ROC 都紧贴左上角、AUROC 虚高
到 0.99 以上，完全丧失区分度。PR 体系不需 TN，只关注检出区间的 Precision 与
Recall，是目标检测的标准选择。

### 4.6 交付物

- bin 级：`EXP2_bin_{auroc,auprc}_bar.pdf`、`EXP2_bin_{roc,pr}.pdf`、
  `EXP2_desert_fp_recall08_bar.pdf`。
- region 级：`EXP2_prcmax_region_{auprc_bar,pr}.pdf`、`EXP2_region_f1_bar.pdf`、
  `EXP2_region_iou_median_bar.pdf`。
- 表：`exp2/summary/EXP2_metrics_bin_all.csv`、
  `EXP2_prcmax_region_metrics_all.csv`、`EXP2_prcmax_significance_tests.csv`、
  `EXP2_desert_fp_all.csv`、`EXP2_region_iou_all.csv`、`EXP2_union_f1_all.csv`。

### 4.7 结果（n = 5 splits，mean ± SEM）

| model | bin AUROC | bin AUPRC | region AUPRC | region F1 | Desert FP/Mb@rec0.8 | TP med IoU |
|---|---|---|---|---|---|---|
| **Cerberus_union** | **0.9249±0.0033** | **0.8875±0.0064** | **0.6430±0.0122** | **0.4977±0.0127** | **0.028±0.0022** | **0.6683±0.0029** |
| DcSE | 0.7730±0.0081 | 0.6476±0.0165 | 0.3323±0.0214 | 0.1458±0.0153 | 0.2649±0.0194 | 0.4994±0.0091 |
| SENet | 0.7711±0.0101 | 0.6493±0.0204 | 0.3407±0.0207 | 0.1433±0.0190 | 0.2715±0.0192 | 0.5020±0.0061 |
| MuSE_official | 0.7667±0.0091 | 0.6450±0.0184 | 0.3076±0.0185 | 0.1445±0.0083 | 0.2892±0.0209 | 0.4886±0.0085 |
| DeepSE_official | 0.7461±0.0109 | 0.6258±0.0217 | 0.2910±0.0172 | 0.1185±0.0055 | 0.3485±0.0222 | 0.4787±0.0078 |

region AUPRC 4/4 配对 t 检验 p < 1e-4。注意 bin 级（prevalence 0.37）与 region 级
（prevalence 0.043）是两个难度与随机基线完全不同的任务，区间数值不可直接比较；
正文以 **region 级为主、bin 级为补充**。

---

## 5. 实验三：谱系特异性本系对决（cluster 级）

### 5.1 命题与谱系

在 4 大代表谱系上重训基线，与 Cerberus 对应通道做**本系正向对决**，并用
pred × GT 交叉热图检验谱系可分性。

| 谱系 | 类型 | Cerberus 通道 idx |
|---|---|---|
| K562 | 红系/造血肿瘤 | 31 |
| HepG2 | 肝细胞癌/内胚层上皮 | 35 |
| GM12878 | 淋巴母细胞/适应性免疫 | 30 |
| WTC11 | iPSC（通道名带 PE 前缀，仅显示名差异） | 2 |

### 5.2 训练口径（纯单细胞标签）

- **正样本** = 与**该细胞自身 SE 行重叠**的 MACS2 peak（3 kb 单元）。
- **负样本** = 该细胞其余（非 SE）peaks ∪ 随机 3 kb 背景区（排除 blacklist、
  >50% N、与自身 SE/peaks 重叠），两类各半，1:1（cap 30k）。
- 4 模型 × 4 细胞 × 5 splits = **80 次重训**。

> **重要修正（2026-09-10）**：首版构建脚本误把"全部 peaks"当正样本
> （K562/split0 实测仅 14.9% 落在 SE，85.1% 落在 TE 簇），80 个基线模型与
> Exp3 基线图表因此作废，已按上式重建并重训/重打分。旧产物归档
> `exp3/_deprecated_buggy_labels/`。Cerberus / Exp1 / Exp2 不受影响。

### 5.3 评估池与打分

- **评估池 = 该细胞自身的 ROSE 行全集**（正 = `isSuper==1` SE 行；负 = TE 行），
  自然分布，不跨细胞借样本；实测 prevalence 0.04–0.088。
- **prcmax 打分**：簇分数 = 构成峰分数的最大值。
  - 基线：峰中心 3 kb 窗分数（谱系特异模型）；
  - Cerberus：该谱系**单通道**在峰 bins 上的均值，读 Exp2 全染色体稠密轨道
    （零新推理）。
- 指标：AUPRC（主）、AUROC 与 val-tuned cluster F1（附）；5-split 配对 t 检验。

### 5.4 交叉热图规范（pred × GT，每模型一张）

- 行 = pred 谱系（在该细胞训练的模型 / Cerberus 通道），列 = GT 谱系评估池；
  颜色 = AUPRC（test，5 split 均值）。
- **同一列的 4 行必须在同一批 region 上评分**；对角线 = 本系对决。
- Cerberus 行读 Exp2 的 90 通道稠密轨道（零新推理），对角格与主表逐簇 524 kb
  前向可差 ~0.02（相关 ~0.95），图注注明。
- 基线行需跨池补推理（80 个 GPU 任务），主表只含对角。
- 样式：seaborn heatmap，**统一色标 0–0.70**（`final_style.AUPRC_VMAX`），
  不加对角线黑框，格内只标均值（白字、无 SD/SEM），x 刻度标签水平。
- 不做阈值化 bleed-through。

### 5.5 交付物

`exp3/fig/EXP3_prcmax_{auprc,auroc,test_f1}_bar.pdf` +
`EXP3_prcmax_cross_heatmap_<model>.pdf`（5 张）；
表 `exp3/summary/EXP3_prcmax_cluster_metrics_all.csv`、显著性检验 CSV。

### 5.6 结果（n = 5 splits，mean ± SEM）

**AUPRC（主指标）**

| model | K562 | HepG2 | GM12878 | WTC11 |
|---|---|---|---|---|
| **Cerberus** | **0.544 ± 0.034** | **0.556 ± 0.030** | **0.587 ± 0.025** | **0.703 ± 0.021** |
| SENet | 0.202 ± 0.023 | 0.228 ± 0.019 | 0.215 ± 0.025 | 0.369 ± 0.017 |
| DcSE | 0.197 ± 0.023 | 0.231 ± 0.024 | 0.218 ± 0.016 | 0.360 ± 0.013 |
| MuSE_official | 0.185 ± 0.019 | 0.205 ± 0.025 | 0.131 ± 0.009 | 0.236 ± 0.011 |
| DeepSE_official | 0.159 ± 0.012 | 0.188 ± 0.013 | 0.160 ± 0.016 | 0.277 ± 0.014 |

- AUROC：Cerberus 0.901–0.965；基线 0.682–0.910。
- F1_vt：Cerberus 0.494–0.622；基线 0.111–0.408。
- 48/48 项配对 t 检验 p < 0.05（最大 p = 8.27e-3）。
- 交叉热图：Cerberus 对角 0.544–0.703 vs 非对角 0.159–0.328（对角优势明确）；
  四基线整个矩阵 0.11–0.37，**无对角结构**（本系模型不优于跨系）。

---

## 6. 统一统计与可视化规范

### 6.1 统计

- 每 split 一组独立结果，n = 5；报 **mean ± SEM**（SEM = SD/√5）。
- Cerberus vs 各基线：**双尾配对 t 检验（df = 4）**，结果落
  `summary/*_significance_tests.csv` 与 `exp2/summary/`、`exp3/summary/`。
- PR 曲线的阴影 = 逐 split 曲线插值到公共网格后取 mean ± SEM。

### 6.2 出图

- **只出 PDF**；TrueType（`pdf.fonttype = 42`）、Arial/Helvetica、`font.size ≈ 7`、
  只留左/下坐标轴、`legend.frameon = False`、**无 Panel Label**。
- 柱状图：Mean 实柱 + 5 split 散点 + SEM 误差线。
- 曲线：5-split mean 实线 + SEM 阴影带；图例标注 AP = mean ± SD。
- **配色唯一来源** `scripts/final_style.py`：Cerberus `#E86043`、
  DeepSE `#404287`、SENet `#F8BAA9`、MuSE `#8A89BE`、DcSE `#3C3C3C`。
  （2026-09-08 旧图集仍为 `benchmark_style.py` 哑光版，仅存档。）
- 每张图带 `EXPx · ...` 图题。

### 6.3 已知绘图坑（必须遵守）

| 问题 | 规范 |
|---|---|
| PR 曲线 recall=0 处 precision 假性 ~0.4 | sklearn 在有负样本分数高于全部正样本时给出多个 `(0,0)` 点 + 哨兵 `(0,1)`；须剔除 `(0,0)`、保留 `(0,1)`、同 recall 取最大 precision 后插值（AUPRC 数值不受影响） |
| log 轴柱状图在 AI 中释放剪切后溢出画板 | log 轴柱体显式设有限基线 `bottom = ymin`（`ymin = min(mean)/3`）+ `ylim=(ymin, ymax)`，禁用默认 `bottom=0` |
| 热图在 Illustrator 中只有左上角色块保留颜色 | 用 seaborn heatmap（矢量 QuadMesh），不用 `imshow`（会写成 4-bit Indexed 栅格） |

### 6.4 命名规范

- 目录：`exp1/ exp2/ exp3/`；文件前缀一律 `EXP1_ / EXP2_ / EXP3_`。
- 主口径文件带 `prcmax`；归档口径带 `_deprecated_*`，不出现于 `figs/` 主入口。
- 图统一走 `figs/` 相对符号链接（不复制实体）。

---

## 7. 交付物与目录结构

```
20260909.final_benchmark/
├── Benchmark_PLAN.md            # 本文件（唯一口径来源）
├── README.md                    # 目录导航与速查
├── EXECUTION_PLAN.md            # 历史执行链（S1–S8，含作业号）
├── GAP_AUDIT_20260909.md        # 资产核对与决策记录 D1–D18
├── _archive/                    # 旧版方案备份
├── assets/                      # 只读符号链接中心（genome/SE_pipeline/ckpt/GT/tracks/eval/legacy_scripts）
├── scripts/                     # 全部执行脚本（含 slurm runner、final_style.py）
├── exp1/{split0..4,curves,fig}  # Exp1 池 → 打分 → 指标 → 图
├── exp2/{masks*,eval_split*,summary,fig}
├── exp3/{pool_split*,ckpt_split*,data_split*,summary,fig} + _deprecated_*
├── summary/                     # 跨 split 汇总、显著性、报告、PR mean±SEM
├── figs/                        # 最终图统一入口（相对符号链接，22 个有效链接）
└── logs/                        # -> <repo>/logs/20260909
```

**`figs/` 布局（22 个有效链接）**

| 子目录 | 数量 | 内容 |
|---|---|---|
| `exp1_prcmax/` | 5 | `EXP1_prcmax_{auprc,auroc,f1vt}_bar.pdf`、`EXP1_prcmax_{roc,pr}.pdf` |
| `exp2_bin/` | 5 | `EXP2_bin_{auroc,auprc}_bar.pdf`、`EXP2_bin_{roc,pr}.pdf`、`EXP2_desert_fp_recall08_bar.pdf` |
| `exp2_region/` | 4 | `EXP2_prcmax_region_{auprc_bar,pr}.pdf`、`EXP2_region_f1_bar.pdf`、`EXP2_region_iou_median_bar.pdf` |
| `exp3_prcmax/` | 8 | `EXP3_prcmax_{auprc,auroc,test_f1}_bar.pdf` + `EXP3_prcmax_cross_heatmap_<model>.pdf` ×5 |
| `LEGACY_final_5split_20260908/` | 目录链接 | 2026-09-08 旧图集（存档对照） |
| `_deprecated_*` | 归档 | full-mean / prc-mean / 峰级试做 / 早期错误热图 |

**主图规划（5 面板建议）**

1. Exp1 PR 曲线（`Binary classification (union ROSE clusters) PR (mean ± SEM)`）
2. Exp1 F1 bar（`F1 score (val-tuned thresholds)`）
3. Exp2 region PR 曲线（`Whole-chromosome SE detection (union ROSE clusters) PR (mean ± SEM)`）
4. Exp2 region F1 bar
5. Exp2 Desert FP/Mb bar（recall = 0.8）

> 两张 F1 图需在图题中区分：Exp1 = 簇级 val-tuned 阈值；Exp2 = bin 阈值 + 缝合 +
> IoU≥0.3 的 region 检测 F1。PR 图建议在图例标 `AP = mean ± SD`，并注明随机基线
> = prevalence（Exp1 ≈ 0.08，Exp2 region ≈ 0.043，bin ≈ 0.37）。

---

## 8. 复现入口

```bash
BASE=test10_eval_new/20260909.final_benchmark

# Exp1 prcmax：块→构成峰映射 → 基线/ Cerberus 打分 → 指标与图
bash $BASE/scripts/run_exp1_prcmax_peaks.sh        # 323714
bash $BASE/scripts/run_exp1_prcmax_baseline.sh     # 323719（20 任务）
bash $BASE/scripts/run_exp1_prcmax_cerberus.sh     # 323720（5 任务）
bash $BASE/scripts/run_exp1_prcmax_metrics.sh      # 323744

# Exp2 prcmax region（bin/F1/desert/IoU 沿用原协议，无需重算）
bash $BASE/scripts/run_exp2_prcmax.sh              # 323745
bash $BASE/scripts/run_exp2_prcmax_plot.sh         # 323750

# Exp3 prcmax（cluster 级；交叉热图另算）
bash $BASE/scripts/run_exp3_baseline_prcmax.sh     # 323617（80 任务）
bash $BASE/scripts/run_exp3_cerberus_prcmax.sh     # 323618（5 任务）
bash $BASE/scripts/run_exp3_metrics_all.sh         # 指标
bash $BASE/scripts/run_exp3_cross_heatmaps.sh      # 322904 / 323712

# 汇总：报告 + PR-AUC mean±SEM
python $BASE/scripts/write_final_report.py
python $BASE/scripts/summary_pr_auc_mean_sem.py
```

计算规则：一律 sbatch 到计算节点（GPU 分区按难度选 H100 / gpu03 / gpu），
Job Array 并行，多阶段用 `--dependency=afterok` 自动串联；日志统一
`logs/20260909/job_%j.{out,err}`。

---

## 9. 口径备注与预设审稿问题

### 9.1 为什么基线不重训

Exp1/Exp2 复用既有的 20 个 Union ckpt（4 模型 × 5 splits），其训练口径为
MACS2 narrowPeak 单元中心 3 kb、`merge -d 3000`、1:1（cap 30k）+ 自然 test
评估。该口径与 Exp1 的评估样本（ROSE 缝合簇）在"单元"层面不同，因此
**prcmax 用构成峰分数做跨尺度对齐**，使基线不必在评估端被非峰窗口惩罚。

### 9.2 训练与评估的"解耦"质疑

基线按 3 kb peak 窗训练、在 cluster/region 上评估，看似解耦。prcmax 的
peak-restricted 变体正是为此设计：评估标量只由构成峰决定。三套聚合
（full-mean / prc-mean / prc-max）结论一致，见 §9.4。

### 9.3 为什么 Exp2 的 region 负样本含 desert

region 负 = typical（`cluster_union − se_union`）∪ desert 瓦片（80 bins），
即"全部非 SE 区域"，与既有 2026-09-08 冻结口径一致；若只用 typical，
任务会明显变简单（本轮未做该敏感性重算，可作为后续可选项）。

### 9.4 已归档口径（不在交付内）

| 口径 | 定义 | 结论 | 归档位置 |
|---|---|---|---|
| full-mean | 簇分数 = 簇覆盖 bins / 全部滑窗的均值 | 系统性压低基线（AUPRC 0.158–0.175） | `figs/_deprecated_non_prcmax/` |
| prc-mean | 簇分数 = 构成峰分数的均值 | 与 full-mean 几乎一致 | 同上 |
| prc-max | 簇分数 = 构成峰分数最大值 | **现行主口径** | 主图主表 |
| Exp3 peak 级 | 以 peak 为评估单元 | 试做后按用户要求撤销 | `exp3/_deprecated_peak_level/` |

三套口径结论方向完全一致（Cerberus 全面领先、对角优势保持），差异只在基线
绝对值，因此正文用 prc-max、以 mean 版作保守补充。

### 9.5 其它注意事项

- **Cerberus 交叉热图对角 ~0.02 偏差**：热图读稠密轨道、主表为逐簇 524 kb
  前向，属两套评分来源的正常差异，图注注明即可。
- **F1 的两种含义**：Exp1/Exp3 是簇级分数直接阈值（评估单元即簇）；
  Exp2 是 bin 阈值 + 缝合 + IoU≥0.3 的区间检测 F1。不可混用同一张图比较。
- **bin 级 vs region 级**：prevalence 0.37 vs 0.043，随机基线不同，数值不可直接
  比较；正文以 region 级为主。
- **WTC11 通道名**带 PE 前缀，仅为显示名差异。
- 5-split 只回答指标稳健性；机制/变异分析仍用固定模型（test10 系列）。

---

## 10. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-09-09 | 建立本目录与 5-split 协议；Exp1 定为 ROSE 簇级、基线零重训；desert 定义冻结；Region ROC 弃用（D1–D11） |
| 2026-09-10 | Exp3 增补 pred×GT 交叉热图（D12/D13）；Exp3 基线正样本 bug 修正并重训（D14）；峰级试做撤销（D15）；prc-mean/prc-max 变体（D16/D17）；**prcmax 定为唯一主口径**（D18） |
| 2026-09-11 | 三个实验 prcmax 结果合流；热图样式定稿（统一色标、只标均值、无对角框）；PR 插值 / log 轴几何修正；`figs/` 收敛为 22 个有效链接；新增 PR-AUC mean±SEM 汇总；本方案文档去冗余、结构化定稿 |
