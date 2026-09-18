# 用自己的 SE annotation 重训四个基线（2026-09-06）

> 与 `20260905.model_benchmark`（DcSE 数据）并列的第二个分析：**用本项目 90
> channel 自己的 ROSE SE/H3K27ac 数据**重新构建 chr-split 数据并重训四个基线。

## 1. 数据源

- `/public/home/liunangroup/liangyan/project/SE_model/SE_pipeline/SuperEnhancer/`
  每个 channel 的 ROSE 输出 `hg38_AllStitched.table.txt`（含 `isSuper`、
  `stitchedPeakRank`）；90 个 channel 与 Cerberus metadata 一一对应（已验证 90/90）；
- `/public/home/liunangroup/liangyan/project/SE_model/SE_pipeline/peaks/`
  每个 channel 的 H3K27ac narrowPeak（10 列 BED）。

## 2. 样本定义（与用户确认）

- **阳性**：H3K27ac peak 落入**任意** channel 的 ROSE SE（`isSuper==1`）区间。
  SE 常含多个 constituent peak，因此**以 peak 为单位**（不取 SE 区域中心），
  避免正样本中心落在无 peak 的 SE 空白区；
- **阴性**：与**全部 90 channel 的 SE 并集均不重叠**的纯典型增强子 peak
  （从未在任何谱系达到 SE 阈值）；
- **去重**：所有 channel 汇总后按 `merge -d 3000` 合并过近/跨 channel 重复峰；
- **输入**：合并单元中心的 3,000 bp 窗口（四基线输入规格）；
- **切分**：train = 非 3/9/11/14（1:1, cap=30,000/类）、val = 9/14（1:1）、
  test = 3/11（`test_balanced` 1:1；`test_all` 全量真实分布）。

## 3. 脚本与作业

| 脚本 | 作用 | 作业 | 状态 |
|---|---|---|---|
| `build_own_se_dataset.py` | 构建数据（SE union、peak 正负池、merge-3000、3kb 窗口、chr-split、序列提取） | ownSE_build 319981 | 完成 |
| `train_baseline.py`（复用 benchmark/） | 六个基线配置重训（官方为主 + 原版对照） | DeepSE_orig 319982 ✅（val 0.6987）、SENet 319983 ✅（0.7240）、DcSE 319985 ✅（0.7222）、DeepSE_official 319996 ✅（0.7008）、MuSE_orig 319984 ✅（0.7182, ep39）、MuSE_official 319998 ✅（0.7201, ep73） | 全部完成 |
| `eval_cerberus_test_ctx.py --no_sam` | Cerberus val 阈值 + chr3/11 test（safety shift，流式 batch） | 320165 ✅（319986 OOM / 320004 缺 no_grad / 320109 边界空切片均已修复并清理日志） | 完成 |
| `run_ownSE_post.sh` | afterok 自动链：6 基线 test eval -> val 阈值 -> 汇总 | 320166 ✅ | 完成 |

## 5.1 w2v 使用口径（2026-09-06 用户确认：一律官方）

- DcSE/SENet/MuSE-X1：6-mer `embedding_matrix.npy` == 官方 dna2vec 6-mer 子集
  （无选择，天然官方）；
- DeepSE / MuSE-X2：主配置 = **官方 k3to8**（`kmer_official_k3to8.npz`）；
  原版 hg19 w2v（`DeepSE_orig`/`MuSE_orig`）保留作对照；
- 上一轮 DcSE 数据基准同理：官方为主，原版作稳健性对照。

## 4. 数据统计（job 319981 完成）

- SE-constituent peak（正，merge -d 3000）：57,150；pure-TE peak（负）：137,049；
- 中心 3 kb 片段总计 194,199；
- train 60,000（30,000/类，封顶）、val 8,338（4,169/类）、
  test_balanced 13,132（6,566/类）、test_all 22,678（真实分布）；
- 阳性 `n_cells` 分布随共享谱系数递减（1:14,505 … 12:144），符合预期。

## 5. Cerberus safety 处理（2026-09-06 用户补充）

- `shifted_center = clamp(center, 262144, chrom_len-262144)`（与
  `se_data_utils.py`/`1__interpret_se.py` 一致）：窗口始终完整落在染色体内部；
- bin 坐标相对**实际窗口起点 + 57,344 bp（crop 224 bins×256）**换算，
  不假设片段位于窗口中心；
- own-SE 数据用 CSV 自带 `chr/unit_start/unit_end`（`--no_sam`）。

## 6. 状态

全部完成（2026-09-06）。结果：`test_eval/test_metrics.csv`、
`test_eval/val_thresholds_ownSE.json`。

### 6.1 chr3/11 盲测汇总（test_balanced n=13,132；test_all n=22,678）

| tag | val AUC/AUPR | test_bal AUC/AUPR | test_bal F1_vt | test_all AUC/AUPR |
|---|---|---|---|---|
| Cerberus_ctx | **0.8825 / 0.8777** | **0.8953 / 0.8943** | **0.8187** | **0.8953 / 0.7957** |
| SENet | 0.7240 / 0.7093 | 0.7300 / 0.7188 | 0.6987 | 0.7260 / 0.5201 |
| MuSE_official | 0.7201 / 0.7039 | 0.7299 / 0.7188 | 0.6995 | 0.7272 / 0.5228 |
| DcSE | 0.7222 / 0.7051 | 0.7299 / 0.7198 | 0.6969 | 0.7275 / 0.5198 |
| MuSE_orig（对照） | 0.7182 / 0.7031 | 0.7261 / 0.7154 | 0.6981 | 0.7241 / 0.5183 |
| DeepSE_official | 0.7008 / 0.6853 | 0.7115 / 0.7025 | 0.6933 | 0.7101 / 0.5026 |
| DeepSE_orig（对照） | 0.6987 / 0.6881 | 0.7065 / 0.6988 | 0.6823 | 0.7055 / 0.4995 |

> 关键结论：在**我们自己的 SE annotation**（SE-constituent peak 阳性 +
> 从未 SE 的 pure-TE peak 阴性）上，Cerberus 用真实 ±260 kb 上下文显著领先
> 四个基线（test_balanced AUC 0.895 vs ~0.71–0.73），与 DcSE 短片段数据上的
> 结论相反——任务输入域决定对比结论，论文中两种设定需分开表述。
