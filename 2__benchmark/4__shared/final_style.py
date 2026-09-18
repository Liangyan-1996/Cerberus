"""20260909 最终 benchmark 唯一配色/顺序（所有 EXP1/2/3 最终图只引用本模块）。

用户 2026-09-10 指定配色（按标准模型顺序映射）：
  Cerberus_union #E86043 / DeepSE_official #404287 / SENet #F8BAA9 /
  MuSE_official #8A89BE / DcSE #3C3C3C
旧 2026-09-08 图集沿用 assets/legacy_scripts/scripts/benchmark_style.py（哑光版），
本目录新图一律用本模块。
"""

MODELS = ["Cerberus_union", "DeepSE_official", "SENet",
          "MuSE_official", "DcSE"]

COLORS = {
    "Cerberus_union": "#E86043",
    "Cerberus_ctx": "#E86043",
    "Cerberus": "#E86043",
    "DeepSE_official": "#404287",
    "SENet": "#F8BAA9",
    "MuSE_official": "#8A89BE",
    "DcSE": "#3C3C3C",
}

# 所有 AUPRC 热图（cluster + peak，共 10 张）统一色标
AUPRC_VMAX = 0.70
