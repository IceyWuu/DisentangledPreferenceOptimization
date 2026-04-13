# -*- coding: utf-8 -*-
"""
Read margin_chain.jsonl logs and save comparison plots (no display).

This version incorporates your latest constraints:
- MAX_STEP = global_max_step // 2 across all runs (auto).
- Fig1..Fig5 follow your finalized definitions:
    Fig1: m_t vs step (NO smoothing; raw m)
    Fig2: |E_w[Δm_t]|              (windowed over RAW delta_m)
    Fig3: Var_w(Δm_t)              (windowed over RAW delta_m)
    Fig4: SNR_t = |E_w[Δm_t]| / sqrt(Var_w(Δm_t))
    Fig5: two-panel scatter:
          left : x=Var_w(m_t),   y=SNR_t
          right: x=|Δm_t| (RAW), y=SNR_t
- delta_m is ALWAYS the raw step-to-step diff and is NEVER smoothed.
- “AllVars vs step” plots:
    * DO NOT plot metadata / unused columns (e.g., grad_accum_steps, micro_batch_size, etc.)
    * DO NOT re-plot columns already covered by Fig1..Fig5
    * Symmetric pairs are grouped on one figure (zw+zl, sw_l2+sl_l2, dw+dl)
    * No duplicate plotting within AllVars (grouped cols won’t be plotted again as singles)
- Adds rho distribution (hist overlay).

Usage:
  1) Edit llm_model_dict / PYTHIA2B_DBEMA_SWEEP_MODE / RUNS / FIG_WINDOW / SMOOTH_* below
  2) cd DIL/outputs && python aexperiment_cw.py
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # IMPORTANT: no GUI
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.ticker import LogFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import numpy as np

USE_LATEX = False  # ← 开发时设为 False，最终出图设为 True
if USE_LATEX:
    # 高质量：调用系统 LaTeX
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        # "font.serif": ["Computer Modern Roman"],
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times"],
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts,amssymb,bm}",
    })
else:
    # 快速模式：使用 Matplotlib 内置 Computer Modern 风格
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times"],
        "mathtext.fontset": "cm",  # 关键：使用 Computer Modern math 字体
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts,amssymb,bm}",
        "axes.formatter.use_mathtext": True,
    })

ICML_MODE = True
ICML_COL_W_IN = 3.25   # single-column width
ICML_TEXT_W_IN = 6.75  # two-column (figure*) width

FIG_WIDTH = 3.6
FIG_HEIGHT = 3.2

# heights
ICML_LINE_H_IN = 2.05          # single plot height (when used alone)
ICML_PANEL_1x4_H_IN = 1.65     # 1x4 panel height
ICML_SCATTER_H_IN = 2.35       # for your Fig5 (2 panels)

DB_PLOT_MODE = "real"  # "real" 或 "algorithm_view"
# "real": 使用真实dw/dl计算DB（当前行为）
# "algorithm_view": 使用EMA平滑后的变量计算DB（算法视角）

def apply_icml_style():
    if not ICML_MODE:
        return
    plt.rcParams.update({
        "font.size": 9.0,          # 比你原来的更大（原来可能用5.x）
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.titlesize": 9.0,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "figure.figsize": (1.8, 1.6),   # ← 关键：单图尺寸小但字体大
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlepad": 4,        # 减少标题与图间距
        "axes.labelpad": 2,        # 减少轴标签与tick间距
    })

apply_icml_style()

# =========================
# User config (edit here)
# =========================
llm_model_dict = {    # pythia / mistral / qwen2.5-7b-instruct
    "1": "pythia-410m",
    "2": "pythia-2b",
    "3": "mistral-7b",
    "4": "qwen2.5-7b-instruct",
}
llm_model = llm_model_dict["2"]
OUT_DIR = r"./figs/aexperiment_{}".format(llm_model)

# ---- Pythia-2.8B：DPO + DB calibration 的 db_ema_beta 消融可视化 ----
PYTHIA2B_DBEMA_SWEEP_MODE = False
if PYTHIA2B_DBEMA_SWEEP_MODE:
    llm_model = "pythia-2b"
    OUT_DIR = r"./figs/aexperiment_pythia-2b_dpo_dbema_sweep"

if llm_model == "pythia-410m":
    # pythia-410m
    RUNS = [
        {"path": r"./pythia-410m-dpo-20251229_215315/margin_chain.jsonl", "label": "dpo"}, # bs32
        {"path": r"./pythia-410m-ipo-20251229_225439/margin_chain.jsonl", "label": "ipo"}, # bs32, tau=0.1
        {"path": r"./pythia-410m-cpo-20251230_163440/margin_chain.jsonl", "label": "cpo"}, # bs32
        {"path": r"./pythia-410m-simpo-20260119_225314/margin_chain.jsonl", "label": "simpo"},
        # {"path": r"./pythia-410m-slic-20251230_173548/margin_chain.jsonl", "label": "slic"}, # bs32
        {"path": r"./pythia-410m-ddro-20260106_124724/margin_chain.jsonl", "label": "ddro"},
        {"path": r"./pythia-410m-bce-20251224_174150/margin_chain.jsonl", "label": "bce"}, # bs32
        {"path": r"./pythia-410m-lsif-20251230_193848/margin_chain.jsonl", "label": "lsif"}, # bs32
        {"path": r"./pythia-410m-ukl-20251230_183721/margin_chain.jsonl", "label": "ukl"}, # bs32
        {"path": r"./pythia-410m-dpo-calib-ema-20260114_004611/margin_chain.jsonl", "label": "dpo-calib"},
        {"path": r"./pythia-410m-ipo-calib-ema-20260114_133257/margin_chain.jsonl", "label": "ipo-calib"},
        {"path": r"./pythia-410m-cpo-calib-ema-20260123_224309/margin_chain.jsonl", "label": "cpo-calib"},
        {"path": r"./pythia-410m-simpo-calib-ema-20260119_174156/margin_chain.jsonl", "label": "simpo-calib"},
        # {"path": r"./pythia-410m-slic-calib-ema2-20260115_152704/margin_chain.jsonl", "label": "slic-calib"},
        {"path": r"./pythia-410m-ddro-calib-ema2-20260115_225824/margin_chain.jsonl", "label": "ddro-calib"},
        {"path": r"./pythia-410m-bce-calib-ema-20260114_030220/margin_chain.jsonl", "label": "bce-calib"},
        {"path": r"./pythia-410m-lsif-calib-ema2-20260115_230220/margin_chain.jsonl", "label": "lsif-calib"},
        {"path": r"./pythia-410m-ukl-calib-ema-20260114_144239/margin_chain.jsonl", "label": "ukl-calib"},
    ]
elif llm_model == "pythia-2b":
    # Pythia-2.8B
    if PYTHIA2B_DBEMA_SWEEP_MODE:
        # db_ema_beta 消融：base DPO + 四个 β（路径按你本地 output 目录名改时间戳即可）
        RUNS = [
            # {"path": r"./pythia-2b-dpo-20260114_215720/margin_chain.jsonl", "label": "dpo"},
            {
                "path": r"./pythia-2b-dpo-calib-dbema0p5-20260325_023455_1_2285911/margin_chain.jsonl",
                "label": "dpo-calib-ema0p5",
            },
            {
                "path": r"./pythia-2b-dpo-calib-dbema0p9-20260325_023455_2_2285911/margin_chain.jsonl",
                "label": "dpo-calib-ema0p9",
            },
            {
                "path": r"./pythia-2b-dpo-calib-dbema0p95-20260325_120010_3_2285911/margin_chain.jsonl",
                "label": "dpo-calib-ema0p95",
            },
            {
                "path": r"./pythia-2b-dpo-calib-dbema0p999-20260325_120220_4_2285911/margin_chain.jsonl",
                "label": "dpo-calib-ema0p999",
            },
        ]
    else:
        RUNS = [
            # {"path": r"./pythia-2b-dpo-20260114_215720/margin_chain.jsonl", "label": "dpo"}, # bs32
            # {"path": r"./pythia-2b-ipo-20260116_004600/margin_chain.jsonl", "label": "ipo"}, # bs32，tau=0.1
            # {"path": r"./pythia-2b-cpo-20260115_022527/margin_chain.jsonl", "label": "cpo"}, # bs32
            # {"path": r"./pythia-2b-tidpo-20260329_034847/margin_chain.jsonl", "label": "tidpo"},
            # {"path": r"./pythia-2b-simpo-20260120_102856/margin_chain.jsonl", "label": "simpo"},
            # # {"path": r"./pythia-2b-slic-20260115_112145/margin_chain.jsonl", "label": "slic"}, # bs32
            # {"path": r"./pythia-2b-ddro-20260115_201741/margin_chain.jsonl", "label": "ddro"},
            # {"path": r"./pythia-2b-bce-20260115_065352/margin_chain.jsonl", "label": "bce"}, # bs32
            # {"path": r"./pythia-2b-lsif-20260115_154949/margin_chain.jsonl", "label": "lsif"}, # bs32
            # {"path": r"./pythia-2b-ukl-20260116_051323/margin_chain.jsonl", "label": "ukl"}, # bs32

            # {"path": r"./pythia-2b-dpo-calib-em2-20260116_011137/margin_chain.jsonl", "label": "dpo-calib"},
            # {"path": r"./pythia-2b-dpo-calib-20260331_224638/margin_chain.jsonl", "label": "dpo-calib"},
            # {"path": r"./pythia-2b-ipo-calib-em2-20260116_144013/margin_chain.jsonl", "label": "ipo-calib"},
            # {"path": r"./pythia-2b-cpo-calib-ema-20260120_171125/margin_chain.jsonl", "label": "cpo-calib"},
            # {"path": r"./pythia-2b-tidpo-calib-ema-20260329_034841/margin_chain.jsonl", "label": "tidpo-calib"},
            # {"path": r"./pythia-2b-simpo-calib-ema-20260120_000735/margin_chain.jsonl", "label": "simpo-calib"},
            # # {"path": r"./pythia-2b-slic-calib-em2-20260116_095749/margin_chain.jsonl", "label": "slic-calib"},
            # {"path": r"./pythia-2b-ddro-calib-ema-20260123_171851/margin_chain.jsonl", "label": "ddro-calib"},
            # {"path": r"./pythia-2b-bce-calib-ema-20260120_043811/margin_chain.jsonl", "label": "bce-calib"},
            # {"path": r"./pythia-2b-lsif-calib-em2-20260116_201625/margin_chain.jsonl", "label": "lsif-calib"},
            # {"path": r"./pythia-2b-ukl-calib-ema-20260122_003450/margin_chain.jsonl", "label": "ukl-calib"},

            # {"path": r"./pythia-2b-dpo-20260404_113903/margin_chain.jsonl", "label": "dpo"},
            {"path": r"./pythia-2b-ipo-20260404_113903/margin_chain.jsonl", "label": "ipo"},
            {"path": r"./pythia-2b-cpo-20260404_113903/margin_chain.jsonl", "label": "cpo"},
            {"path": r"./pythia-2b-tidpo-20260329_034847/margin_chain.jsonl", "label": "tidpo"},
            {"path": r"./pythia-2b-simpo-20260404_010443/margin_chain.jsonl", "label": "simpo"},
            {"path": r"./pythia-2b-ddro-20260404_010443/margin_chain.jsonl", "label": "ddro"},
            {"path": r"./pythia-2b-bce-20260404_113903/margin_chain.jsonl", "label": "bce"},
            {"path": r"./pythia-2b-dpo-calib-20260331_224638/margin_chain.jsonl", "label": "dpo-calib"},
            {"path": r"./pythia-2b-ipo-calib-20260404_113903/margin_chain.jsonl", "label": "ipo-calib"},
            {"path": r"./pythia-2b-cpo-calib-20260404_010443/margin_chain.jsonl", "label": "cpo-calib"},
            {"path": r"./pythia-2b-tidpo-calib-ema-20260329_034841/margin_chain.jsonl", "label": "tidpo-calib"},
            {"path": r"./pythia-2b-simpo-calib-20260404_010443/margin_chain.jsonl", "label": "simpo-calib"},
            {"path": r"./pythia-2b-ddro-calib-20260404_010443/margin_chain.jsonl", "label": "ddro-calib"},
            {"path": r"./pythia-2b-bce-calib-20260404_113903/margin_chain.jsonl", "label": "bce-calib"},
        ]
    
elif llm_model == "mistral-7b":
    # mistral-7b
    RUNS = [
        # {"path": r"./mistral-7b-dpo-lora-20260105_004932/margin_chain.jsonl", "label": "dpo"}, # bs32
        # {"path": r"./mistral-7b-cpo-lora-20260119_063425/margin_chain.jsonl", "label": "cpo"},
        {"path": r"./mistral-7b-tidpo-lora-20260329_032137/margin_chain.jsonl", "label": "tidpo"},
        # {"path": r"./mistral-7b-simpo-lora-20260120_103452/margin_chain.jsonl", "label": "simpo"},
        # {"path": r"./mistral-7b-ddro-lora-20260118_193824/margin_chain.jsonl", "label": "ddro"},
        # {"path": r"./mistral-7b-bce-lora-20260105_115557/margin_chain.jsonl", "label": "bce"}, # bs32
        # {"path": r"./mistral-7b-lsif-lora-20260119_174530/margin_chain.jsonl", "label": "lsif"}, # bs32
        # {"path": r"./mistral-7b-dpo-lora-calib-20260120_231406/margin_chain.jsonl", "label": "dpo-calib"}, # bs32
        # {"path": r"./mistral-7b-cpo-lora-calib-20260121_103541/margin_chain.jsonl", "label": "cpo-calib"},
        {"path": r"./mistral-7b-tidpo-lora-calib-20260329_032116/margin_chain.jsonl", "label": "tidpo-calib"},
        # {"path": r"./mistral-7b-simpo-lora-calib-20260121_220008/margin_chain.jsonl", "label": "simpo-calib"},
        # {"path": r"./mistral-7b-ddro-lora-calib-20260121_091425/margin_chain.jsonl", "label": "ddro-calib"},
        # {"path": r"./mistral-7b-bce-lora-calib-20260121_202327/margin_chain.jsonl", "label": "bce-calib"}, # bs32
        # {"path": r"./mistral-7b-lsif-lora-calib-20260120_220857/margin_chain.jsonl", "label": "lsif-calib"}, # bs32

        # # {"path": r"./mistral-7b-dpo-full-lora-20260107_235806/margin_chain.jsonl", "label": "dpo-full"}, # bs32
        # # {"path": r"./mistral-7b-bce-full-lora-20260109_002210/margin_chain.jsonl", "label": "bce-full"}, # bs32
        # {"path": r"./mistral-7b-dpo-lora-20260325_000807/margin_chain.jsonl", "label": "dpo"}, # bs32
        # {"path": r"./mistral-7b-dpo-lora-calib-20260326_013116/margin_chain.jsonl", "label": "dpo-calib"}, # full-hh
        # {"path": r"./mistral-7b-dpo-lora-calib-20260326_183213/margin_chain.jsonl", "label": "dpo-calib"}, # merge-hh
        # {"path": r"./mistral-7b-dpo-lora-20260327_092032/margin_chain.jsonl", "label": "dpo"}, # merge-hh
        # {"path": r"./mistral-7b-bce-lora-calib-20260326_211459/margin_chain.jsonl", "label": "bce-calib"}, # merge-hh
        # {"path": r"./mistral-7b-bce-lora-20260326_214331/margin_chain.jsonl", "label": "bce"}, # merge-hh
    ]

elif llm_model == "qwen2.5-7b-instruct":
    RUNS = [
        {"path": r"./qwen2.5-7b-instruct-dpo-lora-20260327_013512/margin_chain.jsonl", "label": "dpo"},
        {"path": r"./qwen2.5-7b-instruct-dpo-lora-calib-20260329_041005/margin_chain.jsonl", "label": "dpo-calib"},
        {"path": r"./qwen2.5-7b-instruct-bce-lora-20260327_013512/margin_chain.jsonl", "label": "bce"},
        {"path": r"./qwen2.5-7b-instruct-bce-lora-calib-20260329_041005/margin_chain.jsonl", "label": "bce-calib"},
        {"path": r"./qwen2.5-7b-instruct-cpo-lora-20260331_225523/margin_chain.jsonl", "label": "cpo"},
        {"path": r"./qwen2.5-7b-instruct-cpo-lora-calib-20260331_225523/margin_chain.jsonl", "label": "cpo-calib"},
        {"path": r"./qwen2.5-7b-instruct-simpo-lora-20260331_225523/margin_chain.jsonl", "label": "simpo"},
        {"path": r"./qwen2.5-7b-instruct-simpo-lora-calib-20260331_225523/margin_chain.jsonl", "label": "simpo-calib"},
        {"path": r"./qwen2.5-7b-instruct-tidpo-lora-20260401_190750/margin_chain.jsonl", "label": "tidpo"},
        {"path": r"./qwen2.5-7b-instruct-tidpo-lora-calib-20260401_190750/margin_chain.jsonl", "label": "tidpo-calib"},
        {"path": r"./qwen2.5-7b-instruct-ddro-lora-20260401_190750/margin_chain.jsonl", "label": "ddro"},
        {"path": r"./qwen2.5-7b-instruct-ddro-lora-calib-20260401_190750/margin_chain.jsonl", "label": "ddro-calib"},
        {"path": r"./qwen2.5-7b-instruct-lsif-lora-20260401_190750/margin_chain.jsonl", "label": "lsif"},
        {"path": r"./qwen2.5-7b-instruct-lsif-lora-calib-20260331_225523/margin_chain.jsonl", "label": "lsif-calib"},

        # {"path": r"./qwen2.5-7b-instruct-dpo-lora-20260326_023142/margin_chain.jsonl", "label": "dpo"},
        # {"path": r"./qwen2.5-7b-instruct-dpo-lora-calib-20260326_023142/margin_chain.jsonl", "label": "dpo-calib"},
        # {"path": r"./qwen2.5-7b-instruct-bce-lora-20260326_023142/margin_chain.jsonl", "label": "bce"},
        # {"path": r"./qwen2.5-7b-instruct-bce-lora-calib-20260326_023142/margin_chain.jsonl", "label": "bce-calib"},
    ]

# Base colors (Paul Tol optimized)
BASE = {
    "bce":  "#4477AA",  # 深蓝
    "dpo":  "#EE6677",  # 珊瑚红
    "ipo":  "#228833",  # 森林绿
    "cpo":  "#CCBB44",  # 金黄
    "simpo": "#33BBEE",  # 亮蓝   #CC6677 红褐    #3399FF 亮蓝
    "lsif": "#D55E00",  # 橙红
    "ukl":  "#666666",  # 深灰
    "ddro": "#8c564b",  # brown（与 LSIF 区分）
    "tidpo": "#228833",  # 黑色
}

def lighten(hex_col, f=0.25):
    r, g, b = mcolors.hex2color(hex_col)
    return mcolors.to_hex([min(1, c + f*(1-c)) for c in (r,g,b)])

def darken(hex_col, f=0.25):
    r, g, b = mcolors.hex2color(hex_col)
    return mcolors.to_hex([max(0, c - f*c) for c in (r,g,b)])
    
METHOD_REGISTRY = {
    # Original
    "bce":      {"display": "DIL-BCE",       "color": BASE["bce"]},
    "dpo":      {"display": "DPO",       "color": BASE["dpo"]},
    "ipo":      {"display": "IPO",       "color": BASE["ipo"]},
    "cpo":      {"display": "CPO",       "color": BASE["cpo"]},
    "simpo":     {"display": "SimPO",    "color": BASE["simpo"]},
    "tidpo":     {"display": "TIDPO",    "color": BASE["tidpo"]},
    "lsif":     {"display": "DIL-LSIF",      "color": BASE["lsif"]},
    "ukl":      {"display": "DIL-UKL",       "color": BASE["ukl"]},
    "ddro":     {"display": "DDRO",      "color": BASE["ddro"]},
    # Calibrated (lighter)
    "bce-calib": {"display": "DIL-BCE w/ RC", "color": lighten(BASE["bce"])},
    "dpo-calib": {"display": "DPO w/ RC", "color": lighten(BASE["dpo"])},
    "ipo-calib": {"display": "IPO w/ RC", "color": lighten(BASE["ipo"])},
    "cpo-calib": {"display": "CPO w/ RC", "color": lighten(BASE["cpo"])},
    "simpo-calib":{"display": "Simpo w/ RC","color": lighten(BASE["simpo"])},
    "tidpo-calib":{"display": "TIDPO w/ RC","color": lighten(BASE["tidpo"])},
    "lsif-calib":{"display": "DIL-LSIF w/ RC","color": lighten(BASE["lsif"])},
    "ukl-calib": {"display": "DIL-UKL w/ RC", "color": lighten(BASE["ukl"])},
    "ddro-calib":{"display": "DDRO w/ RC","color": lighten(BASE["ddro"])},
}

SAVE_FORMAT = "pdf"  # "png" or "pdf"
DPI = 300

# ---- window used for Fig2/3/4 and Fig5 scatter ----
FIG_WINDOW = 50 # 50

# 全局下采样策略（可根据需要调整）
SUBSAMPLE_EVERY = {
    "Fig1": 10,
    "Fig2": 10,
    "Fig3": 10,
    "Fig4_SNR": 10,
    "Fig4_tSNR": 10,
    "Fig6": 10,
    "Fig6b": 10,
}
ALLVARS_SUBSAMPLE_EVERY = 5

# For rolling stats, to avoid early-step explosions:
# - "full": min_periods = window (stable; first window-1 become NaN)
# - "half": min_periods = max(5, window//2)
ROLL_MIN_PERIODS_MODE = "full"  # "full" or "half"

# ---- smoothing (applied only where allowed; delta_m is always excluded) ----
SMOOTH_ENABLE = True
SMOOTH_METHOD = "ema"          # "ema" or "rolling"
SMOOTH_EMA_SPAN = 25           # typical: 10~50
SMOOTH_ROLLING_WINDOW = 25
SMOOTH_MIN_PERIODS = 1

# Always keep delta_m raw (per your definition); do not smooth it anywhere.
GLOBAL_NO_SMOOTH_COLS: Set[str] = {"delta_m"}

# ---- output subdirs ----
AUTO_PLOT_ALL_VARS = True
AUTO_PLOT_DIRNAME = "all_vars_vs_step"
RHO_DIST_DIRNAME = "rho_distribution"

# Only plot these “meaningful” variables in AllVars (everything else is ignored).
# (And we will further remove anything already plotted by Fig1..Fig5.)
ALLVARS_ALLOWLIST: Set[str] = {
    "zw", "zl",
    "delta_m",          # raw
    "sw_l2", "sl_l2",
    "sw_sl_dot",
    "rho",
    "dw", "dl",
    "dw_over_dl",     
    "log_dw_over_dl",    
    # "lr",
}

# Symmetric groups (draw on the same figure in AllVars)
ALLVARS_SYMMETRIC_GROUPS = [
    {"slug": "zw_zl",       "cols": ["zw", "zl"]},
    {"slug": "dw_dl",       "cols": ["dw", "dl"]},
    {"slug": "sw_l2_sl_l2", "cols": ["sw_l2", "sl_l2"]},
]


# =========================
# Helpers
# =========================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_slug(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:160] if len(s) > 160 else s

def get_all_variables_label(variable):
    return {"zw_zl": r"Chosen / rejected rewards",
            "dw_dl": r"Incentive coefficients",
            "sw_l2_sl_l2": r"$\|\boldsymbol{s}_w\|, \|\boldsymbol{s}_l\|$",
            "delta_m": r"Displacement $\Delta m$",
            "dw_over_dl": r"Incentive ratio",
            "log_dw_over_dl": r"Incentive log-ratio",
            "rho": r"Cosine similarity",
            "sw_sl_dot": r"$\langle \boldsymbol{s}_{w,t}, \boldsymbol{s}_{l,t}\rangle$"}[variable]

def load_jsonl_as_df(jsonl_path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            # Keep scalars only; ignore huge *_vec lists/dicts to avoid memory blow-up.
            row: Dict[str, Any] = {}
            for k, v in obj.items():
                if k.endswith("_vec"):
                    continue
                if isinstance(v, (list, dict)):
                    continue
                row[k] = v
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure step exists & sorted
    if "step" in df.columns:
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df = df.dropna(subset=["step"]).sort_values("step")
    else:
        df["step"] = range(1, len(df) + 1)
        
    # 如果是矫正模型
    if "calib" in jsonl_path:
        # 根据DB_PLOT_MODE决定如何处理
        if DB_PLOT_MODE == "real":
            # 当前行为：使用真实DB
            alpha = 1.
            df["dw"] = df["dw"] * alpha
            df["dl"] = df["dl"] / alpha
        elif DB_PLOT_MODE == "algorithm_view":
            # 算法视角：使用EMA平滑后的变量
            # 检查是否有EMA平滑的变量
            if "db/dw_ema" in df.columns and "db/dl_ema" in df.columns:
                # 使用EMA平滑后的dw/dl
                df["dw"] = df["db/dw_ema"]
                df["dl"] = df["db/dl_ema"]
            else:
                # 如果没有EMA变量，回退到真实DB
                print(f"[WARN] {jsonl_path}: 缺少EMA平滑变量，使用真实DB")
                alpha = 1.
                df["dw"] = df["dw"] * alpha
                df["dl"] = df["dl"] / alpha
            
            # 如果存在EMA平滑的边界，也使用它们
            if "db/lower" in df.columns and "db/upper" in df.columns:
                # 这些将在后续的add_ratio_band函数中被使用
                pass

    # Coerce numeric columns where possible (keep loss_type as-is)
    for c in df.columns:
        if c in ("loss_type",):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)

def _blend_to_white(color, amount: float):
    """amount in [0,1]. 0 -> original color, 1 -> white."""
    r, g, b = mcolors.to_rgb(color)
    r2 = r + (1.0 - r) * amount
    g2 = g + (1.0 - g) * amount
    b2 = b + (1.0 - b) * amount
    return (r2, g2, b2)

def get_roll_min_periods(window: int) -> int:
    if ROLL_MIN_PERIODS_MODE.lower() == "half":
        return max(5, window // 2)
    return window

def ensure_m_and_delta_m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures:
      - m exists (prefer logged m; else fallback m = zw - zl)
      - delta_m exists as RAW step-to-step diff: delta_m[t] = m[t] - m[t-1]
    """
    df = df.sort_values("step").reset_index(drop=True)

    if "m" not in df.columns:
        if "zw" in df.columns and "zl" in df.columns:
            df["m"] = pd.to_numeric(df["zw"], errors="coerce") - pd.to_numeric(df["zl"], errors="coerce")
        else:
            df["m"] = np.nan

    m = pd.to_numeric(df["m"], errors="coerce")
    df["m"] = m

    # Always (re)compute raw delta_m from m
    df["delta_m"] = m.diff()
    return df

def add_dw_dl_ratio(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Adds:
      - dw_over_dl      = dw / dl
      - log_dw_over_dl  = log(dw/dl)
    Robust to dl<=0 or non-finite values (set to NaN).
    """
    if ("dw" not in df.columns) or ("dl" not in df.columns):
        df["dw_over_dl"] = np.nan
        df["log_dw_over_dl"] = np.nan
        return df

    dw = pd.to_numeric(df["dw"], errors="coerce")
    dl = pd.to_numeric(df["dl"], errors="coerce")

    ratio = np.full(len(df), np.nan, dtype=np.float64)
    mask = np.isfinite(dw.values) & np.isfinite(dl.values) & (np.abs(dl.values) > eps)
    ratio[mask] = (dw.values[mask] / dl.values[mask]).astype(np.float64)

    # optional: reject negative ratios (shouldn't happen if both are >0, but be safe)
    ratio[ratio <= 0] = np.nan

    df["dw_over_dl"] = ratio
    df["log_dw_over_dl"] = np.log(ratio)

    return df

def add_ratio_band(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    For ratio r = dw/dl, define norm ratio a = ||s_l|| / ||s_w|| from sl_l2, sw_l2.
    The (disentanglement) band:
        lower = rho * a
        upper = (1/rho) * a
    (requires rho>0).
    Adds:
      - sl_over_sw
      - band_lower, band_upper
      - in_band (0/1)
      - band_slack_log: signed log-distance to band (>=0 inside, <0 outside)
        slack = min(log(r)-log(lower), log(upper)-log(r)).
    """
    # prerequisites
    for c in ["dw_over_dl", "rho", "sw_l2", "sl_l2"]:
        if c not in df.columns:
            df["sl_over_sw"] = np.nan
            df["band_lower"] = np.nan
            df["band_upper"] = np.nan
            df["in_band"] = np.nan
            df["band_slack_log"] = np.nan
            return df

    r = pd.to_numeric(df["dw_over_dl"], errors="coerce").to_numpy(dtype=np.float64)
    rho = pd.to_numeric(df["rho"], errors="coerce").to_numpy(dtype=np.float64)

    sw = pd.to_numeric(df["sw_l2"], errors="coerce").to_numpy(dtype=np.float64)
    sl = pd.to_numeric(df["sl_l2"], errors="coerce").to_numpy(dtype=np.float64)

    a = np.full(len(df), np.nan, dtype=np.float64)
    mask_a = np.isfinite(sw) & np.isfinite(sl) & (sw > eps)
    a[mask_a] = sl[mask_a] / sw[mask_a]

    lower = np.full(len(df), np.nan, dtype=np.float64)
    upper = np.full(len(df), np.nan, dtype=np.float64)

    # only define the band when rho is positive and not too small
    mask_rho = np.isfinite(rho) & (rho > eps) & np.isfinite(a)
    lower[mask_rho] = rho[mask_rho] * a[mask_rho]
    upper[mask_rho] = (1.0 / rho[mask_rho]) * a[mask_rho]

    # ratio validity
    mask_r = np.isfinite(r) & (r > 0)
    in_band = np.full(len(df), np.nan, dtype=np.float64)
    mask_all = mask_rho & mask_r & np.isfinite(lower) & np.isfinite(upper) & (lower > 0) & (upper > 0)

    in_band[mask_all] = ((r[mask_all] >= lower[mask_all]) & (r[mask_all] <= upper[mask_all])).astype(np.float64)

    df["sl_over_sw"] = a
    df["band_lower"] = np.log(lower)
    df["band_upper"] = np.log(upper)
    # df["band_lower"] = pd.to_numeric(df["db/lower"], errors="coerce").to_numpy(dtype=np.float64)
    # df["band_upper"] = pd.to_numeric(df["db/upper"], errors="coerce").to_numpy(dtype=np.float64)
    df["in_band"] = in_band
    df["in_band_ratio"] = np.cumsum(in_band) / in_band.size
    return df  # pd.to_numeric(g_full["db/lower"], errors="coerce")

def add_windowed_stats(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Computes windowed stats from RAW delta_m (NOT smoothed):
      mu_w      = E_w[delta_m]
      var_w     = Var_w(delta_m)
      drift_mag = |mu_w|
      snr_w     = |mu_w| / sqrt(var_w)
    Also computes Var_w(m) for Fig5 scatter (x-axis).
    """
    df = df.sort_values("step").reset_index(drop=True)
    df = ensure_m_and_delta_m(df)
    df = add_dw_dl_ratio(df)
    df = add_ratio_band(df)

    delta = pd.to_numeric(df["delta_m"], errors="coerce")
    m = pd.to_numeric(df["m"], errors="coerce")

    minp = get_roll_min_periods(window)

    mu = delta.rolling(window=window, min_periods=minp).mean()
    var = delta.rolling(window=window, min_periods=minp).var(ddof=0)

    drift_mag = mu.abs()
    denom = np.sqrt(var)
    snr = drift_mag / denom * np.sqrt(window) # 乘以一个根号windowsize
    snr[(denom.isna()) | (denom < 1e-8)] = np.nan  # avoid blow-ups

    df[f"drift_mag_w{window}"] = drift_mag
    df[f"noise_var_w{window}"] = var
    df[f"snr_w{window}"] = snr
    
    # HAC-corrected tSNR: resolvability of mean drift under autocorrelation
    df[f"tsnr_hac_w{window}"] = rolling_hac_tsnr(delta, window)

    df[f"m_var_w{window}"] = m.rolling(window=window, min_periods=minp).var(ddof=0)
    return df

def get_hac_lag(window: int) -> int:
    # 经验上足够稳：H <= w-1；小数据不宜太大
    return max(1, min(10, window // 5, window - 1))

def rolling_hac_tsnr(delta: pd.Series, window: int) -> pd.Series:
    """
    HAC/Newey–West t-stat of mean drift over rolling window:
      tSNR_k = mu_hat / sqrt(Var_hat(mu_hat)),
    where Var_hat(mu_hat) is HAC-corrected with Bartlett kernel.
    """
    x = pd.to_numeric(delta, errors="coerce").to_numpy()
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    H = get_hac_lag(window)

    # start only when we have a full window (consistent with your "full" mode)
    start = window - 1
    for t in range(start, n):
        seg = x[t - window + 1 : t + 1]
        if np.any(~np.isfinite(seg)):
            continue
        mu = float(seg.mean())
        xc = seg - mu

        # gamma_0
        gamma0 = float(np.dot(xc, xc) / window)
        S = gamma0

        # HAC sum
        for h in range(1, H + 1):
            wgt = 1.0 - h / (H + 1.0)  # Bartlett
            cov = float(np.dot(xc[h:], xc[:-h]) / window)
            S += 2.0 * wgt * cov

        var_mu = S / window
        if var_mu > 1e-12:
            out[t] = np.abs(mu) / np.sqrt(var_mu)
        else:
            out[t] = np.nan

    return pd.Series(out, index=delta.index, name=f"tsnr_hac_w{window}")

def smooth_series(
    s: pd.Series,
    force_no_smooth: bool = False,
    no_smooth_cols: Optional[Set[str]] = None,
) -> pd.Series:
    if no_smooth_cols is None:
        no_smooth_cols = set()

    colname = str(s.name)
    if force_no_smooth:
        return s
    if (not SMOOTH_ENABLE) or (colname in no_smooth_cols) or (colname in GLOBAL_NO_SMOOTH_COLS):
        return s

    y = pd.to_numeric(s, errors="coerce")
    if SMOOTH_METHOD.lower() == "ema":
        return y.ewm(span=FIG_WINDOW, adjust=False).mean()
    if SMOOTH_METHOD.lower() == "rolling":
        return y.rolling(window=FIG_WINDOW, min_periods=SMOOTH_MIN_PERIODS).mean()
    raise ValueError(f"Unknown SMOOTH_METHOD: {SMOOTH_METHOD}")

def plot_compare_lines(
    run_dfs: List[Tuple[str, pd.DataFrame]],
    ys: List[str],
    out_path: str,
    title: Optional[str],
    x: str = "step",
    xlabel: str = "Step",
    ylabel: str = "value",
    force_no_smooth: bool = False,
    no_smooth_cols: Optional[Set[str]] = None,
    subsample_every: int = 1,
) -> None:
    if no_smooth_cols is None:
        no_smooth_cols = set()
        
    IS_PAIR_MODE = (
        len(ys) == 2
        and ((ys[0].startswith("z") and ys[1].startswith("z"))
             or (ys[0].startswith("d") and ys[1].startswith("d"))
             or (ys[0].endswith("_l2") and ys[1].endswith("_l2")))
    )
    if len(ys) == 2:
        if "log_r" in ys[0]:
            IS_PAIR_MODE = False
    PRIMARY_VAR = ys[0] if len(ys) == 2 else None
    SECONDARY_VAR = ys[1] if len(ys) == 2 else None

    # Collect existing columns
    ys_exist = []
    for y in ys:
        for _, df in run_dfs:
            if y in df.columns:
                ys_exist.append(y)
                break
    ys_exist = list(dict.fromkeys(ys_exist))
    if not ys_exist:
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    all_series = []

    for run_label, df in run_dfs:
        if df.empty or x not in df.columns:
            continue
        g = df.sort_values(x)
        if subsample_every > 1:
            g = g.iloc[::subsample_every].reset_index(drop=True)

        for y in ys_exist:
            if y not in g.columns:
                continue
            series = g[y]
            if series.dropna().empty:
                continue
            s_sm = smooth_series(series.rename(y), force_no_smooth=force_no_smooth, no_smooth_cols=no_smooth_cols)
            all_series.append((run_label, y, g[x].values, s_sm.values))

    if not all_series:
        plt.close(fig)
        return
    
    # Plot
    plotted_labels = set()
    for run_label, var, xs, ys_vals in all_series:
        meta = METHOD_REGISTRY.get(run_label, {"display": run_label, "color": "C0"})
        display_name = meta["display"]
        base_color = meta["color"]
        
        if "-calib" in run_label:
            linestyle = "--"
        elif "-reg" in run_label:
            linestyle = "-."  # 点划线，可选
        else:
            linestyle = "-"

        if IS_PAIR_MODE:
            if var == PRIMARY_VAR:
                color = base_color
                label = display_name if display_name not in plotted_labels else None
                if label is not None:
                    plotted_labels.add(display_name)
            elif var == SECONDARY_VAR:
                color = _blend_to_white(base_color, 0.5)  # 50% towards white
                label = None  # never show in legend
            else:
                color = base_color
                label = f"{display_name}:{var}"
        else:
            # Normal mode: each variable gets its own entry (but usually only one var)
            label = display_name if display_name not in plotted_labels else None
            if label is not None:
                plotted_labels.add(display_name)
            color = base_color

        ax.plot(xs, ys_vals, color=color, label=label, linestyle=linestyle, linewidth=1.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # if title:
    #     ax.set_title(title)
    ax.grid(True, linewidth=0.3, linestyle=':', alpha=0.6)
    if any(lbl is not None for line in ax.lines for lbl in [line.get_label()] if lbl != "_nolegend_"):
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
def plot_two_series_compare(
    run_dfs: List[Tuple[str, pd.DataFrame]],
    ys: List[str],
    out_path: str,
    xlabel: str = "Step",
    ylabel: str = "Value",
    series_labels: Optional[List[str]] = None,
    force_no_smooth: bool = False,
    no_smooth_cols: Optional[Set[str]] = None,
    subsample_every: int = 1,
    use_method_colors: bool = True,
) -> None:
    if len(ys) != 2:
        raise ValueError("ys must contain exactly two column names")
    
    if no_smooth_cols is None:
        no_smooth_cols = set()
    if series_labels is None:
        series_labels = [ys[0], ys[1]]

    # Base color for unified mode (when use_method_colors=False)
    base_color_primary = "#1f77b4"
    base_color_secondary = "#d62728"  

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plotted_any = False

    for run_label, df in run_dfs:
        if df.empty or "step" not in df.columns:
            continue

        g = df.sort_values("step")
        if subsample_every > 1:
            g = g.iloc[::subsample_every].reset_index(drop=True)

        # Get data
        y1 = pd.to_numeric(g.get(ys[0], pd.Series([np.nan]*len(g))), errors="coerce")
        y2 = pd.to_numeric(g.get(ys[1], pd.Series([np.nan]*len(g))), errors="coerce")
        xs = g["step"].values

        if y1.isna().all() and y2.isna().all():
            continue

        # Choose color
        if use_method_colors:
            meta = METHOD_REGISTRY.get(run_label, {"display": run_label, "color": "#000000"})
            primary_color = meta["color"]
            secondary_color = _blend_to_white(primary_color, 0.5)
            label1 = f"{meta['display']}: {series_labels[0]}"
            label2 = f"{meta['display']}: {series_labels[1]}"
        else:
            primary_color = base_color_primary
            secondary_color = base_color_secondary
            label1 = series_labels[0]
            label2 = series_labels[1]

        # Smooth if needed
        if not force_no_smooth and ys[0] not in no_smooth_cols:
            y1 = smooth_series(y1.rename(ys[0]), force_no_smooth=False, no_smooth_cols=no_smooth_cols)
        if not force_no_smooth and ys[1] not in no_smooth_cols:
            y2 = smooth_series(y2.rename(ys[1]), force_no_smooth=False, no_smooth_cols=no_smooth_cols)

        # Plot
        ax.plot(xs, y1, color=primary_color, linestyle="-", linewidth=1.2, label=label1)
        ax.plot(xs, y2, color=secondary_color, linestyle="-", linewidth=1.2, label=label2)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def plot_zw_zl_per_method(
    base_label: str,
    run_dfs: List[Tuple[str, pd.DataFrame]],
    out_dir: str,
    model_name: str,
    subsample_every: int = 10,
) -> None:
    """
    Plot zw/zl for a base method and its calibrated variant (if exists).
    Saves to: out_dir/zw_zl_sv_step_{base_label}_{model_name}.pdf
    Curves:
        - z_w, z_l         (original)
        - z_w^rc, z_l^rc   (calibrated, if available)
    """
    # Collect relevant runs
    original_df = None
    calib_df = None

    for lbl, df in run_dfs:
        if lbl == base_label:
            original_df = df
        elif lbl == f"{base_label}-calib":
            calib_df = df

    if original_df is None:
        return  # Skip if no original

    # Prepare data
    def prepare_data(df, is_calib=False):
        if df is None:
            return None, None
        df_plot = df.iloc[::subsample_every].reset_index(drop=True)
        xs = df_plot["step"].values
        zw = pd.to_numeric(df_plot.get("zw", pd.Series([np.nan]*len(df_plot))), errors="coerce").values
        zl = pd.to_numeric(df_plot.get("zl", pd.Series([np.nan]*len(df_plot))), errors="coerce").values
        return xs, (zw, zl)

    orig_xs, (orig_zw, orig_zl) = prepare_data(original_df)
    calib_xs, (calib_zw, calib_zl) = prepare_data(calib_df, is_calib=True)

    # Plot
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Original: blue
    ax.plot(orig_xs, orig_zw, color="#1f77b4", linestyle="-", linewidth=1.2, label=r"$z_{w,t}$")
    ax.plot(orig_xs, orig_zl, color="#7fbfff", linestyle="-", linewidth=1.2, label=r"$z_{l,t}$")
    # ax.scatter(orig_xs, orig_zw, color="#1f77b4", s=12, zorder=5, label=r"$z_{w,t}$")
    # ax.scatter(orig_xs, orig_zl, color="#7fbfff", s=12, zorder=5, label=r"$z_{l,t}$")

    # Calibrated: red (if exists)
    if calib_df is not None:
        ax.plot(calib_xs, calib_zw, color="#d62728", linestyle="--", linewidth=1.2, label=r"$z_{w,t}^{\text{rc}}$")
        ax.plot(calib_xs, calib_zl, color="#e99696", linestyle="--", linewidth=1.2, label=r"$z_{l,t}^{\text{rc}}$")
        # ax.scatter(calib_xs, calib_zw, color="#d62728", s=12, zorder=5, label=r"$z_{w,t}^{\text{rc}}$")
        # ax.scatter(calib_xs, calib_zl, color="#e99696", s=12, zorder=5, label=r"$z_{l,t}^{\text{rc}}$")

    ax.set_xlabel("Step")
    ax.set_ylabel(r"Chosen / rejected rewards $z_{w,t}, z_{l,t}$")
    ax.grid(True, linewidth=0.3, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"zw_zl_sv_step_{base_label}_{model_name}.{SAVE_FORMAT}")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
def integrate_margin_ode(df: pd.DataFrame) -> pd.Series:
    """
    Numerically integrate dm/dt = dw*||sw||^2 + dl*||sl||^2 - (dw+dl)*<sw, sl>
    using forward Euler on the full-resolution df.
    Returns a Series of m_ode with same index as input df.
    """
    required = ["dw", "dl", "sw_l2", "sl_l2", "sw_sl_dot", "step", "m", "lr"]
    if not all(c in df.columns for c in required):
        return pd.Series(np.nan, index=df.index)

    df = df.sort_values("step").reset_index(drop=True)
    step = df["step"].values.astype(float)
    dw = pd.to_numeric(df["dw"], errors="coerce").values
    dl = pd.to_numeric(df["dl"], errors="coerce").values
    sw = pd.to_numeric(df["sw_l2"], errors="coerce").values
    sl = pd.to_numeric(df["sl_l2"], errors="coerce").values
    dot = pd.to_numeric(df["sw_sl_dot"], errors="coerce").values
    lr = pd.to_numeric(df["lr"], errors="coerce").values
    m_true = pd.to_numeric(df["m"], errors="coerce").values

    rhs = dw * sw**2 + dl * sl**2 - (dw + dl) * dot
    m_ode = np.full_like(rhs, np.nan)

    valid = np.isfinite(m_true) & np.isfinite(rhs) & np.isfinite(lr) & (lr > 0)
    if not valid.any():
        return pd.Series(np.nan, index=df.index)

    start = np.argmax(valid)
    m_ode[start] = m_true[start]

    for t in range(start + 1, len(rhs)):
        if not (np.isfinite(rhs[t-1]) and np.isfinite(lr[t-1]) and lr[t-1] > 0):
            continue
        dt = float(lr[t-1])  # ← 关键：用学习率作为时间步长
        m_ode[t] = m_ode[t-1] + rhs[t-1] * dt

    return pd.Series(m_ode, index=df.index)

def plot_margin_with_ode(
    run_dfs: List[Tuple[str, pd.DataFrame]],
    out_path: str,
    subsample_every: int = 20,
    show_ode: bool = False,
) -> None:
    """
    Plot margin m_t vs step for multiple methods.
    Optionally overlay numerical integration of the ODE:
        dm/dt = dw*||sw||^2 + dl*||sl||^2 - (dw+dl)*<sw, sl>
    Empirical curve: light solid line.
    ODE curve: dark dashed line.
    """
    # TARGET_METHODS = {"bce", "dpo"}
    # run_dfs = [(lbl, df) for lbl, df in run_dfs if lbl in TARGET_METHODS]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    # base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    # run_to_color = {lbl: base_colors[i % len(base_colors)] for i, (lbl, _) in enumerate(run_dfs)}

    for run_label, df in run_dfs:
        if "m" not in df.columns or "step" not in df.columns:
            continue
        
        meta = METHOD_REGISTRY.get(run_label, {"display": run_label, "color": "#000000"})
        display_name = meta["display"]
        color = meta["color"]
        
        if "-calib" in run_label:
            linestyle = "--"
        elif "-reg" in run_label:
            linestyle = "-."  # 点划线，可选
        else:
            linestyle = "-"

        # Empirical (subsampled)
        df_plot = df.iloc[::subsample_every].reset_index(drop=True)
        xs = df_plot["step"].values
        m_true = pd.to_numeric(df_plot["m"], errors="coerce").values
        ax.plot(xs, m_true, color=color, linestyle=linestyle, label=display_name)   # , alpha=0.6, linewidth=1.0

        # ODE (integrate on full res, then subsample)
        if show_ode:
            m_ode_full = integrate_margin_ode(df)
            if not m_ode_full.isna().all():
                m_ode_plot = m_ode_full.iloc[df_plot.index].values
                ax.plot(xs, m_ode_plot, color=color, linestyle="--", linewidth=1.2)

    ax.set_xlabel("Step")
    ax.set_ylabel(r"Margin $m$")
    ax.grid(True, linewidth=0.3, linestyle=':', alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def adjust_log_scale(ax, num_ticks=4):
    """
    Adjust the y-axis of the log scale plot to show `num_ticks` ticks in log space.
    The ticks will be chosen as evenly spaced in log scale (e.g., -3, -1, 1, 3 for 10^-3 to 10^3).
    """
   
    def custom_formatter(x, pos):
        if x <= 0:
            return ""
        exp = int(np.floor(np.log10(x)))
        mantissa = x / (10 ** exp)
        if mantissa == 1.0:
            return f"1e{exp}" if exp != 0 else "1"
        else:
            return f"{mantissa:.1f}e{exp}"
 
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter()) 

    # Return the adjusted ax
    return ax

# 辅助函数
# def plot_db(ax, df, title, color, is_calib=False, subsample_every=1):
#     """绘制Disentanglement Band"""
#     if df.empty or "step" not in df.columns:
#         return
    
#     g = df.sort_values("step")
#     xs_fill = g["step"].values
    
#     if is_calib:
#         # 校准后的DB
#         alpha = pd.to_numeric(g["db/alpha"], errors="coerce")
#         dw_corrected = pd.to_numeric(g["dw"], errors="coerce") * alpha
#         dl_corrected = pd.to_numeric(g["dl"], errors="coerce") / alpha
#         r = dw_corrected / dl_corrected
#         lo = np.exp(pd.to_numeric(g["db/lower"], errors="coerce"))
#         up = np.exp(pd.to_numeric(g["db/upper"], errors="coerce"))
#     else:
#         # 校准前的DB
#         r = pd.to_numeric(g["dw_over_dl"], errors="coerce")
#         lo = np.exp(pd.to_numeric(g["band_lower"], errors="coerce"))
#         up = np.exp(pd.to_numeric(g["band_upper"], errors="coerce"))
    
#     # 填充带区域
#     ax.fill_between(xs_fill, lo.values, up.values, color=color, alpha=0.15, label="Disentanglement band")
    
#     # 子采样用于绘制曲线
#     if subsample_every > 1:
#         g = g.iloc[::subsample_every].reset_index(drop=True)
    
#     xs = g["step"].values
    
#     if is_calib:
#         # 重新计算子采样后的数据
#         alpha = pd.to_numeric(g["db/alpha"], errors="coerce")
#         dw_corrected = pd.to_numeric(g["dw"], errors="coerce") * alpha
#         dl_corrected = pd.to_numeric(g["dl"], errors="coerce") / alpha
#         r = dw_corrected / dl_corrected
#         lo = np.exp(pd.to_numeric(g["db/lower"], errors="coerce"))
#         up = np.exp(pd.to_numeric(g["db/upper"], errors="coerce"))
#     else:
#         r = pd.to_numeric(g["dw_over_dl"], errors="coerce")
#         lo = np.exp(pd.to_numeric(g["band_lower"], errors="coerce"))
#         up = np.exp(pd.to_numeric(g["band_upper"], errors="coerce"))
    
#     # 绘制激励比率
#     label = r"$\log r_t$ w/ RC" if is_calib else r"$\log r_t$ w/o RC"
#     ax.plot(xs, r.values, color=color, linewidth=1.2, label=label)
    
#     # 绘制带中心
#     band_center = np.sqrt(lo.values * up.values)
#     band_center_smooth = pd.Series(band_center).ewm(span=FIG_WINDOW, adjust=False).mean()
#     center_label = r"Band center $\log r_t^{\star}$"
#     ax.plot(xs, band_center_smooth.values, color='tab:orange', 
#             linestyle='--', linewidth=1.2, label=center_label)
    
#     # 设置坐标轴
#     ax.set_title(title, fontsize=10)
#     ax.set_xlabel("Step")
#     ax.set_ylabel("Incentive log-ratio")
#     ax.set_yscale("log")
#     ax = adjust_log_scale(ax)
#     ax.grid(True, linewidth=0.3, linestyle=":", alpha=0.6)
#     ax.legend(fontsize=8)

def plot_db(ax, df, title, color, is_calib=False, subsample_every=1):
    """绘制Disentanglement Band：平滑在完整数据上，绘图时子采样"""
    if df.empty or "step" not in df.columns:
        return

    # === 1. 使用完整数据计算 ratio 和 band 边界 ===
    g_full = df.sort_values("step").reset_index(drop=True)
    xs_full = g_full["step"].values

    if is_calib:
        alpha_full = pd.to_numeric(g_full["db/alpha"], errors="coerce")
        dw_corr_full = pd.to_numeric(g_full["dw"], errors="coerce") * alpha_full
        dl_corr_full = pd.to_numeric(g_full["dl"], errors="coerce") / alpha_full
        r_full = dw_corr_full / dl_corr_full
        lo_full_raw = pd.to_numeric(g_full["db/lower"], errors="coerce")
        up_full_raw = pd.to_numeric(g_full["db/upper"], errors="coerce")
    else:
        r_full = pd.to_numeric(g_full["dw_over_dl"], errors="coerce")
        lo_full_raw = pd.to_numeric(g_full["band_lower"], errors="coerce")
        up_full_raw = pd.to_numeric(g_full["band_upper"], errors="coerce")

    # 指数还原（log → linear）
    lo_full = np.exp(lo_full_raw)
    up_full = np.exp(up_full_raw)
    band_center_full = np.sqrt(lo_full * up_full)

    # === 2. 在完整数据上做 EWM 平滑 ===
    r_smooth_full = pd.Series(r_full).ewm(span=FIG_WINDOW, adjust=False).mean()
    band_center_smooth_full = pd.Series(band_center_full).ewm(span=FIG_WINDOW, adjust=False).mean()

    # === 3. 子采样用于绘图（从平滑后序列中取点）===
    if subsample_every > 1:
        idx_plot = np.arange(0, len(xs_full), subsample_every)
        xs_plot = xs_full[idx_plot]
        r_smooth_plot = r_smooth_full.iloc[idx_plot].values
        band_center_smooth_plot = band_center_smooth_full.iloc[idx_plot].values
    else:
        xs_plot = xs_full
        r_smooth_plot = r_smooth_full.values
        band_center_smooth_plot = band_center_smooth_full.values

    # === 4. 绘制 band 区域（使用原始完整数据，非平滑）===
    ax.fill_between(xs_full, lo_full, up_full, color=color, alpha=0.35, label="Disentanglement band")

    # === 5. 绘制平滑曲线（已子采样）===
    label = r"$\log r_t$ w/ RC" if is_calib else r"$\log r_t$ w/o RC"
    ax.plot(xs_plot, r_smooth_plot, color=color, linewidth=1.2, label=label)
    ax.plot(xs_plot, band_center_smooth_plot, color='tab:orange',
            linestyle='--', linewidth=1.2, label=r"Band center $\log r_t^{\star}$")

    # === 6. 坐标轴设置 ===
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Incentive log-ratio")
    ax.set_yscale("log")
    ax = adjust_log_scale(ax)
    ax.grid(True, linewidth=0.3, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)
    
def plot_zw_zl_trajectories(ax, base_df, calib_df, subsample_every=1):
    """绘制zw/zl轨迹"""
    # 绘制Base objective
    if base_df is not None and not base_df.empty and "step" in base_df.columns:
        g = base_df.sort_values("step")
        if subsample_every > 1:
            g = g.iloc[::subsample_every].reset_index(drop=True)
        
        xs = g["step"].values
        
        if "zw" in g.columns:
            zw_sm = smooth_series(pd.to_numeric(g["zw"], errors="coerce").rename("zw"), 
                                 force_no_smooth=False)
            ax.plot(xs, zw_sm.values, color="#1f77b4", linestyle="-",
                   linewidth=1.2, label=r"Chosen (base w/o RC)")
        
        if "zl" in g.columns:
            zl_sm = smooth_series(pd.to_numeric(g["zl"], errors="coerce").rename("zl"),
                                 force_no_smooth=False)
            ax.plot(xs, zl_sm.values, color="#7fbfff", linestyle="-",
                   linewidth=1.2, label=r"Rejected (base w/o RC)")
    
    # 绘制Calibrated
    if calib_df is not None and not calib_df.empty and "step" in calib_df.columns:
        g = calib_df.sort_values("step")
        if subsample_every > 1:
            g = g.iloc[::subsample_every].reset_index(drop=True)
        
        xs = g["step"].values
        
        if "zw" in g.columns:
            zw_sm = smooth_series(pd.to_numeric(g["zw"], errors="coerce").rename("zw"),
                                 force_no_smooth=False)
            ax.plot(xs, zw_sm.values, color="#d62728", linestyle="--",
                   linewidth=1.2, label=r"Chosen (base w/ RC)")
        
        if "zl" in g.columns:
            zl_sm = smooth_series(pd.to_numeric(g["zl"], errors="coerce").rename("zl"),
                                 force_no_smooth=False)
            ax.plot(xs, zl_sm.values, color="#e99696", linestyle="--",
                   linewidth=1.2, label=r"Rejected (base w/ RC)")
    
    ax.set_title(r"Rewards Over Steps", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Chosen/rejected rewards")
    ax.grid(True, linewidth=0.3, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="best")
    
def plot_margin(ax, base_df, calib_df, subsample_every=1):
    """绘制margin变化"""
    # 计算并绘制Base margin
    if base_df is not None and not base_df.empty:
        g = base_df.sort_values("step")
        if subsample_every > 1:
            g = g.iloc[::subsample_every].reset_index(drop=True)
        
        xs = g["step"].values
        
        if "zw" in g.columns and "zl" in g.columns:
            zw = pd.to_numeric(g["zw"], errors="coerce")
            zl = pd.to_numeric(g["zl"], errors="coerce")
            margin = zw - zl
            margin_sm = smooth_series(margin.rename("margin"), force_no_smooth=False)
            ax.plot(xs, margin_sm.values, color="#1f77b4", linestyle="-",
                   linewidth=1.5, label="Base w/o RC")
    
    # 计算并绘制Calibrated margin
    if calib_df is not None and not calib_df.empty:
        g = calib_df.sort_values("step")
        if subsample_every > 1:
            g = g.iloc[::subsample_every].reset_index(drop=True)
        
        xs = g["step"].values
        
        if "zw" in g.columns and "zl" in g.columns:
            zw = pd.to_numeric(g["zw"], errors="coerce")
            zl = pd.to_numeric(g["zl"], errors="coerce")
            margin = zw - zl
            margin_sm = smooth_series(margin.rename("margin"), force_no_smooth=False)
            ax.plot(xs, margin_sm.values, color="#d62728", linestyle="--",
                   linewidth=1.5, label="Base w/ RC")
    
    ax.set_title(r"Margin Over Steps", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Margin")
    ax.grid(True, linewidth=0.3, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="best")


def _lookup_run_df(run_dfs: List[Tuple[str, pd.DataFrame]], label: str) -> Optional[pd.DataFrame]:
    for lbl, df in run_dfs:
        if lbl == label:
            return df
    return None


def plot_db_calib_sweep_overlay(
    ax,
    calib_entries: List[Tuple[str, pd.DataFrame]],
    subsample_every: int = 1,
) -> None:
    """多条 RC 配置：只画平滑后的 incentive ratio 曲线（不画 band 填充，避免重叠）。"""
    if not calib_entries:
        ax.text(0.5, 0.5, "No calibration runs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("DB w/ RC (db_ema β sweep)", fontsize=10)
        return
    for idx, (leg, df) in enumerate(calib_entries):
        if df is None or df.empty or "step" not in df.columns:
            continue
        g_full = df.sort_values("step").reset_index(drop=True)
        xs_full = g_full["step"].values
        alpha_full = pd.to_numeric(g_full["db/alpha"], errors="coerce")
        dw_corr_full = pd.to_numeric(g_full["dw"], errors="coerce") * alpha_full
        dl_corr_full = pd.to_numeric(g_full["dl"], errors="coerce") / alpha_full
        r_full = dw_corr_full / dl_corr_full
        r_smooth_full = pd.Series(r_full).ewm(span=FIG_WINDOW, adjust=False).mean()
        if subsample_every > 1:
            idx_plot = np.arange(0, len(xs_full), subsample_every)
            xs_plot = xs_full[idx_plot]
            r_plot = r_smooth_full.iloc[idx_plot].values
        else:
            xs_plot = xs_full
            r_plot = r_smooth_full.values
        c = plt.cm.tab10(idx % 10)
        ax.plot(xs_plot, r_plot, color=c, linewidth=1.2, label=leg)
    ax.set_title("DB w/ RC (db_ema β sweep)", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Incentive log-ratio")
    ax.set_yscale("log")
    ax = adjust_log_scale(ax)
    ax.grid(True, linewidth=0.3, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7, loc="best")


def plot_zw_zl_trajectories_multicalib(
    ax,
    base_df: Optional[pd.DataFrame],
    calib_entries: List[Tuple[str, pd.DataFrame]],
    subsample_every: int = 1,
) -> None:
    if base_df is not None and not base_df.empty and "step" in base_df.columns:
        g = base_df.sort_values("step")
        if subsample_every > 1:
            g = g.iloc[::subsample_every].reset_index(drop=True)
        xs = g["step"].values
        if "zw" in g.columns:
            zw_sm = smooth_series(pd.to_numeric(g["zw"], errors="coerce").rename("zw"), force_no_smooth=False)
            ax.plot(xs, zw_sm.values, color="#1f77b4", linestyle="-", linewidth=1.2, label=r"Chosen (w/o RC)")
        if "zl" in g.columns:
            zl_sm = smooth_series(pd.to_numeric(g["zl"], errors="coerce").rename("zl"), force_no_smooth=False)
            ax.plot(xs, zl_sm.values, color="#7fbfff", linestyle="-", linewidth=1.2, label=r"Rejected (w/o RC)")
    for idx, (leg, df) in enumerate(calib_entries):
        if df is None or df.empty or "step" not in df.columns:
            continue
        g = df.sort_values("step")
        if subsample_every > 1:
            g = g.iloc[::subsample_every].reset_index(drop=True)
        xs = g["step"].values
        c = plt.cm.tab10(idx % 10)
        if "zw" in g.columns:
            zw_sm = smooth_series(pd.to_numeric(g["zw"], errors="coerce").rename("zw"), force_no_smooth=False)
            ax.plot(xs, zw_sm.values, color=c, linestyle="--", linewidth=1.1, label=f"Ch. {leg}")
        if "zl" in g.columns:
            zl_sm = smooth_series(pd.to_numeric(g["zl"], errors="coerce").rename("zl"), force_no_smooth=False)
            ax.plot(xs, zl_sm.values, color=c, linestyle=":", linewidth=1.0, label=f"Rej. {leg}")
    ax.set_title(r"Rewards Over Steps", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Chosen/rejected rewards")
    ax.grid(True, linewidth=0.3, linestyle=":", alpha=0.6)
    ax.legend(fontsize=6, loc="best")


def plot_margin_multicalib(
    ax,
    base_df: Optional[pd.DataFrame],
    calib_entries: List[Tuple[str, pd.DataFrame]],
    subsample_every: int = 1,
) -> None:
    if base_df is not None and not base_df.empty:
        g = base_df.sort_values("step")
        if subsample_every > 1:
            g = g.iloc[::subsample_every].reset_index(drop=True)
        xs = g["step"].values
        if "zw" in g.columns and "zl" in g.columns:
            zw = pd.to_numeric(g["zw"], errors="coerce")
            zl = pd.to_numeric(g["zl"], errors="coerce")
            margin = zw - zl
            margin_sm = smooth_series(margin.rename("margin"), force_no_smooth=False)
            ax.plot(xs, margin_sm.values, color="#1f77b4", linestyle="-", linewidth=1.5, label=r"Margin w/o RC")
    for idx, (leg, df) in enumerate(calib_entries):
        if df is None or df.empty:
            continue
        g = df.sort_values("step")
        if subsample_every > 1:
            g = g.iloc[::subsample_every].reset_index(drop=True)
        xs = g["step"].values
        if "zw" not in g.columns or "zl" not in g.columns:
            continue
        zw = pd.to_numeric(g["zw"], errors="coerce")
        zl = pd.to_numeric(g["zl"], errors="coerce")
        margin = zw - zl
        margin_sm = smooth_series(margin.rename("margin"), force_no_smooth=False)
        c = plt.cm.tab10(idx % 10)
        ax.plot(xs, margin_sm.values, color=c, linestyle="--", linewidth=1.2, label=f"M {leg}")
    ax.set_title(r"Margin Over Steps", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Margin")
    ax.grid(True, linewidth=0.3, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7, loc="best")


def plot_objective_comprehensive_dpo_dbema_sweep(
    run_dfs: List[Tuple[str, pd.DataFrame]],
    out_dir: str,
    model_name: str,
    subsample_every: int = 5,
) -> None:
    """
    与 plot_objective_comprehensive 相同 1x4 版面：DPO base + 多种 db_ema_beta 的 RC 曲线对比。
    """
    sweep_meta = [
        ("dpo-calib-ema0p5", r"$\beta=0.5$"),
        ("dpo-calib-ema0p9", r"$\beta=0.9$"),
        ("dpo-calib-ema0p95", r"$\beta=0.95$"),
        ("dpo-calib-ema0p999", r"$\beta=0.999$"),
    ]
    base_data = _lookup_run_df(run_dfs, "dpo")
    calib_entries: List[Tuple[str, pd.DataFrame]] = []
    for lbl, disp in sweep_meta:
        dfc = _lookup_run_df(run_dfs, lbl)
        if dfc is not None and not dfc.empty:
            calib_entries.append((disp, dfc))
        else:
            print(f"[WARN] db_ema sweep: missing run label={lbl!r}")

    if base_data is None or base_data.empty:
        print("[WARN] db_ema sweep: missing base `dpo`; skip comprehensive_dpo_dbema_sweep")
        return

    ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 4, figsize=(ICML_COL_W_IN * 4, ICML_LINE_H_IN * 1.2), constrained_layout=False)

    plot_db(axes[0], base_data, "DB w/o Reward Calibration", "tab:blue", is_calib=False, subsample_every=subsample_every)
    plot_db_calib_sweep_overlay(axes[1], calib_entries, subsample_every=subsample_every)
    plot_zw_zl_trajectories_multicalib(axes[2], base_data, calib_entries, subsample_every=subsample_every)
    plot_margin_multicalib(axes[3], base_data, calib_entries, subsample_every=subsample_every)

    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        bottom=0.15,
        top=0.88,
        wspace=0.2,
        hspace=0.0,
    )
    out_path = os.path.join(out_dir, f"comprehensive_dpo_dbema_sweep_{model_name}.{SAVE_FORMAT}")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"[OK] Saved db_ema sweep comprehensive: {out_path}")


def plot_objective_comprehensive(
    base_method: str,
    run_dfs: List[Tuple[str, pd.DataFrame]],
    out_dir: str,
    model_name: str,
    subsample_every: int = 10,
) -> None:
    """
    为每个objective绘制综合图：1x4布局
    使用三个辅助函数实现
    """
    # 获取数据
    base_data = None
    calib_data = None
    
    for lbl, df in run_dfs:
        if lbl == base_method:
            base_data = df
        elif lbl == f"{base_method}-calib":
            calib_data = df
    
    if base_data is None or base_data.empty:
        print(f"[WARN] No base data for {base_method}")
        return
    
    # 创建1x4的图形
    fig, axes = plt.subplots(1, 4, figsize=(ICML_COL_W_IN*4, ICML_LINE_H_IN*1.2), constrained_layout=False)
    
    # ===== 子图1: DB Before (校准前) =====
    if base_data is not None:
        plot_db(axes[0], base_data, "DB w/o Reward Calibration", 'tab:blue', 
                is_calib=False, subsample_every=subsample_every)
    else:
        axes[0].text(0.5, 0.5, "No base data", 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("DB w/o Reward Calibration", fontsize=10)
    
    # ===== 子图2: DB After (校准后) =====
    if calib_data is not None:
        plot_db(axes[1], calib_data, "DB w/ Reward Calibration", 'tab:red', 
                is_calib=True, subsample_every=subsample_every)
    else:
        axes[1].text(0.5, 0.5, "No calibration data", 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("DB w/ Reward Calibration", fontsize=10)
    
    # ===== 子图3: zw/zl轨迹 =====
    plot_zw_zl_trajectories(axes[2], base_data, calib_data, subsample_every=subsample_every)
    
    # ===== 子图4: margin变化 =====
    plot_margin(axes[3], base_data, calib_data, subsample_every=subsample_every)
    
    # 调整布局
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(
        left=0.04,    # 若 y 轴标签不长，可更小
        right=0.99,
        bottom=0.15,  # 保留足够空间给 x 轴标签
        top=0.88,     # 标题高度
        wspace=0.2,  # ← 主要调节项：减小此值让子图更紧凑
        hspace=0.0
    )
        
    # 保存
    out_path = os.path.join(out_dir, f"comprehensive_{base_method}_{model_name}.{SAVE_FORMAT}")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.)
    plt.close(fig)
    print(f"[OK] Saved comprehensive plot for {base_method}: {out_path}")
    
def plot_all_objectives_comprehensive(
    run_dfs: List[Tuple[str, pd.DataFrame]],
    out_dir: str,
    model_name: str,
    methods_to_plot: Optional[List[str]] = None,
    subsample_every: int = 10,
) -> None:

    if methods_to_plot is None:
        methods_to_plot = sorted({lbl for lbl, _ in run_dfs if "-calib" not in lbl and "-reg" not in lbl})
    
    ensure_dir(out_dir)
    
    for base_method in methods_to_plot:
        plot_objective_comprehensive(
            base_method=base_method,
            run_dfs=run_dfs,
            out_dir=out_dir,
            model_name=model_name,
            subsample_every=subsample_every,
        )


def plot_ratio_with_band_focused(
    run_label: str,
    df: pd.DataFrame,
    out_path: str,
    x: str = "step",
    use_log_y: bool = True,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    draw_band_edges: bool = False,
    draw_running_ratio: bool = True,
    right_axis_col: Optional[str] = None,
    subsample_every: int = 1,
) -> None:
    if df.empty or (x not in df.columns):
        return

    is_calib = "calib" in run_label

    if DB_PLOT_MODE == "algorithm_view" and is_calib:
        has_ema_vars = all(col in df.columns for col in ["db/log_r_ema", "db/log_r_star_ema", "db/lower", "db/upper"])
        if not has_ema_vars:
            print(f"[WARN] {run_label}: 算法视角模式但缺少EMA变量，回退到真实DB计算")
            required_cols = ["dw_over_dl", "band_lower", "band_upper"]
    else:
        required_cols = ["dw_over_dl", "band_lower", "band_upper"]

    if not all(col in df.columns for col in required_cols):
        return

    # === 1. 使用完整数据（不子采样）进行平滑 ===
    g_full = df.sort_values(x).reset_index(drop=True)
    xs_full = g_full[x].values

    # 计算完整数据上的 ratio 和 band center
    if is_calib:
        alpha_full = pd.to_numeric(g_full["db/alpha"], errors="coerce")
        dw_corr_full = pd.to_numeric(g_full["dw"], errors="coerce") * alpha_full
        dl_corr_full = pd.to_numeric(g_full["dl"], errors="coerce") / alpha_full
        r_full = dw_corr_full / dl_corr_full

        lo_full_raw = pd.to_numeric(g_full["db/lower"], errors="coerce")
        up_full_raw = pd.to_numeric(g_full["db/upper"], errors="coerce")
    else:
        r_full = pd.to_numeric(g_full["dw_over_dl"], errors="coerce")
        lo_full_raw = pd.to_numeric(g_full["band_lower"], errors="coerce")
        up_full_raw = pd.to_numeric(g_full["band_upper"], errors="coerce")

    # 指数还原（注意：band_lower/upper 是 log 形式）
    lo_full = np.exp(lo_full_raw)
    up_full = np.exp(up_full_raw)

    # 计算 band center（几何平均）
    band_center_full = np.sqrt(lo_full * up_full)

    # === 2. 在完整数据上做 EWM 平滑 ===
    r_smooth_full = pd.Series(r_full).ewm(span=FIG_WINDOW, adjust=False).mean()
    band_center_smooth_full = pd.Series(band_center_full).ewm(span=FIG_WINDOW, adjust=False).mean()

    # === 3. 子采样用于绘图（从平滑后的完整序列中取点）===
    if subsample_every > 1:
        idx_plot = np.arange(0, len(xs_full), subsample_every)
        xs_plot = xs_full[idx_plot]
        r_smooth_plot = r_smooth_full.iloc[idx_plot].values
        band_center_smooth_plot = band_center_smooth_full.iloc[idx_plot].values
    else:
        xs_plot = xs_full
        r_smooth_plot = r_smooth_full.values
        band_center_smooth_plot = band_center_smooth_full.values

    # === 4. 绘图 ===
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # 填充 band 区域（使用原始完整数据，非平滑）
    ax.fill_between(xs_full, lo_full, up_full, color='tab:blue', alpha=0.4,
                    label="Disentanglement band", zorder=1)

    if draw_band_edges:
        ax.plot(xs_full, lo_full, linestyle="--", linewidth=0.9, zorder=2, label="DB lower")
        ax.plot(xs_full, up_full, linestyle="--", linewidth=0.9, zorder=2, label="DB upper")

    # 绘制平滑后的曲线（已子采样）
    ax.plot(xs_plot, r_smooth_plot, color='tab:blue', linewidth=1.2,
            label=r"Realized log-ratio $\log r_t$", zorder=3)
    ax.plot(xs_plot, band_center_smooth_plot, color='tab:orange', linewidth=1.2,
            linestyle="--", label=r"Band center $\log r_t^{\star}$", zorder=4)

    # 坐标轴设置
    ax.set_xlabel("Step")
    ax.set_ylabel(r"Incentive log-ratio")
    ax.set_yscale("log")
    ax = adjust_log_scale(ax, num_ticks=4)
    ax.grid(True, linewidth=0.3, linestyle=':', alpha=0.6)

    # 右侧 y 轴（如果需要）
    if right_axis_col is not None and right_axis_col in g_full.columns:
        y2_full = pd.to_numeric(g_full[right_axis_col], errors="coerce")
        y2_smooth_full = y2_full.ewm(span=FIG_WINDOW, adjust=False).mean()
        y2_plot = y2_smooth_full.iloc[idx_plot].values if subsample_every > 1 else y2_smooth_full.values

        ax2 = ax.twinx()
        ax2.plot(xs_plot, y2_plot, linewidth=1.3, linestyle="-.", label="mode(iii) rate (rolling)")
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("mode(iii) rate")
        ax2.grid(False)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best",
                  handlelength=1.2, handletextpad=0.3, frameon=True)
    else:
        ax.legend(loc="best", handlelength=1.2, handletextpad=0.3, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
def plot_rho_distribution(
    run_dfs: List[Tuple[str, pd.DataFrame]],
    rho_col: str,
    out_path: str,
    title: str,
    bins: int = 60,
) -> None:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    any_ok = False
    for run_label, df in run_dfs:
        if rho_col not in df.columns:
            continue
        vals = pd.to_numeric(df[rho_col], errors="coerce").dropna()
        if vals.empty:
            continue
        
        meta = METHOD_REGISTRY.get(run_label, {"display": run_label, "color": "C0"})
        label = meta["display"]
        color = meta["color"]

        any_ok = True
        ax.hist(vals.values, bins=bins, density=True, alpha=0.3, label=label, color=color)

    if not any_ok:
        plt.close(fig)
        return

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Density")
    ax.grid(True, linewidth=0.3, linestyle=':', alpha=0.6)
    ax.legend(loc="best", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def plot_fig5_scatter_two_panels(
    run_dfs: List[Tuple[str, pd.DataFrame]],
    x_left_col: str,
    x_right_col: str,
    y_col: str,
    out_path: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH*2, FIG_HEIGHT))
    axL, axR = axes[0], axes[1]
    any_ok = False

    for run_label, df in run_dfs:
  
        meta = METHOD_REGISTRY.get(run_label, {"display": run_label, "color": "C0"})
        label = meta["display"]
        color = meta["color"]

        if (x_left_col in df.columns) and (y_col in df.columns):
            xL = pd.to_numeric(df[x_left_col], errors="coerce")
            y = pd.to_numeric(df[y_col], errors="coerce")
            mask = (~xL.isna()) & (~y.isna())
            if mask.any():
                any_ok = True
                axL.scatter(xL[mask].values, y[mask].values, s=10, alpha=0.35, label=label, color=color, rasterized=True)

        if (x_right_col in df.columns) and (y_col in df.columns):
            xR = pd.to_numeric(df[x_right_col], errors="coerce")
            y = pd.to_numeric(df[y_col], errors="coerce")
            mask = (~xR.isna()) & (~y.isna())
            if mask.any():
                any_ok = True
                axR.scatter(xR[mask].values, y[mask].values, s=10, alpha=0.35, label=label, color=color, rasterized=True)

    if not any_ok:
        plt.close(fig)
        return

    axL.set_xlabel(x_left_col)
    axL.set_ylabel(y_col)
    # axL.set_title("Left: x=Var_w(m_t), y=SNR_t")
    axL.grid(True, linewidth=0.3)
    axL.legend(loc="best", fontsize=9)

    axR.set_xlabel(x_right_col)
    axR.set_ylabel(y_col)
    # axR.set_title("Right: x=|Δm_t| (raw), y=SNR_t")
    axR.grid(True, linewidth=0.3)
    axR.legend(loc="best", fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def flatten_group_cols(groups: List[Dict[str, Any]]) -> Set[str]:
    s = set()
    for g in groups:
        for c in g.get("cols", []):
            s.add(c)
    return s

def is_constant_across_all_runs(run_dfs: List[Tuple[str, pd.DataFrame]], col: str) -> bool:
    vals = []
    for _, df in run_dfs:
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce").dropna().values
        if v.size > 0:
            vals.append(v)
    if not vals:
        return True
    cat = np.concatenate(vals)
    if cat.size == 0:
        return True
    uniq = np.unique(cat)
    return uniq.size <= 1

def compute_unified_y_range(run_dfs, target_labels=("bce", "dpo"), col="dw_over_dl", padding_factor=1.5):
    values = []
    for label, df in run_dfs:
        if label in target_labels and col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v) > 0:
                values.extend(v.values)
    if not values:
        return None, None
    arr = np.array(values)
    r_min, r_max = arr.min(), arr.max()
    r_range = r_max - r_min
    y_pad = r_range * padding_factor
    y_min = max(r_min - y_pad, r_min * 0.75)
    y_max = min(r_max + y_pad, r_max * 1.25)
    return y_min, y_max

def scientific_formatter(x, pos):
    if x <= 0:
        return ""
    if x >= 1:
        return f"{int(round(x))}"  # 四舍五入到整数（避免 2.999 显示为 2）
    else:
        # 动态计算小数位数：确保至少显示 1 位非零数字
        if x < 1e-5:  # 极小值用科学计数法（避免 0.00000...）
            return f"{x:.1e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}"
        else:
            # 保留足够小数位（例如 0.0005 → 0.0005, 0.01 → 0.01）
            s = f"{x:.6f}".rstrip('0').rstrip('.')
            return s if s else "0"

def plot_abstract_figure(
    run_dfs: List[Tuple[str, pd.DataFrame]],
    methods_to_plot: List[str],
    zw_colors: Dict[str, str],  # 方法名 -> zw颜色
    zl_color: str = "#db7977",  # 所有zl使用相同颜色（灰色）
    out_path: str = None,
    subsample_every: int = 10,
    linewidth: float = 2.0,
    figsize: Tuple[float, float] = (4, 3),
) -> None:
    """
    绘制摘要图：简洁的zw/zl趋势图，无坐标轴和边框
    
    Args:
        run_dfs: 运行数据列表 [(label, df), ...]
        methods_to_plot: 要绘制的方法列表，如 ["dpo", "bce", "dpo-calib", "bce-calib"]
        zw_colors: 各方法zw曲线的颜色映射
        zl_color: zl曲线的颜色（所有方法相同）
        out_path: 输出路径
        subsample_every: 子采样间隔
        linewidth: 线宽
        figsize: 图形尺寸
    """
    # 创建图形，去除所有边框
    fig, ax = plt.subplots(figsize=figsize)
    
    # 去除所有坐标轴、边框、网格
    ax.set_axis_off()
    ax.set_frame_on(False)
    
    # 收集要绘制的方法数据
    for method in methods_to_plot:
        # 查找对应的数据
        df = None
        for label, data in run_dfs:
            if label == method:
                df = data
                break
        
        if df is None or df.empty:
            print(f"[WARN] 方法 {method} 无数据，跳过")
            continue
        
        # 子采样
        df_plot = df.iloc[::subsample_every].reset_index(drop=True)
        
        # 确保有step列
        if "step" not in df_plot.columns:
            continue
        
        xs = df_plot["step"].values
        
        # 绘制zw曲线
        if "zw" in df_plot.columns:
            zw = pd.to_numeric(df_plot["zw"], errors="coerce")
            if not zw.isna().all():
                zw_sm = zw.ewm(span=5, adjust=False).mean()
                zw_color = zw_colors.get(method, "#000000")
                linestyle = "-" if "-calib" in method else "-"
                ax.plot(xs, zw_sm.values, color=zw_color, linestyle=linestyle, linewidth=linewidth)
        
        # 绘制zl曲线
        if "zl" in df_plot.columns:
            zl = pd.to_numeric(df_plot["zl"], errors="coerce")
            if not zl.isna().all():
                zl_sm = zl.ewm(span=5, adjust=False).mean()
                linestyle = "-" if "-calib" in method else "-"
                ax.plot(xs, zl_sm.values,color=zl_color,linestyle=linestyle,linewidth=linewidth * 0.8,alpha=0.7)
    
    ax.autoscale_view()
    if out_path:
        plt.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0,transparent=True)  
    
    plt.close(fig)

def plot_abstract_db_figure(
    run_label: str,
    df: pd.DataFrame,
    out_path: str,
    subsample_every: int = 10,
    linewidth: float = 2.0,
    figsize: Tuple[float, float] = (4, 3),
    color: str = "tab:blue",
    center_color: str = "tab:orange",
) -> None:
    if df.empty or "step" not in df.columns:
        return

    # === 1. 使用完整数据计算 ratio 和 band ===
    g_full = df.sort_values("step").reset_index(drop=True)
    xs_full = g_full["step"].values

    is_calib = "calib" in run_label
    if is_calib:
        alpha_full = pd.to_numeric(g_full["db/alpha"], errors="coerce")
        dw_corr_full = pd.to_numeric(g_full["dw"], errors="coerce") * alpha_full
        dl_corr_full = pd.to_numeric(g_full["dl"], errors="coerce") / alpha_full
        r_full = dw_corr_full / dl_corr_full

        lo_full_raw = pd.to_numeric(g_full["db/lower"], errors="coerce")
        up_full_raw = pd.to_numeric(g_full["db/upper"], errors="coerce")
    else:
        r_full = pd.to_numeric(g_full["dw_over_dl"], errors="coerce")
        lo_full_raw = pd.to_numeric(g_full["band_lower"], errors="coerce")
        up_full_raw = pd.to_numeric(g_full["band_upper"], errors="coerce")

    # 指数还原（log → linear）
    lo_full = np.exp(lo_full_raw)
    up_full = np.exp(up_full_raw)
    band_center_full = np.sqrt(lo_full * up_full)

    # === 2. 在完整数据上做 EWM 平滑 ===
    r_smooth_full = pd.Series(r_full).ewm(span=FIG_WINDOW, adjust=False).mean()
    band_center_smooth_full = pd.Series(band_center_full).ewm(span=FIG_WINDOW, adjust=False).mean()

    # === 3. 子采样用于绘图 ===
    if subsample_every > 1:
        idx_plot = np.arange(0, len(xs_full), subsample_every)
        xs_plot = xs_full[idx_plot]
        r_smooth_plot = r_smooth_full.iloc[idx_plot].values
        band_center_smooth_plot = band_center_smooth_full.iloc[idx_plot].values
    else:
        xs_plot = xs_full
        r_smooth_plot = r_smooth_full.values
        band_center_smooth_plot = band_center_smooth_full.values

    # === 4. 创建无边框图形 ===
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_frame_on(False)
    ax.set_yscale("log")

    # === 5. 填充 band 区域（使用完整原始数据，非平滑）===
    # 使用更深的颜色 + 更高 alpha 提升打印可见性
    fill_color = mcolors.to_hex(mcolors.to_rgb(color) if color != "tab:blue" else "#1f77b4")
    ax.fill_between(xs_full, lo_full, up_full, color=fill_color, alpha=0.35, zorder=1)

    # === 6. 绘制平滑曲线（已子采样）===
    ax.plot(xs_plot, r_smooth_plot, color=color, linewidth=linewidth, linestyle="-", zorder=3)
    ax.plot(xs_plot, band_center_smooth_plot, color=center_color, linewidth=linewidth * 0.8, linestyle="--", zorder=4)

    ax.autoscale_view()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    
def generate_abstract_figures(
    run_dfs: List[Tuple[str, pd.DataFrame]],
    model_name: str,
    out_dir: str,
) -> None:
    """
    为指定模型生成所有摘要图
    
    Args:
        run_dfs: 运行数据
        model_name: 模型名称
        out_dir: 输出目录
    """
    # 创建摘要图目录
    abstract_dir = os.path.join(out_dir, "abstract_figures")
    ensure_dir(abstract_dir)
    
    # 定义要绘制的方法
    # methods = ["dpo", "bce", "dpo-calib", "bce-calib"]
    methods = ["dpo", "cpo", "dpo-calib", "cpo-calib"]  
    
    # 定义zw颜色（使用现有BASE颜色）
    zw_colors = {
        "dpo": "#61b283", # BASE["dpo"],        # 珊瑚红
        "cpo": "#61b283", # BASE["bce"],        # 深蓝
        "dpo-calib": "#61b283", #lighten(BASE["dpo"]),  # 浅珊瑚红
        "cpo-calib": "#61b283", #lighten(BASE["bce"]),  # 浅深蓝
    }
    # zw_colors = {
    #     "dpo": "#61b283", # BASE["dpo"],        # 珊瑚红
    #     "bce": "#61b283", # BASE["bce"],        # 深蓝
    #     "dpo-calib": "#61b283", #lighten(BASE["dpo"]),  # 浅珊瑚红
    #     "bce-calib": "#61b283", #lighten(BASE["bce"]),  # 浅深蓝
    # }
    
    # 生成单个摘要图（所有方法在一张图）
    out_path = os.path.join(abstract_dir, f"abstract_all_methods_{model_name}.{SAVE_FORMAT}")
    plot_abstract_figure(
        run_dfs=run_dfs,
        methods_to_plot=methods,
        zw_colors=zw_colors,
        zl_color="#db7977",  # 深灰色
        out_path=out_path,
        subsample_every=20,  # 较大的子采样，使曲线更平滑
        linewidth=2.5,
        figsize=(6, 4),
    )
    
    # 也可以为每个方法单独生成摘要图
    for method in ["dpo", "cpo"]:
        # 原始版本
        out_path_single = os.path.join(abstract_dir, f"abstract_{method}_{model_name}.{SAVE_FORMAT}")
        plot_abstract_figure(
            run_dfs=run_dfs,
            methods_to_plot=[method],
            zw_colors={method: zw_colors[method]},
            zl_color="#db7977",
            out_path=out_path_single,
            subsample_every=20,
            linewidth=3.0,
            figsize=(4, 3),
        )
        
        # 如果有calib版本，也单独生成
        calib_method = f"{method}-calib"
        if any(lbl == calib_method for lbl, _ in run_dfs):
            out_path_calib = os.path.join(abstract_dir, f"abstract_{calib_method}_{model_name}.{SAVE_FORMAT}")
            plot_abstract_figure(
                run_dfs=run_dfs,
                methods_to_plot=[calib_method],
                zw_colors={calib_method: zw_colors[calib_method]},
                zl_color="#db7977",
                out_path=out_path_calib,
                subsample_every=20,
                linewidth=3.0,
                figsize=(4, 3),
            )
        
    for method in ["dpo", "cpo"]:
        # 原始版本
        df_orig = None
        for lbl, data in run_dfs:
            if lbl == method:
                df_orig = data
                break
        if df_orig is not None and not df_orig.empty:
            out_db = os.path.join(abstract_dir, f"abstract_db_{method}_{model_name}.{SAVE_FORMAT}")
            plot_abstract_db_figure(
                run_label=method,
                df=df_orig,
                out_path=out_db,
                subsample_every=20,
                linewidth=2.5,
                figsize=(4, 3),
                color="#1f77b4",
            )

        # 校准版本
        calib_method = f"{method}-calib"
        df_calib = None
        for lbl, data in run_dfs:
            if lbl == calib_method:
                df_calib = data
                break
        if df_calib is not None and not df_calib.empty:
            out_db_calib = os.path.join(abstract_dir, f"abstract_db_{calib_method}_{model_name}.{SAVE_FORMAT}")
            plot_abstract_db_figure(
                run_label=calib_method,
                df=df_calib,
                out_path=out_db_calib,
                subsample_every=20,
                linewidth=2.5,
                figsize=(4, 3),
                color="#1f77b4",
            )
    
    print(f"[DONE] 摘要图已保存至: {abstract_dir}")


# =========================
# Main
# =========================
def main() -> None:
    ensure_dir(OUT_DIR)

    # 1) load all runs first (no truncation yet)
    loaded: List[Tuple[str, pd.DataFrame]] = []
    max_steps: List[int] = []

    for item in RUNS:
        path = item["path"]
        label = item.get("label") or os.path.basename(os.path.dirname(path)) or os.path.basename(path)

        df = load_jsonl_as_df(path)
        if df.empty:
            print(f"[WARN] Empty/invalid jsonl: {path}")
            continue
        if "step" not in df.columns:
            print(f"[WARN] Missing step column: {path}")
            continue

        ms = int(pd.to_numeric(df["step"], errors="coerce").dropna().max())
        max_steps.append(ms)
        loaded.append((label, df))

    if not loaded:
        raise RuntimeError("No valid runs loaded. Please check RUNS paths.")

    # 2) MAX_STEP = (global max step) / 2
    global_max = max(max_steps)
    MAX_STEP = global_max
    print(f"[INFO] global_max_step={global_max} => MAX_STEP ={MAX_STEP}")
    
    # 子目录：存放每个 run 的处理后数据
    PROCESSED_RUNS_DIR = os.path.join(OUT_DIR, "processed_runs")
    ensure_dir(PROCESSED_RUNS_DIR)

    # 3) truncate + compute required derived columns + save scalars
    run_dfs: List[Tuple[str, pd.DataFrame]] = []
    for label, df in loaded:
        df = df[df["step"] <= MAX_STEP].copy().sort_values("step").reset_index(drop=True)
        if df.empty:
            print(f"[WARN] {label}: no rows <= MAX_STEP={MAX_STEP}")
            continue

        df = ensure_m_and_delta_m(df)
        df = add_windowed_stats(df, FIG_WINDOW)

        # Fig5 needs |Δm_t| raw
        df["abs_delta_m_raw"] = pd.to_numeric(df["delta_m"], errors="coerce").abs()

        run_out = os.path.join(PROCESSED_RUNS_DIR, f"run_{label}")
        ensure_dir(run_out)
        df.to_csv(os.path.join(run_out, f"scalars_w{FIG_WINDOW}.csv"), index=False)

        run_dfs.append((label, df))

    if not run_dfs:
        raise RuntimeError("All runs became empty after truncation. Check data.")

    # combined scalars
    combined = []
    for label, df in run_dfs:
        tmp = df.copy()
        tmp["run_label"] = label
        combined.append(tmp)
    combined_df = pd.concat(combined, ignore_index=True)
    combined_df.to_csv(os.path.join(PROCESSED_RUNS_DIR, f"combined_scalars_w{FIG_WINDOW}.csv"), index=False)

    # 4) PDF-aligned figures
    PLOT_CONFIGS = [
        {
            "name": "Fig1",
            "ys": ["m"],
            "ylabel": r"Margin $m$",
            "force_no_smooth": True,
            "suffix": "Margin_m_t_vs_step",    # Fig1: m_t vs step (NO smoothing)
            "is_margin": True,
        },
        {
            "name": "Fig2",
            "ys": [f"drift_mag"],
            "ylabel": r"Drift E$[\Delta m_t]$",
            "force_no_smooth": True,
            "suffix": f"DriftMagnitude_absE_delta_m"  # Fig2: |E_w[Δm_t]|
        },
        {
            "name": "Fig3",
            "ys": [f"noise_var_w{FIG_WINDOW}"],
            "ylabel": r"Variance $\text{Var}(\Delta m_t)$",
            "force_no_smooth": True,
            "suffix": f"NoiseScale_Var_delta_m"   # Fig3: Var_w(Δm_t)
        },
        {
            "name": "Fig4_SNR",
            "ys": [f"snr_w{FIG_WINDOW}"],
            "ylabel": r"SNR",
            "force_no_smooth": True,
            "suffix": f"SNR_absE_over_sqrtVar"    # Fig4: SNR
        },
        {
            "name": "Fig4_tSNR",
            "ys": [f"tsnr_hac_w{FIG_WINDOW}"],
            "ylabel": r"tSNR",
            "force_no_smooth": True,
            "suffix": f"tSNR_HAC"
        },
        {
            "name": "Fig6",
            "ys": ["log_dw_over_dl"],
            "ylabel": r"Log incentive ratio $\log (d_w/d_l)$",
            "force_no_smooth": True,
            "suffix": "log_dw_over_dl_vs_step"   
        },
        {
            "name": "Fig7",
            "ys": ["db/log_r_ema", "db/log_r_star_ema"],
            "ylabel": r"Log incentive ratio $\log r_t$",
            "force_no_smooth": True,
            "suffix": "log_ratio_vs_step"
        },
        {
        "name": "Fig7",
        "ys": ["db/log_r_ema", "db/log_r_star_ema"],
        "ylabel": r"Log incentive ratio $\log r_t$",
        "force_no_smooth": True,
        "suffix": "log_ratio_vs_step",
        "is_pair_plot": True, 
        "series_labels": [r"$\log r_t$", r"$\log r_t^*$"],
    },
        {
            "name": "Fig8",
            "ys": ["db/alpha"],
            "ylabel": r"Alpha",
            "force_no_smooth": True,
            "suffix": "alpha_vs_step"
        }
    ]
    
    # for cfg in PLOT_CONFIGS:
    #     out_path = os.path.join(OUT_DIR, f"{cfg['name']}_{cfg['suffix']}_{llm_model}.{SAVE_FORMAT}")
    #     subsample = SUBSAMPLE_EVERY.get(cfg["name"], 1)
        
    #     if cfg.get("is_margin", False):
    #         # Special handling for Fig1: margin + ODE + subsample
    #         plot_margin_with_ode(
    #             run_dfs=run_dfs,
    #             out_path=out_path,
    #             subsample_every=subsample,
    #             show_ode=False,
    #         )
    #     elif cfg.get("is_pair_plot", False):
    #         plot_two_series_compare(
    #             run_dfs=run_dfs,
    #             ys=cfg["ys"],
    #             out_path=out_path,
    #             xlabel="Step",
    #             ylabel=cfg["ylabel"],
    #             series_labels=cfg["series_labels"],
    #             force_no_smooth=cfg["force_no_smooth"],
    #             no_smooth_cols=GLOBAL_NO_SMOOTH_COLS,
    #             subsample_every=SUBSAMPLE_EVERY.get(cfg["name"], 1),
    #             use_method_colors=False,  # 多方法比较
    #         )
    #     else:
    #         plot_compare_lines(
    #             run_dfs=run_dfs,
    #             ys=cfg["ys"],
    #             out_path=out_path,
    #             title=f"PDF {cfg['name']} | {cfg.get('title_extra', '')} | uptoHalfMax{MAX_STEP}",
    #             x="step",
    #             xlabel="Step",
    #             ylabel=cfg["ylabel"],
    #             force_no_smooth=cfg["force_no_smooth"],
    #             no_smooth_cols=GLOBAL_NO_SMOOTH_COLS,
    #             subsample_every=subsample,
    #         )
    #     print(f"[OK] Saved: {out_path}")

    # drift_col = f"drift_mag"
    # var_col = f"noise_var"
    # snr_col = f"snr"
    # tsnr_col = f"tsnr_hac"
    # mvar_col = f"m_var"

    # # Fig5: scatter two panels
    # out5 = os.path.join(OUT_DIR, f"Fig5_Scatter_SNR_vs_VarM_and_absDeltaM_{llm_model}.{SAVE_FORMAT}")
    # plot_fig5_scatter_two_panels(
    #     run_dfs=run_dfs,
    #     x_left_col=mvar_col,
    #     x_right_col="abs_delta_m_raw",
    #     y_col=snr_col,
    #     out_path=out5,
    #     title=f"PDF Fig5 | scatter | w={FIG_WINDOW}, minp={get_roll_min_periods(FIG_WINDOW)} | uptoHalfMax{MAX_STEP}",
    # )
    # print(f"[OK] Saved: {out5}")
    
    # # 5) rho distribution (hist)
    # rho_dir = os.path.join(OUT_DIR, RHO_DIST_DIRNAME)
    # ensure_dir(rho_dir)
    # out_rho = os.path.join(rho_dir, f"RhoDist_rho_hist_{llm_model}.{SAVE_FORMAT}")
    # plot_rho_distribution(
    #     run_dfs=run_dfs,
    #     rho_col="rho",
    #     out_path=out_rho,
    #     title=f"rho distribution | uptoHalfMax{MAX_STEP}",
    #     bins=60,
    # )
    # print(f"[OK] Saved: {out_rho}")
    
    # 逐一画出不同方法下dw/dl及对应的band
    band_dir = os.path.join(OUT_DIR, "ratio_band_per_run")
    ensure_dir(band_dir)
    
    y_min, y_max = compute_unified_y_range(run_dfs)
    for run_label, df in run_dfs:
        outp = os.path.join(band_dir, f"Band_dw_over_dl_{safe_slug(run_label)}_{llm_model}.{SAVE_FORMAT}")
        plot_ratio_with_band_focused(
            run_label, df, outp,
            x="step",
            use_log_y=True,
            y_min=y_min,
            y_max=y_max
        )
        print(f"[OK] Saved: {outp}")

    # # 6) AllVars vs step (filtered, non-duplicate)
    # if AUTO_PLOT_ALL_VARS:
    #     auto_dir = os.path.join(OUT_DIR, AUTO_PLOT_DIRNAME)
    #     ensure_dir(auto_dir)
    
    #     # 已绘制列（Fig1–Fig5）
    #     already_plotted_cols = {"m", drift_col, var_col, snr_col, mvar_col, "abs_delta_m_raw"}
    
    #     # 收集所有存在的、允许的、非常量的列
    #     cols_exist = {col for _, df in run_dfs for col in df.columns}
    #     candidate_cols = (ALLVARS_ALLOWLIST & cols_exist) - already_plotted_cols
    #     allvars_cols = {c for c in candidate_cols if not is_constant_across_all_runs(run_dfs, c)}
    
    #     grouped_cols = flatten_group_cols(ALLVARS_SYMMETRIC_GROUPS)
    
    #     def _plot_vars(cols: List[str], slug: str, is_group: bool = False):
    #         if not cols:
    #             return
    #         out_name = f"AllVars{'_GROUP' if is_group else ''}_{slug}_vs_step_{llm_model}.{SAVE_FORMAT}"
    #         out_path = os.path.join(auto_dir, out_name)
    #         ylabel = get_all_variables_label(slug)
    #         title = f"AllVars{' GROUP' if is_group else ''} | {slug} vs step | smooth={SMOOTH_METHOD} (delta_m raw) | uptoHalfMax{MAX_STEP}"
    #         plot_compare_lines(
    #             run_dfs=run_dfs,
    #             ys=cols,
    #             out_path=out_path,
    #             title=title,
    #             x="step",
    #             xlabel="Step",
    #             ylabel=ylabel,
    #             force_no_smooth=False,
    #             subsample_every=ALLVARS_SUBSAMPLE_EVERY,
    #         )
    #         print(f"[OK] Saved: {out_path}")
    
    #     # 1. 绘制对称组
    #     for g in ALLVARS_SYMMETRIC_GROUPS:
    #         print(g)
    #         cols = [c for c in g["cols"] if c in allvars_cols]
    #         if cols:
    #             _plot_vars(cols, g["slug"], is_group=True)
    
    #     # 2. 绘制剩余单列
    #     remaining = sorted(allvars_cols - grouped_cols)
    #     for col in remaining:
    #         _plot_vars([col], col, is_group=False)
    
    #     print(f"[DONE] AllVars plots saved to: {auto_dir}")

    # === Per-method zw/zl plots (like ratio_band_per_run) ===
    zw_zl_dir = os.path.join(OUT_DIR, "zw_zl_per_method")
    ensure_dir(zw_zl_dir)
    
    base_methods = set()
    for lbl, _ in run_dfs:
        if "-calib" in lbl:
            base = lbl.split("-")[0]
            base_methods.add(base)
        else:
            base_methods.add(lbl)
    
    # Plot each base method
    for base_label in sorted(base_methods):
        plot_zw_zl_per_method(
            base_label=base_label,
            run_dfs=run_dfs,
            out_dir=zw_zl_dir,
            model_name=llm_model,
            subsample_every=20,
        )
        print(f"[OK] Saved zw/zl plot for: {base_label}")
    
    # 6) 绘制每个objective的综合图（新增）
    print("\n[INFO] Plotting comprehensive objective plots...")
    comprehensive_dir = os.path.join(OUT_DIR, "comprehensive")

    if PYTHIA2B_DBEMA_SWEEP_MODE:
        plot_objective_comprehensive_dpo_dbema_sweep(
            run_dfs=run_dfs,
            out_dir=comprehensive_dir,
            model_name=llm_model,
            subsample_every=5,
        )
    else:
        # 根据模型决定要绘制的方法
        if llm_model == "pythia-410m":
            methods_to_plot = ["bce", "dpo", "ipo", "cpo", "simpo", "lsif", "ukl", "ddro"]
        elif llm_model == "pythia-2b":
            methods_to_plot = ["bce", "dpo", "ipo", "cpo", "lsif", "ukl", "ddro", "simpo", "tidpo"]
        elif llm_model == "mistral-7b":
            methods_to_plot = ["bce", "dpo", "cpo", "simpo", "ddro", "lsif", "tidpo"]
        else:
            # 默认：所有非校准方法
            methods_to_plot = sorted({lbl for lbl, _ in run_dfs if "-calib" not in lbl})

        plot_all_objectives_comprehensive(
            run_dfs=run_dfs,
            out_dir=comprehensive_dir,
            model_name=llm_model,
            methods_to_plot=methods_to_plot,
            subsample_every=5,
        )
    
    # # 8) 生成摘要图（新增功能）
    # print("\n[INFO] Generating abstract figures for paper...")
    # generate_abstract_figures(
    #     run_dfs=run_dfs,
    #     model_name=llm_model,
    #     out_dir=OUT_DIR,
    # )


    print(f"[DONE] All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
