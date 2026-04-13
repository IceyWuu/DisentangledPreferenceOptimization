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
  1) Edit RUNS / OUT_DIR / FIG_WINDOW / SMOOTH_* below
  2) python plot_jsonl_metrics_compare.py
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


ICML_MODE = True
ICML_COL_W_IN = 3.25   # single-column width
ICML_TEXT_W_IN = 6.75  # two-column (figure*) width

# FIG_WIDTH = 6
# FIG_HEIGHT = 4

FIG_WIDTH = 3.6
FIG_HEIGHT = 3.2

# heights
ICML_LINE_H_IN = 2.05          # single plot height (when used alone)
ICML_PANEL_1x4_H_IN = 1.65     # 1x4 panel height
ICML_SCATTER_H_IN = 2.35       # for your Fig5 (2 panels)

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
llm_model = "mistral-7b"   # mistral-7b, pythia-410m
OUT_DIR = r"./figs/aexperiment_s_list_{}".format(llm_model)

if llm_model == "pythia-410m":
    # pythia-410m
    RUNS = [
        # {"path": r"./pythia-410m-dpo-20251221_111826/margin_chain.jsonl", "label": "dpo"}, # bs32, 但是有些量只算了最后一个minibatch的
        # {"path": r"./pythia-410m-bce-20251221_134129/margin_chain.jsonl", "label": "bce"}, # bs32, 但是有些量只算了最后一个minibatch的
        # {"path": r"./pythia-410m-dpo-20251221_030839/margin_chain.jsonl", "label": "dpo"}, # bs2
        # {"path": r"./pythia-410m-bce-20251221_045557/margin_chain.jsonl", "label": "bce"}, # bs2
        # {"path": r"./pythia-410m-dpo-20251224_164049/margin_chain.jsonl", "label": "dpo"}, # bs32, dw,dl算了m而不是tilde m

        # {"path": r"./pythia-410m-bce-20251224_174150/margin_chain.jsonl", "label": "bce"}, # bs32
        # {"path": r"./pythia-410m-dpo-20251229_215315/margin_chain.jsonl", "label": "dpo"}, # bs32
        # {"path": r"./pythia-410m-ipo-20251229_225439/margin_chain.jsonl", "label": "ipo"}, # bs32，tau=0.1
        # {"path": r"./pythia-410m-cpo-20251230_163440/margin_chain.jsonl", "label": "cpo"}, # bs32
        # {"path": r"./pythia-410m-slic-20251230_173548/margin_chain.jsonl", "label": "slic"}, # bs32
        # {"path": r"./pythia-410m-lsif-20251230_193848/margin_chain.jsonl", "label": "lsif"}, # bs32
        # {"path": r"./pythia-410m-ukl-20251230_183721/margin_chain.jsonl", "label": "ukl"}, # bs32
        # {"path": r"./pythia-410m-ddro-20260106_124724/margin_chain.jsonl", "label": "ddro"},

        {"path": r"./pythia-410m-bce-s_list-20260108_202348/margin_chain.jsonl", "label": "bce"}, # bs32, sw,sl,sw_sl_dot每个minibatch都记录，即一共16个量（累计16个minibatch一步更新）
        {"path": r"./pythia-410m-dpo-s_list-20260108_192103/margin_chain.jsonl", "label": "dpo"}, # bs32, sw,sl,sw_sl_dot每个minibatch都记录，即一共16个量（累计16个minibatch一步更新）
    ]
elif llm_model == "mistral-7b":
    # mistral-7b
    RUNS = [
        # {"path": r"./mistral-7b-dpo-20260105_004932/margin_chain.jsonl", "label": "dpo"}, # bs32
        # {"path": r"./mistral-7b-bce-20260105_115557/margin_chain.jsonl", "label": "bce"}, # bs32
        {"path": r"./mistral-7b-dpo-full-lora-20260107_235806/margin_chain.jsonl", "label": "dpo"}, # bs32
        {"path": r"./mistral-7b-bce-full-lora-20260109_002210/margin_chain.jsonl", "label": "bce"}, # bs32
    ]

SAVE_FORMAT = "pdf"  # "png" or "pdf"
DPI = 300

# ---- window used for Fig2/3/4 and Fig5 scatter ----
FIG_WINDOW = 50 # 50

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
    "lr",
}

# Keep only these list-type columns (others lists/dicts are dropped).
KEEP_LIST_COLS: Set[str] = {
    "sw_l2_mb_vec", "sl_l2_mb_vec", "sw_sl_dot_mb_vec",
    "dw_vec", "dl_vec",
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

            # Keep scalars; keep only selected list columns; drop all other *_vec / lists / dicts.
            row: Dict[str, Any] = {}
            for k, v in obj.items():
                # explicitly keep selected list columns
                if k in KEEP_LIST_COLS:
                    row[k] = v
                    continue

                # drop any other *_vec (these can be huge)
                if k.endswith("_vec"):
                    continue

                # drop any other lists/dicts (including other *_mb_vec not in whitelist)
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

    # Coerce numeric columns where possible (keep loss_type as-is)
    for c in df.columns:
        if c in ("loss_type",):
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")

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
    # sw = np.sqrt(pd.to_numeric(df["sw_l2"], errors="coerce").to_numpy(dtype=np.float64)) # sw，sl是开过根号的
    # sl = np.sqrt(pd.to_numeric(df["sl_l2"], errors="coerce").to_numpy(dtype=np.float64))

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

    # log-slack: how far inside the band (or outside if negative)
    slack = np.full(len(df), np.nan, dtype=np.float64)
    if np.any(mask_all):
        lr = np.log(r[mask_all])
        ll = np.log(lower[mask_all])
        lu = np.log(upper[mask_all])
        slack[mask_all] = np.minimum(lr - ll, lu - lr)  # >=0 inside; <0 outside

    df["sl_over_sw"] = a
    df["band_lower"] = lower
    df["band_upper"] = upper
    df["in_band"] = in_band
    df["band_slack_log"] = slack
    return df

def add_mb_ratio_and_bandcenter_stats(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    For each step, compute micro-batch-level:
      - r_mb_mean, r_mb_std where r_mb[i] = (mean(dw over samples in micro-batch i) /
                                            mean(dl over samples in micro-batch i))
      - band_center_mb_mean, band_center_mb_std where band_center_mb[i] is computed from
        sw_l2_mb_vec/sl_l2_mb_vec/sw_sl_dot_mb_vec:
            rho_i = dot_i / (sw_i * sl_i)
            a_i   = sl_i / sw_i
            center_i = a_i * 0.5 * (rho_i + 1/rho_i)

    Requires:
      - sw_l2_mb_vec, sl_l2_mb_vec, sw_sl_dot_mb_vec (length = grad_accum_steps)
      - dw_vec, dl_vec (length = micro_batch_size * grad_accum_steps)
      - micro_batch_size, grad_accum_steps (scalars)
    """
    if df.empty:
        df["r_mb_mean"] = np.nan
        df["r_mb_std"] = np.nan
        df["band_center_mb_mean"] = np.nan
        df["band_center_mb_std"] = np.nan
        return df

    # prepare outputs
    r_mean = np.full(len(df), np.nan, dtype=np.float64)
    r_std = np.full(len(df), np.nan, dtype=np.float64)
    c_mean = np.full(len(df), np.nan, dtype=np.float64)
    c_std = np.full(len(df), np.nan, dtype=np.float64)

    for idx, row in df.iterrows():
        # ---- band center from *_mb_vec ----
        sw_mb = row.get("sw_l2_mb_vec", None)
        sl_mb = row.get("sl_l2_mb_vec", None)
        dot_mb = row.get("sw_sl_dot_mb_vec", None)

        centers = None
        if isinstance(sw_mb, list) and isinstance(sl_mb, list) and isinstance(dot_mb, list):
            sw_arr = np.asarray(sw_mb, dtype=np.float64)
            sl_arr = np.asarray(sl_mb, dtype=np.float64)
            dot_arr = np.asarray(dot_mb, dtype=np.float64)

            mask = np.isfinite(sw_arr) & np.isfinite(sl_arr) & np.isfinite(dot_arr) & (sw_arr > eps) & (sl_arr > eps)
            if mask.any():
                rho_arr = np.full_like(sw_arr, np.nan, dtype=np.float64)
                rho_arr[mask] = dot_arr[mask] / (sw_arr[mask] * sl_arr[mask])

                # require rho>0 for center (because of 1/rho)
                mask2 = mask & np.isfinite(rho_arr) & (rho_arr > eps)
                if mask2.any():
                    a_arr = np.full_like(sw_arr, np.nan, dtype=np.float64)
                    a_arr[mask2] = sl_arr[mask2] / sw_arr[mask2]
                    centers = np.full_like(sw_arr, np.nan, dtype=np.float64)
                    centers[mask2] = a_arr[mask2] * 0.5 * (rho_arr[mask2] + 1.0 / rho_arr[mask2])

        if centers is not None:
            v = centers[np.isfinite(centers)]
            if v.size > 0:
                c_mean[idx] = float(v.mean())
                c_std[idx] = float(v.std(ddof=0))

        # ---- r from dw_vec/dl_vec folded into micro-batches ----
        dw_vec = row.get("dw_vec", None)
        dl_vec = row.get("dl_vec", None)
        mbs = row.get("micro_batch_size", None)
        gas = row.get("grad_accum_steps", None)

        if isinstance(dw_vec, list) and isinstance(dl_vec, list):
            try:
                mbs_i = int(mbs) if (mbs is not None and np.isfinite(float(mbs))) else None
                gas_i = int(gas) if (gas is not None and np.isfinite(float(gas))) else None
            except Exception:
                mbs_i, gas_i = None, None

            if mbs_i is not None and gas_i is not None and mbs_i > 0 and gas_i > 0:
                dw_arr = np.asarray(dw_vec, dtype=np.float64)
                dl_arr = np.asarray(dl_vec, dtype=np.float64)
                expect_len = mbs_i * gas_i

                if dw_arr.size == expect_len and dl_arr.size == expect_len:
                    dw_mb = dw_arr.reshape(gas_i, mbs_i).mean(axis=1)
                    dl_mb = dl_arr.reshape(gas_i, mbs_i).mean(axis=1)

                    mask = np.isfinite(dw_mb) & np.isfinite(dl_mb) & (np.abs(dl_mb) > eps)
                    if mask.any():
                        r_mb = np.full_like(dw_mb, np.nan, dtype=np.float64)
                        r_mb[mask] = dw_mb[mask] / dl_mb[mask]
                        r_mb = r_mb[np.isfinite(r_mb) & (r_mb > 0)]
                        if r_mb.size > 0:
                            r_mean[idx] = float(r_mb.mean())
                            r_std[idx] = float(r_mb.std(ddof=0))

    df["r_mb_mean"] = r_mean
    df["r_mb_std"] = r_std
    df["band_center_mb_mean"] = c_mean
    df["band_center_mb_std"] = c_std
    return df


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
        return y.ewm(span=SMOOTH_EMA_SPAN, adjust=False).mean()
    if SMOOTH_METHOD.lower() == "rolling":
        return y.rolling(window=SMOOTH_ROLLING_WINDOW, min_periods=SMOOTH_MIN_PERIODS).mean()
    raise ValueError(f"Unknown SMOOTH_METHOD: {SMOOTH_METHOD}")

def plot_compare_lines(
    run_dfs: List[Tuple[str, pd.DataFrame]],
    ys: List[str],
    out_path: str,
    title: Optional[str],
    x: str = "step",
    force_no_smooth: bool = False,
    no_smooth_cols: Optional[Set[str]] = None,
) -> None:
    if no_smooth_cols is None:
        no_smooth_cols = set()

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
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

    # Build all series (for shading per run)
    all_series = []  # (run_label, series_label, xs, ys_vals)
    for run_label, df in run_dfs:
        if df.empty or x not in df.columns:
            continue
        g = df.sort_values(x)

        if "loss_type" in g.columns and g["loss_type"].nunique(dropna=True) > 1:
            for lt, gg in g.groupby("loss_type", dropna=True):
                gg = gg.sort_values(x)
                for y in ys_exist:
                    if y not in gg.columns:
                        continue
                    series = gg[y]
                    if series.dropna().empty:
                        continue
                    s_sm = smooth_series(series.rename(y), force_no_smooth=force_no_smooth, no_smooth_cols=no_smooth_cols)
                    all_series.append((run_label, f"{run_label}:{lt}:{y}", gg[x].values, s_sm.values))
        else:
            for y in ys_exist:
                if y not in g.columns:
                    continue
                series = g[y]
                if series.dropna().empty:
                    continue
                s_sm = smooth_series(series.rename(y), force_no_smooth=force_no_smooth, no_smooth_cols=no_smooth_cols)
                all_series.append((run_label, f"{run_label}", g[x].values, s_sm.values))    # f"{run_label}:{y}"

    if not all_series:
        plt.close(fig)
        return

    # Assign shades per run_label
    run_to_series_idx: Dict[str, List[int]] = {}
    for idx, (run_label, _, _, _) in enumerate(all_series):
        run_to_series_idx.setdefault(run_label, []).append(idx)

    run_labels_order = [lbl for (lbl, _) in run_dfs]
    run_to_color = {lbl: base_colors[i % len(base_colors)] for i, lbl in enumerate(run_labels_order)}

    for run_label, idxs in run_to_series_idx.items():
        base = run_to_color.get(run_label, "C0")
        n = len(idxs)
        amounts = [0.0] if n == 1 else np.linspace(0.0, 0.70, n)

        for amount, idx in zip(amounts, idxs):
            _, series_label, xs, ys_vals = all_series[idx]
            ax.plot(xs, ys_vals, label=series_label, color=_blend_to_white(base, float(amount)))

    ax.set_xlabel(x)
    ax.set_ylabel("value")
    # if title:
    #     ax.set_title(title)
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
def plot_ratio_with_band_per_run(
    run_label: str,
    df: pd.DataFrame,
    out_path: str,
    x: str = "step",
    use_log_y: bool = False,  # 新增参数：是否使用对数 y 轴
) -> None:
    """
    Per-run plot: Incentive ratio (d_w / d_l) with Ratio band.
    Optimized for ICML single-column layout.
    """
    if df.empty or (x not in df.columns):
        return
    for c in ["dw_over_dl", "band_lower", "band_upper"]:
        if c not in df.columns:
            return

    g = df.sort_values(x)
    xs = g[x].values

    r = pd.to_numeric(g["dw_over_dl"], errors="coerce")
    lo = pd.to_numeric(g["band_lower"], errors="coerce")
    up = pd.to_numeric(g["band_upper"], errors="coerce")

    if r.dropna().empty and lo.dropna().empty and up.dropna().empty:
        return

    # Use wider figure for band plots to show more detail
    fig, ax = plt.subplots(figsize=(FIG_WIDTH *1.5, FIG_HEIGHT))

    # Main curve
    ax.plot(xs, r.values, color='tab:blue', linewidth=1.2, label=r"Incentive ratio ($d_w / d_l$)")

    # Shaded band
    ax.fill_between(xs, lo.values, up.values, color='tab:blue', alpha=0.2, label="Disentanglement band")

    ax.set_xlabel("Step")
    ax.set_ylabel("Incentive ratio")

    # Conditionally apply log scale
    if use_log_y:
        ax.set_yscale("log")
        # For log scale, we still want clean tick labels (avoid "1e-1")
        from matplotlib.ticker import LogFormatter
        ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False, minor_thresholds=(2, 0.4)))
    else:
        # Use plain float formatting: e.g., 0.2 instead of 2×10⁻¹
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(style='plain', axis='y')

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid(True, linewidth=0.3, linestyle=':', alpha=0.6)

    ax.legend(loc="best", handlelength=1.2, handletextpad=0.3, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def plot_ratio_and_bandcenter_mb_var(
    run_label: str,
    df: pd.DataFrame,
    out_path: str,
    x: str = "step",
    use_log_y: bool = False,
    k_std: float = 1.0,  # shading = mean ± k_std*std
) -> None:
    """
    Plot two curves with micro-batch variability bands:
      1) r_mb_mean with ± std band
      2) band_center_mb_mean with ± std band
    All in the same base color family (different alpha/linestyle).
    """
    if df.empty or (x not in df.columns):
        return

    need_cols = ["r_mb_mean", "r_mb_std", "band_center_mb_mean", "band_center_mb_std"]
    for c in need_cols:
        if c not in df.columns:
            return

    g = df.sort_values(x).reset_index(drop=True)
    xs = g[x].values

    r_mu = pd.to_numeric(g["r_mb_mean"], errors="coerce").to_numpy(dtype=np.float64)
    r_sd = pd.to_numeric(g["r_mb_std"], errors="coerce").to_numpy(dtype=np.float64)
    c_mu = pd.to_numeric(g["band_center_mb_mean"], errors="coerce").to_numpy(dtype=np.float64)
    c_sd = pd.to_numeric(g["band_center_mb_std"], errors="coerce").to_numpy(dtype=np.float64)

    if np.all(~np.isfinite(r_mu)) and np.all(~np.isfinite(c_mu)):
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    base = "tab:blue"  # single-run plot: consistent with your other band plots

    # r mean ± std
    m1 = np.isfinite(r_mu) & np.isfinite(r_sd)
    if m1.any():
        ax.plot(xs[m1], r_mu[m1], color=base, linewidth=1.2, label=r"$\mathbb{E}_{mb}[d_w/d_l]$")
        ax.fill_between(
            xs[m1],
            (r_mu[m1] - k_std * r_sd[m1]),
            (r_mu[m1] + k_std * r_sd[m1]),
            color=base,
            alpha=0.18,
            linewidth=0.0,
            label=rf"$\pm {k_std}\sigma$ (mb)"
        )
    elif np.isfinite(r_mu).any():
        m = np.isfinite(r_mu)
        ax.plot(xs[m], r_mu[m], color=base, linewidth=1.2, label=r"$\mathbb{E}_{mb}[d_w/d_l]$")

    # band-center mean ± std (same color family; dashed + lighter)
    m2 = np.isfinite(c_mu) & np.isfinite(c_sd)
    if m2.any():
        ax.plot(xs[m2], c_mu[m2], color=base, linewidth=1.2, linestyle="--", alpha=0.85, label=r"$\mathbb{E}_{mb}[\mathrm{center}]$")
        ax.fill_between(
            xs[m2],
            (c_mu[m2] - k_std * c_sd[m2]),
            (c_mu[m2] + k_std * c_sd[m2]),
            color=base,
            alpha=0.10,
            linewidth=0.0,
        )
    elif np.isfinite(c_mu).any():
        m = np.isfinite(c_mu)
        ax.plot(xs[m], c_mu[m], color=base, linewidth=1.2, linestyle="--", alpha=0.85, label=r"$\mathbb{E}_{mb}[\mathrm{center}]$")

    ax.set_xlabel("Step")
    ax.set_ylabel("Value")

    if use_log_y:
        ax.set_yscale("log")
        from matplotlib.ticker import LogFormatter
        ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False, minor_thresholds=(2, 0.4)))
    else:
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(style='plain', axis='y')

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid(True, linewidth=0.3, linestyle=':', alpha=0.6)
    ax.legend(loc="best", handlelength=1.2, handletextpad=0.3, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_ratio_with_band_focused(
    run_label: str,
    df: pd.DataFrame,
    out_path: str,
    x: str = "step",
    use_log_y: bool = False,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    """
    Focused plot on the main curve (dw_over_dl) to highlight trends.
    Y-axis is adjusted to focus on the curve, may not show full band.
    Optimized to show BCE's upward trend vs DPO's flat trend.
    If y_min and y_max are provided, use them; otherwise calculate from data.
    """
    if df.empty or (x not in df.columns):
        return
    for c in ["dw_over_dl", "band_lower", "band_upper"]:
        if c not in df.columns:
            return

    g = df.sort_values(x)
    xs = g[x].values

    r = pd.to_numeric(g["dw_over_dl"], errors="coerce")
    lo = pd.to_numeric(g["band_lower"], errors="coerce")
    up = pd.to_numeric(g["band_upper"], errors="coerce")

    if r.dropna().empty and lo.dropna().empty and up.dropna().empty:
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Calculate y-axis range focused on the main curve
    # If y_min/y_max are provided, use them; otherwise calculate from this data
    if y_min is None or y_max is None:
        r_valid = r.dropna()
        if len(r_valid) > 0:
            r_min = r_valid.min()
            r_max = r_valid.max()
            r_range = r_max - r_min
            # Focus on the curve with some padding, may not show full band
            y_padding = r_range * 1  # 100% padding
            y_min = max(r_min - y_padding, r_valid.min() * 0.85)  # Allow more range downward
            y_max = min(r_max + y_padding, r_valid.max() * 1.15)  # Allow more range upward
        else:
            y_min = None
            y_max = None

    # Shaded band (may be partially visible)
    ax.fill_between(xs, lo.values, up.values, color='tab:blue', alpha=0.15, label="Disentanglement band", zorder=1)

    # Calculate and smooth the band center (mean of upper and lower bounds)
    band_center = (lo.values + up.values) / 2.0
    # Smooth the band center using rolling window
    # Use min_periods=FIG_WINDOW to ignore points with insufficient window size
    band_center_series = pd.Series(band_center)
    band_center_smooth = band_center_series.rolling(window=FIG_WINDOW, min_periods=FIG_WINDOW, center=False).mean()
    
    # Plot smoothed band center line (NaN values will be automatically skipped)
    ax.plot(xs, band_center_smooth.values, color='tab:orange', linewidth=1.2, linestyle='--', 
            alpha=0.8, label="Band center", zorder=4)

    # Main curve - emphasized with thicker line
    ax.plot(xs, r.values, color='tab:blue', linewidth=1.2, label=r"Incentive ratio ($d_{w,t} / d_{l,t}$)", zorder=3)

    ax.set_xlabel("Step")
    ax.set_ylabel("Incentive ratio")

    # Conditionally apply log scale
    if use_log_y:
        ax.set_yscale("log")
        from matplotlib.ticker import LogFormatter
        ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False, minor_thresholds=(2, 0.4)))
    else:
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(style='plain', axis='y')
        
        # Set y-axis limits to focus on the curve
        if y_min is not None and y_max is not None:
            blank = 0.2
            ax.set_ylim(y_min-blank, y_max+blank)

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid(True, linewidth=0.3, linestyle=':', alpha=0.6)

    ax.legend(loc="best", handlelength=1.2, handletextpad=0.3, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def plot_ratio_with_band_per_run(
    run_label: str,
    df: pd.DataFrame,
    out_path: str,
    x: str = "step",
    use_log_y: bool = False,  # 是否使用对数 y 轴
    draw_band_edges: bool = False,
    draw_running_ratio: bool = True, 
    right_axis_col: Optional[str] = None,
) -> None:
    """
    Per-run plot: Incentive ratio (d_w / d_l) with Ratio band.
    Optimized for ICML single-column layout.
    """
    if df.empty or (x not in df.columns):
        return
    for c in ["dw_over_dl", "band_lower", "band_upper"]:
        if c not in df.columns:
            return

    g = df.sort_values(x)
    xs = g[x].values

    r = pd.to_numeric(g["dw_over_dl"], errors="coerce")
    lo = pd.to_numeric(g["band_lower"], errors="coerce")
    up = pd.to_numeric(g["band_upper"], errors="coerce")

    if r.dropna().empty and lo.dropna().empty and up.dropna().empty:
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Shaded band
    band_mask = np.isfinite(lo) & np.isfinite(up)
    if band_mask.any():
        # ax.plot(xs, lo.values, linestyle="--", linewidth=0.9, zorder=2, label="DB lower")
        # ax.plot(xs, up.values, linestyle="--", linewidth=0.9, zorder=2, label="DB upper")
        ax.fill_between(xs, lo.values, up.values, color='tab:blue', zorder=1, alpha=0.2, linewidth=0.2, label="Disentanglement band")
    
    # Optional band edges (usually OFF to avoid covering the ratio curve)
    if draw_band_edges and band_mask.any():
        ax.plot(xs, lo, linestyle="--", linewidth=0.9, zorder=2, label="DB lower")
        ax.plot(xs, up, linestyle="--", linewidth=0.9, zorder=2, label="DB upper")

    # Main curve
    ax.plot(xs, r.values, color='tab:blue', linewidth=1.2, zorder=3, label=r"Incentive ratio ($d_w / d_l$)")
    # ax.scatter(xs, r.values, color='tab:blue', s=3, alpha=1.0, zorder=3, label=r"Incentive ratio ($d_w / d_l$)")
    
    # # Highlight out-of-band points (uses your existing in_band)
    # if "in_band" in g.columns:
    #     inb = pd.to_numeric(g["in_band"], errors="coerce").values
    #     mask = np.isfinite(inb) & np.isfinite(r)
    #     out_mask = mask & (inb < 0.5)
    #     if out_mask.any():
    #         ax.scatter(xs[out_mask], r[out_mask], marker="+", s=2, alpha=0.6, zorder=4, label="out of band", color="red")

    ax.set_xlabel("Step")
    ax.set_ylabel("Incentive ratio")

    # Conditionally apply log scale
    if use_log_y:
        ax.set_yscale("log")
        # For log scale, we still want clean tick labels (avoid "1e-1")
        from matplotlib.ticker import LogFormatter
        ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False, minor_thresholds=(2, 0.4)))
    else:
        # Use plain float formatting: e.g., 0.2 instead of 2×10⁻¹
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(style='plain', axis='y')

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid(True, linewidth=0.3, linestyle=':', alpha=0.6)

    # # right axis: rolling mode(iii) rate (much more discriminative than in-band ratio)
    # if right_axis_col is not None and (right_axis_col in g.columns):
    #     y2 = pd.to_numeric(g[right_axis_col], errors="coerce").values
    #     ax2 = ax.twinx()
    #     ax2.plot(xs, y2, linewidth=1.3, linestyle="-.", label="mode(iii) rate (rolling)")
    #     ax2.set_ylim(0.0, 1.0)
    #     ax2.set_ylabel("mode(iii) rate")
    #     ax2.grid(False)

    #     h1, l1 = ax.get_legend_handles_labels()
    #     h2, l2 = ax2.get_legend_handles_labels()
    #     ax.legend(h1 + h2, l1 + l2, loc="best",
    #               handlelength=1.2, handletextpad=0.3, frameon=True)
    # else:
    #     ax.legend(loc="best", handlelength=1.2, handletextpad=0.3, frameon=True)
    
    # right axis running in-band ratio
    # if draw_running_ratio and ("in_band" in g.columns):
    #     inb = pd.to_numeric(g["in_band"], errors="coerce").values
    #     run_ratio = running_inband_ratio(inb)
    #     ax2 = ax.twinx()
    #     ax2.plot(xs, run_ratio, linewidth=1.2, linestyle="-.", zorder=2, label="running in-band ratio")
    #     ax2.set_ylim(0.0, 1.0)
    #     ax2.set_ylabel("In-band ratio (cumulative)")
    #     ax2.grid(False)

    #     # Combine legends from both axes
    #     h1, l1 = ax.get_legend_handles_labels()
    #     h2, l2 = ax2.get_legend_handles_labels()
    #     ax.legend(h1 + h2, l1 + l2, loc="best", handlelength=1.2, handletextpad=0.3, frameon=True)
    # else:
    #     ax.legend(loc="best", handlelength=1.2, handletextpad=0.3, frameon=True)
    
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
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    run_to_color = {lbl: base_colors[i % len(base_colors)] for i, (lbl, _) in enumerate(run_dfs)}

    any_ok = False
    for run_label, df in run_dfs:
        if rho_col not in df.columns:
            continue
        vals = pd.to_numeric(df[rho_col], errors="coerce").dropna()
        if vals.empty:
            continue
        any_ok = True
        ax.hist(
            vals.values,
            bins=bins,
            density=True,
            alpha=0.35,
            label=run_label,
            color=run_to_color.get(run_label, "C0"),
        )

    if not any_ok:
        plt.close(fig)
        return

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Density")
    # ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")

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
    base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    run_to_color = {lbl: base_colors[i % len(base_colors)] for i, (lbl, _) in enumerate(run_dfs)}

    axL, axR = axes[0], axes[1]
    any_ok = False

    for run_label, df in run_dfs:
        c = run_to_color.get(run_label, "C0")

        if (x_left_col in df.columns) and (y_col in df.columns):
            xL = pd.to_numeric(df[x_left_col], errors="coerce")
            y = pd.to_numeric(df[y_col], errors="coerce")
            mask = (~xL.isna()) & (~y.isna())
            if mask.any():
                any_ok = True
                axL.scatter(xL[mask].values, y[mask].values, s=10, alpha=0.35, label=run_label, color=c, rasterized=True)

        if (x_right_col in df.columns) and (y_col in df.columns):
            xR = pd.to_numeric(df[x_right_col], errors="coerce")
            y = pd.to_numeric(df[y_col], errors="coerce")
            mask = (~xR.isna()) & (~y.isna())
            if mask.any():
                any_ok = True
                axR.scatter(xR[mask].values, y[mask].values, s=10, alpha=0.35, label=run_label, color=c, rasterized=True)

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
    # MAX_STEP = global_max // 2
    # print(f"[INFO] global_max_step={global_max} => MAX_STEP (half)={MAX_STEP}")
    MAX_STEP = global_max
    print(f"[INFO] global_max_step={global_max} => MAX_STEP ={MAX_STEP}")

    # 3) truncate + compute required derived columns + save scalars
    run_dfs: List[Tuple[str, pd.DataFrame]] = []
    for label, df in loaded:
        df = df[df["step"] <= MAX_STEP].copy().sort_values("step").reset_index(drop=True)
        if df.empty:
            print(f"[WARN] {label}: no rows <= MAX_STEP={MAX_STEP}")
            continue

        df = ensure_m_and_delta_m(df)
        df = add_windowed_stats(df, FIG_WINDOW)
        df = add_mb_ratio_and_bandcenter_stats(df)


        # Fig5 needs |Δm_t| raw
        df["abs_delta_m_raw"] = pd.to_numeric(df["delta_m"], errors="coerce").abs()

        run_out = os.path.join(OUT_DIR, f"run_{label}")
        ensure_dir(run_out)
        df.to_csv(os.path.join(run_out, f"scalars_uptoHalfMax{MAX_STEP}_w{FIG_WINDOW}.csv"), index=False)

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
    combined_df.to_csv(os.path.join(OUT_DIR, f"combined_scalars_uptoHalfMax{MAX_STEP}_w{FIG_WINDOW}.csv"), index=False)

    # 4) PDF-aligned figures

    # Fig1: m_t vs step (NO smoothing)
    out1 = os.path.join(OUT_DIR, f"Fig1_Margin_m_t_vs_step_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
    plot_compare_lines(
        run_dfs=run_dfs,
        ys=["m"],
        out_path=out1,
        title=f"PDF Fig1 | m_t vs step | NO smoothing | uptoHalfMax{MAX_STEP}",
        x="step",
        force_no_smooth=True,
    )
    print(f"[OK] Saved: {out1}")

    drift_col = f"drift_mag_w{FIG_WINDOW}"
    var_col = f"noise_var_w{FIG_WINDOW}"
    snr_col = f"snr_w{FIG_WINDOW}"
    tsnr_col = f"tsnr_hac_w{FIG_WINDOW}"
    mvar_col = f"m_var_w{FIG_WINDOW}"

    # Fig2: |E_w[Δm_t]|
    out2 = os.path.join(OUT_DIR, f"Fig2_DriftMagnitude_absE_delta_m_w{FIG_WINDOW}_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
    plot_compare_lines(
        run_dfs=run_dfs,
        ys=[drift_col],
        out_path=out2,
        title=f"PDF Fig2 | |E_w[Δm_t]| (w={FIG_WINDOW}, minp={get_roll_min_periods(FIG_WINDOW)}) | uptoHalfMax{MAX_STEP}",
        x="step",
        force_no_smooth=True,
    )
    print(f"[OK] Saved: {out2}")

    # Fig3: Var_w(Δm_t)
    out3 = os.path.join(OUT_DIR, f"Fig3_NoiseScale_Var_delta_m_w{FIG_WINDOW}_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
    plot_compare_lines(
        run_dfs=run_dfs,
        ys=[var_col],
        out_path=out3,
        title=f"PDF Fig3 | Var_w(Δm_t) (w={FIG_WINDOW}, minp={get_roll_min_periods(FIG_WINDOW)}) | uptoHalfMax{MAX_STEP}",
        x="step",
        force_no_smooth=True,
    )
    print(f"[OK] Saved: {out3}")

    # Fig4: SNR
    out4 = os.path.join(OUT_DIR, f"Fig4_SNR_absE_over_sqrtVar_w{FIG_WINDOW}_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
    plot_compare_lines(
        run_dfs=run_dfs,
        ys=[snr_col],
        out_path=out4,
        title=f"PDF Fig4 | SNR (w={FIG_WINDOW}, minp={get_roll_min_periods(FIG_WINDOW)}) | uptoHalfMax{MAX_STEP}",
        x="step",
        force_no_smooth=True,
    )
    
    out4 = os.path.join(OUT_DIR, f"Fig4_tSNR_HAC_w{FIG_WINDOW}_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
    plot_compare_lines(
        run_dfs=run_dfs,
        ys=[tsnr_col],
        out_path=out4,
        title=f"PDF Fig4 | tSNR (HAC, w={FIG_WINDOW}, H={get_hac_lag(FIG_WINDOW)}) | uptoHalfMax{MAX_STEP}",
        x="step",
        force_no_smooth=True,
    )
    
    print(f"[OK] Saved: {out4}")

    # Fig5: scatter two panels
    out5 = os.path.join(OUT_DIR, f"Fig5_Scatter_SNR_vs_VarM_and_absDeltaM_w{FIG_WINDOW}_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
    plot_fig5_scatter_two_panels(
        run_dfs=run_dfs,
        x_left_col=mvar_col,
        x_right_col="abs_delta_m_raw",
        y_col=snr_col,
        out_path=out5,
        title=f"PDF Fig5 | scatter | w={FIG_WINDOW}, minp={get_roll_min_periods(FIG_WINDOW)} | uptoHalfMax{MAX_STEP}",
    )
    print(f"[OK] Saved: {out5}")
    
    # Fig6: dw/dl ratio vs step
    out6 = os.path.join(OUT_DIR, f"Fig6_dw_over_dl_vs_step_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
    plot_compare_lines(
        run_dfs=run_dfs,
        ys=["dw_over_dl"],
        out_path=out6,
        title=f"Fig6 | dw/dl vs step | uptoHalfMax{MAX_STEP}",
        x="step",
        force_no_smooth=True,   # ratio 这种量我建议不平滑，避免引入假象
    )
    print(f"[OK] Saved: {out6}")
    
    # Fig6b: log(dw/dl) vs step (recommended)
    out6b = os.path.join(OUT_DIR, f"Fig6b_log_dw_over_dl_vs_step_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
    plot_compare_lines(
        run_dfs=run_dfs,
        ys=["log_dw_over_dl"],
        out_path=out6b,
        title=f"Fig6b | log(dw/dl) vs step | uptoHalfMax{MAX_STEP}",
        x="step",
        force_no_smooth=True,
    )
    print(f"[OK] Saved: {out6b}")
    
    # 逐一画出不同方法下dw/dl及对应的band
    band_dir = os.path.join(OUT_DIR, "ratio_band_per_run")
    ensure_dir(band_dir)

    # ensure_dir(band_dir)
    # for run_label, df in run_dfs:
    #     outp = os.path.join(band_dir, f"Band_dw_over_dl_{safe_slug(run_label)}_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
    #     plot_ratio_with_band_per_run(run_label, df, outp, x="step", use_log_y=safe_slug(run_label) in ["cpo"])
    #     print(f"[OK] Saved: {outp}")
    
    # 生成聚焦中间线的band图（突出趋势）
    # First, calculate unified y-axis range from all BCE and DPO data
    all_r_values = []
    for run_label, df in run_dfs:
        if run_label in ["bce", "dpo"]:
            if "dw_over_dl" in df.columns:
                r = pd.to_numeric(df["dw_over_dl"], errors="coerce").dropna()
                if len(r) > 0:
                    all_r_values.extend(r.values)
    
    # Calculate unified y-axis range
    unified_y_min = None
    unified_y_max = None
    if len(all_r_values) > 0:
        all_r_array = np.array(all_r_values)
        r_min = all_r_array.min()
        r_max = all_r_array.max()
        r_range = r_max - r_min
        y_padding = r_range * 1.5  # 150% padding (increased from 100% to show more range)
        unified_y_min = max(r_min - y_padding, r_min * 0.75)  # Allow more range downward (changed from 0.85)
        unified_y_max = min(r_max + y_padding, r_max * 1.25)  # Allow more range upward (changed from 1.15)
    
    # Generate focused plots with unified y-axis range
    for run_label, df in run_dfs:
        outp_focused = os.path.join(band_dir, f"Band_dw_over_dl_{safe_slug(run_label)}_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
        plot_ratio_with_band_focused(
            run_label, df, outp_focused,
            x="step",
            use_log_y=safe_slug(run_label) in ["cpo"],
            y_min=unified_y_min,
            y_max=unified_y_max
        )
        print(f"[OK] Saved: {outp_focused}")

        outp_mb = os.path.join(band_dir, f"MBVar_ratio_and_center_{safe_slug(run_label)}_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
        plot_ratio_and_bandcenter_mb_var(
            run_label, df, outp_mb,
            x="step",
            use_log_y=safe_slug(run_label) in ["cpo"],
            k_std=1.0,
        )
        print(f"[OK] Saved: {outp_mb}")


    # 5) rho distribution (hist)
    rho_dir = os.path.join(OUT_DIR, RHO_DIST_DIRNAME)
    ensure_dir(rho_dir)
    out_rho = os.path.join(rho_dir, f"RhoDist_rho_hist_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
    plot_rho_distribution(
        run_dfs=run_dfs,
        rho_col="rho",
        out_path=out_rho,
        title=f"rho distribution | uptoHalfMax{MAX_STEP}",
        bins=60,
    )
    print(f"[OK] Saved: {out_rho}")

    # 6) AllVars vs step (filtered, non-duplicate)
    if AUTO_PLOT_ALL_VARS:
        auto_dir = os.path.join(OUT_DIR, AUTO_PLOT_DIRNAME)
        ensure_dir(auto_dir)

        # Columns already plotted in Fig1..Fig5 (do not re-plot in AllVars)
        already_plotted_cols = {
            "m",               # Fig1
            drift_col,         # Fig2
            var_col,           # Fig3
            snr_col,           # Fig4
            mvar_col,          # Fig5 (scatter only)
            "abs_delta_m_raw", # Fig5 (scatter only)
        }

        # Build final AllVars target set:
        # - allowlist only
        # - remove already plotted
        # - remove constant columns across runs
        # - only keep columns that exist in at least one run
        cols_exist = set()
        for _, df in run_dfs:
            cols_exist.update(df.columns)

        allvars_cols = set(c for c in ALLVARS_ALLOWLIST if c in cols_exist)
        allvars_cols = allvars_cols.difference(already_plotted_cols)

        # Drop constants
        allvars_cols = {c for c in allvars_cols if not is_constant_across_all_runs(run_dfs, c)}

        # Symmetric groups first
        grouped_cols = flatten_group_cols(ALLVARS_SYMMETRIC_GROUPS)
        for g in ALLVARS_SYMMETRIC_GROUPS:
            cols = [c for c in g["cols"] if c in allvars_cols]
            if not cols:
                continue
            outp = os.path.join(auto_dir, f"AllVars_GROUP_{safe_slug(g['slug'])}_vs_step_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
            plot_compare_lines(
                run_dfs=run_dfs,
                ys=cols,
                out_path=outp,
                title=f"AllVars GROUP | {g['slug']} | smooth={SMOOTH_METHOD} (delta_m raw) | uptoHalfMax{MAX_STEP}",
                x="step",
                force_no_smooth=False,
            )
            print(f"[OK] Saved: {outp}")

        # Single-variable plots for remaining (exclude grouped ones to avoid duplicates)
        remaining = sorted([c for c in allvars_cols if c not in grouped_cols])
        for c in remaining:
            outp = os.path.join(auto_dir, f"AllVars_{safe_slug(c)}_vs_step_uptoHalfMax{MAX_STEP}.{SAVE_FORMAT}")
            plot_compare_lines(
                run_dfs=run_dfs,
                ys=[c],
                out_path=outp,
                title=f"AllVars | {c} vs step | smooth={SMOOTH_METHOD} (delta_m raw) | uptoHalfMax{MAX_STEP}",
                x="step",
                force_no_smooth=False,
            )
            print(f"[OK] Saved: {outp}")

        print(f"[DONE] AllVars plots saved to: {auto_dir}")

    print(f"[DONE] All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
