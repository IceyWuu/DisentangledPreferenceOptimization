"""
比较calib前后各方法的评测结果变化。

默认读取 outputs/results/mistral-7b-base/ 和 outputs/results/mistral-7b-calib/ 目录下的
评测结果文件，对比每个方法在 calib 前后的指标变化，并输出表格。

你也可以通过环境变量切换模型/目录：
- DIL_MODEL_NAME: 模型名（如 "pythia-2b" 或 "mistral-7b"），会自动拼成 "{model}-base"/"{model}-calib"
- DIL_BASE_SUBDIR: 覆盖 base 子目录名（默认 "{model}-base"）
- DIL_CALIB_SUBDIR: 覆盖 calib 子目录名（默认 "{model}-calib"）

支持五种评测类型：
- arc: 读取 **/out_arc_*.json 文件
- bbh: 读取 bbh/lb2_bbh_*.json/**/results_*.json 文件
- mmlu_pro: 读取 mmlu_pro/lb2_mmlu_pro_*.json/**/results_*.json 文件
- math_hard: 读取 math_hard/lb2_math_hard_*.json/**/results_*.json 文件
- musr: 读取 musr/lb2_musr_*.json/**/results_*.json；单文件内对三个子任务 acc_norm 按样本数加权
    (250*murder + 256*object_placements + 250*team_allocation)/756；stderr 按独立估计做加权方差合成
- gsm8k: 读取 **/out_gsm8k_*.json 文件

通过修改文件顶部的 EVAL_TYPE 变量选择类型。

命令行示例（在 DIL/outputs/ 下）：
  python compare_calib_results.py --eval-type musr --model pythia-2b
  python compare_calib_results.py --eval-type bbh,mmlu_pro,musr --model qwen2.5-7b

可选：
  --aggregate-calib-runs
      同一 loss 有多个「Method 带 _YYYYMMDD_HHMMSS」的 calib 结果时，对 Calib 侧主指标做
      跨运行均值 ± 样本标准差（与 lm-eval 单次 stderr 不同，见 main() 内说明）。

不确定度说明：
  - lm-eval 的 acc_norm_stderr 等是「单次评测」的标准误，脚本会单独成表，并可合并为「主±stderr」列。
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from numbers import Number
import pandas as pd
import numpy as np
import re

# ============================================================================
# 配置：选择评测类型
# ============================================================================
# 可选值: "arc", "bbh", "mmlu_pro", "math_hard", "musr" 或 "gsm8k"
EVAL_TYPE = "bbh"
# ============================================================================

# MuSR 子任务全量测试样本数（用于 acc_norm 加权平均：分母 250+256+250=756）
MUSR_SUBTASK_SAMPLE_WEIGHTS: Dict[str, int] = {
    "leaderboard_musr_murder_mysteries": 250,
    "leaderboard_musr_object_placements": 256,
    "leaderboard_musr_team_allocation": 250,
}


def load_json_results(file_path: Path) -> Optional[Dict]:
    """加载JSON评测结果文件。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # 文件不存在时不打印警告，静默返回None
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析JSON文件 {file_path}: {e}")
        return None


def _pick_acc_from_sample(ex: dict) -> Optional[float]:
    """
    从 lm-eval 的 samples jsonl 单条样本里提取 0/1 acc。
    兼容两种常见结构：
    - 顶层 "acc"
    - "metrics" 里 "acc" / "acc,none" / 其他以 "acc" 开头的 key
    """
    if "acc" in ex and ex["acc"] is not None:
        try:
            return float(ex["acc"])
        except (TypeError, ValueError):
            return None

    metrics = ex.get("metrics") or {}
    for k in ("acc", "acc,none"):
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                return None

    for k, v in metrics.items():
        if isinstance(k, str) and k.startswith("acc") and v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _summarize_bernoulli(xs: List[float]) -> Dict[str, Optional[float]]:
    n = len(xs)
    if n == 0:
        return {"n": 0, "acc": None, "var": None, "stderr": None}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    stderr = math.sqrt(mean * (1 - mean) / n)  # 0/1 Bernoulli
    return {"n": n, "acc": mean, "var": var, "stderr": stderr}


def compute_mmlu_pro_breakdown_from_samples(samples_jsonl: Path) -> Optional[Dict]:
    """
    复用 mmlu_pro_sub_tasks.py 的聚合逻辑：
    从 samples_leaderboard_mmlu_pro_*.jsonl 流式统计 by_category / by_src。
    """
    if not samples_jsonl.exists():
        return None

    # 按 category 聚合
    cat_bucket = defaultdict(list)
    cat_order: List[str] = []
    cat_seen = set()

    # 按 src 聚合（更细粒度）
    src_bucket = defaultdict(list)
    src_order: List[str] = []
    src_seen = set()

    try:
        with open(samples_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    # 跳过坏行，不中断
                    continue

                doc = ex.get("doc") or {}
                category = doc.get("category") or "UNKNOWN"
                src = doc.get("src") or "UNKNOWN"

                acc = _pick_acc_from_sample(ex)
                if acc is None:
                    continue

                if category not in cat_seen:
                    cat_seen.add(category)
                    cat_order.append(category)
                cat_bucket[category].append(acc)

                if src not in src_seen:
                    src_seen.add(src)
                    src_order.append(src)
                src_bucket[src].append(acc)
    except FileNotFoundError:
        return None

    out = {
        "by_category": {c: _summarize_bernoulli(cat_bucket[c]) for c in cat_order},
        "by_src": {s: _summarize_bernoulli(src_bucket[s]) for s in src_order},
    }
    return out


def load_or_compute_mmlu_pro_breakdown(run_dir: Path) -> Tuple[Optional[Dict], List[str]]:
    """
    优先读取 run_dir 下的 mmlu_pro_breakdown.json。
    如果没有，则尝试从同目录下的 samples_leaderboard_mmlu_pro_*.jsonl 计算。
    返回 (breakdown, missing_file_hints)，missing 不包含本机绝对路径。
    """
    missing: List[str] = []

    breakdown_json = run_dir / "mmlu_pro_breakdown.json"
    if breakdown_json.exists():
        data = load_json_results(breakdown_json)
        if data:
            return data, missing

    sample_files = sorted(run_dir.glob("samples_leaderboard_mmlu_pro_*.jsonl"), key=lambda p: p.name)
    if not sample_files:
        missing.append("mmlu_pro_breakdown.json")
        missing.append("samples_leaderboard_mmlu_pro_*.jsonl")
        return None, missing

    samples_jsonl = sample_files[-1]
    breakdown = compute_mmlu_pro_breakdown_from_samples(samples_jsonl)
    if breakdown is None:
        missing.append(samples_jsonl.name)
    return breakdown, missing


def _subtask_names_for_group(data: Dict, group_key: str, task_prefix: str) -> List[str]:
    """
    列出「多种评测情形」对应的子任务名：优先 group_subtasks[group_key]，
    否则取 results 下所有以 task_prefix 开头且不等于 group_key 的键。
    """
    results = data.get("results") or {}
    if not isinstance(results, dict):
        return []
    gs = (data.get("group_subtasks") or {}).get(group_key)
    names: List[str] = []
    if isinstance(gs, list) and len(gs) > 0:
        names = [
            n
            for n in gs
            if isinstance(n, str) and n in results and isinstance(results[n], dict)
        ]
    if not names:
        for k in sorted(results.keys()):
            if (
                isinstance(k, str)
                and k.startswith(task_prefix)
                and k != group_key
                and isinstance(results[k], dict)
            ):
                names.append(k)
    return names


def _aggregate_leaderboard_metrics_from_results(
    data: Dict,
    group_key: str,
    task_prefix: str,
    metric_pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    """
    单个 JSON 的 results 内：
    - 若存在多个子任务（多种情形），对每个 canonical 指标，在子任务上对原始字段取算术平均；
      stderr 类字段同样对各子任务取平均（与 acc 相同操作）。
    - 若无子任务，则只读 group_key 对应行的汇总值（单情形）。
    """
    results = data.get("results") or {}
    if not isinstance(results, dict):
        return {}

    sub_names = _subtask_names_for_group(data, group_key, task_prefix)
    out: Dict[str, float] = {}

    if sub_names:
        for canon, raw_key in metric_pairs:
            vals: List[float] = []
            for sn in sub_names:
                row = results.get(sn)
                if not isinstance(row, dict):
                    continue
                v = row.get(raw_key)
                if isinstance(v, Number) and np.isfinite(v):
                    vals.append(float(v))
            if vals:
                out[canon] = float(np.mean(vals))
        return out

    group = results.get(group_key)
    if isinstance(group, dict):
        for canon, raw_key in metric_pairs:
            v = group.get(raw_key)
            if isinstance(v, Number) and np.isfinite(v):
                out[canon] = float(v)
    return out


def _aggregate_musr_metrics_with_sample_weights(data: Dict) -> Dict[str, float]:
    """
    MuSR 汇总 acc_norm：按各子任务测试集样本数加权
        (250*murder_mysteries + 256*object_placements + 250*team_allocation) / 756
    若仅部分子任务有数值，则只对「有权重的、且出现结果的」子任务用其权重做归一（分母为有效权重之和）。

    acc_norm_stderr：各子任务估计视为独立，对加权均值用
        sqrt(sum_i (w_i/W)^2 * se_i^2) 合成（非对 stderr 做算术平均）。
    若无子任务或无法加权，退回读 leaderboard_musr 汇总行（与旧逻辑一致）。
    """
    results = data.get("results") or {}
    if not isinstance(results, dict):
        return {}

    sub_names = _subtask_names_for_group(data, "leaderboard_musr", "leaderboard_musr_")
    acc_key = "acc_norm,none"
    se_key = "acc_norm_stderr,none"

    if sub_names:
        terms: List[Tuple[float, float, float]] = []
        for sn in sub_names:
            w = float(MUSR_SUBTASK_SAMPLE_WEIGHTS.get(sn, 0.0))
            if w <= 0:
                continue
            row = results.get(sn)
            if not isinstance(row, dict):
                continue
            a = row.get(acc_key)
            if not isinstance(a, Number) or not np.isfinite(a):
                continue
            se_v = row.get(se_key)
            se_f = (
                float(se_v)
                if isinstance(se_v, Number) and np.isfinite(se_v)
                else float("nan")
            )
            terms.append((w, float(a), se_f))

        if terms:
            W = sum(t[0] for t in terms)
            acc_w = sum(t[0] * t[1] for t in terms) / W
            se_sq = 0.0
            for w, _a, se in terms:
                if np.isfinite(se) and W > 0:
                    se_sq += (w / W) ** 2 * (se ** 2)
            out_m: Dict[str, float] = {"acc_norm": acc_w}
            if se_sq > 0 and np.isfinite(se_sq):
                out_m["acc_norm_stderr"] = float(math.sqrt(se_sq))
            return out_m

    return _aggregate_leaderboard_metrics_from_results(
        data,
        group_key="leaderboard_musr",
        task_prefix="leaderboard_musr_",
        metric_pairs=[
            ("acc_norm", "acc_norm,none"),
            ("acc_norm_stderr", "acc_norm_stderr,none"),
        ],
    )


def extract_metrics(data: Dict, eval_type: str = "arc") -> Dict[str, float]:
    """从评测结果中提取指标（leaderboard 类：多子任务时 bbh 等算术平均；musr 为样本加权）。"""
    if not data or "results" not in data:
        return {}

    results = data["results"]
    metrics: Dict[str, float] = {}

    if eval_type == "bbh":
        metrics = _aggregate_leaderboard_metrics_from_results(
            data,
            group_key="leaderboard_bbh",
            task_prefix="leaderboard_bbh_",
            metric_pairs=[
                ("acc_norm", "acc_norm,none"),
                ("acc_norm_stderr", "acc_norm_stderr,none"),
            ],
        )
    elif eval_type == "musr":
        metrics = _aggregate_musr_metrics_with_sample_weights(data)
    elif eval_type == "math_hard":
        metrics = _aggregate_leaderboard_metrics_from_results(
            data,
            group_key="leaderboard_math_hard",
            task_prefix="leaderboard_math_hard_",
            metric_pairs=[
                ("exact_match", "exact_match,none"),
                ("exact_match_stderr", "exact_match_stderr,none"),
            ],
        )
    elif eval_type == "mmlu_pro":
        metrics = _aggregate_leaderboard_metrics_from_results(
            data,
            group_key="leaderboard_mmlu_pro",
            task_prefix="leaderboard_mmlu_pro_",
            metric_pairs=[
                ("acc", "acc,none"),
                ("acc_stderr", "acc_stderr,none"),
            ],
        )
    else:
        # 对于arc，遍历所有任务
        for task_name, task_results in results.items():
            for key, value in task_results.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
    
    return metrics


def get_method_name(filename: str, eval_type: str = "arc") -> str:
    """从文件名中提取方法名。"""
    # 支持可选时间戳后缀：..._{YYYYMMDD_HHMMSS}.json
    def strip_ts(s: str) -> str:
        m = re.match(r"^(.*)_\d{8}_\d{6}$", s)
        return m.group(1) if m else s

    if eval_type == "arc":
        # out_arc_bce.json / out_arc_bce_20260121_123456.json -> bce
        prefix = "out_arc_"
        if filename.startswith(prefix) and filename.endswith(".json"):
            core = filename[len(prefix):-5]
            return strip_ts(core)
    elif eval_type == "gsm8k":
        # out_gsm8k_bce.json / out_gsm8k_bce_20260121_123456.json -> bce
        prefix = "out_gsm8k_"
        if filename.startswith(prefix) and filename.endswith(".json"):
            core = filename[len(prefix):-5]
            return strip_ts(core)
    elif eval_type in ("bbh", "mmlu_pro", "math_hard", "musr"):
        # lb2_{eval_type}_bce.json / lb2_{eval_type}_bce_20260121_123456.json -> bce
        prefix = f"lb2_{eval_type}_"
        if filename.startswith(prefix) and filename.endswith(".json"):
            core = filename[len(prefix):-5]
            return strip_ts(core)
    return filename


def find_latest_results_file(method_dir: Path) -> Optional[Path]:
    """在方法目录下找到最新的 results_*.json（单目录多实验时请用 enumerate_eval_results_under_run_dir）。"""
    entries = enumerate_eval_results_under_run_dir(method_dir)
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0][1]
    # 兼容：多个实验目录时仍返回「最新」的一个，避免遗漏调用方崩溃；对比表应走 enumerate 逻辑
    return max((p for _, p in entries), key=lambda f: (f.name, f.stat().st_mtime))


def parse_ablation_slug_from_results_path(results_path: Path) -> str:
    """
    从 results_*.json 向上查找目录名，解析
    ``{family}-{loss}-calib-<消融片段>-{YYYYMMDD_HHMMSS}`` 中的 <消融片段>。
    例如：...-dpo-calib-dbema0p5-20260325_023455... -> ``dbema0p5``；
    ``...-calib-seed1-20260326_023439...`` -> ``seed1``；
    多段用 ``-`` 连接：``dbema0p5-seed1``。
    若路径中无 ``-calib-`` 或未匹配到时间戳，返回空串（保持与旧布局兼容）。
    """
    cur: Optional[Path] = results_path.parent
    for _ in range(32):
        if cur is None:
            break
        name = cur.name
        m = re.search(r"-calib-(.+?)-(\d{8}_\d{6})", name)
        if m:
            raw = m.group(1).strip("-")
            return raw if raw else ""
        cur = cur.parent
    return ""


def enumerate_eval_results_under_run_dir(run_dir: Optional[Path]) -> List[Tuple[str, Path]]:
    """
    在某个 ``lb2_*_dpo_*.json`` 运行目录下，枚举所有评测落盘（按 results 父目录去重）。
    返回 [(ablation_slug, results_json), ...]；ablation_slug 可能为空（无消融或旧路径）。
    同一父目录下多个 results_*.json 时取按文件名+mtime 最新的一条。
    """
    if run_dir is None or not run_dir.exists():
        return []
    files = list(run_dir.glob("**/results_*.json"))
    if not files:
        return []
    by_parent: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        by_parent[str(f.parent.resolve())].append(f)
    rows: List[Tuple[str, Path]] = []
    for parent in sorted(by_parent.keys()):
        group = by_parent[parent]
        group.sort(key=lambda p: (p.name, p.stat().st_mtime), reverse=True)
        best = group[0]
        slug = parse_ablation_slug_from_results_path(best)
        rows.append((slug, best))
    rows.sort(key=lambda r: (r[0], r[1].name))
    return rows


def format_method_label_with_ablation(
    method_name: str, ts: Optional[str], ablation_slug: str
) -> str:
    """
    Method 列：``LOSS`` / ``LOSS_ts`` / ``LOSS_ablation_ts``（ablation 中 ``-`` 换成 ``_``）。
    """
    parts: List[str] = [method_name.upper()]
    slug = (ablation_slug or "").strip()
    if slug:
        parts.append(slug.replace("-", "_"))
    if ts:
        parts.append(ts)
    return "_".join(parts)


def _pick_base_results_file(
    base_entries: List[Tuple[str, Path]], calib_ablation_slug: str
) -> Optional[Path]:
    """base 侧多份结果时：优先与 calib 消融 slug 同名匹配，否则用第一份。"""
    if not base_entries:
        return None
    if len(base_entries) == 1:
        return base_entries[0][1]
    want = (calib_ablation_slug or "").strip()
    if want:
        for s, p in base_entries:
            if s == want:
                return p
    return base_entries[0][1]


def _compare_dirbased_eval_runs(
    base_dir: Path, calib_dir: Path, eval_type: str
) -> List[Dict]:
    """
    用于 bbh / musr / math_hard / mmlu_pro：支持同一 run 目录下多个消融子目录各有一份 results。
    """
    base_task = base_dir / eval_type
    calib_task = calib_dir / eval_type
    base_method_dirs = _select_latest_run_dir_by_method(base_task, eval_type)
    calib_runs = _collect_runs_by_method_in_dir(calib_task, eval_type)
    all_method_names = set(base_method_dirs.keys()) | set(calib_runs.keys())

    out_rows: List[Dict] = []
    for method_name in sorted(all_method_names):
        base_method_dir = base_method_dirs.get(method_name) if base_task.exists() else None
        base_entries = (
            enumerate_eval_results_under_run_dir(base_method_dir)
            if base_method_dir
            else []
        )

        items = calib_runs.get(method_name) or []
        if not items:
            items = [(None, None)]  # type: ignore[list-item]
        items = sorted(
            items,
            key=lambda x: ((x[0] or ""), x[1].name if x[1] else ""),  # type: ignore[union-attr]
            reverse=True,
        )

        for ts, calib_method_dir in items:
            if calib_method_dir:
                calib_entries = enumerate_eval_results_under_run_dir(calib_method_dir)
                if not calib_entries:
                    calib_entries = [("", None)]  # type: ignore[list-item]
            else:
                calib_entries = [("", None)]  # type: ignore[list-item]

            for ablation_slug, calib_file in calib_entries:
                if base_entries:
                    base_file = _pick_base_results_file(base_entries, ablation_slug)
                elif base_method_dir:
                    base_file = find_latest_results_file(base_method_dir)
                else:
                    base_file = None

                method_label = format_method_label_with_ablation(method_name, ts, ablation_slug)

                base_metrics: Dict[str, float] = {}
                if base_file is not None and base_file.exists():
                    base_data = load_json_results(base_file)
                    base_metrics = extract_metrics(base_data, eval_type) if base_data else {}

                calib_metrics: Dict[str, float] = {}
                if calib_file is not None and calib_file.exists():
                    calib_data = load_json_results(calib_file)
                    calib_metrics = extract_metrics(calib_data, eval_type) if calib_data else {}

                if not base_metrics and not calib_metrics:
                    continue
                out_rows.extend(
                    _compare_metric_dicts_to_rows(method_label, base_metrics, calib_metrics)
                )
    return out_rows


def _build_breakdown_method_file_map(
    base_dir: Path, calib_dir: Path, eval_sub: str
) -> Dict[str, Tuple[Optional[Path], Optional[Path]]]:
    """
    供 mmlu_pro breakdown / bbh|math_hard 子任务表使用：与主对比表一致地展开消融维度。
    """
    base_latest_dirs = _select_latest_run_dir_by_method(base_dir / eval_sub, eval_sub)
    calib_runs = _collect_runs_by_method_in_dir(calib_dir / eval_sub, eval_sub)
    method_files: Dict[str, Tuple[Optional[Path], Optional[Path]]] = {}
    all_methods = set(base_latest_dirs.keys()) | set(calib_runs.keys())
    for m in sorted(all_methods):
        b_dir = base_latest_dirs.get(m)
        b_entries = enumerate_eval_results_under_run_dir(b_dir) if b_dir else []
        items = calib_runs.get(m) or []
        if not items:
            bf: Optional[Path] = None
            if b_entries:
                bf = _pick_base_results_file(b_entries, "")
            elif b_dir:
                bf = find_latest_results_file(b_dir)
            method_files[m.upper()] = (bf, None)
            continue
        items = sorted(
            items,
            key=lambda x: ((x[0] or ""), x[1].name if x[1] else ""),  # type: ignore[union-attr]
            reverse=True,
        )
        for ts, c_dir in items:
            cent = enumerate_eval_results_under_run_dir(c_dir) if c_dir else []
            if not cent:
                cent = [("", None)]  # type: ignore[list-item]
            for ablation_slug, c_file in cent:
                label = format_method_label_with_ablation(m, ts, ablation_slug)
                if b_entries:
                    b_file = _pick_base_results_file(b_entries, ablation_slug)
                elif b_dir:
                    b_file = find_latest_results_file(b_dir)
                else:
                    b_file = None
                method_files[label] = (b_file, c_file)
    return method_files


def _extract_last_timestamp_from_name(name: str) -> Optional[str]:
    matches = re.findall(r"\d{8}_\d{6}", name)
    return matches[-1] if matches else None


def _pick_latest_path_by_timestamp(paths: List[Path]) -> Optional[Path]:
    """
    Pick the latest path by the last YYYYMMDD_HHMMSS found in its name.
    If none have timestamp, fall back to lexicographic name.
    """
    if not paths:
        return None
    scored = []
    for p in paths:
        ts = _extract_last_timestamp_from_name(p.name) or ""
        scored.append((ts, p.name, p))
    scored.sort(reverse=True)
    return scored[0][2]


def _collect_runs_by_method_in_dir(task_dir: Path, eval_type: str) -> Dict[str, List[Tuple[Optional[str], Path]]]:
    """
    For dir-based evals (bbh/mmlu_pro/math_hard/musr): collect run directories by method.
    Returns {method: [(timestamp, run_dir), ...]} where timestamp may be None for legacy names.
    """
    runs: Dict[str, List[Tuple[Optional[str], Path]]] = defaultdict(list)
    if not task_dir.exists():
        return runs
    for run_dir in task_dir.glob(f"lb2_{eval_type}_*.json"):
        method = get_method_name(run_dir.name, eval_type)
        core = run_dir.name[len(f"lb2_{eval_type}_"):-5]
        ts = _extract_last_timestamp_from_name(core)
        runs[method].append((ts, run_dir))
    return runs


def _select_latest_run_dir_by_method(task_dir: Path, eval_type: str) -> Dict[str, Path]:
    runs = _collect_runs_by_method_in_dir(task_dir, eval_type)
    out: Dict[str, Path] = {}
    for method, items in runs.items():
        out[method] = _pick_latest_path_by_timestamp([p for _, p in items])  # type: ignore[arg-type]
    return {k: v for k, v in out.items() if v is not None}


def _collect_files_by_method_in_dir(root_dir: Path, eval_type: str) -> Dict[str, List[Tuple[Optional[str], Path]]]:
    """
    For file-based evals (arc/gsm8k): collect output json files by method.
    Returns {method: [(timestamp, file_path), ...]}.
    """
    runs: Dict[str, List[Tuple[Optional[str], Path]]] = defaultdict(list)
    if not root_dir.exists():
        return runs
    if eval_type == "arc":
        pattern = "**/out_arc_*.json"
        prefix = "out_arc_"
    elif eval_type == "gsm8k":
        pattern = "**/out_gsm8k_*.json"
        prefix = "out_gsm8k_"
    else:
        return runs

    for f in root_dir.glob(pattern):
        core = f.name[len(prefix):-5]
        ts = _extract_last_timestamp_from_name(core)
        method = get_method_name(f.name, eval_type)
        runs[method].append((ts, f))
    return runs


def _select_latest_file_by_method(root_dir: Path, eval_type: str) -> Dict[str, Path]:
    runs = _collect_files_by_method_in_dir(root_dir, eval_type)
    out: Dict[str, Path] = {}
    for method, items in runs.items():
        out[method] = _pick_latest_path_by_timestamp([p for _, p in items])  # type: ignore[arg-type]
    return {k: v for k, v in out.items() if v is not None}


def _compare_metric_dicts_to_rows(
    method_name: str, base_metrics: Dict[str, float], calib_metrics: Dict[str, float]
) -> List[Dict]:
    rows: List[Dict] = []
    all_metrics = set(base_metrics.keys()) | set(calib_metrics.keys())
    for metric in sorted(all_metrics):
        base_value = base_metrics.get(metric, None)
        calib_value = calib_metrics.get(metric, None)
        if base_value is not None and calib_value is not None:
            diff = calib_value - base_value
            diff_percent = (diff / base_value * 100) if base_value != 0 else 0
        else:
            diff = None
            diff_percent = None
        rows.append(
            {
                "Method": method_name.upper(),
                "Metric": metric,
                "Base": base_value,
                "Calib": calib_value,
                "Change": diff,
                "Change (%)": diff_percent,
            }
        )
    return rows


def _write_multi_calib_comparison_jsons(
    eval_type: str,
    base_dir: Path,
    calib_dir: Path,
    output_dir: Path,
) -> List[Path]:
    """
    当 calib 侧存在带时间戳的输出（*_YYYYMMDD_HHMMSS.json）时：
    - 每个 method 的每个 calib 时间戳都单独与 base（该 method 最新一份）对比
    - 输出对比结果为 json（不影响现有 md 输出）
    - 如果检测不到时间戳：不做任何事，保持兼容
    """
    written: List[Path] = []

    def dump_json(out_path: Path, payload: Dict) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as w:
            json.dump(payload, w, ensure_ascii=False, indent=2)

    if eval_type in ("arc", "gsm8k"):
        base_latest = _select_latest_file_by_method(base_dir, eval_type)
        calib_runs = _collect_files_by_method_in_dir(calib_dir, eval_type)
        for method, items in calib_runs.items():
            for ts, calib_file in items:
                if not ts:
                    continue
                base_file = base_latest.get(method)
                base_metrics: Dict[str, float] = {}
                calib_metrics: Dict[str, float] = {}
                if base_file and base_file.exists():
                    base_data = load_json_results(base_file)
                    base_metrics = extract_metrics(base_data, eval_type) if base_data else {}
                if calib_file and calib_file.exists():
                    calib_data = load_json_results(calib_file)
                    calib_metrics = extract_metrics(calib_data, eval_type) if calib_data else {}
                rows = _compare_metric_dicts_to_rows(method, base_metrics, calib_metrics)
                out_path = output_dir / f"calib_comparison_{eval_type}_{method}_{ts}.json"
                dump_json(
                    out_path,
                    {
                        "eval_type": eval_type,
                        "method": method,
                        "calib_timestamp": ts,
                        "rows": rows,
                    },
                )
                written.append(out_path)

    elif eval_type in ("bbh", "mmlu_pro", "math_hard", "musr"):
        base_task_dir = base_dir / eval_type
        calib_task_dir = calib_dir / eval_type

        base_latest_dirs = _select_latest_run_dir_by_method(base_task_dir, eval_type)
        calib_runs = _collect_runs_by_method_in_dir(calib_task_dir, eval_type)

        for method, items in calib_runs.items():
            for ts, calib_run_dir in items:
                if not ts:
                    continue

                base_run_dir = base_latest_dirs.get(method)
                base_entries = (
                    enumerate_eval_results_under_run_dir(base_run_dir) if base_run_dir else []
                )
                calib_entries = enumerate_eval_results_under_run_dir(calib_run_dir)
                if not calib_entries:
                    continue

                for ablation_slug, calib_file in calib_entries:
                    if base_entries:
                        base_file = _pick_base_results_file(base_entries, ablation_slug)
                    elif base_run_dir:
                        base_file = find_latest_results_file(base_run_dir)
                    else:
                        base_file = None

                    base_metrics: Dict[str, float] = {}
                    calib_metrics: Dict[str, float] = {}

                    if base_file and base_file.exists():
                        base_data = load_json_results(base_file)
                        base_metrics = extract_metrics(base_data, eval_type) if base_data else {}

                    if calib_file and calib_file.exists():
                        calib_data = load_json_results(calib_file)
                        calib_metrics = extract_metrics(calib_data, eval_type) if calib_data else {}

                    rows = _compare_metric_dicts_to_rows(method, base_metrics, calib_metrics)
                    slug_key = re.sub(r"[^\w.-]+", "_", ablation_slug) if ablation_slug else ""
                    if slug_key:
                        out_path = (
                            output_dir
                            / f"calib_comparison_{eval_type}_{method}_{ts}_{slug_key}.json"
                        )
                    else:
                        out_path = output_dir / f"calib_comparison_{eval_type}_{method}_{ts}.json"
                    dump_json(
                        out_path,
                        {
                            "eval_type": eval_type,
                            "method": method,
                            "calib_timestamp": ts,
                            "ablation_slug": ablation_slug or None,
                            "rows": rows,
                        },
                    )
                    written.append(out_path)

    return written


def _collect_calib_timestamps(eval_type: str, calib_dir: Path) -> List[str]:
    """
    Collect all calib-side timestamps (YYYYMMDD_HHMMSS) for the given eval_type.
    If none found, returns [] and caller should keep legacy single-md behavior.
    """
    ts_set = set()
    if eval_type in ("arc", "gsm8k"):
        runs = _collect_files_by_method_in_dir(calib_dir, eval_type)
        for _, items in runs.items():
            for ts, _ in items:
                if ts:
                    ts_set.add(ts)
    elif eval_type in ("bbh", "mmlu_pro", "math_hard", "musr"):
        task_dir = calib_dir / eval_type
        runs = _collect_runs_by_method_in_dir(task_dir, eval_type)
        for _, items in runs.items():
            for ts, _ in items:
                if ts:
                    ts_set.add(ts)
    return sorted(ts_set)


def _select_calib_for_timestamp(
    eval_type: str, calib_dir: Path, calib_ts: str
) -> Dict[str, Path]:
    """
    Select calib output (file or run_dir) per method for a specific timestamp.
    Returns {method: path} where path is a json file (arc/gsm8k) or a run directory (bbh/mmlu_pro/math_hard).
    """
    out: Dict[str, Path] = {}
    if eval_type in ("arc", "gsm8k"):
        runs = _collect_files_by_method_in_dir(calib_dir, eval_type)
        for method, items in runs.items():
            cands = [p for ts, p in items if ts == calib_ts]
            if cands:
                out[method] = _pick_latest_path_by_timestamp(cands)  # type: ignore[assignment]
    elif eval_type in ("bbh", "mmlu_pro", "math_hard", "musr"):
        task_dir = calib_dir / eval_type
        runs = _collect_runs_by_method_in_dir(task_dir, eval_type)
        for method, items in runs.items():
            cands = [p for ts, p in items if ts == calib_ts]
            if cands:
                out[method] = _pick_latest_path_by_timestamp(cands)  # type: ignore[assignment]
    return {k: v for k, v in out.items() if v is not None}


def _compare_results_for_calib_timestamp(
    base_dir: Path, calib_dir: Path, eval_type: str, calib_ts: str
) -> pd.DataFrame:
    """
    Compare base vs calib for a specific calib timestamp.
    Base side uses the latest available output per method (legacy behavior),
    Calib side is filtered to the given timestamp.
    """
    base_dir = Path(base_dir)
    calib_dir = Path(calib_dir)

    rows: List[Dict] = []

    if eval_type in ("arc", "gsm8k"):
        base_latest = _select_latest_file_by_method(base_dir, eval_type)
        calib_sel = _select_calib_for_timestamp(eval_type, calib_dir, calib_ts)
        all_methods = set(base_latest.keys()) | set(calib_sel.keys())
        for method in sorted(all_methods):
            base_file = base_latest.get(method)
            calib_file = calib_sel.get(method)
            base_metrics: Dict[str, float] = {}
            calib_metrics: Dict[str, float] = {}
            if base_file and base_file.exists():
                base_data = load_json_results(base_file)
                base_metrics = extract_metrics(base_data, eval_type) if base_data else {}
            if calib_file and calib_file.exists():
                calib_data = load_json_results(calib_file)
                calib_metrics = extract_metrics(calib_data, eval_type) if calib_data else {}
            rows.extend(_compare_metric_dicts_to_rows(method, base_metrics, calib_metrics))

    elif eval_type in ("bbh", "mmlu_pro", "math_hard", "musr"):
        base_task_dir = base_dir / eval_type
        base_latest_dirs = _select_latest_run_dir_by_method(base_task_dir, eval_type)
        calib_sel = _select_calib_for_timestamp(eval_type, calib_dir, calib_ts)
        all_methods = set(base_latest_dirs.keys()) | set(calib_sel.keys())
        for method in sorted(all_methods):
            base_run_dir = base_latest_dirs.get(method)
            calib_run_dir = calib_sel.get(method)
            base_entries = (
                enumerate_eval_results_under_run_dir(base_run_dir) if base_run_dir else []
            )
            calib_entries = (
                enumerate_eval_results_under_run_dir(calib_run_dir) if calib_run_dir else []
            )
            if not calib_entries:
                calib_entries = [("", None)]  # type: ignore[list-item]

            for ablation_slug, calib_file in calib_entries:
                if base_entries:
                    base_file = _pick_base_results_file(base_entries, ablation_slug)
                elif base_run_dir:
                    base_file = find_latest_results_file(base_run_dir)
                else:
                    base_file = None

                method_label = format_method_label_with_ablation(method, calib_ts, ablation_slug)

                base_metrics: Dict[str, float] = {}
                calib_metrics: Dict[str, float] = {}

                if base_file and base_file.exists():
                    base_data = load_json_results(base_file)
                    base_metrics = extract_metrics(base_data, eval_type) if base_data else {}

                if calib_file and calib_file.exists():
                    calib_data = load_json_results(calib_file)
                    calib_metrics = extract_metrics(calib_data, eval_type) if calib_data else {}

                rows.extend(
                    _compare_metric_dicts_to_rows(method_label, base_metrics, calib_metrics)
                )

    else:
        # Fallback: use legacy comparator
        return compare_results(base_dir, calib_dir, eval_type)

    return pd.DataFrame(rows)


def _has_legacy_calib_outputs(eval_type: str, calib_dir: Path) -> bool:
    """
    Return True if calib side has any outputs WITHOUT timestamp suffix.
    """
    if eval_type in ("arc", "gsm8k"):
        runs = _collect_files_by_method_in_dir(calib_dir, eval_type)
        for _, items in runs.items():
            for ts, _ in items:
                if not ts:
                    return True
        return False
    if eval_type in ("bbh", "mmlu_pro", "math_hard", "musr"):
        task_dir = calib_dir / eval_type
        runs = _collect_runs_by_method_in_dir(task_dir, eval_type)
        for _, items in runs.items():
            for ts, _ in items:
                if not ts:
                    return True
        return False
    return False


def _select_legacy_calib_per_method(eval_type: str, calib_dir: Path) -> Dict[str, Path]:
    """
    Select calib output per method among entries WITHOUT timestamp.
    For file-based evals: returns {method: json_file}
    For dir-based evals: returns {method: run_dir}
    """
    out: Dict[str, Path] = {}
    if eval_type in ("arc", "gsm8k"):
        runs = _collect_files_by_method_in_dir(calib_dir, eval_type)
        for method, items in runs.items():
            cands = [p for ts, p in items if not ts]
            if cands:
                # legacy files don't have ts; pick latest by name
                out[method] = _pick_latest_path_by_timestamp(cands)  # type: ignore[assignment]
        return out
    if eval_type in ("bbh", "mmlu_pro", "math_hard", "musr"):
        task_dir = calib_dir / eval_type
        runs = _collect_runs_by_method_in_dir(task_dir, eval_type)
        for method, items in runs.items():
            cands = [p for ts, p in items if not ts]
            if cands:
                out[method] = _pick_latest_path_by_timestamp(cands)  # type: ignore[assignment]
        return out
    return out


def _compare_results_for_legacy_calib_only(base_dir: Path, calib_dir: Path, eval_type: str) -> pd.DataFrame:
    """
    Compare base vs calib but ONLY using calib outputs without timestamp.
    Base side still uses latest available per method.
    """
    base_dir = Path(base_dir)
    calib_dir = Path(calib_dir)

    rows: List[Dict] = []

    if eval_type in ("arc", "gsm8k"):
        base_latest = _select_latest_file_by_method(base_dir, eval_type)
        calib_sel = _select_legacy_calib_per_method(eval_type, calib_dir)
        all_methods = set(base_latest.keys()) | set(calib_sel.keys())
        for method in sorted(all_methods):
            base_file = base_latest.get(method)
            calib_file = calib_sel.get(method)
            base_metrics: Dict[str, float] = {}
            calib_metrics: Dict[str, float] = {}
            if base_file and base_file.exists():
                base_data = load_json_results(base_file)
                base_metrics = extract_metrics(base_data, eval_type) if base_data else {}
            if calib_file and calib_file.exists():
                calib_data = load_json_results(calib_file)
                calib_metrics = extract_metrics(calib_data, eval_type) if calib_data else {}
            rows.extend(_compare_metric_dicts_to_rows(method, base_metrics, calib_metrics))
        return pd.DataFrame(rows)

    if eval_type in ("bbh", "mmlu_pro", "math_hard", "musr"):
        base_task_dir = base_dir / eval_type
        base_latest_dirs = _select_latest_run_dir_by_method(base_task_dir, eval_type)
        calib_sel = _select_legacy_calib_per_method(eval_type, calib_dir)
        all_methods = set(base_latest_dirs.keys()) | set(calib_sel.keys())
        for method in sorted(all_methods):
            base_run_dir = base_latest_dirs.get(method)
            calib_run_dir = calib_sel.get(method)
            base_entries = (
                enumerate_eval_results_under_run_dir(base_run_dir) if base_run_dir else []
            )
            calib_entries = (
                enumerate_eval_results_under_run_dir(calib_run_dir) if calib_run_dir else []
            )
            if not calib_entries:
                calib_entries = [("", None)]  # type: ignore[list-item]

            for ablation_slug, calib_file in calib_entries:
                if base_entries:
                    base_file = _pick_base_results_file(base_entries, ablation_slug)
                elif base_run_dir:
                    base_file = find_latest_results_file(base_run_dir)
                else:
                    base_file = None

                method_label = format_method_label_with_ablation(method, None, ablation_slug)

                base_metrics: Dict[str, float] = {}
                calib_metrics: Dict[str, float] = {}

                if base_file and base_file.exists():
                    base_data = load_json_results(base_file)
                    base_metrics = extract_metrics(base_data, eval_type) if base_data else {}

                if calib_file and calib_file.exists():
                    calib_data = load_json_results(calib_file)
                    calib_metrics = extract_metrics(calib_data, eval_type) if calib_data else {}

                rows.extend(
                    _compare_metric_dicts_to_rows(method_label, base_metrics, calib_metrics)
                )
        return pd.DataFrame(rows)

    # Fallback
    return compare_results(base_dir, calib_dir, eval_type)


def _find_mmlu_pro_latest_method_files(
    base_dir: Path, calib_dir: Path
) -> Dict[str, Tuple[Optional[Path], Optional[Path]]]:
    """
    返回 {method_name: (base_results_json, calib_results_json)}，仅用于 mmlu_pro。
    method_name 为 lb2_mmlu_pro_*.json 的后缀（小写）。
    """
    base_task_dir = base_dir / "mmlu_pro"
    calib_task_dir = calib_dir / "mmlu_pro"

    base_latest_dirs = _select_latest_run_dir_by_method(base_task_dir, "mmlu_pro")
    calib_latest_dirs = _select_latest_run_dir_by_method(calib_task_dir, "mmlu_pro")

    all_method_names = set(base_latest_dirs.keys()) | set(calib_latest_dirs.keys())
    out: Dict[str, Tuple[Optional[Path], Optional[Path]]] = {}
    for method_name in sorted(all_method_names):
        base_method_dir = base_latest_dirs.get(method_name)
        calib_method_dir = calib_latest_dirs.get(method_name)
        base_file = find_latest_results_file(base_method_dir) if base_method_dir else None
        calib_file = find_latest_results_file(calib_method_dir) if calib_method_dir else None
        out[method_name] = (base_file, calib_file)
    return out


def _find_group_latest_method_files(
    base_dir: Path, calib_dir: Path, eval_type: str
) -> Dict[str, Tuple[Optional[Path], Optional[Path]]]:
    """
    返回 {method_name: (base_results_json, calib_results_json)}，用于 bbh / math_hard。
    method_name 为 lb2_{eval_type}_*.json 的后缀（小写）。
    """
    base_task_dir = base_dir / eval_type
    calib_task_dir = calib_dir / eval_type

    base_latest_dirs = _select_latest_run_dir_by_method(base_task_dir, eval_type)
    calib_latest_dirs = _select_latest_run_dir_by_method(calib_task_dir, eval_type)

    all_method_names = set(base_latest_dirs.keys()) | set(calib_latest_dirs.keys())
    out: Dict[str, Tuple[Optional[Path], Optional[Path]]] = {}
    for method_name in sorted(all_method_names):
        base_method_dir = base_latest_dirs.get(method_name)
        calib_method_dir = calib_latest_dirs.get(method_name)
        base_file = find_latest_results_file(base_method_dir) if base_method_dir else None
        calib_file = find_latest_results_file(calib_method_dir) if calib_method_dir else None
        out[method_name] = (base_file, calib_file)
    return out


def compare_results(base_dir: Path, calib_dir: Path, eval_type: str = "arc"):
    """比较base和calib目录下的评测结果。"""
    base_dir = Path(base_dir)
    calib_dir = Path(calib_dir)
    
    if eval_type == "arc":
        # ARC格式：base 用每个 method 最新一份；calib 若存在多个时间戳版本，则展开为多行（Method=LOSS_时间戳）
        base_latest = _select_latest_file_by_method(base_dir, "arc")
        calib_runs = _collect_files_by_method_in_dir(calib_dir, "arc")
        all_method_names = set(base_latest.keys()) | set(calib_runs.keys())
        
        results = []
        for method_name in sorted(all_method_names):
            base_file = base_latest.get(method_name)
            items = calib_runs.get(method_name) or []
            if not items:
                items = [(None, None)]  # type: ignore[list-item]
            # 时间戳优先倒序，其次按文件名
            items = sorted(items, key=lambda x: ((x[0] or ""), x[1].name if x[1] else ""), reverse=True)  # type: ignore[union-attr]
            for ts, calib_file in items:
                method_label = method_name.upper() if not ts else f"{method_name.upper()}_{ts}"
            
                base_metrics = {}
                if base_file and base_file.exists():
                    base_data = load_json_results(base_file)
                    base_metrics = extract_metrics(base_data, eval_type) if base_data else {}

                calib_metrics = {}
                if calib_file and calib_file.exists():
                    calib_data = load_json_results(calib_file)
                    calib_metrics = extract_metrics(calib_data, eval_type) if calib_data else {}

                if not base_metrics and not calib_metrics:
                    continue

                results.extend(_compare_metric_dicts_to_rows(method_label, base_metrics, calib_metrics))
    elif eval_type == "gsm8k":
        # GSM8K格式：base 用每个 method 最新一份；calib 若存在多个时间戳版本，则展开为多行（Method=LOSS_时间戳）
        base_latest = _select_latest_file_by_method(base_dir, "gsm8k")
        calib_runs = _collect_files_by_method_in_dir(calib_dir, "gsm8k")
        all_method_names = set(base_latest.keys()) | set(calib_runs.keys())
        
        results = []
        for method_name in sorted(all_method_names):
            base_file = base_latest.get(method_name)
            items = calib_runs.get(method_name) or []
            if not items:
                items = [(None, None)]  # type: ignore[list-item]
            items = sorted(items, key=lambda x: ((x[0] or ""), x[1].name if x[1] else ""), reverse=True)  # type: ignore[union-attr]
            for ts, calib_file in items:
                method_label = method_name.upper() if not ts else f"{method_name.upper()}_{ts}"
            
                base_metrics = {}
                if base_file and base_file.exists():
                    base_data = load_json_results(base_file)
                    base_metrics = extract_metrics(base_data, eval_type) if base_data else {}

                calib_metrics = {}
                if calib_file and calib_file.exists():
                    calib_data = load_json_results(calib_file)
                    calib_metrics = extract_metrics(calib_data, eval_type) if calib_data else {}

                if not base_metrics and not calib_metrics:
                    continue

                results.extend(_compare_metric_dicts_to_rows(method_label, base_metrics, calib_metrics))
    
    elif eval_type in ("bbh", "musr", "math_hard", "mmlu_pro"):
        results = _compare_dirbased_eval_runs(base_dir, calib_dir, eval_type)
    else:
        raise ValueError(
            f"不支持的评测类型: {eval_type}，支持的类型: arc, bbh, mmlu_pro, math_hard, musr, gsm8k"
        )
    
    return pd.DataFrame(results)


def format_table(df: pd.DataFrame) -> str:
    """格式化DataFrame为可读的表格字符串。"""
    # 按方法和指标排序（合并表可能没有 Metric 列）
    sort_cols = ["Method", "Metric"] if "Metric" in df.columns else ["Method"]
    df_sorted = df.sort_values(sort_cols).copy()
    
    # 格式化数值显示
    def format_value(val):
        if val is None:
            return 'N/A'
        if isinstance(val, float):
            if abs(val) < 0.01:
                return f'{val:.6f}'
            else:
                return f'{val:.4f}'
        return str(val)
    
    df_formatted = df_sorted.copy()
    for col in ['Base', 'Calib', 'Change', 'Change (%)']:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(format_value)
    
    return df_formatted.to_string(index=False)


def df_to_markdown(df: pd.DataFrame) -> str:
    """将DataFrame转换为Markdown表格格式（不依赖tabulate）。"""
    if df.empty:
        return ""
    
    # 格式化数值显示
    def format_value(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 'N/A'
        if isinstance(val, float):
            if abs(val) < 0.01:
                return f'{val:.6f}'
            else:
                return f'{val:.4f}'
        return str(val)
    
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if df_formatted[col].dtype in ['float64', 'float32']:
            df_formatted[col] = df_formatted[col].apply(format_value)
        else:
            df_formatted[col] = df_formatted[col].astype(str)
    
    # 生成Markdown表格
    lines = []
    
    # 表头
    headers = list(df_formatted.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # 数据行
    for _, row in df_formatted.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


def _is_stderr_metric_name(metric: str) -> bool:
    """
    统一判定哪些 metric 属于 stderr/方差/不确定性类。
    兼容：
    - bbh: acc_norm_stderr
    - mmlu_pro: acc_stderr
    - math_hard: exact_match_stderr
    - arc: acc_stderr, acc_norm_stderr, ...（可能带逗号后缀）
    """
    if metric is None:
        return False
    m = str(metric).lower()
    return "stderr" in m


def _split_main_and_stderr_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将总表按 Metric 拆为 (main_metrics_df, stderr_metrics_df)。"""
    if df.empty or "Metric" not in df.columns:
        return df.copy(), df.iloc[0:0].copy()
    mask_stderr = df["Metric"].apply(_is_stderr_metric_name)
    return df[~mask_stderr].copy(), df[mask_stderr].copy()


def build_mean_pm_stderr_table(
    df: pd.DataFrame, main_key: str, stderr_key: str
) -> pd.DataFrame:
    """
    将同一 Method 下的主指标与 lm-eval stderr 合并为一列「值 ± stderr」（单次评测不确定度）。
    stderr 缺失时只显示主指标数值。
    """
    if df.empty or "Metric" not in df.columns:
        return pd.DataFrame()
    dm = df[df["Metric"] == main_key][["Method", "Base", "Calib", "Change", "Change (%)"]].copy()
    ds = df[df["Metric"] == stderr_key][["Method", "Base", "Calib"]].copy()
    if dm.empty:
        return dm
    ds = ds.rename(columns={"Base": "Base_stderr", "Calib": "Calib_stderr"})
    out = dm.merge(ds, on="Method", how="left")

    def fmt_pm(val: object, se: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        if se is None or (isinstance(se, float) and pd.isna(se)):
            return f"{float(val):.4f}"
        return f"{float(val):.4f} ± {float(se):.4f}"

    out["Base (主±stderr)"] = [
        fmt_pm(v, s) for v, s in zip(out["Base"], out["Base_stderr"])
    ]
    out["Calib (主±stderr)"] = [
        fmt_pm(v, s) for v, s in zip(out["Calib"], out["Calib_stderr"])
    ]
    return out[
        ["Method", "Base (主±stderr)", "Calib (主±stderr)", "Change", "Change (%)"]
    ]


def aggregate_calib_timestamps_for_metric(df_main: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    对 Method 形如「LOSS_YYYYMMDD_HHMMSS」的多条 calib 结果，按 loss 分组合并，
    报告 Calib 列的样本均值与样本标准差（跨运行离散度，非 lm-eval stderr）。
    """
    sub = df_main[df_main["Metric"] == metric_name].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["Loss", "n", "Calib_mean", "Calib_std", "Calib mean±std"]
        )
    ts_pat = re.compile(r"^(.+)_(\d{8})_(\d{6})$")

    def strip_ts(meth: object) -> tuple[str, bool]:
        s = str(meth)
        m = ts_pat.match(s)
        if m:
            return m.group(1), True
        return s, False

    pairs = sub["Method"].map(strip_ts)
    sub["_loss"] = pairs.map(lambda x: x[0])
    sub["_has_ts"] = pairs.map(lambda x: x[1])
    multi = sub[sub["_has_ts"]].copy()
    if multi.empty:
        return pd.DataFrame(
            columns=["Loss", "n", "Calib_mean", "Calib_std", "Calib mean±std"]
        )

    rows: List[Dict] = []
    for loss, grp in multi.groupby("_loss"):
        calibs = pd.to_numeric(grp["Calib"], errors="coerce").dropna()
        if calibs.empty:
            continue
        n = int(calibs.shape[0])
        mean = float(calibs.mean())
        std = float(calibs.std(ddof=1)) if n > 1 else float("nan")

        def fmt_row(mu: float, sd: float, count: int) -> str:
            if count <= 1 or not np.isfinite(sd):
                return f"{mu:.4f} (n={count})"
            return f"{mu:.4f} ± {sd:.4f} (n={count})"

        rows.append(
            {
                "Loss": loss,
                "n": n,
                "Calib_mean": mean,
                "Calib_std": std if n > 1 else None,
                "Calib mean±std": fmt_row(mean, std, n),
            }
        )
    return pd.DataFrame(rows)


def _build_baseline_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comparison table against BASELINE if BASELINE exists in df.

    Input df schema (like compare_results output):
      Method, Metric, Base, Calib, Change, Change (%)

    Output schema:
      Method, Metric, Baseline, Base, Calib, Base-Baseline, Calib-Baseline, Base-Baseline (%), Calib-Baseline (%)
    """
    if df.empty:
        return df.iloc[0:0].copy()
    if "Method" not in df.columns or "Metric" not in df.columns:
        return df.iloc[0:0].copy()

    method_col = df["Method"].astype(str)
    baseline_rows = df[(method_col.str.upper() == "BASELINE") & df["Base"].notna()].copy()
    if baseline_rows.empty:
        return df.iloc[0:0].copy()

    # Baseline is taken from Base side (we don't expect a calib baseline)
    baseline_by_metric: Dict[str, float] = {}
    for _, r in baseline_rows.iterrows():
        metric = r.get("Metric")
        b = r.get("Base")
        if isinstance(metric, str) and isinstance(b, Number):
            baseline_by_metric[metric] = float(b)

    if not baseline_by_metric:
        return df.iloc[0:0].copy()

    out_rows: List[Dict] = []
    for _, r in df.iterrows():
        method = str(r.get("Method", ""))
        if method.upper() == "BASELINE":
            continue
        metric = r.get("Metric")
        if not isinstance(metric, str):
            continue

        baseline = baseline_by_metric.get(metric, None)
        base = r.get("Base")
        calib = r.get("Calib")

        base_minus = None
        calib_minus = None
        base_minus_pct = None
        calib_minus_pct = None
        if isinstance(baseline, (int, float)) and np.isfinite(baseline):
            if isinstance(base, (int, float)) and np.isfinite(base):
                base_minus = float(base) - float(baseline)
                base_minus_pct = (base_minus / float(baseline) * 100.0) if baseline != 0 else None
            if isinstance(calib, (int, float)) and np.isfinite(calib):
                calib_minus = float(calib) - float(baseline)
                calib_minus_pct = (calib_minus / float(baseline) * 100.0) if baseline != 0 else None

        out_rows.append(
            {
                "Method": method,
                "Metric": metric,
                "Baseline": baseline,
                "Base": base,
                "Calib": calib,
                "Base-Baseline": base_minus,
                "Calib-Baseline": calib_minus,
                "Base-Baseline (%)": base_minus_pct,
                "Calib-Baseline (%)": calib_minus_pct,
            }
        )

    return pd.DataFrame(out_rows)


def _build_mmlu_pro_breakdown_delta_tables(
    method_files: Dict[str, Tuple[Optional[Path], Optional[Path]]]
) -> Tuple[str, List[str]]:
    """
    为每个 method 构建子任务 breakdown 的 Base/Calib/Δ 表（by_category/by_src）。
    返回 (markdown_text, missing_warnings)。
    注意：missing_warnings 不包含本机绝对路径（避免出现 __home__...）。
    """
    md_lines: List[str] = []
    missing_warnings: List[str] = []

    for method_label, (base_results, calib_results) in method_files.items():
        if base_results is None and calib_results is None:
            continue

        md_lines.append(f"\n\n## MMLU-PRO 子任务变化：{method_label}\n")

        def load_breakdown_side(results_path: Optional[Path], side: str) -> Optional[Dict]:
            if results_path is None:
                missing_warnings.append(f"{method_label} ({side}): results_*.json 缺失，跳过子任务 breakdown")
                return None
            run_dir = results_path.parent
            breakdown, missing = load_or_compute_mmlu_pro_breakdown(run_dir)
            if breakdown is None:
                missing_warnings.append(
                    f"{method_label} ({side}): 子任务 breakdown 缺失（mmlu_pro_breakdown.json / samples_leaderboard_mmlu_pro_*.jsonl 不存在或不可用）"
                )
            else:
                # 即便 breakdown 有，也把 side 上缺失信息记录到最后汇总（不打印子任务表）
                for m in missing:
                    if m:
                        missing_warnings.append(f"{method_label} ({side}): 缺少 {m}")
            return breakdown

        base_breakdown = load_breakdown_side(base_results, "Base")
        calib_breakdown = load_breakdown_side(calib_results, "Calib")

        if not base_breakdown and not calib_breakdown:
            md_lines.append("\n（Base/Calib 都没有可用的子任务 breakdown，已跳过）\n")
            continue

        for key, title in (("by_category", "按 category"), ("by_src", "按 src")):
            base_map = (base_breakdown or {}).get(key) or {}
            calib_map = (calib_breakdown or {}).get(key) or {}
            # union + 保序：base 在前，calib 新增追加
            all_tasks = list(dict.fromkeys(list(base_map.keys()) + list(calib_map.keys())))

            rows = []
            for task in all_tasks:
                b = base_map.get(task) or {}
                c = calib_map.get(task) or {}
                b_acc = b.get("acc")
                c_acc = c.get("acc")

                change = None
                change_pct = None
                if isinstance(b_acc, (int, float)) and isinstance(c_acc, (int, float)):
                    change = c_acc - b_acc
                    change_pct = (change / b_acc * 100.0) if b_acc != 0 else None

                rows.append(
                    {
                        "Subtask": task,
                        "Base_n": b.get("n"),
                        "Base_acc": b_acc,
                        "Calib_n": c.get("n"),
                        "Calib_acc": c_acc,
                        "Change": change,
                        "Change (%)": change_pct,
                    }
                )

            df = pd.DataFrame(rows)
            if not df.empty and "Change" in df.columns:
                df = df.sort_values(["Change", "Subtask"], ascending=[False, True], na_position="last")

            md_lines.append(f"\n### {title}\n\n")
            md_lines.append(df_to_markdown(df))
            md_lines.append("\n")

    return "".join(md_lines), missing_warnings


def _build_group_subtask_delta_tables(
    eval_type: str,
    method_files: Dict[str, Tuple[Optional[Path], Optional[Path]]],
) -> str:
    """
    为 bbh / math_hard 构建子任务变化表：不打印，只用于写入 md。
    子任务列表优先来自 group_subtasks。
    """
    if eval_type == "bbh":
        group_name = "leaderboard_bbh"
        metric_key = "acc_norm,none"
        stderr_key = "acc_norm_stderr,none"
        metric_name = "acc_norm"
    elif eval_type == "math_hard":
        group_name = "leaderboard_math_hard"
        metric_key = "exact_match,none"
        stderr_key = "exact_match_stderr,none"
        metric_name = "exact_match"
    else:
        return ""

    md_lines: List[str] = []
    md_lines.append("\n\n---\n")
    md_lines.append(f"\n# {eval_type.upper()} 子任务变化（Base vs Calib）\n")

    for method_label, (base_results, calib_results) in method_files.items():
        if base_results is None and calib_results is None:
            continue

        def load_json(path: Optional[Path]) -> Optional[Dict]:
            if path is None or not path.exists():
                return None
            return load_json_results(path)

        base_data = load_json(base_results)
        calib_data = load_json(calib_results)

        base_results_map = (base_data or {}).get("results") or {}
        calib_results_map = (calib_data or {}).get("results") or {}

        # 取子任务列表：优先 group_subtasks[group_name]，否则用 results key 前缀推断
        base_sub = ((base_data or {}).get("group_subtasks") or {}).get(group_name) or []
        calib_sub = ((calib_data or {}).get("group_subtasks") or {}).get(group_name) or []
        subtasks = list(dict.fromkeys(list(base_sub) + list(calib_sub)))
        if not subtasks:
            # fallback：从 results key 推断
            if eval_type == "bbh":
                prefix = "leaderboard_bbh_"
            else:
                prefix = "leaderboard_math_"
            all_keys = list(dict.fromkeys(list(base_results_map.keys()) + list(calib_results_map.keys())))
            subtasks = [k for k in all_keys if isinstance(k, str) and k.startswith(prefix) and k != group_name]

        md_lines.append(f"\n\n## 子任务变化：{method_label}\n")

        if not subtasks:
            md_lines.append("\n（未找到子任务列表：group_subtasks 缺失，且无法从 results keys 推断）\n")
            continue

        rows = []
        for task in subtasks:
            b = base_results_map.get(task) or {}
            c = calib_results_map.get(task) or {}
            b_val = b.get(metric_key)
            c_val = c.get(metric_key)
            b_se = b.get(stderr_key)
            c_se = c.get(stderr_key)

            change = None
            change_pct = None
            if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
                change = c_val - b_val
                change_pct = (change / b_val * 100.0) if b_val != 0 else None

            rows.append(
                {
                    "Subtask": task,
                    f"Base_{metric_name}": b_val,
                    "Base_stderr": b_se,
                    f"Calib_{metric_name}": c_val,
                    "Calib_stderr": c_se,
                    "Change": change,
                    "Change (%)": change_pct,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty and "Change" in df.columns:
            df = df.sort_values(["Change", "Subtask"], ascending=[False, True], na_position="last")
        md_lines.append("\n")
        md_lines.append(df_to_markdown(df))
        md_lines.append("\n")

    return "".join(md_lines)


_VALID_COMPARE_EVAL_TYPES = frozenset(
    {"arc", "bbh", "mmlu_pro", "math_hard", "musr", "gsm8k"}
)


def _parse_eval_types_arg(s: str) -> list[str]:
    """支持单个类型或逗号分隔多个，如 bbh,mmlu_pro,musr。"""
    raw = (s or "").strip().lower()
    if not raw:
        return ["arc"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return ["arc"]
    bad = [p for p in parts if p not in _VALID_COMPARE_EVAL_TYPES]
    if bad:
        raise SystemExit(
            f"错误: 不支持的 --eval-type: {bad}\n"
            f"       允许的值: {', '.join(sorted(_VALID_COMPARE_EVAL_TYPES))}"
        )
    return parts


def _run_compare_eval_type(
    eval_type: str,
    args: argparse.Namespace,
    script_dir: Path,
    base_dir: Path,
    calib_dir: Path,
    inferred_model_name: str,
) -> None:
    """对单个 eval_type 执行对比、终端输出与写入 calib_comparison_{eval_type}.md。"""
    # 比较结果
    df = compare_results(base_dir, calib_dir, eval_type)

    if df.empty:
        print(f"[{eval_type}] 未找到任何评测结果，跳过。")
        return

    df_main, df_stderr = _split_main_and_stderr_metrics(df)

    if not df_main.empty:
        print("\n[主指标 / mean 类指标]\n")
        print(format_table(df_main))
    else:
        print("\n（未找到主指标 / mean 类指标）")

    if not df_stderr.empty:
        print("\n[不确定性 / stderr 类指标]\n")
        print(format_table(df_stderr))
    else:
        print("\n（未找到 stderr 类指标）")

    # 主指标 ± lm-eval stderr（单次评测）
    pm_info = [
        ("arc", "acc_norm", "acc_norm_stderr"),
        ("gsm8k", "acc", "acc_stderr"),
        ("bbh", "acc_norm", "acc_norm_stderr"),
        ("musr", "acc_norm", "acc_norm_stderr"),
        ("mmlu_pro", "acc", "acc_stderr"),
        ("math_hard", "exact_match", "exact_match_stderr"),
    ]
    pm_main, pm_stderr = None, None
    for et, mk, sk in pm_info:
        if eval_type == et:
            pm_main, pm_stderr = mk, sk
            break
    if pm_main and pm_stderr:
        df_pm = build_mean_pm_stderr_table(df, pm_main, pm_stderr)
        if not df_pm.empty:
            print("\n[主指标 ± lm-eval stderr（单次评测不确定度）]\n")
            print(format_table(df_pm))
        else:
            print("\n（无法生成 主±stderr 合并表：缺少主指标行）")

    if args.aggregate_calib_runs and pm_main:
        df_agg = aggregate_calib_timestamps_for_metric(df_main, pm_main)
        if not df_agg.empty:
            print("\n[跨多次 calib 时间戳：Calib 主指标 mean ± 样本std]\n")
            print(
                df_agg[["Loss", "n", "Calib mean±std"]].to_string(index=False)
            )
        elif args.aggregate_calib_runs:
            print("\n（--aggregate-calib-runs：未找到带 _YYYYMMDD_HHMMSS 的多条记录）")
    print()

    # 保存为Markdown表格（按模型分类：outputs/results/compare_results/{model_name}/）
    output_dir = script_dir / "results" / "compare_results" / inferred_model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_md = output_dir / f"calib_comparison_{eval_type}.md"

    breakdown_md = ""
    missing_samples: List[str] = []
    if eval_type == "mmlu_pro":
        method_files = _build_breakdown_method_file_map(base_dir, calib_dir, "mmlu_pro")
        breakdown_md, missing_samples = _build_mmlu_pro_breakdown_delta_tables(method_files)

    group_subtask_md = ""
    if eval_type in ("bbh", "math_hard"):
        method_files = _build_breakdown_method_file_map(base_dir, calib_dir, eval_type)
        group_subtask_md = _build_group_subtask_delta_tables(eval_type, method_files)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(f"# Calib前后评测结果对比 ({eval_type.upper()}, 变化 = Calib - Base)\n\n")
        df_main_md, df_stderr_md = _split_main_and_stderr_metrics(df)
        if not df_main_md.empty:
            f.write("## 主指标（mean 类）\n\n")
            f.write(df_to_markdown(df_main_md))
        else:
            f.write("## 主指标（mean 类）\n\n（无）\n")

        if not df_stderr_md.empty:
            f.write("\n\n## 不确定性（stderr 类）\n\n")
            f.write(df_to_markdown(df_stderr_md))
        else:
            f.write("\n\n## 不确定性（stderr 类）\n\n（无）\n")

        if pm_main and pm_stderr:
            df_pm_md = build_mean_pm_stderr_table(df, pm_main, pm_stderr)
            if not df_pm_md.empty:
                f.write("\n\n## 主指标 ± lm-eval stderr（单次评测）\n\n")
                f.write(
                    "以下为同一 Method 下主指标与标准误合并展示；stderr 来自 lm-eval，"
                    "不是多次训练之间的标准差。\n\n"
                )
                f.write(df_to_markdown(df_pm_md))

        if args.aggregate_calib_runs and pm_main:
            df_agg_md = aggregate_calib_timestamps_for_metric(df_main, pm_main)
            if not df_agg_md.empty:
                f.write("\n\n## 跨 calib 时间戳汇总（样本 mean ± 样本 std）\n\n")
                f.write(
                    "仅统计 Method 以 `_YYYYMMDD_HHMMSS` 结尾的行（中间可有消融段，如 `DPO_dbema0p5_20260325_023455`）；"
                    "std 为多次评测结果之间的样本标准差。\n\n"
                )
                f.write(df_to_markdown(df_agg_md[["Loss", "n", "Calib mean±std"]]))

        base_vs_bl_main = _build_baseline_comparison_df(df_main_md)
        base_vs_bl_stderr = _build_baseline_comparison_df(df_stderr_md)
        if (not base_vs_bl_main.empty) or (not base_vs_bl_stderr.empty):
            f.write("\n\n---\n\n## 与 BASELINE 的对比（Base / Calib / Baseline）\n")
            if not base_vs_bl_main.empty:
                f.write("\n\n### 主指标（mean 类）\n\n")
                f.write(df_to_markdown(base_vs_bl_main))
            else:
                f.write("\n\n### 主指标（mean 类）\n\n（无）\n")
            if not base_vs_bl_stderr.empty:
                f.write("\n\n### 不确定性（stderr 类）\n\n")
                f.write(df_to_markdown(base_vs_bl_stderr))
            else:
                f.write("\n\n### 不确定性（stderr 类）\n\n（无）\n")

        if eval_type == "mmlu_pro" and missing_samples:
            uniq = list(dict.fromkeys(missing_samples))
            f.write("\n\n---\n\n## 提示（缺失文件/跳过项）\n")
            for msg in uniq:
                f.write(f"- {msg}\n")

        if breakdown_md.strip():
            f.write("\n\n---\n")
            f.write("\n# MMLU-PRO 子任务 Breakdown（Base vs Calib）\n")
            f.write(breakdown_md)
        if group_subtask_md.strip():
            f.write(group_subtask_md)
    print(f"[{eval_type}] 结果已保存到: {output_md}")

    if eval_type == "mmlu_pro" and missing_samples:
        uniq = list(dict.fromkeys(missing_samples))
        print("\n" + "=" * 80)
        print("提示：部分 mmlu_pro 子任务 breakdown 计算被跳过（缺少文件）")
        print("=" * 80)
        for msg in uniq:
            print(f"- {msg}")


def main():
    """主函数。"""
    # 命令行参数（可覆盖文件顶部配置）
    parser = argparse.ArgumentParser(description="Compare base vs calib evaluation results.")
    parser.add_argument(
        "--eval-type",
        type=str,
        default=EVAL_TYPE,
        metavar="TYPE[,TYPE,...]",
        help=(
            f"评测类型，单个或逗号分隔多个，如 bbh,mmlu_pro,musr "
            f"(默认来自文件: {EVAL_TYPE})。允许: {', '.join(sorted(_VALID_COMPARE_EVAL_TYPES))}。"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help='Model name used to locate results dirs (e.g. "pythia-2b", "mistral-7b"). '
             'Overrides env DIL_MODEL_NAME if provided.',
    )
    parser.add_argument(
        "--aggregate-calib-runs",
        action="store_true",
        help=(
            "对 Method 形如 LOSS_YYYYMMDD_HHMMSS 的多条记录，按 loss 汇总 Calib 主指标的 "
            "样本均值 ± 样本标准差（跨多次 calib 跑分；与 lm-eval 单次 stderr 不同）。"
        ),
    )
    args = parser.parse_args()

    eval_types = _parse_eval_types_arg(str(args.eval_type))
    print(f"评测类型: {', '.join(et.upper() for et in eval_types)}")
    
    # 设置路径
    script_dir = Path(__file__).parent
    # 目录选择优先级：
    # CLI --model > 环境变量 DIL_MODEL_NAME > 默认 "mistral-7b"
    model_name = (args.model or os.getenv("DIL_MODEL_NAME") or "mistral-7b").strip()
    # 如需覆盖具体子目录名，可使用环境变量：
    # - DIL_BASE_SUBDIR / DIL_CALIB_SUBDIR
    base_subdir = (os.getenv("DIL_BASE_SUBDIR") or f"{model_name}-base").strip()
    calib_subdir = (os.getenv("DIL_CALIB_SUBDIR") or f"{model_name}-calib").strip()

    base_dir = script_dir / "results" / base_subdir
    calib_dir = script_dir / "results" / calib_subdir
    
    print(f"Base目录: {base_dir}")
    print(f"Calib目录: {calib_dir}")
    print()
    
    # 检查目录是否存在（如果都不存在则报错）
    if not base_dir.exists() and not calib_dir.exists():
        print(f"错误: Base和Calib目录都不存在")
        print(f"  Base目录: {base_dir}")
        print(f"  Calib目录: {calib_dir}")
        return
    
    if not base_dir.exists():
        print(f"提示: Base目录不存在，将只显示Calib结果: {base_dir}")
    
    if not calib_dir.exists():
        print(f"提示: Calib目录不存在，将只显示Base结果: {calib_dir}")
    
    # 从目录名中提取模型名称（去掉-base或-calib后缀）
    inferred_model_name = None
    if base_dir.exists():
        inferred_model_name = base_dir.name.replace('-base', '')
    elif calib_dir.exists():
        inferred_model_name = calib_dir.name.replace('-calib', '')
    
    if inferred_model_name is None:
        print("错误: 无法从目录名中提取模型名称")
        return
    
    print(f"检测到模型: {inferred_model_name}")
    print()

    for eval_type in eval_types:
        print("\n" + "#" * 80)
        print(f"### {eval_type.upper()}")
        print("#" * 80 + "\n")
        _run_compare_eval_type(
            eval_type,
            args,
            script_dir,
            base_dir,
            calib_dir,
            inferred_model_name,
        )


if __name__ == '__main__':
    main()
