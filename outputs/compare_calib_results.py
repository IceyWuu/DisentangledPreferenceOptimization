"""
Compare evaluation metrics before and after calibration across methods.

By default reads result files under outputs/results/mistral-7b-base/ and
outputs/results/mistral-7b-calib/, compares each method's metrics before vs after calib,
and prints tables.

You can switch model/directory via environment variables:
- DPO_MODEL_NAME: model name (e.g. "pythia-2b" or "mistral-7b"), expanded to "{model}-base" / "{model}-calib"
- DPO_BASE_SUBDIR: override base subdir name (default "{model}-base")
- DPO_CALIB_SUBDIR: override calib subdir name (default "{model}-calib")

Supported eval types:
- arc: **/out_arc_*.json
- bbh: bbh/lb2_bbh_*.json/**/results_*.json
- mmlu_pro: mmlu_pro/lb2_mmlu_pro_*.json/**/results_*.json
- math_hard: math_hard/lb2_math_hard_*.json/**/results_*.json
- musr: musr/lb2_musr_*.json/**/results_*.json; within one file, acc_norm for three subtasks is
    sample-weighted (250*murder + 256*object_placements + 250*team_allocation)/756; stderr is
    combined via weighted variance of independent estimates
- gsm8k: **/out_gsm8k_*.json

Select type by editing EVAL_TYPE at the top of this file.

CLI examples (from DPO/outputs/):
  python compare_calib_results.py --eval-type musr --model pythia-2b
  python compare_calib_results.py --eval-type bbh,mmlu_pro,musr --model qwen2.5-7b

Optional:
  --aggregate-calib-runs
      When the same loss has multiple calib runs with Method suffix _YYYYMMDD_HHMMSS, report
      cross-run mean ± sample std for Calib primary metrics (unlike lm-eval single-run stderr; see main()).

Uncertainty:
  - lm-eval's acc_norm_stderr etc. are standard errors for a single evaluation run; this script
    lists them in a separate table and can merge primary ± stderr into one column.
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
# Config: select eval type
# ============================================================================
# Allowed: "arc", "bbh", "mmlu_pro", "math_hard", "musr", or "gsm8k"
EVAL_TYPE = "bbh"
# ============================================================================

# MuSR subtask full test set sizes (for acc_norm weighting; denominator 250+256+250=756)
MUSR_SUBTASK_SAMPLE_WEIGHTS: Dict[str, int] = {
    "leaderboard_musr_murder_mysteries": 250,
    "leaderboard_musr_object_placements": 256,
    "leaderboard_musr_team_allocation": 250,
}


def load_json_results(file_path: Path) -> Optional[Dict]:
    """Load a JSON evaluation results file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # Missing file: no warning, return None
        return None
    except json.JSONDecodeError as e:
        print(f"Error: failed to parse JSON file {file_path}: {e}")
        return None


def _pick_acc_from_sample(ex: dict) -> Optional[float]:
    """
    Extract 0/1 acc from one lm-eval samples jsonl record.
    Supports two common layouts:
    - top-level "acc"
    - under "metrics": "acc" / "acc,none" / any key starting with "acc"
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
    Same aggregation as mmlu_pro_sub_tasks.py:
    stream samples_leaderboard_mmlu_pro_*.jsonl and aggregate by_category / by_src.
    """
    if not samples_jsonl.exists():
        return None

    # Aggregate by category
    cat_bucket = defaultdict(list)
    cat_order: List[str] = []
    cat_seen = set()

    # Aggregate by src (finer granularity)
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
                    # Skip bad lines without aborting
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
    Prefer mmlu_pro_breakdown.json under run_dir; else compute from
    samples_leaderboard_mmlu_pro_*.jsonl in the same directory.
    Returns (breakdown, missing_file_hints); hints omit absolute paths.
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
    List subtask names for multi-setting evals: prefer group_subtasks[group_key],
    else all results keys starting with task_prefix and not equal to group_key.
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
    Within one JSON's results:
    - If multiple subtasks exist, take arithmetic mean of raw fields per canonical metric across
      subtasks; stderr-like fields are averaged the same way as acc.
    - If no subtasks, read only the aggregate row for group_key (single setting).
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
    MuSR aggregate acc_norm: weighted by subtask test set sizes
        (250*murder_mysteries + 256*object_placements + 250*team_allocation) / 756
    If only some subtasks have values, normalize using weights only for subtasks that have both
    a weight and a result (denominator = sum of effective weights).

    acc_norm_stderr: treat subtask estimates as independent; for the weighted mean use
        sqrt(sum_i (w_i/W)^2 * se_i^2) (not arithmetic mean of stderr).

    If no subtasks or weighting fails, fall back to leaderboard_musr aggregate row (legacy).
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
    """Extract metrics from eval JSON (leaderboard: mean over subtasks for bbh etc.; musr is sample-weighted)."""
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
        # ARC: iterate all tasks
        for task_name, task_results in results.items():
            for key, value in task_results.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
    
    return metrics


def get_method_name(filename: str, eval_type: str = "arc") -> str:
    """Extract method name from filename."""
    # Optional timestamp suffix: ..._{YYYYMMDD_HHMMSS}.json
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
    """Latest results_*.json under method dir (use enumerate_eval_results_under_run_dir if multiple runs)."""
    entries = enumerate_eval_results_under_run_dir(method_dir)
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0][1]
    # Back-compat: if multiple experiment dirs, still return the newest to avoid caller crashes; use enumerate for comparison tables
    return max((p for _, p in entries), key=lambda f: (f.name, f.stat().st_mtime))


def parse_ablation_slug_from_results_path(results_path: Path) -> str:
    """
    Walk up from results_*.json and parse <ablation> from directory names like
    ``{family}-{loss}-calib-<ablation>-{YYYYMMDD_HHMMSS}``.
    E.g. ...-dpo-calib-dbema0p5-20260325_023455... -> ``dbema0p5``;
    ``...-calib-seed1-20260326_023439...`` -> ``seed1``;
    multiple segments joined with ``-``: ``dbema0p5-seed1``.
    If no ``-calib-`` or timestamp match, return "" (legacy layouts).
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
    Under a ``lb2_*_dpo_*.json`` run directory, list all eval artifacts (dedupe by results parent).
    Returns [(ablation_slug, results_json), ...]; ablation_slug may be empty (no ablation or old path).
    If multiple results_*.json share a parent, keep the newest by filename + mtime.
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
    Method column: ``LOSS`` / ``LOSS_ts`` / ``LOSS_ablation_ts`` (replace ``-`` in ablation with ``_``).
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
    """When base has multiple results: match calib ablation slug if possible, else first entry."""
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
    For bbh / musr / math_hard / mmlu_pro: multiple ablation subdirs under one run, each with results.
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
    For mmlu_pro breakdown and bbh|math_hard subtask tables: same ablation expansion as main table.
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
    When calib has timestamped outputs (*_YYYYMMDD_HHMMSS.json):
    - Each method × each calib timestamp vs base (latest per method for base)
    - Write comparison JSON (does not change existing md output)
    - If no timestamp detected: no-op, backward compatible
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
    Return {method_name: (base_results_json, calib_results_json)} for mmlu_pro only.
    method_name is the suffix of lb2_mmlu_pro_*.json (lowercase).
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
    Return {method_name: (base_results_json, calib_results_json)} for bbh / math_hard.
    method_name is the suffix of lb2_{eval_type}_*.json (lowercase).
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
    """Compare evaluation results under base vs calib directories."""
    base_dir = Path(base_dir)
    calib_dir = Path(calib_dir)
    
    if eval_type == "arc":
        # ARC: base uses latest per method; calib expands multiple timestamps to rows (Method=LOSS_timestamp)
        base_latest = _select_latest_file_by_method(base_dir, "arc")
        calib_runs = _collect_files_by_method_in_dir(calib_dir, "arc")
        all_method_names = set(base_latest.keys()) | set(calib_runs.keys())
        
        results = []
        for method_name in sorted(all_method_names):
            base_file = base_latest.get(method_name)
            items = calib_runs.get(method_name) or []
            if not items:
                items = [(None, None)]  # type: ignore[list-item]
            # Sort by timestamp desc, then filename
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
        # GSM8K: base uses latest per method; calib expands multiple timestamps to rows (Method=LOSS_timestamp)
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
            f"Unsupported eval type: {eval_type}. Allowed: arc, bbh, mmlu_pro, math_hard, musr, gsm8k"
        )
    
    return pd.DataFrame(results)


def format_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a readable table string."""
    # Sort by Method and Metric (merged tables may lack Metric column)
    sort_cols = ["Method", "Metric"] if "Metric" in df.columns else ["Method"]
    df_sorted = df.sort_values(sort_cols).copy()
    
    # Format numeric display
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
    """Convert DataFrame to Markdown table (no tabulate dependency)."""
    if df.empty:
        return ""
    
    # Format numeric display
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
    
    # Build Markdown table
    lines = []
    
    # Header
    headers = list(df_formatted.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Data rows
    for _, row in df_formatted.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


def _is_stderr_metric_name(metric: str) -> bool:
    """
    Classify metrics as stderr / variance / uncertainty.
    Covers:
    - bbh: acc_norm_stderr
    - mmlu_pro: acc_stderr
    - math_hard: exact_match_stderr
    - arc: acc_stderr, acc_norm_stderr, ... (optional comma suffix)
    """
    if metric is None:
        return False
    m = str(metric).lower()
    return "stderr" in m


def _split_main_and_stderr_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split full table into (main_metrics_df, stderr_metrics_df) by Metric."""
    if df.empty or "Metric" not in df.columns:
        return df.copy(), df.iloc[0:0].copy()
    mask_stderr = df["Metric"].apply(_is_stderr_metric_name)
    return df[~mask_stderr].copy(), df[mask_stderr].copy()


def build_mean_pm_stderr_table(
    df: pd.DataFrame, main_key: str, stderr_key: str
) -> pd.DataFrame:
    """
    Merge primary metric and lm-eval stderr per Method into one column (value ± stderr) for a single run.
    If stderr is missing, show primary value only.
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

    out["Base (main ± stderr)"] = [
        fmt_pm(v, s) for v, s in zip(out["Base"], out["Base_stderr"])
    ]
    out["Calib (main ± stderr)"] = [
        fmt_pm(v, s) for v, s in zip(out["Calib"], out["Calib_stderr"])
    ]
    return out[
        ["Method", "Base (main ± stderr)", "Calib (main ± stderr)", "Change", "Change (%)"]
    ]


def aggregate_calib_timestamps_for_metric(df_main: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    For multiple calib rows with Method like LOSS_YYYYMMDD_HHMMSS, group by loss and report
    sample mean and sample std of the Calib column (cross-run spread, not lm-eval stderr).
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
    Build Base/Calib/delta subtask breakdown tables per method (by_category/by_src).
    Returns (markdown_text, missing_warnings).
    missing_warnings omit absolute paths (avoid __home__... in output).
    """
    md_lines: List[str] = []
    missing_warnings: List[str] = []

    for method_label, (base_results, calib_results) in method_files.items():
        if base_results is None and calib_results is None:
            continue

        md_lines.append(f"\n\n## MMLU-PRO subtask changes: {method_label}\n")

        def load_breakdown_side(results_path: Optional[Path], side: str) -> Optional[Dict]:
            if results_path is None:
                missing_warnings.append(
                    f"{method_label} ({side}): missing results_*.json; skipping subtask breakdown"
                )
                return None
            run_dir = results_path.parent
            breakdown, missing = load_or_compute_mmlu_pro_breakdown(run_dir)
            if breakdown is None:
                missing_warnings.append(
                    f"{method_label} ({side}): subtask breakdown missing "
                    "(no mmlu_pro_breakdown.json / samples_leaderboard_mmlu_pro_*.jsonl or unusable)"
                )
            else:
                # Record missing hints for this side even when breakdown exists (for final summary)
                for m in missing:
                    if m:
                        missing_warnings.append(f"{method_label} ({side}): missing {m}")
            return breakdown

        base_breakdown = load_breakdown_side(base_results, "Base")
        calib_breakdown = load_breakdown_side(calib_results, "Calib")

        if not base_breakdown and not calib_breakdown:
            md_lines.append("\n(No usable subtask breakdown on Base/Calib; skipped.)\n")
            continue

        for key, title in (("by_category", "By category"), ("by_src", "By src")):
            base_map = (base_breakdown or {}).get(key) or {}
            calib_map = (calib_breakdown or {}).get(key) or {}
            # Union preserving order: base keys first, then new calib keys
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
    Build bbh / math_hard subtask delta tables for md only (not printed to terminal).
    Subtask list prefers group_subtasks.
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
    md_lines.append(f"\n# {eval_type.upper()} subtask changes (Base vs Calib)\n")

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

        # Subtask list: prefer group_subtasks[group_name], else infer from results key prefix
        base_sub = ((base_data or {}).get("group_subtasks") or {}).get(group_name) or []
        calib_sub = ((calib_data or {}).get("group_subtasks") or {}).get(group_name) or []
        subtasks = list(dict.fromkeys(list(base_sub) + list(calib_sub)))
        if not subtasks:
            # Fallback: infer from results keys
            if eval_type == "bbh":
                prefix = "leaderboard_bbh_"
            else:
                prefix = "leaderboard_math_"
            all_keys = list(dict.fromkeys(list(base_results_map.keys()) + list(calib_results_map.keys())))
            subtasks = [k for k in all_keys if isinstance(k, str) and k.startswith(prefix) and k != group_name]

        md_lines.append(f"\n\n## Subtask changes: {method_label}\n")

        if not subtasks:
            md_lines.append(
                "\n(No subtask list: group_subtasks missing and could not infer from results keys.)\n"
            )
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
    """Single eval type or comma-separated list, e.g. bbh,mmlu_pro,musr."""
    raw = (s or "").strip().lower()
    if not raw:
        return ["arc"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return ["arc"]
    bad = [p for p in parts if p not in _VALID_COMPARE_EVAL_TYPES]
    if bad:
        raise SystemExit(
            f"Error: unsupported --eval-type: {bad}\n"
            f"       Allowed: {', '.join(sorted(_VALID_COMPARE_EVAL_TYPES))}"
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
    """Run comparison for one eval_type: terminal output and calib_comparison_{eval_type}.md."""
    # Comparison result
    df = compare_results(base_dir, calib_dir, eval_type)

    if df.empty:
        print(f"[{eval_type}] No evaluation results found; skipping.")
        return

    df_main, df_stderr = _split_main_and_stderr_metrics(df)

    if not df_main.empty:
        print("\n[Primary / mean-like metrics]\n")
        print(format_table(df_main))
    else:
        print("\n(No primary / mean-like metrics found.)")

    if not df_stderr.empty:
        print("\n[Uncertainty / stderr-like metrics]\n")
        print(format_table(df_stderr))
    else:
        print("\n(No stderr-like metrics found.)")

    # Primary ± lm-eval stderr (single run)
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
            print("\n[Primary ± lm-eval stderr (single-run uncertainty)]\n")
            print(format_table(df_pm))
        else:
            print("\n(Cannot build primary±stderr table: missing primary metric rows.)")

    if args.aggregate_calib_runs and pm_main:
        df_agg = aggregate_calib_timestamps_for_metric(df_main, pm_main)
        if not df_agg.empty:
            print("\n[Multiple calib timestamps: Calib primary mean ± sample std]\n")
            print(
                df_agg[["Loss", "n", "Calib mean±std"]].to_string(index=False)
            )
        elif args.aggregate_calib_runs:
            print("\n(--aggregate-calib-runs: no multiple rows with _YYYYMMDD_HHMMSS.)")
    print()

    # Save Markdown under outputs/results/compare_results/{model_name}/
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
        f.write(f"# Base vs calib comparison ({eval_type.upper()}, change = Calib - Base)\n\n")
        df_main_md, df_stderr_md = _split_main_and_stderr_metrics(df)
        if not df_main_md.empty:
            f.write("## Primary metrics (mean-like)\n\n")
            f.write(df_to_markdown(df_main_md))
        else:
            f.write("## Primary metrics (mean-like)\n\n(none)\n")

        if not df_stderr_md.empty:
            f.write("\n\n## Uncertainty (stderr-like)\n\n")
            f.write(df_to_markdown(df_stderr_md))
        else:
            f.write("\n\n## Uncertainty (stderr-like)\n\n(none)\n")

        if pm_main and pm_stderr:
            df_pm_md = build_mean_pm_stderr_table(df, pm_main, pm_stderr)
            if not df_pm_md.empty:
                f.write("\n\n## Primary ± lm-eval stderr (single run)\n\n")
                f.write(
                    "Primary metric and standard error merged per Method; stderr comes from lm-eval, "
                    "not std across multiple training runs.\n\n"
                )
                f.write(df_to_markdown(df_pm_md))

        if args.aggregate_calib_runs and pm_main:
            df_agg_md = aggregate_calib_timestamps_for_metric(df_main, pm_main)
            if not df_agg_md.empty:
                f.write("\n\n## Across calib timestamps (sample mean ± sample std)\n\n")
                f.write(
                    "Only rows whose Method ends with `_YYYYMMDD_HHMMSS` (ablation segments allowed, "
                    "e.g. `DPO_dbema0p5_20260325_023455`); std is sample std across multiple eval runs.\n\n"
                )
                f.write(df_to_markdown(df_agg_md[["Loss", "n", "Calib mean±std"]]))

        base_vs_bl_main = _build_baseline_comparison_df(df_main_md)
        base_vs_bl_stderr = _build_baseline_comparison_df(df_stderr_md)
        if (not base_vs_bl_main.empty) or (not base_vs_bl_stderr.empty):
            f.write("\n\n---\n\n## Comparison vs BASELINE (Base / Calib / Baseline)\n")
            if not base_vs_bl_main.empty:
                f.write("\n\n### Primary metrics (mean-like)\n\n")
                f.write(df_to_markdown(base_vs_bl_main))
            else:
                f.write("\n\n### Primary metrics (mean-like)\n\n(none)\n")
            if not base_vs_bl_stderr.empty:
                f.write("\n\n### Uncertainty (stderr-like)\n\n")
                f.write(df_to_markdown(base_vs_bl_stderr))
            else:
                f.write("\n\n### Uncertainty (stderr-like)\n\n(none)\n")

        if eval_type == "mmlu_pro" and missing_samples:
            uniq = list(dict.fromkeys(missing_samples))
            f.write("\n\n---\n\n## Notes (missing files / skipped)\n")
            for msg in uniq:
                f.write(f"- {msg}\n")

        if breakdown_md.strip():
            f.write("\n\n---\n")
            f.write("\n# MMLU-PRO subtask breakdown (Base vs Calib)\n")
            f.write(breakdown_md)
        if group_subtask_md.strip():
            f.write(group_subtask_md)
    print(f"[{eval_type}] Saved to: {output_md}")

    if eval_type == "mmlu_pro" and missing_samples:
        uniq = list(dict.fromkeys(missing_samples))
        print("\n" + "=" * 80)
        print("Note: some mmlu_pro subtask breakdown steps were skipped (missing files).")
        print("=" * 80)
        for msg in uniq:
            print(f"- {msg}")


def main():
    """Entry point."""
    # CLI overrides file-top config
    parser = argparse.ArgumentParser(description="Compare base vs calib evaluation results.")
    parser.add_argument(
        "--eval-type",
        type=str,
        default=EVAL_TYPE,
        metavar="TYPE[,TYPE,...]",
        help=(
            f"Eval type(s), one or comma-separated, e.g. bbh,mmlu_pro,musr "
            f"(default from file: {EVAL_TYPE}). Allowed: {', '.join(sorted(_VALID_COMPARE_EVAL_TYPES))}."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help='Model name used to locate results dirs (e.g. "pythia-2b", "mistral-7b"). '
             'Overrides env DPO_MODEL_NAME if provided.',
    )
    parser.add_argument(
        "--aggregate-calib-runs",
        action="store_true",
        help=(
            "For multiple rows with Method like LOSS_YYYYMMDD_HHMMSS, aggregate Calib primary metrics "
            "by loss as sample mean ± sample std (across calib runs; unlike lm-eval single-run stderr)."
        ),
    )
    args = parser.parse_args()

    eval_types = _parse_eval_types_arg(str(args.eval_type))
    print(f"Eval type(s): {', '.join(et.upper() for et in eval_types)}")
    
    # Paths
    script_dir = Path(__file__).parent
    # Directory resolution: CLI --model > env DPO_MODEL_NAME > default "mistral-7b"
    model_name = (args.model or os.getenv("DPO_MODEL_NAME") or "mistral-7b").strip()
    # Override subdir names via env: DPO_BASE_SUBDIR / DPO_CALIB_SUBDIR
    base_subdir = (os.getenv("DPO_BASE_SUBDIR") or f"{model_name}-base").strip()
    calib_subdir = (os.getenv("DPO_CALIB_SUBDIR") or f"{model_name}-calib").strip()

    base_dir = script_dir / "results" / base_subdir
    calib_dir = script_dir / "results" / calib_subdir
    
    print(f"Base dir: {base_dir}")
    print(f"Calib dir: {calib_dir}")
    print()
    
    # Require at least one of base/calib
    if not base_dir.exists() and not calib_dir.exists():
        print(f"Error: neither Base nor Calib directory exists")
        print(f"  Base dir: {base_dir}")
        print(f"  Calib dir: {calib_dir}")
        return
    
    if not base_dir.exists():
        print(f"Note: Base dir missing; showing Calib only: {base_dir}")
    
    if not calib_dir.exists():
        print(f"Note: Calib dir missing; showing Base only: {calib_dir}")
    
    # Infer model name from directory (strip -base or -calib)
    inferred_model_name = None
    if base_dir.exists():
        inferred_model_name = base_dir.name.replace('-base', '')
    elif calib_dir.exists():
        inferred_model_name = calib_dir.name.replace('-calib', '')
    
    if inferred_model_name is None:
        print("Error: could not infer model name from directory names")
        return
    
    print(f"Inferred model: {inferred_model_name}")
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
