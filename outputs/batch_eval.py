"""
Batch evaluate all merged models using lm_eval.

This script scans output_merged/ for merged model directories, extracts the
method name (bce, dpo, ddro, etc.), and runs lm_eval tests.

Supported model families (configurable): pythia-2b, mistral-7b, qwen2.5-7b (or all).
Models with "calib" in the name go to <family>-calib/, others to <family>-base/.
For adapter-based families, this script uses lazy merge cache under output_merged/.

QUICK START - Edit configuration at the top of this file:
  Look for the "CONFIGURATION" section (around line 50) and set:
  - METHODS_TO_EVAL: which methods to evaluate (e.g., "bce,dpo" or "all")
  - CATEGORY_TO_EVAL: "all", "base", or "calib"
  - EVAL_METHOD_TO_USE: single method like "mmlu_pro" or comma-separated like "math_hard,mmlu_pro"

# Step 1: simply run
  python batch_eval.py

  or

  nohup python batch_eval.py --eval-method "mmlu_pro" --methods "dpo" --category calib --model-family pythia-2b --device cuda:0 > /dev/null 2>&1 &

# Step 2: compare the results
  python compare_calib_results.py --eval-type musr --model pythia-2b

Command-line options (override config settings):
  --eval-method: Evaluation method(s) to use, comma-separated for multiple (e.g., "math_hard,mmlu_pro")
  --methods: Comma-separated list of methods or "all"
  --category: "all", "base", or "calib"
  --model-family: "pythia-2b", "mistral-7b", or "all"
  --dry-run: Only print what would be evaluated, don't actually run
  --skip-existing: Skip models that already have evaluation results
  --batch-size: Comma list paired with --eval-method order, or one int for all; omitted → math_hard 32, else 1
  --results-base: Base directory for results (default: ./results)
  --device: lm_eval HuggingFace device, e.g. cuda:0, cuda:3, cpu.

GPU selection:
  1) Environment variable (inherited by subprocesses):
       CUDA_VISIBLE_DEVICES=3 python batch_eval.py ...
  2) Explicit logical device:
       python batch_eval.py ... --device cuda:2

For hf-causal-experimental tasks (arc / gsm8k / mmlu):
  - No --device -> use_accelerate=True (device_map=auto), may span visible GPUs.
  - With --device -> use_accelerate=False, force single-device placement.

For qwen2.5-7b:
  - Training outputs are usually LoRA adapters under --adapter-dir
    (e.g. qwen2.5-7b-instruct-dpo-lora-*), not pre-merged dirs.
  - The script merges adapters before evaluation.
  - Default base model path:
      ../../ModelAndDatasets/alignment-handbook/qwen2.5-7b-instruct
    Override via QWEN25_7B_INSTRUCT_BASE_MODEL or --base-model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

# Delay heavy imports to worker subprocesses after CUDA_VISIBLE_DEVICES is set.


# ============================================================================
# CONFIGURATION: Edit these settings to control what gets evaluated
# ============================================================================

# Which model family to evaluate
# Options: "pythia-2b", "mistral-7b", or "all"
MODEL_FAMILY_TO_EVAL = "mistral-7b"

# Methods to evaluate (loss types)
METHODS_TO_EVAL = "dpo, bce, cpo, ddro, lsif, simpo, tidpo"

# Category to evaluate
# Options: "all", "base", or "calib"
CATEGORY_TO_EVAL = "base"  

# Evaluation method
# Options: "mmlu_pro", "bbh", "arc", "gsm8k", "mmlu", or "math_hard"
EVAL_METHOD_TO_USE = "bbh"

# LM Eval command style (will be auto-overridden by eval method unless explicitly set)
# Options: "module" (python -m lm_eval) or "main" (python main.py)
# Use "main" for lmeval_v1 environment, "module" for newer versions
# Usually leave this as-is: style is auto-switched by eval method.
LM_EVAL_STYLE = "main"

# Conda environments to auto-activate per eval method
CONDA_ENV_MAP = {
    "arc": "lmeval_v1",
    "gsm8k": "lmeval_v1",
    "mmlu": "lmeval_v1",
    "bbh": "lmeval_v2",
    "math_hard": "lmeval_v2",
    "mmlu_pro": "lmeval_v2",
    "musr": "lmeval_v2",
}

# Path to legacy lm_eval v1 entrypoint (arc / gsm8k / mmlu use lmeval_v1 + main.py).
# Expected sibling layout: LeaderBoardV1/lm-evaluation-harness/main.py
# If it doesn't exist, the script will automatically fall back to "module".
LM_EVAL_MAIN_PATH = "../../LeaderBoardV1/lm-evaluation-harness/main.py"

# For module-style tasks (mmlu_pro / bbh / ...), use local LeaderBoard V2 harness.
# Override with LM_EVAL_V2_HARNESS_ROOT when directory layout differs.
_LM_EVAL_V2_HARNESS_DEFAULT = Path(__file__).resolve().parent.parent.parent / "LeaderBoardV2" / "lm-evaluation-harness"
LM_EVAL_V2_HARNESS_ROOT = Path(
    os.environ.get("LM_EVAL_V2_HARNESS_ROOT", str(_LM_EVAL_V2_HARNESS_DEFAULT))
).expanduser().resolve()

# Root for merged models; set env OUTPUT_MERGED_ROOT to override
OUTPUT_MERGED_ROOT = os.environ.get("OUTPUT_MERGED_ROOT", "output_merged")

# Root for adapter models (LoRA); set env OUTPUT_ADAPTER_ROOT to override
OUTPUT_ADAPTER_ROOT = os.environ.get("OUTPUT_ADAPTER_ROOT", ".")

# Base model paths for each model family
BASE_MODEL_PATHS = {
    "mistral-7b": os.environ.get(
        "MISTRAL_7B_BASE_MODEL",
        # "../../ModelAndDatasets/local_models/mistralai/Mistral-7B-v0.1"
        "../../ModelAndDatasets/alignment-handbook/local_models/mistralai/Mistral-7B-v0.1"
    ),
    "pythia-2b": os.environ.get(
        "PYTHIA_2B_BASE_MODEL",
        "../../ModelAndDatasets/alignment-handbook/pythia-2.8b",
    ),
    # Same convention as other families: relative to DIL/outputs working directory.
    "qwen2.5-7b": os.environ.get(
        "QWEN25_7B_INSTRUCT_BASE_MODEL",
        "../../ModelAndDatasets/alignment-handbook/qwen2.5-7b-instruct",
    ),
}

# Per-family adapter glob override (relative to --adapter-dir).
ADAPTER_DIR_GLOB_BY_FAMILY: dict[str, str] = {
    "qwen2.5-7b": "qwen2.5-7b-instruct-*",
}

# Per-family merged-dir glob patterns (deduplicated on resolve()).
MERGED_DIR_GLOBS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "qwen2.5-7b": ("qwen2.5-7b-*-merged", "qwen2.5-7b-instruct-*-merged"),
}

# Families that use "scan adapters -> lazy merge cache -> evaluate".
ADAPTER_MODEL_FAMILIES = frozenset({"pythia-2b", "mistral-7b", "qwen2.5-7b"})

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

# Common method names to extract from model names
KNOWN_METHODS = [
    "bce", "dpo", "ddro", "cpo", "ipo", "simpo", "kto", "slic", "lsif", "ukl", "tidpo",
    # Evaluate the raw base model (no adapter / no merge)
    "baseline",
]

KNOWN_MODEL_FAMILIES = ["pythia-2b", "mistral-7b", "qwen2.5-7b"]

# MMLU subtask list (HendrycksTest)
MMLU_SUBTASKS = [
    "abstract_algebra","anatomy","astronomy","business_ethics","clinical_knowledge",
    "college_biology","college_chemistry","college_computer_science","college_mathematics",
    "college_medicine","college_physics","computer_security","conceptual_physics","econometrics",
    "electrical_engineering","elementary_mathematics","formal_logic","global_facts",
    "high_school_biology","high_school_chemistry","high_school_computer_science",
    "high_school_european_history","high_school_geography","high_school_government_and_politics",
    "high_school_macroeconomics","high_school_mathematics","high_school_microeconomics",
    "high_school_physics","high_school_psychology","high_school_statistics","high_school_us_history",
    "high_school_world_history","human_aging","human_sexuality","international_law","jurisprudence",
    "logical_fallacies","machine_learning","management","marketing","medical_genetics",
    "miscellaneous","moral_disputes","moral_scenarios","nutrition","philosophy","prehistory",
    "professional_accounting","professional_law","professional_medicine","professional_psychology",
    "public_relations","security_studies","sociology","us_foreign_policy","virology","world_religions"
]
MMLU_TASKS = ",".join([f"hendrycksTest-{s}" for s in MMLU_SUBTASKS])


def _postprocess_mmlu_results_inplace(result_file: Path) -> None:
    """
    lm-eval v1 writes per-subtask MMLU scores but may omit an aggregate.
    Add a simple mean aggregate and keep subtask membership in-place.
    """
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    results = data.get("results") or {}
    if not isinstance(results, dict):
        return

    subtask_keys = [k for k in results.keys() if isinstance(k, str) and k.startswith("hendrycksTest-")]
    if not subtask_keys:
        return

    accs = []
    stderrs = []
    for k in subtask_keys:
        m = results.get(k) or {}
        if not isinstance(m, dict):
            continue
        if isinstance(m.get("acc"), (int, float)):
            accs.append(float(m["acc"]))
        if isinstance(m.get("acc_stderr"), (int, float)):
            stderrs.append(float(m["acc_stderr"]))

    if not accs:
        return

    mean_acc = sum(accs) / len(accs)
    mean_stderr = (sum(stderrs) / len(stderrs)) if stderrs else None

    # Add a group-like summary entry for easy downstream parsing
    results["mmlu"] = {
        "acc": mean_acc,
        "acc_stderr": mean_stderr,
        "alias": "mmlu",
        "num_subtasks": len(subtask_keys),
    }
    data["groups"] = data.get("groups") or {}
    if isinstance(data["groups"], dict):
        data["groups"]["mmlu"] = results["mmlu"]
    data["group_subtasks"] = data.get("group_subtasks") or {}
    if isinstance(data["group_subtasks"], dict):
        data["group_subtasks"]["mmlu"] = subtask_keys

    try:
        with open(result_file, "w", encoding="utf-8") as w:
            json.dump(data, w, ensure_ascii=False, indent=2)
    except Exception:
        return

# Evaluation method configurations
# Maps eval method name to (task_name, file_prefix, model_type, num_fewshot)
# model_type: "hf" or "hf-causal-experimental"
EVAL_METHODS = {
    "mmlu_pro": ("leaderboard_mmlu_pro", "lb2_mmlu_pro", "hf", 5),
    "bbh": ("leaderboard_bbh", "lb2_bbh", "hf", 3),
    # leaderboard_musr: https://github.com/EleutherAI/lm-evaluation-harness (leaderboard task)
    # Example:
    #   python -m lm_eval --model hf --model_args "pretrained=YOUR_MODEL_PATH,dtype=float16" \
    #     --tasks leaderboard_musr --batch_size 1 --num_fewshot 0 --output_path ./results/lb2_musr.json
    "musr": ("leaderboard_musr", "lb2_musr", "hf", 0),
    "arc": ("arc_challenge", "out_arc", "hf-causal-experimental", 25),
    "gsm8k": ("gsm8k", "out_gsm8k", "hf-causal-experimental", 5),
    # mmlu: many subtasks (hendrycksTest-*) with simple mean aggregation in postprocess
    "mmlu": (MMLU_TASKS, "out_mmlu", "hf-causal-experimental", 5),
    "math_hard": ("leaderboard_math_hard", "lb2_math_hard", "hf", 4),
}

# lm-eval v0.4+ writes under --output_path via EvaluationTracker: a subdir named after
# sanitized pretrained, containing results_<timestamp>.json (not a single flat .json).
LM_EVAL_V2_DIR_LAYOUT_METHODS = frozenset({"bbh", "mmlu_pro", "math_hard", "musr"})

# When --batch-size is omitted, use per-method defaults (math_hard is heavy; larger batch is usually safe).
_DEFAULT_BATCH_SIZE_MATH_HARD = 32
_DEFAULT_BATCH_SIZE_OTHER = 1


def _default_batch_size_for_eval_method(eval_method: str) -> int:
    """When --batch-size is omitted."""
    return _DEFAULT_BATCH_SIZE_MATH_HARD if eval_method == "math_hard" else _DEFAULT_BATCH_SIZE_OTHER


def _parse_batch_sizes_list(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--batch-size is empty")
    out: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as e:
            raise ValueError(f"invalid --batch-size token: {p!r}") from e
        if v < 1:
            raise ValueError(f"--batch-size must be >= 1, got {v}")
        out.append(v)
    return out


def _build_batch_size_map(eval_methods: list[str], batch_size_arg: str | None) -> dict[str, int]:
    """
    If batch_size_arg is None: per-method defaults (math_hard→32, others→1).
    If one integer: broadcast to every eval method.
    If comma list: length must equal len(eval_methods), paired by order with --eval-method.
    """
    if batch_size_arg is None:
        return {em: _default_batch_size_for_eval_method(em) for em in eval_methods}
    bs_list = _parse_batch_sizes_list(batch_size_arg)
    n = len(eval_methods)
    if len(bs_list) == 1:
        v = bs_list[0]
        return {em: v for em in eval_methods}
    if len(bs_list) != n:
        raise ValueError(
            f"--batch-size has {len(bs_list)} value(s) but --eval-method has {n} method(s). "
            f"Pass one integer to use the same batch for all methods, or exactly {n} values in the same order as: "
            f"{eval_methods}"
        )
    return dict(zip(eval_methods, bs_list))


def _lm_eval_v2_has_results(run_root: Path) -> bool:
    """True if run_root is a directory containing at least one results_*.json (v2 tracker layout)."""
    if not run_root.is_dir():
        return False
    return any(run_root.rglob("results_*.json"))


def extract_method_name(model_name: str, model_family: str) -> str:
    """Extract method name from model directory name.
    
    Examples:
        pythia-2b-bce-20260115_065352-merged -> bce
        pythia-2b-dpo-calib-ema-20260114_030220-merged -> dpo
        pythia-2b-lsif-20260115_154949-merged -> lsif
    """
    # Remove common prefixes and suffixes
    prefix = f"{model_family}-"
    name = model_name
    if name.startswith(prefix):
        name = name[len(prefix):]
    name = name.replace("-merged", "")
    
    # Try to find known methods
    for method in KNOWN_METHODS:
        # Match method at the start or after a dash
        pattern = rf"(?:^|-)({method})(?:-|$)"
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return method.lower()
    
    # Fallback: try to extract first meaningful word after pythia-2b-
    parts = name.split("-")
    if parts:
        # Skip common suffixes like "calib", "ema", timestamps
        for part in parts:
            if part and part not in ["calib", "ema", "ema2", "log1p", "db"]:
                # Check if it looks like a method name (short, lowercase)
                if len(part) <= 6 and part.isalpha():
                    return part.lower()
    
    # Last resort: use first part
    return parts[0].lower() if parts else "unknown"


def is_calib_model(model_name: str) -> bool:
    """Check if model name contains 'calib'."""
    return "calib" in model_name.lower()


def extract_model_family(model_name: str) -> str | None:
    """Infer model family from directory name."""
    name = model_name.lower()
    for fam in KNOWN_MODEL_FAMILIES:
        if name.startswith(fam + "-"):
            return fam
    return None


def _extract_last_timestamp_from_path(p: Path) -> str | None:
    """
    Extract the last YYYYMMDD_HHMMSS timestamp from a model directory name.
    Falls back to parent directory name (useful when evaluating checkpoint-* dirs).
    """
    for name in (p.name, p.parent.name):
        matches = re.findall(r"\d{8}_\d{6}", name)
        if matches:
            return matches[-1]
    return None


def find_adapter_models(adapter_dir: Path, model_family: str) -> list[tuple[str, Path]]:
    """Find LoRA adapter directories for a given family (or all).
    
    Returns adapters sorted by method name, then by category (base before calib).
    """
    families: list[str]
    if model_family == "all":
        families = KNOWN_MODEL_FAMILIES[:]
    else:
        families = [model_family]

    adapters: list[tuple[str, Path]] = []
    for fam in families:
        glob_pat = ADAPTER_DIR_GLOB_BY_FAMILY.get(fam, f"{fam}-*")
        for item in adapter_dir.glob(glob_pat):
            if not item.is_dir():
                continue
            # Skip merged directories
            if item.name.endswith("-merged"):
                continue
            # Check if it's a valid LoRA adapter (has adapter_config.json and adapter_model files)
            has_adapter = (item / "adapter_model.safetensors").exists() or \
                         (item / "adapter_model.bin").exists()
            has_config = (item / "adapter_config.json").exists()
            
            if has_adapter and has_config:
                adapters.append((fam, item))
            else:
                # Also check checkpoint subdirectories
                for checkpoint_dir in item.glob("checkpoint-*"):
                    if checkpoint_dir.is_dir():
                        has_adapter_ckpt = (checkpoint_dir / "adapter_model.safetensors").exists() or \
                                          (checkpoint_dir / "adapter_model.bin").exists()
                        has_config_ckpt = (checkpoint_dir / "adapter_config.json").exists()
                        if has_adapter_ckpt and has_config_ckpt:
                            adapters.append((fam, checkpoint_dir))
    
    # Sort by method name first, then by category (base before calib)
    def sort_key(adapter_entry: tuple[str, Path]) -> tuple[str, int]:
        fam, adapter_path = adapter_entry
        method = extract_method_name(adapter_path.name, fam)
        is_calib = is_calib_model(adapter_path.name)
        # base = 0, calib = 1, so base comes before calib
        category_order = 1 if is_calib else 0
        return (method, category_order)
    
    return sorted(adapters, key=sort_key)


def find_merged_models(merged_dir: Path, model_family: str) -> list[tuple[str, Path]]:
    """Find merged model directories for a given family (or all).
    
    Returns models sorted by method name, then by category (base before calib).
    """
    families: list[str]
    if model_family == "all":
        families = KNOWN_MODEL_FAMILIES[:]
    else:
        families = [model_family]

    models: list[tuple[str, Path]] = []
    for fam in families:
        patterns = MERGED_DIR_GLOBS_BY_FAMILY.get(fam, (f"{fam}-*-merged",))
        seen_resolved: set[Path] = set()
        for pattern in patterns:
            for item in merged_dir.glob(pattern):
                if not item.is_dir():
                    continue
                key = item.resolve()
                if key in seen_resolved:
                    continue
                # Check if it's a valid merged model (has config.json)
                if (item / "config.json").exists():
                    seen_resolved.add(key)
                    models.append((fam, item))
    
    # Sort by method name first, then by category (base before calib)
    def sort_key(model_entry: tuple[str, Path]) -> tuple[str, int]:
        fam, model_path = model_entry
        method = extract_method_name(model_path.name, fam)
        is_calib = is_calib_model(model_path.name)
        # base = 0, calib = 1, so base comes before calib
        category_order = 1 if is_calib else 0
        return (method, category_order)
    
    return sorted(models, key=sort_key)


def _pick_dtype(dtype_str: str) -> Any:
    """Pick torch dtype from string."""
    import torch

    s = (dtype_str or "auto").lower()
    if s == "auto":
        # Prefer bf16 if available; otherwise fp16.
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {dtype_str!r}. Use auto|bf16|fp16|fp32.")


def _resolve_maybe_local_model_ref(model_ref: str) -> tuple[str, bool]:
    """
    Return (resolved_ref, is_local_path).
    
    - If model_ref looks like a local path (contains path separators, starts with '.', '..', '~'),
      we expand/resolve it. If it exists, treat as local. If it does NOT exist, raise a clear error
      instead of letting HF Hub validation explode on '..'.
    - Otherwise, treat as a Hub id.
    """
    s = (model_ref or "").strip()
    if not s:
        raise ValueError("base_model is empty.")

    looks_like_path = (
        s.startswith(".")
        or s.startswith("~")
        or ("/" in s)
        or ("\\" in s)
        or (os.path.sep in s)
        or (os.path.altsep is not None and os.path.altsep in s)
    )

    if looks_like_path:
        p = Path(s).expanduser()
        try:
            p = p.resolve()
        except Exception:
            # On some platforms/permission setups, resolve() can fail; fallback to absolute().
            p = p.absolute()

        if not p.exists():
            raise FileNotFoundError(
                f"base_model looks like a local path but was not found: {p}\n"
                f"Fix: point it to your downloaded base model directory (must contain config.json), e.g.\n"
                f"  --base_model /home/icshy/LLM_alignment/ModelAndDatasets/local_models/mistralai/Mistral-7B-v0.1\n"
                f"Or use a Hub id:\n"
                f"  --base_model mistralai/Mistral-7B-v0.1"
            )
        return str(p), True

    return s, False


def _merge_cache_payload(
    adapter_dir: Path,
    base_model: str,
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    return {
        "adapter_dir": str(adapter_dir.resolve()),
        "base_model": str(base_model),
        "dtype": str(dtype),
        "device_map": str(device_map),
        "trust_remote_code": bool(trust_remote_code),
    }


def _merge_cache_dir_name(adapter_name: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    short = hashlib.sha256(raw).hexdigest()[:10]
    return f"{adapter_name}-{short}-merged"


def _is_valid_merged_cache(merged_dir: Path, payload: dict[str, Any]) -> bool:
    cfg_ok = (merged_dir / "config.json").exists()
    if not cfg_ok:
        return False
    meta_file = merged_dir / "merge_meta.json"
    if not meta_file.exists():
        return False
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    return bool(isinstance(meta, dict) and meta.get("cache_payload") == payload)


def merge_adapter_with_lazy_cache(
    adapter_dir: Path,
    base_model: str,
    dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = False,
    merged_root: Path | None = None,
) -> Path:
    """
    Merge LoRA adapter to a cached merged directory under output_merged.
    Reuse existing cache when payload matches.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(adapter_dir).resolve()
    base_model, _ = _resolve_maybe_local_model_ref(base_model)
    merged_root = Path(merged_root or OUTPUT_MERGED_ROOT).resolve()
    merged_root.mkdir(parents=True, exist_ok=True)

    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")

    payload = _merge_cache_payload(
        adapter_dir=adapter_dir,
        base_model=base_model,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    target_name = _merge_cache_dir_name(adapter_dir.name, payload)
    target_dir = merged_root / target_name
    lock_file = merged_root / f".{target_name}.lock"

    if _is_valid_merged_cache(target_dir, payload):
        print(f"[INFO] Reusing merged cache: {target_dir}")
        return target_dir

    # Backward compatibility: reuse legacy "<adapter>-merged" if it exists.
    legacy_dir = merged_root / f"{adapter_dir.name}-merged"
    if legacy_dir.exists() and (legacy_dir / "config.json").exists():
        print(f"[INFO] Reusing legacy merged directory: {legacy_dir}")
        return legacy_dir

    # Acquire lock to avoid duplicate merges from parallel workers.
    wait_start = time.time()
    while True:
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if _is_valid_merged_cache(target_dir, payload):
                print(f"[INFO] Reusing merged cache (built by other worker): {target_dir}")
                return target_dir
            # Handle stale lock defensively (e.g., crashed worker).
            try:
                if (time.time() - lock_file.stat().st_mtime) > 1800:
                    print(f"[WARNING] Removing stale merge lock: {lock_file}")
                    os.remove(lock_file)
                    continue
            except OSError:
                pass
            if (time.time() - wait_start) > 2400:
                raise TimeoutError(f"Timeout waiting merge lock: {lock_file}")
            time.sleep(0.5)

    tmp_merge_dir: Path | None = None
    try:
        if _is_valid_merged_cache(target_dir, payload):
            print(f"[INFO] Reusing merged cache: {target_dir}")
            return target_dir

        torch_dtype = _pick_dtype(dtype)
        print(f"[INFO] Merging adapter: {adapter_dir}")
        print(f"[INFO] Base model: {base_model}")
        print(f"[INFO] dtype: {torch_dtype}")
        print(f"[INFO] device_map: {device_map}")

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model)

        model = PeftModel.from_pretrained(base, adapter_dir)
        model = model.merge_and_unload()

        tmp_merge_dir = Path(
            tempfile.mkdtemp(prefix=f"{target_name}.tmp.", dir=str(merged_root))
        )
        model.save_pretrained(tmp_merge_dir, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(tmp_merge_dir)

        chat_template_src = adapter_dir / "chat_template.jinja"
        if chat_template_src.exists() and chat_template_src.is_file():
            shutil.copy2(chat_template_src, tmp_merge_dir / "chat_template.jinja")

        with open(tmp_merge_dir / "merge_meta.json", "w", encoding="utf-8") as f:
            json.dump({"cache_payload": payload}, f, ensure_ascii=False, indent=2)

        # Free GPU memory before evaluation subprocess loads the model.
        del model, tokenizer, base
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(tmp_merge_dir), str(target_dir))
        tmp_merge_dir = None
        print(f"[INFO] Merged model cached at: {target_dir}")
        return target_dir
    finally:
        if tmp_merge_dir is not None and tmp_merge_dir.exists():
            shutil.rmtree(tmp_merge_dir, ignore_errors=True)
        try:
            os.remove(lock_file)
        except OSError:
            pass


def check_result_exists(
    model_family: str,
    model_path: Path,
    results_base: Path,
    method: str,
    is_calib: bool,
    eval_method: str = "mmlu_pro",
) -> bool:
    """Check if evaluation result already exists."""
    if eval_method not in EVAL_METHODS:
        raise ValueError(f"Unknown eval_method: {eval_method}. Choose from: {list(EVAL_METHODS.keys())}")
    
    category = f"{model_family}-calib" if is_calib else f"{model_family}-base"
    file_prefix = EVAL_METHODS[eval_method][1]
    ts = _extract_last_timestamp_from_path(model_path)
    eval_dir = results_base / category / eval_method

    if eval_method in LM_EVAL_V2_DIR_LAYOUT_METHODS:
        candidates: list[Path] = []
        if ts:
            candidates.append(eval_dir / f"{file_prefix}_{method}_{ts}.json")
        candidates.append(eval_dir / f"{file_prefix}_{method}.json")
        for p in candidates:
            if _lm_eval_v2_has_results(p):
                return True
        if eval_dir.is_dir():
            for p in eval_dir.glob(f"{file_prefix}_{method}_*.json"):
                if _lm_eval_v2_has_results(p):
                    return True
        return False

    if ts:
        result_file_ts = eval_dir / f"{file_prefix}_{method}_{ts}.json"
        if result_file_ts.is_file() and result_file_ts.stat().st_size > 0:
            return True

    result_file = eval_dir / f"{file_prefix}_{method}.json"
    if result_file.is_file() and result_file.stat().st_size > 0:
        return True

    if eval_dir.is_dir():
        for p in eval_dir.glob(f"{file_prefix}_{method}_*.json"):
            if p.is_file() and p.stat().st_size > 0:
                return True

    return False


def _parallel_worker_run_json(job_path: Path) -> None:
    """
    Worker entrypoint.
    Parent process binds one physical GPU via CUDA_VISIBLE_DEVICES;
    worker always evaluates with device=cuda:0.
    """
    job_path = Path(job_path)
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    job.pop("_job_index", None)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"[parallel-worker] CUDA_VISIBLE_DEVICES={vis!r} job={job_path.name}")
    run_eval._lm_eval_style = job["lm_eval_style"]
    run_eval._lm_eval_main_path = job["lm_eval_main_path"]
    run_eval._conda_env = job.get("conda_env")
    lim = job.get("limit")
    lim_f = float(lim) if lim is not None else None
    success, msg = run_eval(
        job["fam"],
        Path(job["model_path"]),
        Path(job["results_base"]),
        job["method"],
        job["is_calib"],
        eval_method=job["eval_method"],
        batch_size=int(job["batch_size"]),
        dtype=job["dtype"],
        dry_run=bool(job["dry_run"]),
        conda_env=job.get("conda_env"),
        use_adapter=bool(job["use_adapter"]),
        base_model=job.get("merge_base"),
        device="cuda:0",
        limit=lim_f,
    )
    print(f"[parallel-worker] success={success} {msg}")
    sys.exit(0 if success else 1)


def _run_parallel_gpu_pool(
    script_path: Path,
    job_payloads: list[dict[str, Any]],
    physical_gpu_ids: list[int],
) -> list[tuple[Any, ...]]:
    """
    GPU-bound subprocess scheduler (no multiprocessing pool).
    At most len(physical_gpu_ids) workers run concurrently, one physical GPU each.
    Returns ordered tuples:
    (fam, Path(model), method, success, msg, eval_method).
    """
    if not physical_gpu_ids:
        raise ValueError("physical_gpu_ids must not be empty")
    n = len(job_payloads)
    ordered: list[tuple[Any, ...] | None] = [None] * n
    queue: deque[int] = deque(range(n))
    available: deque[int] = deque(int(x) for x in physical_gpu_ids)
    running: list[tuple[subprocess.Popen, int, int, str]] = []
    temp_paths: list[str] = []

    try:
        while queue or running:
            while queue and available:
                gpu = available.popleft()
                ji = queue.popleft()
                payload = dict(job_payloads[ji])
                payload["_job_index"] = ji
                fd, tpath = tempfile.mkstemp(suffix=".json", prefix="batch_eval_job_")
                os.close(fd)
                temp_paths.append(tpath)
                with open(tpath, "w", encoding="utf-8") as wf:
                    json.dump(payload, wf, ensure_ascii=False, indent=2)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                cmd = [sys.executable, str(script_path), "--parallel-worker-job", tpath]
                p = subprocess.Popen(cmd, env=env)
                running.append((p, gpu, ji, tpath))
                print(f"[gpu-subproc] start {ji + 1}/{n} on physical GPU {gpu} -> {Path(tpath).name}")

            for item in list(running):
                proc, gpu, ji, tpath = item
                rc = proc.poll()
                if rc is None:
                    continue
                running.remove(item)
                available.append(gpu)
                pl = job_payloads[ji]
                fam = pl["fam"]
                model_p = Path(pl["model_path"])
                method = pl["method"]
                ev = pl["eval_method"]
                ok = rc == 0
                msg = "success" if ok else f"worker exit code {rc}"
                ordered[ji] = (fam, model_p, method, ok, msg, ev)
                print(f"[gpu-subproc] done {ji + 1}/{n} on physical GPU {gpu} ok={ok}")

            if queue or running:
                time.sleep(0.25)
    finally:
        for tp in temp_paths:
            try:
                os.remove(tp)
            except OSError:
                pass

    out: list[tuple[Any, ...]] = []
    for i, row in enumerate(ordered):
        if row is None:
            raise RuntimeError(f"worker job {i} did not return a result (scheduler inconsistency)")
        out.append(row)
    return out


def _make_parallel_job_payload(
    *,
    fam: str,
    model_path: Path,
    results_base: Path,
    method: str,
    is_calib: bool,
    eval_method: str,
    batch_size: int,
    dtype: str,
    dry_run: bool,
    conda_env: str | None,
    use_adapter: bool,
    merge_base: str | None,
    lm_eval_style: str,
    lm_eval_main_path: str,
    limit: float | None = None,
) -> dict[str, Any]:
    mb = merge_base
    if mb is not None:
        mb = str(Path(mb).expanduser().resolve())
    return {
        "fam": fam,
        "model_path": str(Path(model_path).resolve()),
        "results_base": str(Path(results_base).resolve()),
        "method": method,
        "is_calib": bool(is_calib),
        "eval_method": eval_method,
        "batch_size": batch_size,
        "dtype": dtype,
        "dry_run": bool(dry_run),
        "conda_env": conda_env,
        "use_adapter": bool(use_adapter),
        "merge_base": mb,
        "lm_eval_style": lm_eval_style,
        "lm_eval_main_path": lm_eval_main_path,
        "limit": limit,
    }


def _parse_parallel_gpus(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        q = p
        if q.lower().startswith("cuda:"):
            q = q.split(":", 1)[1]
        out.append(int(q))
    return out


def _resolve_eval_runtime_for_method(
    eval_method: str, lm_eval_style_arg: str
) -> tuple[str, str | None]:
    """Resolve lm_eval style and conda env from eval_method."""
    auto_style = lm_eval_style_arg
    auto_conda_env = None
    if eval_method in CONDA_ENV_MAP:
        auto_conda_env = CONDA_ENV_MAP[eval_method]
        if auto_conda_env == "lmeval_v1":
            auto_style = "main"
        elif auto_conda_env == "lmeval_v2":
            auto_style = "module"
    return auto_style, auto_conda_env


def run_eval(
    model_family: str,
    model_path: Path,
    results_base: Path,
    method: str,
    is_calib: bool,
    eval_method: str = "mmlu_pro",
    batch_size: int = 1,
    dtype: str = "float16",
    dry_run: bool = False,
    conda_env: str | None = None,
    use_adapter: bool = False,
    base_model: str | None = None,
    device: str | None = None,
    limit: float | None = None,
) -> tuple[bool, str]:
    """Run lm_eval for a single model.
    
    Args:
        use_adapter: If True, treat model_path as an adapter directory and lazy-merge it first.
        base_model: Base model path for merging (required if use_adapter=True).
    """
    if eval_method not in EVAL_METHODS:
        raise ValueError(f"Unknown eval_method: {eval_method}. Choose from: {list(EVAL_METHODS.keys())}")
    
    task_name, file_prefix, model_type, num_fewshot = EVAL_METHODS[eval_method]
    category = f"{model_family}-calib" if is_calib else f"{model_family}-base"
    ts = _extract_last_timestamp_from_path(model_path)
    if ts:
        result_file = results_base / category / eval_method / f"{file_prefix}_{method}_{ts}.json"
    else:
        result_file = results_base / category / eval_method / f"{file_prefix}_{method}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle adapter lazy-merge cache (output_merged/<adapter>-<hash>-merged)
    model_path_to_use = model_path
    
    if use_adapter:
        if not base_model:
            raise ValueError("base_model is required when use_adapter=True")
        
        if not dry_run:
            print(f"[INFO] Resolving lazy merge cache for adapter: {model_path.name}...")
            try:
                model_path_to_use = merge_adapter_with_lazy_cache(
                    adapter_dir=model_path,
                    base_model=base_model,
                    dtype=dtype,
                    device_map="auto",
                    trust_remote_code=False,
                )
            except Exception as e:
                error_msg = f"Failed to merge adapter for {model_path.name}: {e}"
                print(f"[ERROR] {error_msg}")
                return False, error_msg
        else:
            print(f"[DRY-RUN] Would resolve/merge adapter cache: {model_path.name}")
            model_path_to_use = model_path  # Use original path for dry-run
    
    # Build command
    # Use absolute path for model to avoid HF Hub lookup of relative path
    model_path_str = str(Path(model_path_to_use).resolve())
    
    # Build model_args by backend:
    # - hf-causal-experimental without --device uses accelerate(device_map=auto)
    # - with --device, force single-device placement
    if model_type == "hf-causal-experimental":
        if device:
            model_args = f"pretrained={model_path_str},use_accelerate=False,device={device}"
        else:
            model_args = f"pretrained={model_path_str},use_accelerate=True"
    else:
        model_args = f"pretrained={model_path_str},dtype={dtype}"
        if device:
            model_args = f"{model_args},device={device}"
    
    # Build command based on lm_eval style
    # For lmeval_v1, use "python main.py", for newer versions use "python -m lm_eval"
    if hasattr(run_eval, "_lm_eval_style"):
        lm_eval_style = run_eval._lm_eval_style
    else:
        # Default to "main" for backward compatibility
        lm_eval_style = "main"

    lm_eval_main_path = getattr(run_eval, "_lm_eval_main_path", None) or LM_EVAL_MAIN_PATH

    if lm_eval_style == "main":
        main_path = Path(lm_eval_main_path)
        if not main_path.exists():
            print(
                f"[WARNING] lm_eval main.py not found at '{lm_eval_main_path}'. "
                f"Falling back to 'python -m lm_eval'."
            )
            lm_eval_style = "module"

    if lm_eval_style == "main":
        base_cmd = [
            "python", lm_eval_main_path,
            "--model", model_type,
            "--model_args", model_args,
            "--tasks", task_name,
            "--batch_size", str(batch_size),
            "--output_path", str(result_file),
        ]
    else:
        base_cmd = [
            "python", "-m", "lm_eval",
            "--model", model_type,
            "--model_args", model_args,
            "--tasks", task_name,
            "--batch_size", str(batch_size),
            "--output_path", str(result_file),
        ]

    # Wrap with conda run if a conda_env is specified
    # Also check the function attribute as fallback
    if not conda_env:
        conda_env = getattr(run_eval, "_conda_env", None)
    
    if conda_env:
        # Use conda run with --no-capture-output to see real-time output
        # --live-stream helps with real-time output, --no-capture-output prevents buffering
        cmd = ["conda", "run", "--no-capture-output", "--live-stream", "-n", conda_env] + base_cmd
    else:
        cmd = base_cmd
    
    # Add num_fewshot if specified
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    # Add --log_samples for mmlu_pro
    if eval_method == "mmlu_pro":
        cmd.append("--log_samples")
    
    if dry_run:
        print(f"[DRY-RUN] Would evaluate: {model_path.name}")
        print(f"         Method: {method}, Category: {category}")
        print(f"         Output: {result_file}")
        print(f"         Command: {' '.join(cmd)}")
        return True, "dry-run"
    
    print(f"\n{'='*80}")
    print(f"[INFO] Evaluating: {model_path.name}")
    print(f"[INFO] Method: {method}, Category: {category}")
    print(f"[INFO] Output: {result_file}")
    if conda_env:
        print(f"[INFO] Using conda environment: {conda_env}")
    print(f"[INFO] Full command: {' '.join(cmd)}")
    lm_style = getattr(run_eval, "_lm_eval_style", None) or "main"
    run_env: dict[str, str] | None = None
    if lm_style == "module":
        if LM_EVAL_V2_HARNESS_ROOT.is_dir():
            run_env = os.environ.copy()
            root_s = str(LM_EVAL_V2_HARNESS_ROOT)
            prev = run_env.get("PYTHONPATH", "")
            run_env["PYTHONPATH"] = root_s + (os.pathsep + prev if prev else "")
            print(
                f"[INFO] python -m lm_eval: prepended PYTHONPATH with {root_s} "
                f"(use local harness when lm_eval is not installed in env)"
            )
        else:
            print(
                f"[WARNING] LeaderBoard V2 harness not found: {LM_EVAL_V2_HARNESS_ROOT}\n"
                f"          If you hit 'No module named lm_eval', set LM_EVAL_V2_HARNESS_ROOT "
                f"or install lm-eval in the selected conda env."
            )
    print(f"{'='*80}\n")
    
    try:
        # For conda run, we might need to use shell=True on some systems
        # But first try without shell
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
            env=run_env,
        )
        if eval_method in LM_EVAL_V2_DIR_LAYOUT_METHODS:
            written = sorted(
                result_file.rglob("results_*.json"),
                key=lambda p: p.stat().st_mtime,
            )
            if not written:
                err = (
                    f"lm-eval v2 did not write results_*.json under {result_file}. "
                    "Tracker layout is: <output_path>/<sanitized_pretrained>/results_<time>.json — "
                    "look for a directory named like lb2_*_*.json under "
                    f"{result_file.parent}/"
                )
                print(f"\n[ERROR] {err}")
                return False, err
            print(f"\n[SUCCESS] Evaluated: {model_path.name}")
            print(f"[INFO] Run root (dir): {result_file}/")
            print(f"[INFO] Aggregated metrics JSON: {written[-1]}")
        else:
            print(f"\n[SUCCESS] Evaluated: {model_path.name} -> {result_file}")
        # Postprocess: add mmlu aggregated score + subtasks list
        if eval_method == "mmlu":
            _postprocess_mmlu_results_inplace(result_file)
        success = True
        msg = "success"
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to evaluate {model_path.name}: {e}"
        print(f"\n[ERROR] {error_msg}")
        success = False
        msg = error_msg
    except Exception as e:
        error_msg = f"Unexpected error evaluating {model_path.name}: {e}"
        print(f"\n[ERROR] {error_msg}")
        success = False
        msg = error_msg
    
    return success, msg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluate all merged models using lm_eval."
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default=OUTPUT_MERGED_ROOT,
        help=f"Directory containing merged models (default: {OUTPUT_MERGED_ROOT}).",
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default="./results",
        help="Base directory for evaluation results (default: ./results).",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        choices=["all"] + KNOWN_MODEL_FAMILIES,
        default=MODEL_FAMILY_TO_EVAL,
        help=(
            f"Which model family to evaluate (default: {MODEL_FAMILY_TO_EVAL} from config). "
            f"Choices: {['all'] + KNOWN_MODEL_FAMILIES}"
        ),
    )
    parser.add_argument(
        "--eval-method",
        type=str,
        default=EVAL_METHOD_TO_USE,
        help=f"Evaluation method(s) to use, comma-separated for multiple (default: {EVAL_METHOD_TO_USE} from config). Choices: {', '.join(EVAL_METHODS.keys())}",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=METHODS_TO_EVAL,
        help=f"Comma-separated list of methods to evaluate, or 'all' for all methods (default: {METHODS_TO_EVAL} from config). "
             f"Available methods: {', '.join(KNOWN_METHODS)}",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default=None,
        metavar="N|N1,N2,...",
        help=(
            "lm_eval batch size(s). One integer: same batch for every --eval-method. "
            "Comma-separated list: must have the same length and order as --eval-method (e.g. "
            f"'arc,gsm8k,math_hard' with '1,1,32'). If omitted: math_hard defaults to {_DEFAULT_BATCH_SIZE_MATH_HARD}, "
            f"other eval methods default to {_DEFAULT_BATCH_SIZE_OTHER}."
        ),
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        metavar="N",
        help=(
            "Forwarded to lm-eval --limit: cap examples per task (smoke test). "
            "e.g. 8 with --batch-size 4 ≈ 2 forward steps. "
            "Use a separate --results-base (e.g. ./results_smoke) so you do not mix with full runs."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Model dtype for evaluation (default: float16).",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["all", "base", "calib"],
        default=CATEGORY_TO_EVAL,
        help=f"Filter models by category (default: {CATEGORY_TO_EVAL} from config). Options: 'all', 'base', or 'calib'.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have evaluation results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be evaluated, don't actually run.",
    )
    parser.add_argument(
        "--lm-eval-style",
        type=str,
        choices=["main", "module"],
        default=LM_EVAL_STYLE,
        help=f"LM eval command style: 'main' (python main.py for v1) or 'module' (python -m lm_eval for newer). Default: {LM_EVAL_STYLE}",
    )
    parser.add_argument(
        "--lm-eval-main-path",
        type=str,
        default=LM_EVAL_MAIN_PATH,
        help=(
            "Path to legacy lm_eval v1 main.py (only used when --lm-eval-style=main). "
            f"Default: {LM_EVAL_MAIN_PATH}"
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=OUTPUT_ADAPTER_ROOT,
        help=f"Directory containing LoRA adapter models (default: {OUTPUT_ADAPTER_ROOT}).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help=(
            "Base model path or Hub id for merging adapters. "
            "If not specified, uses default paths from BASE_MODEL_PATHS config."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Passed to lm_eval HuggingFace model as model_args device=..., e.g. cuda:0, cuda:2, cpu. "
            "Alternatively set CUDA_VISIBLE_DEVICES before running this script."
        ),
    )
    parser.add_argument(
        "--parallel-gpus",
        type=str,
        default=None,
        metavar="IDS",
        help=(
            "Comma-separated physical GPU IDs, e.g. 0,1,2,7. "
            "Each eval runs in an OS subprocess (not multiprocessing.Pool), "
            "bound to one GPU via CUDA_VISIBLE_DEVICES and using device=cuda:0. "
            "N GPUs => up to N concurrent eval jobs. "
            "Jobs from all --eval-method entries share one queue. "
            "When set, --device is ignored."
        ),
    )
    args = parser.parse_args()

    parallel_gpu_ids: list[int] | None = None
    if args.parallel_gpus:
        parallel_gpu_ids = _parse_parallel_gpus(args.parallel_gpus)
        if args.device:
            print(
                f"[WARNING] --parallel-gpus={parallel_gpu_ids} is set; ignoring --device={args.device!r}"
            )
    
    merged_dir = Path(args.merged_dir).resolve()
    adapter_dir = Path(args.adapter_dir).resolve()
    results_base = Path(args.results_base).resolve()
    
    # Determine base model path
    base_model_path = args.base_model
    if not base_model_path and args.model_family in BASE_MODEL_PATHS:
        base_model_path = BASE_MODEL_PATHS[args.model_family]
    
    # Adapter families are discovered in --adapter-dir; others in --merged-dir.
    # When model_family is "all", we'll handle each model individually
    use_adapters_for_all = args.model_family in ADAPTER_MODEL_FAMILIES
    
    if use_adapters_for_all or args.model_family == "all":
        if not adapter_dir.exists():
            print(f"[WARNING] Adapter models directory not found: {adapter_dir}")
            if use_adapters_for_all:
                print(f"[ERROR] Adapter directory is required for {args.model_family}")
                sys.exit(1)
        if use_adapters_for_all and not base_model_path:
            print(f"[WARNING] Base model path not configured for {args.model_family}")
            print(
                f"[ERROR] Base model path is required for LoRA merge. "
                f"Set --base-model or configure BASE_MODEL_PATHS / env "
                f"(e.g. QWEN25_7B_INSTRUCT_BASE_MODEL)."
            )
            sys.exit(1)
        if use_adapters_for_all:
            print(f"[INFO] Using adapter directory for {args.model_family}: {adapter_dir}")
            print(f"[INFO] Base model for merging: {base_model_path}")
    
    if not use_adapters_for_all or args.model_family == "all":
        if not merged_dir.exists():
            print(f"[WARNING] Merged models directory not found: {merged_dir}")
            if not use_adapters_for_all and args.model_family != "all":
                print(f"[ERROR] Merged directory is required")
                sys.exit(1)
    
    # Parse comma-separated eval methods
    eval_methods = [m.strip() for m in args.eval_method.split(",")]
    # Validate eval methods
    invalid_eval_methods = [m for m in eval_methods if m not in EVAL_METHODS]
    if invalid_eval_methods:
        print(f"[ERROR] Invalid evaluation methods: {invalid_eval_methods}")
        print(f"       Available methods: {', '.join(EVAL_METHODS.keys())}")
        sys.exit(1)

    try:
        batch_size_map = _build_batch_size_map(eval_methods, args.batch_size)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    print(
        "[INFO] batch_size by eval-method: "
        + ", ".join(f"{m}={batch_size_map[m]}" for m in eval_methods)
    )

    if args.dry_run:
        print(f"[INFO] DRY-RUN mode: no actual evaluation will be performed")
    
    # Discover models by family-specific source (adapter vs merged dirs).
    models = []
    if use_adapters_for_all:
        models = find_adapter_models(adapter_dir, args.model_family)
    elif args.model_family == "all":
        models = []
        for fam in KNOWN_MODEL_FAMILIES:
            if fam in ADAPTER_MODEL_FAMILIES:
                models.extend(find_adapter_models(adapter_dir, fam))
            else:
                models.extend(find_merged_models(merged_dir, fam))
    else:
        models = find_merged_models(merged_dir, args.model_family)
    
    if not models:
        if use_adapters_for_all:
            print(f"[WARNING] No adapter models found in: {adapter_dir}")
        elif args.model_family == "all":
            print(f"[WARNING] No models found in adapter directory ({adapter_dir}) or merged directory ({merged_dir})")
        else:
            print(f"[WARNING] No merged models found in: {merged_dir}")
        sys.exit(0)
    
    # Filter by category if requested
    if args.category != "all":
        filtered_models = []
        for fam, model in models:
            is_calib = is_calib_model(model.name)
            if args.category == "calib" and is_calib:
                filtered_models.append((fam, model))
            elif args.category == "base" and not is_calib:
                filtered_models.append((fam, model))
        models = filtered_models
    
    # Filter by methods if requested
    want_baseline = False
    if args.methods.lower() != "all":
        method_list = [m.strip().lower() for m in args.methods.split(",")]
        if "baseline" in method_list:
            want_baseline = True
            # baseline is not a directory-scanned model; remove it from the filter for real models
            method_list = [m for m in method_list if m != "baseline"]
        # Validate methods
        invalid_methods = [m for m in method_list if m not in KNOWN_METHODS]
        if invalid_methods:
            print(f"[ERROR] Invalid methods: {invalid_methods}")
            print(f"       Available methods: {', '.join(KNOWN_METHODS)}")
            sys.exit(1)
        
        filtered_models = []
        for fam, model in models:
            method = extract_method_name(model.name, fam)
            if method in method_list:
                filtered_models.append((fam, model))
        models = filtered_models
    else:
        # Keep backward-compatible behavior: "all" means all scanned methods.
        # baseline is opt-in via explicit --methods baseline (or list containing baseline).
        want_baseline = False
    
    if not models:
        category_msg = f"category filter: {args.category}" if args.category != "all" else ""
        method_msg = f"method filter: {args.methods}" if args.methods.lower() != "all" else ""
        filter_msgs = [m for m in [category_msg, method_msg] if m]
        filter_msg = ", ".join(filter_msgs) if filter_msgs else "filters"
        if want_baseline:
            print(f"[INFO] No adapter/merged models found matching {filter_msg}; will run baseline only.")
            models = []
        else:
            print(f"[WARNING] No models found matching {filter_msg}")
            sys.exit(0)
    
    print(f"[INFO] Found {len(models)} model(s), eval-methods: {', '.join(eval_methods)}")
    
    # Process evaluation methods in user order; each method evaluates all selected models.
    # With --parallel-gpus, all (eval_method x model) jobs go into one shared queue.
    all_results: list[tuple[Any, ...]] = []
    script_path = Path(__file__).resolve()
    resolved_lm_main = str(Path(args.lm_eval_main_path).expanduser().resolve())
    use_parallel = parallel_gpu_ids is not None
    all_parallel_payloads: list[dict[str, Any]] = []

    for eval_idx, eval_method in enumerate(eval_methods, 1):
        print(f"\n{'#'*80}")
        print(f"[EVAL METHOD {eval_idx}/{len(eval_methods)}] {eval_method}")
        print(f"{'#'*80}\n")

        task_name, file_prefix, model_type, num_fewshot = EVAL_METHODS[eval_method]
        batch_size_this = batch_size_map[eval_method]
        print(f"[INFO] Task name: {task_name}")
        print(f"[INFO] File prefix: {file_prefix}")
        print(f"[INFO] batch_size: {batch_size_this}")

        auto_style, auto_conda_env = _resolve_eval_runtime_for_method(
            eval_method, args.lm_eval_style
        )
        if not use_parallel:
            run_eval._lm_eval_style = auto_style
            run_eval._lm_eval_main_path = args.lm_eval_main_path
            run_eval._conda_env = auto_conda_env

        # Filter out existing if requested
        models_to_eval = models
        if args.skip_existing:
            filtered = []
            skipped = []
            for fam, model in models:
                method = extract_method_name(model.name, fam)
                is_calib = is_calib_model(model.name)
                if check_result_exists(fam, model, results_base, method, is_calib, eval_method):
                    skipped.append((fam, model))
                else:
                    filtered.append((fam, model))
            if skipped:
                print(f"[INFO] Skipping {len(skipped)} already-evaluated model(s) for {eval_method}:")
                for fam, model in skipped:
                    method = extract_method_name(model.name, fam)
                    print(f"  - {model.name} (family={fam}, method={method})")
                print()
            models_to_eval = filtered

        if not models_to_eval:
            print(f"[INFO] All models already evaluated for {eval_method} (or none to evaluate).")

        print(f"[INFO] Will evaluate {len(models_to_eval)} model(s) for {eval_method}\n")

        # Optionally evaluate the raw base model (no adapter / no merge)
        if want_baseline:
            baseline_fams = KNOWN_MODEL_FAMILIES[:] if args.model_family == "all" else [args.model_family]
            for fam in baseline_fams:
                base_ref = None
                if args.base_model and args.model_family != "all":
                    base_ref = args.base_model
                else:
                    base_ref = BASE_MODEL_PATHS.get(fam)

                if not base_ref:
                    print(f"[WARNING] baseline requested but base model path not configured for {fam}; skipping baseline.")
                    continue

                base_path = Path(base_ref)
                if not base_path.exists():
                    print(
                        f"[WARNING] baseline requested but base model path does not exist for {fam}: {base_ref}\n"
                        f"          Fix: set BASE_MODEL_PATHS[{fam!r}] or pass --base-model (when evaluating a single family)."
                    )
                    continue

                if args.skip_existing and check_result_exists(
                    fam,
                    base_path,
                    results_base,
                    method="baseline",
                    is_calib=False,
                    eval_method=eval_method,
                ):
                    print(f"[INFO] Skipping already-evaluated baseline for {fam} ({eval_method}).")
                    continue

                if use_parallel:
                    all_parallel_payloads.append(
                        _make_parallel_job_payload(
                            fam=fam,
                            model_path=base_path,
                            results_base=results_base,
                            method="baseline",
                            is_calib=False,
                            eval_method=eval_method,
                            batch_size=batch_size_this,
                            dtype=args.dtype,
                            dry_run=args.dry_run,
                            conda_env=auto_conda_env,
                            use_adapter=False,
                            merge_base=None,
                            lm_eval_style=auto_style,
                            lm_eval_main_path=resolved_lm_main,
                            limit=args.limit,
                        )
                    )
                    print(
                        f"[BASELINE] queued #{len(all_parallel_payloads)}: "
                        f"{eval_method} | family={fam} -> {base_path}"
                    )
                else:
                    print(f"\n[BASELINE] Evaluating base model for family={fam} -> {base_path}")
                    success, msg = run_eval(
                        fam,
                        base_path,
                        results_base,
                        method="baseline",
                        is_calib=False,
                        eval_method=eval_method,
                        batch_size=batch_size_this,
                        dtype=args.dtype,
                        dry_run=args.dry_run,
                        conda_env=auto_conda_env,
                        use_adapter=False,
                        base_model=None,
                        device=args.device,
                        limit=args.limit,
                    )
                    all_results.append((fam, base_path, "baseline", success, msg, eval_method))

        if use_parallel:
            if parallel_gpu_ids is None:
                raise RuntimeError("internal: parallel_gpu_ids is None")
            for i, (fam, model) in enumerate(models_to_eval, 1):
                method = extract_method_name(model.name, fam)
                is_calib = is_calib_model(model.name)
                print(
                    f"[QUEUE #{len(all_parallel_payloads) + 1}] {eval_method} | "
                    f"{i}/{len(models_to_eval)} {model.name} (family={fam}, loss={method})"
                )
                use_adapter_for_this = fam in ADAPTER_MODEL_FAMILIES
                merge_base = None
                if use_adapter_for_this:
                    merge_base = (
                        base_model_path
                        if args.model_family != "all"
                        else BASE_MODEL_PATHS.get(fam)
                    )
                all_parallel_payloads.append(
                    _make_parallel_job_payload(
                        fam=fam,
                        model_path=model,
                        results_base=results_base,
                        method=method,
                        is_calib=is_calib,
                        eval_method=eval_method,
                        batch_size=batch_size_this,
                        dtype=args.dtype,
                        dry_run=args.dry_run,
                        conda_env=auto_conda_env,
                        use_adapter=use_adapter_for_this,
                        merge_base=merge_base,
                        lm_eval_style=auto_style,
                        lm_eval_main_path=resolved_lm_main,
                        limit=args.limit,
                    )
                )
        else:
            for i, (fam, model) in enumerate(models_to_eval, 1):
                method = extract_method_name(model.name, fam)
                is_calib = is_calib_model(model.name)
                print(f"\n[{i}/{len(models_to_eval)}] Processing: {model.name} (family={fam}, method={method})")
                use_adapter_for_this = fam in ADAPTER_MODEL_FAMILIES
                merge_base = None
                if use_adapter_for_this:
                    merge_base = (
                        base_model_path
                        if args.model_family != "all"
                        else BASE_MODEL_PATHS.get(fam)
                    )

                success, msg = run_eval(
                    fam,
                    model,
                    results_base,
                    method,
                    is_calib,
                    eval_method=eval_method,
                    batch_size=batch_size_this,
                    dtype=args.dtype,
                    dry_run=args.dry_run,
                    conda_env=auto_conda_env,
                    use_adapter=use_adapter_for_this,
                    base_model=merge_base,
                    device=args.device,
                    limit=args.limit,
                )
                all_results.append((fam, model, method, success, msg, eval_method))

    if use_parallel:
        if parallel_gpu_ids is None:
            raise RuntimeError("internal: parallel_gpu_ids is None")
        if all_parallel_payloads:
            print(
                f"\n[INFO] --parallel-gpus: {len(all_parallel_payloads)} total jobs across methods; "
                f"up to {len(parallel_gpu_ids)} concurrent workers (physical GPUs {parallel_gpu_ids})\n"
            )
            pr = _run_parallel_gpu_pool(script_path, all_parallel_payloads, parallel_gpu_ids)
            all_results.extend(pr)
        else:
            print("\n[INFO] --parallel-gpus: no pending jobs (possibly all skipped by --skip-existing).")

    failed = [r for r in all_results if not r[3]]
    if failed:
        print(f"\n[WARNING] {len(failed)} evaluation(s) failed:")
        for fam, model, method, _, msg, eval_method in failed:
            print(f"  - {model.name} (family={fam}, method={method}, eval={eval_method}): {msg}")

    if args.dry_run:
        print("\n[NOTE] This was a dry-run. Use without --dry-run to actually evaluate.")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--parallel-worker-job":
        _parallel_worker_run_json(Path(sys.argv[2]))
    else:
        main()
