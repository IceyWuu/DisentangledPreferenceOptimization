#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch OOD evaluation on AlpacaEval 2.0 (805 instructions) for base vs calib models
trained with different preference losses, and run pairwise "tournament" using AlpacaEval.

Changes vs your current draft:
- DO NOT rely on HuggingFace datasets format. Your local folder contains alpaca_eval.json etc.
- Load instructions directly from local alpaca_eval.json (805).
- Keep a fallback to HF datasets only if you explicitly pass --use-hf-datasets.
- Add --alpaca-eval-json to point directly to ../ModelAndDatasets/tatsu-lab/alpaca_eval/alpaca_eval.json
- Instruction builder supports (instruction + optional input).

Pipeline:
1) Scan model dirs (merged or LoRA adapters) and group by (family, method).
2) Pick latest base and latest calib per method.
3) Generate responses on AlpacaEval set (full 805 or random sample N=200).
4) Run `alpaca_eval evaluate --model_outputs <calib> --reference_outputs <base>`.
5) Parse annotations.json and summarize win-rate (calib beats base).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False


# =============================================================================
# CONFIGURATION (defaults; can be overridden by CLI)
# =============================================================================

# Resolve repo root robustly (so script works no matter your current working directory).
# This file lives in <repo_root>/outputs/.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

KNOWN_METHODS = [
    "bce", "dpo", "ddro", "cpo", "ipo", "simpo", "kto", "slic", "lsif", "ukl",
]
KNOWN_MODEL_FAMILIES = ["pythia-2b", "mistral-7b"]

# Repo defaults (your workspace layout puts everything under ./outputs/)
OUTPUT_MERGED_ROOT = os.environ.get("OUTPUT_MERGED_ROOT", str(_REPO_ROOT / "outputs" / "output_merged"))
OUTPUT_ADAPTER_ROOT = os.environ.get("OUTPUT_ADAPTER_ROOT", str(_REPO_ROOT / "outputs"))

MISTRAL_7B_BASE_PRIMARY = "../ModelAndDatasets/alignment-handbook/local_models/mistralai/Mistral-7B-v0.1"
# If primary path is missing, try this (common alternate layout under ModelAndDatasets/)
MISTRAL_7B_BASE_FALLBACK = "../ModelAndDatasets/local_models/mistralai/Mistral-7B-v0.1"

BASE_MODEL_PATHS = {
    "mistral-7b": os.environ.get(
        "MISTRAL_7B_BASE_MODEL",
        MISTRAL_7B_BASE_PRIMARY,
    ),
    "pythia-2b": os.environ.get(
        "PYTHIA_2B_BASE_MODEL",
        "../ModelAndDatasets/alignment-handbook/pythia-2.8b",
    ),
}

DEFAULT_RESULTS_BASE = str(_REPO_ROOT / "outputs" / "results" / "compare_results")
# AlpacaEval / batch outputs alongside other hh-dataset-output artifacts
HH_DATASET_OUTPUT_RESULTS_BASE = str(_REPO_ROOT / "outputs" / "hh-dataset-output" / "results")

# Generation defaults (keep consistent across all models)
DEFAULT_DTYPE = "float16"     # auto/bfloat16/float16/float32
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_BATCH_SIZE = 1

# Local AlpacaEval folder (your case)
ALPACA_LOCAL_DIR_DEFAULT = "../ModelAndDatasets/tatsu-lab/alpaca_eval"
ALPACA_EVAL_JSON_DEFAULT = os.path.join(ALPACA_LOCAL_DIR_DEFAULT, "alpaca_eval.json")

# Optional HF fallback (only used if --use-hf-datasets)
ALPACA_DATASET_ID = "tatsu-lab/alpaca_eval"
ALPACA_CONFIG = "alpaca_eval"
ALPACA_SPLIT = "eval"

# AlpacaEval annotator config (recommended for 2.0)
DEFAULT_ANNOTATORS_CONFIG = "weighted_alpaca_eval_gpt4_turbo"  # may require logprobs
# DEFAULT_ANNOTATORS_CONFIG = "alpaca_eval_gpt4_turbo_fn"  # may require logprobs


# =============================================================================
# Helpers
# =============================================================================

def resolve_pretrained_local_or_hub(p: str) -> str:
    """
    Expand repo-root-relative paths to absolute for transformers/huggingface_hub.
    Strings like org/model are left as-is unless that path exists under the repo.
    """
    p = (p or "").strip()
    if not p:
        return p
    expanded = Path(p).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve())
    candidate = (_REPO_ROOT / expanded).resolve()
    if candidate.exists():
        return str(candidate)
    if ".." in p or p.startswith("./") or p.startswith("../"):
        return str(candidate)
    return p


def _is_usable_local_model_dir(p: str) -> bool:
    path = Path(p)
    try:
        return path.is_dir() and (path / "config.json").is_file()
    except OSError:
        return False


def _looks_like_hf_repo_id(s: str) -> bool:
    """Two-segment Hub ids only (e.g. mistralai/Mistral-7B-v0.1); not local paths."""
    s = (s or "").strip()
    if not s or ".." in s or s.startswith(("/", "\\", ".", "~")):
        return False
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", s))


def resolve_mistral_base_with_fallback(raw: str) -> str:
    """
    Try MISTRAL_7B_BASE_MODEL / default handbook path first, then local_models layout.
    Skips fallback when raw resolves to a Hub repo id.
    """
    raw = (raw or "").strip()
    if not raw:
        return raw
    primary = resolve_pretrained_local_or_hub(raw)
    if _looks_like_hf_repo_id(primary):
        return primary
    if _is_usable_local_model_dir(primary):
        return primary
    if raw != MISTRAL_7B_BASE_FALLBACK:
        alt = resolve_pretrained_local_or_hub(MISTRAL_7B_BASE_FALLBACK)
        if _is_usable_local_model_dir(alt):
            print(
                f"[INFO] Mistral base not found at {primary}; using fallback {alt}",
                file=sys.stderr,
            )
            return alt
    return primary


def is_calib_model(name: str) -> bool:
    # Keep this intentionally simple but robust to common naming variants.
    s = name.lower()
    return ("calib" in s) or ("calibration" in s)

def extract_method_name(model_name: str, model_family: str) -> str:
    prefix = f"{model_family}-"
    name = model_name
    if name.startswith(prefix):
        name = name[len(prefix):]
    name = name.replace("-merged", "")

    for method in KNOWN_METHODS:
        pattern = rf"(?:^|-)({method})(?:-|$)"
        m = re.search(pattern, name, re.IGNORECASE)
        if m:
            return method.lower()

    parts = name.split("-")
    for p in parts:
        if p and p not in ["calib", "ema", "ema2", "log1p", "db"] and p.isalpha() and len(p) <= 8:
            return p.lower()
    return parts[0].lower() if parts else "unknown"

def _extract_last_timestamp_from_name(s: str) -> Optional[str]:
    matches = re.findall(r"\d{8}_\d{6}", s)
    return matches[-1] if matches else None

def _extract_last_timestamp_from_path(p: Path) -> Optional[str]:
    for name in (p.name, p.parent.name):
        ts = _extract_last_timestamp_from_name(name)
        if ts:
            return ts
    return None

def _pick_torch_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "auto").lower()
    if s == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")

def _looks_like_adapter_dir(p: Path) -> bool:
    return (p / "adapter_config.json").exists() and (
        (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()
    )

def _looks_like_merged_model_dir(p: Path) -> bool:
    return (p / "config.json").exists()

def find_merged_models(merged_dir: Path, model_family: str) -> List[Tuple[str, Path]]:
    families = KNOWN_MODEL_FAMILIES[:] if model_family == "all" else [model_family]
    out: List[Tuple[str, Path]] = []
    for fam in families:
        for item in merged_dir.glob(f"{fam}-*-merged"):
            if item.is_dir() and _looks_like_merged_model_dir(item):
                out.append((fam, item))

    def sort_key(x: Tuple[str, Path]) -> Tuple[str, int, str]:
        fam, p = x
        method = extract_method_name(p.name, fam)
        cat = 1 if is_calib_model(p.name) else 0
        ts = _extract_last_timestamp_from_path(p) or ""
        return (method, cat, ts)

    return sorted(out, key=sort_key)

def find_adapter_models(adapter_dir: Path, model_family: str) -> List[Tuple[str, Path]]:
    families = KNOWN_MODEL_FAMILIES[:] if model_family == "all" else [model_family]
    out: List[Tuple[str, Path]] = []
    for fam in families:
        for item in adapter_dir.glob(f"{fam}-*"):
            if not item.is_dir():
                continue
            if item.name.endswith("-merged"):
                continue
            if _looks_like_adapter_dir(item):
                out.append((fam, item))
            else:
                # also accept checkpoint-* dirs
                for ckpt in item.glob("checkpoint-*"):
                    if ckpt.is_dir() and _looks_like_adapter_dir(ckpt):
                        out.append((fam, ckpt))

    def sort_key(x: Tuple[str, Path]) -> Tuple[str, int, str]:
        fam, p = x
        method = extract_method_name(p.name, fam)
        cat = 1 if is_calib_model(p.name) else 0
        ts = _extract_last_timestamp_from_path(p) or ""
        return (method, cat, ts)

    return sorted(out, key=sort_key)

@dataclass
class ModelEntry:
    family: str
    method: str
    is_calib: bool
    path: Path
    ts: str

def group_latest_base_calib(models: List[Tuple[str, Path]]) -> Dict[Tuple[str, str], Dict[str, ModelEntry]]:
    """
    Return mapping:
      (family, method) -> {"base": ModelEntry, "calib": ModelEntry}
    Pick the latest by timestamp (if present), otherwise by name order.
    """
    groups: Dict[Tuple[str, str], Dict[str, List[ModelEntry]]] = {}
    for fam, p in models:
        method = extract_method_name(p.name, fam)
        calib = is_calib_model(p.name)
        ts = _extract_last_timestamp_from_path(p) or ""
        e = ModelEntry(family=fam, method=method, is_calib=calib, path=p, ts=ts)
        key = (fam, method)
        groups.setdefault(key, {"base": [], "calib": []})
        groups[key]["calib" if calib else "base"].append(e)

    out: Dict[Tuple[str, str], Dict[str, ModelEntry]] = {}
    for key, bucket in groups.items():
        result: Dict[str, ModelEntry] = {}
        for cat in ["base", "calib"]:
            lst = bucket[cat]
            if not lst:
                continue
            lst_sorted = sorted(lst, key=lambda x: (x.ts, x.path.name))
            result[cat] = lst_sorted[-1]  # latest
        if "base" in result and "calib" in result:
            out[key] = result
    return out


def scan_models(
    merged_dir: Path,
    adapter_dir: Path,
    model_family: str,
    *,
    scan_merged: bool = True,
    scan_adapters: bool = True,
) -> List[Tuple[str, Path]]:
    """
    Scan and combine model candidates from both merged models and LoRA adapter dirs.
    This avoids the common failure mode where base+calib live in different roots.
    """
    models: List[Tuple[str, Path]] = []
    if scan_merged and merged_dir.exists():
        models.extend(find_merged_models(merged_dir, model_family))
    if scan_adapters and adapter_dir.exists():
        models.extend(find_adapter_models(adapter_dir, model_family))
    # de-dup by resolved path
    seen = set()
    uniq: List[Tuple[str, Path]] = []
    for fam, p in models:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((fam, p))
    return uniq


# ------------------------------
# Dataset loading (LOCAL JSON first)
# ------------------------------

def load_alpacaeval_examples_from_json(json_path: str) -> List[dict]:
    p = Path(json_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"alpaca_eval.json not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"alpaca_eval.json should be a list[dict], got {type(data)}")

    cleaned: List[dict] = []
    for ex in data:
        if not isinstance(ex, dict):
            continue
        inst = ex.get("instruction")
        if isinstance(inst, str) and inst.strip():
            cleaned.append(ex)
    if not cleaned:
        raise ValueError("No valid examples with 'instruction' found in alpaca_eval.json")

    return cleaned

def load_alpacaeval_examples_hf(local_repo_or_id: str) -> List[dict]:
    if not _HAS_DATASETS:
        raise RuntimeError("Missing dependency: datasets. Install via `pip install datasets`.")

    lp = Path(local_repo_or_id).expanduser().resolve()
    if lp.exists():
        ds = load_dataset(str(lp), ALPACA_CONFIG, split=ALPACA_SPLIT)
    else:
        ds = load_dataset(ALPACA_DATASET_ID, ALPACA_CONFIG, split=ALPACA_SPLIT)

    # convert to list[dict]
    return [dict(x) for x in ds]

def example_to_user_text(ex: dict) -> str:
    inst = ex.get("instruction", "")
    inp = ex.get("input", "")
    if isinstance(inp, str) and inp.strip():
        return f"{inst}\n\n{inp}"
    return inst


# ------------------------------
# Sampling
# ------------------------------

def sample_indices(n_total: int, n: Optional[int], seed: int, random_sample: bool) -> List[int]:
    if n is None or n <= 0 or n >= n_total:
        return list(range(n_total))
    if not random_sample:
        return list(range(n))
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    return perm[:n]


# ------------------------------
# Prompting / generation
# ------------------------------

def build_prompt(tokenizer, user_text: str) -> str:
    """
    Prefer chat template if available.
    Fallback: raw text.
    """
    try:
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": user_text}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return user_text

def generate_outputs_hf(
    model_dir: Path,
    out_json: Path,
    examples: List[dict],
    indices: List[int],
    *,
    dtype: str,
    device_map: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    base_model_for_adapter: Optional[str] = None,
    merge_adapter: bool = False,
) -> None:
    """
    Generate outputs and save as JSON list with fields expected by AlpacaEval:
      {"instruction": ..., "output": ..., "generator": ...}
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # resume if exists
    done_map: Dict[str, str] = {}
    if out_json.exists():
        try:
            old = json.loads(out_json.read_text(encoding="utf-8"))
            if isinstance(old, list):
                for ex in old:
                    inst = ex.get("instruction")
                    out = ex.get("output")
                    if isinstance(inst, str) and isinstance(out, str):
                        done_map[inst] = out
        except Exception:
            pass

    torch_dtype = _pick_torch_dtype(dtype)
    is_adapter = _looks_like_adapter_dir(model_dir)
    if is_adapter and not _HAS_PEFT:
        raise RuntimeError("This looks like a LoRA adapter dir but peft is not installed. Install via `pip install peft`.")

    # Load tokenizer (prefer adapter dir; fallback to base model if needed)
    if is_adapter and base_model_for_adapter:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model_for_adapter, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model
    if is_adapter:
        if not base_model_for_adapter:
            raise ValueError(f"Adapter model detected but base model not provided: {model_dir}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_for_adapter,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, str(model_dir))
        if merge_adapter:
            model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

    model.eval()

    results: List[dict] = []
    generator_name = model_dir.name

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # selected user texts
    selected_user_texts = [example_to_user_text(examples[i]) for i in indices]
    selected_instructions = [examples[i].get("instruction", "") for i in indices]

    pbar = tqdm(range(0, len(selected_user_texts), batch_size), desc=f"GEN {generator_name}", ncols=100)
    for start in pbar:
        batch_user = selected_user_texts[start:start + batch_size]
        batch_inst_key = selected_instructions[start:start + batch_size]

        need_gen: List[Tuple[int, str, str]] = []  # (pos, inst_key, user_text)
        batch_outputs: List[Optional[str]] = [None] * len(batch_user)

        for j, (inst_key, user_text) in enumerate(zip(batch_inst_key, batch_user)):
            # key by *instruction* for compatibility with AlpacaEval outputs format
            if inst_key in done_map:
                batch_outputs[j] = done_map[inst_key]
            else:
                need_gen.append((j, inst_key, user_text))

        if need_gen:
            prompts = [build_prompt(tokenizer, user_text) for _, _, user_text in need_gen]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}

            with torch.inference_mode():
                out_ids = model.generate(**enc, **gen_kwargs)

            input_lens = enc["attention_mask"].sum(dim=1).tolist()
            decoded: List[str] = []
            for k in range(out_ids.size(0)):
                gen_part = out_ids[k, input_lens[k]:]
                decoded.append(tokenizer.decode(gen_part, skip_special_tokens=True).strip())

            for (pos, inst_key, _), out_text in zip(need_gen, decoded):
                batch_outputs[pos] = out_text
                done_map[inst_key] = out_text

        for inst_key, out_text in zip(batch_inst_key, batch_outputs):
            results.append({
                "instruction": inst_key,
                "output": out_text if out_text is not None else "",
                "generator": generator_name,
            })

        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------
# AlpacaEval
# ------------------------------

def run_alpaca_eval_pairwise(
    alpaca_eval_bin: str,
    model_outputs_json: Path,
    reference_outputs_json: Path,
    out_dir: Path,
    annotators_config: str,
    name: str,
    dry_run: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        alpaca_eval_bin, "evaluate",
        "--model_outputs", str(model_outputs_json),
        "--reference_outputs", str(reference_outputs_json),
        "--annotators_config", annotators_config,
        "--output_path", str(out_dir),
        "--name", name,
    ]
    print(f"[ALPACA_EVAL] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)

def find_annotations_json(eval_out_dir: Path) -> Optional[Path]:
    cand = list(eval_out_dir.rglob("annotations.json"))
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime)
    return cand[-1]

def parse_winrate_from_annotations(ann_path: Path) -> dict:
    """
    win_rate(output_2) = mean(preference - 1)
    preference in [1, 2], closer to 2 => output_2 preferred.
    """
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"annotations.json is not a list: {ann_path}")

    prefs: List[float] = []
    len1: List[int] = []
    len2: List[int] = []
    for ex in data:
        if not isinstance(ex, dict):
            continue
        pref = ex.get("preference")
        if isinstance(pref, (int, float)):
            prefs.append(float(pref))
        o1 = ex.get("output_1")
        o2 = ex.get("output_2")
        if isinstance(o1, str):
            len1.append(len(o1.split()))
        if isinstance(o2, str):
            len2.append(len(o2.split()))

    if not prefs:
        raise ValueError("No valid preference values found in annotations.")

    scores = [p - 1.0 for p in prefs]
    n = len(scores)
    mean = sum(scores) / n
    var = sum((x - mean) ** 2 for x in scores) / max(n - 1, 1)
    stderr = (var ** 0.5) / (n ** 0.5)

    return {
        "n": n,
        "win_rate": mean,
        "win_rate_stderr": stderr,
        "avg_len_reference": (sum(len1) / len(len1)) if len1 else None,
        "avg_len_model": (sum(len2) / len(len2)) if len2 else None,
    }

def judge_pairwise_simple(
    *,
    out_path: Path,
    instructions: List[str],
    base_outputs: List[str],
    calib_outputs: List[str],
    judge_model: str,
    max_concurrency: int = 8,
    retry_ties: bool = True,
    max_answer_chars: int = 4000,
) -> Path:
    """
    Pure-text judge via OpenAI-compatible /chat/completions.
    Writes annotations.json with fields: instruction, output_1, output_2, preference.
      preference: 1.0 (base wins), 2.0 (calib wins), 1.5 (tie/invalid/filtered)
    This format is compatible with parse_winrate_from_annotations().
    """
    import asyncio
    import time

    try:
        import httpx
    except Exception as e:
        raise RuntimeError("Missing dependency httpx. Install: `pip install -U httpx`") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is empty. For OpenRouter set: https://openrouter.ai/api/v1")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty.")

    url = f"{base_url}/chat/completions"

    # Optional OpenRouter headers (harmless for others)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if os.environ.get("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
    if os.environ.get("OPENROUTER_X_TITLE"):
        headers["X-Title"] = os.environ["OPENROUTER_X_TITLE"]

    # resume (by default we will *retry ties* because ties are often caused by
    # parse failures / API failures / too-long prompts, and otherwise reruns
    # won't change anything.)
    done = {}
    if out_path.exists():
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(old, list):
                for ex in old:
                    i = ex.get("idx")
                    pref = ex.get("preference")
                    if not isinstance(i, int) or pref is None:
                        continue
                    # If retry_ties=True, don't treat 1.5 as done.
                    if retry_ties and float(pref) == 1.5:
                        continue
                    done[i] = ex
        except Exception:
            pass

    system = (
        "You are a strict judge comparing two answers to the same instruction.\n"
        "Judge by helpfulness, correctness, and instruction-following.\n"
        "If you cannot judge or content is unsafe/filtered, output 0.\n"
        "Return ONLY one character among: 1, 2, 0.\n"
        "1 means Answer A is better; 2 means Answer B is better; 0 means tie.\n"
    )

    def _clip(s: str) -> str:
        s = s or ""
        if max_answer_chars and len(s) > max_answer_chars:
            return s[:max_answer_chars] + "\n\n[TRUNCATED]"
        return s

    def user_prompt(inst: str, a: str, b: str) -> str:
        return (
            f"Instruction:\n{inst}\n\n"
            f"Answer A:\n{_clip(a)}\n\n"
            f"Answer B:\n{_clip(b)}\n\n"
            "Which is better? Return only 1, 2, or 0."
        )

    n = len(instructions)
    if not (len(base_outputs) == len(calib_outputs) == n):
        raise ValueError("Length mismatch among instructions/base_outputs/calib_outputs.")

    sem = asyncio.Semaphore(max_concurrency)
    results = [None] * n

    def _truncate_err(s: str, n: int = 800) -> str:
        s = str(s)
        return s if len(s) <= n else (s[:n] + "...[TRUNCATED]")

    async def one(i: int, client):
        if i in done:
            results[i] = done[i]
            return

        payload = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt(instructions[i], base_outputs[i], calib_outputs[i])},
            ],
            "temperature": 0.0,
            # Encourage a single-token answer on OpenAI-compatible APIs.
            "max_tokens": 1,
            "stream": False,
        }

        # retry a bit for transient failures
        max_retry = 3
        for t in range(max_retry):
            try:
                async with sem:
                    r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                j = r.json()
                txt = (j.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

                # Robust parse: find first digit 0/1/2 anywhere.
                m = re.search(r"[012]", txt)
                vote = m.group(0) if m else ""

                if vote == "1":
                    pref = 1.0
                elif vote == "2":
                    pref = 2.0
                elif vote == "0":
                    pref = 1.5
                else:
                    # empty / filtered / weird => tie
                    pref = 1.5

                ex = {
                    "idx": i,
                    "instruction": instructions[i],
                    "output_1": base_outputs[i],
                    "output_2": calib_outputs[i],
                    "preference": pref,
                    "judge_text": txt,
                }
                results[i] = ex

                # incremental save
                out_path.write_text(
                    json.dumps([x for x in results if x is not None], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return
            except Exception as e:
                if t == max_retry - 1:
                    # final failure => tie, but do not crash whole run
                    err_payload = {"error": repr(e)}
                    # If it's an HTTP error, also store status + response body for debugging.
                    try:
                        if isinstance(e, httpx.HTTPStatusError):
                            err_payload["status_code"] = int(e.response.status_code)
                            err_payload["response_text"] = _truncate_err(e.response.text)
                            err_payload["response_headers"] = dict(e.response.headers)
                    except Exception:
                        pass
                    ex = {
                        "idx": i,
                        "instruction": instructions[i],
                        "output_1": base_outputs[i],
                        "output_2": calib_outputs[i],
                        "preference": 1.5,
                        **err_payload,
                    }
                    results[i] = ex
                    out_path.write_text(
                        json.dumps([x for x in results if x is not None], ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    return
                await asyncio.sleep(1.0 + t)

    async def run_all():
        async with httpx.AsyncClient(timeout=120) as client:
            await asyncio.gather(*[one(i, client) for i in range(n)])

    asyncio.run(run_all())

    out_path.write_text(
        json.dumps([x for x in results if x is not None], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser("AlpacaEval2 OOD pairwise tournament: calib vs base")
    parser.add_argument("--merged-dir", type=str, default=OUTPUT_MERGED_ROOT)
    parser.add_argument("--adapter-dir", type=str, default=OUTPUT_ADAPTER_ROOT)
    parser.add_argument(
        "--results-base",
        type=str,
        default=None,
        help=(
            "Base results dir. Final outputs under "
            "<results-base>/<model-family>/alpacaeval2_ood/... "
            f"Default: {DEFAULT_RESULTS_BASE}, or {HH_DATASET_OUTPUT_RESULTS_BASE} if --hh-dataset-results."
        ),
    )
    parser.add_argument(
        "--hh-dataset-results",
        action="store_true",
        help=(
            "Write results under outputs/hh-dataset-output/results/<model-family>/alpacaeval2_ood/ "
            "(overrides default base; ignored if --results-base is set)."
        ),
    )

    parser.add_argument("--model-family", type=str, choices=["all"] + KNOWN_MODEL_FAMILIES, default="pythia-2b")
    parser.add_argument("--methods", type=str, default="dpo,bce,cpo,ddro,lsif,simpo",
                        help="Comma-separated or 'all'")
    parser.add_argument(
        "--scan",
        type=str,
        default="both",
        choices=["both", "merged", "adapters"],
        help="Where to scan models from. Default 'both' (recommended).",
    )

    # dataset
    parser.add_argument(
        "--alpaca-eval-json",
        type=str,
        default=ALPACA_EVAL_JSON_DEFAULT,
        help="Local path to alpaca_eval.json (805 examples).",
    )
    parser.add_argument(
        "--use-hf-datasets",
        action="store_true",
        help="If set, load AlpacaEval via datasets.load_dataset (HF). Otherwise use local alpaca_eval.json.",
    )
    parser.add_argument(
        "--alpaca-local-path",
        type=str,
        default=ALPACA_LOCAL_DIR_DEFAULT,
        help="Local path to tatsu-lab/alpaca_eval repo, only used when --use-hf-datasets is set.",
    )

    # sampling
    parser.add_argument("--num-samples", type=int, default=0,
                        help="0 means full 805; otherwise evaluate on N samples.")
    parser.add_argument("--sample-random", action="store_true",
                        help="If set, randomly sample N (default is first N).")
    parser.add_argument("--seed", type=int, default=42)

    # generation
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument("--device-map", type=str, default=DEFAULT_DEVICE_MAP)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--merge-adapter", action="store_true",
                        help="If adapter model, merge_and_unload before generation (more VRAM, sometimes faster).")
    parser.add_argument("--base-model", type=str, default="",
                        help="Base model path for adapters (repo-root-relative or absolute). "
                             "If empty, uses BASE_MODEL_PATHS[family]. Hub ids like org/model are supported.")

    # alpaca_eval judge
    parser.add_argument("--alpaca-eval-bin", type=str, default="alpaca_eval")
    parser.add_argument("--annotators-config", type=str, default=DEFAULT_ANNOTATORS_CONFIG)
    parser.add_argument("--judge-api-key", type=str, default="",
                        help="Leave empty to use env OPENAI_API_KEY. Fill with your 智增增 key.")
    parser.add_argument("--judge-base-url", type=str, default="",
                        help="Optional. If your endpoint is OpenAI-compatible, set base url here.")
        # >>> ADD THESE (simple judge recommended)
    parser.add_argument(
        "--judge-mode",
        type=str,
        choices=["simple", "alpaca_eval"],
        default="simple",
        help="simple: use OpenAI-compatible API judge (robust, no logprobs/tools). "
             "alpaca_eval: call `alpaca_eval evaluate` with annotators_config (may require logprobs/tools).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.environ.get("ALPACA_JUDGE_MODEL", "openai/gpt-4o-mini"),
        help="Used when --judge-mode=simple. Example for OpenRouter: openai/gpt-4o-mini.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=8,
        help="Used when --judge-mode=simple. Increase for speed, decrease if rate-limited.",
    )
    parser.add_argument(
        "--judge-retry-ties",
        action="store_true",
        help="Used when --judge-mode=simple. If set, rerun will re-judge previous ties (preference=1.5).",
    )
    parser.add_argument(
        "--judge-max-answer-chars",
        type=int,
        default=4000,
        help="Used when --judge-mode=simple. Truncate each answer to this many characters in the judge prompt.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help=(
            "Optional run name override used in output filenames. "
            "If empty, an auto tag is generated from subset+methods+judge."
        ),
    )
    # <<< END ADD
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    # pipeline control
    parser.add_argument("--gen-only", action="store_true",
                        help="Only generate model outputs, skip any evaluation.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation, assume generation jsons already exist.")


    # Some terminals / IME / chat apps may paste non-breaking spaces (NBSP) before flags,
    # which makes argparse treat them as different tokens (e.g. "\xa0--model-family").
    # Normalize common unicode spaces so pasted commands still work.
    def _normalize_cli_token(s: str) -> str:
        if not isinstance(s, str):
            return s
        for ch in ("\u00a0", "\u2007", "\u202f"):  # NBSP / Figure space / Narrow NBSP
            s = s.replace(ch, " ")
        return s.strip()

    argv = [_normalize_cli_token(x) for x in sys.argv[1:]]
    args = parser.parse_args(argv)
    if args.gen_only and args.eval_only:
        raise ValueError("Cannot set both --gen-only and --eval-only.")


    def _resolve_user_path(p: str) -> Path:
        pp = Path(p).expanduser()
        if pp.is_absolute():
            return pp.resolve()
        # resolve relative to repo root rather than cwd
        return (_REPO_ROOT / pp).resolve()

    merged_dir = _resolve_user_path(args.merged_dir)
    adapter_dir = _resolve_user_path(args.adapter_dir)
    if args.results_base is not None:
        results_base = _resolve_user_path(args.results_base)
    elif args.hh_dataset_results:
        results_base = _resolve_user_path(HH_DATASET_OUTPUT_RESULTS_BASE)
    else:
        results_base = _resolve_user_path(DEFAULT_RESULTS_BASE)
    results_root = results_base / args.model_family / "alpacaeval2_ood"
    results_root.mkdir(parents=True, exist_ok=True)

    # judge env
    if args.judge_api_key:
        os.environ["OPENAI_API_KEY"] = args.judge_api_key
    if args.judge_base_url:
        os.environ["OPENAI_BASE_URL"] = args.judge_base_url
        os.environ["OPENAI_API_BASE"] = args.judge_base_url

    # load dataset (resolve like other flags: relative paths are from repo root, not cwd)
    if args.use_hf_datasets:
        examples = load_alpacaeval_examples_hf(str(_resolve_user_path(args.alpaca_local_path)))
    else:
        examples = load_alpacaeval_examples_from_json(str(_resolve_user_path(args.alpaca_eval_json)))

    n_total = len(examples)
    n = args.num_samples if args.num_samples and args.num_samples > 0 else None
    indices = sample_indices(n_total, n, args.seed, args.sample_random)

    subset_tag = (
        f"full{n_total}"
        if (n is None)
        else f"n{len(indices)}_{'rand' if args.sample_random else 'head'}_seed{args.seed}"
    )
    subset_file = results_root / f"subset_indices_{subset_tag}.json"
    if not subset_file.exists():
        subset_file.write_text(json.dumps(indices, indent=2), encoding="utf-8")

    # scan models (merged/adapters/both)
    scan_merged = args.scan in ("both", "merged")
    scan_adapters = args.scan in ("both", "adapters")
    models_scanned = scan_models(
        merged_dir=merged_dir,
        adapter_dir=adapter_dir,
        model_family=args.model_family,
        scan_merged=scan_merged,
        scan_adapters=scan_adapters,
    )

    if not models_scanned:
        print("[ERROR] No models found (merged or adapters). Check paths.")
        sys.exit(1)

    # filter methods
    want_all = (args.methods.strip().lower() == "all")
    method_set = set([m.strip().lower() for m in args.methods.split(",")]) if not want_all else None

    def _slug(s: str) -> str:
        """
        Make a filename-safe short tag. Keep alnum and [-._], convert others to '_'.
        """
        s = str(s or "").strip()
        if not s:
            return "na"
        s = s.replace(" ", "_")
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:120] if len(s) > 120 else s

    methods_tag = "all" if want_all else "-".join(sorted(method_set))  # type: ignore[arg-type]
    if args.judge_mode == "simple":
        judge_tag = _slug(args.judge_model)
    else:
        judge_tag = _slug(args.annotators_config)

    run_tag = _slug(args.run_name) if args.run_name.strip() else _slug(f"{subset_tag}__{methods_tag}__{args.judge_mode}-{judge_tag}")

    models_filtered: List[Tuple[str, Path]] = []
    for fam, p in models_scanned:
        m = extract_method_name(p.name, fam)
        if want_all or (m in method_set):
            models_filtered.append((fam, p))

    if not models_filtered:
        print("[ERROR] No models after method filter.")
        sys.exit(1)

    pairs = group_latest_base_calib(models_filtered)
    if not pairs:
        print("[ERROR] No (base, calib) pairs found. Ensure your calib models contain 'calib' in name.")
        sys.exit(1)

    print(f"[INFO] AlpacaEval subset: {subset_tag} (total={n_total}, used={len(indices)})")
    print(f"[INFO] Run tag: {run_tag}")
    print(f"[INFO] Annotators config: {args.annotators_config}")
    print(f"[INFO] Found {len(pairs)} method pairs (base vs calib).\n")

    summary = []
    for (fam, method), bc in pairs.items():
        base_e = bc["base"]
        calib_e = bc["calib"]

        # base model path for adapters (must be absolute if local; HF hub ids unchanged)
        raw_base = args.base_model.strip() or BASE_MODEL_PATHS.get(fam, "")
        if args.base_model.strip():
            base_model_for_adapter = resolve_pretrained_local_or_hub(raw_base)
        elif fam == "mistral-7b":
            base_model_for_adapter = resolve_mistral_base_with_fallback(raw_base)
        else:
            base_model_for_adapter = resolve_pretrained_local_or_hub(raw_base)
        if _looks_like_adapter_dir(base_e.path) or _looks_like_adapter_dir(calib_e.path):
            if not base_model_for_adapter:
                raise ValueError(
                    f"Adapter detected but base model path missing for family={fam}. "
                    f"Set --base-model or env var in BASE_MODEL_PATHS."
                )

        # generation outputs paths
        gen_dir = results_root / "generations" / subset_tag / fam / method
        base_out = gen_dir / f"base_{base_e.ts or base_e.path.name}.json"
        calib_out = gen_dir / f"calib_{calib_e.ts or calib_e.path.name}.json"

        # generate base
        do_gen = (not args.eval_only)
        do_eval = (not args.gen_only)

        # generate base
        if do_gen:
            if args.skip_existing and base_out.exists():
                print(f"[SKIP] base generation exists: {base_out}")
            else:
                print(f"\n[GEN] {fam}/{method} BASE -> {base_out}")
                if not args.dry_run:
                    generate_outputs_hf(
                        model_dir=base_e.path,
                        out_json=base_out,
                        examples=examples,
                        indices=indices,
                        dtype=args.dtype,
                        device_map=args.device_map,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        batch_size=args.batch_size,
                        base_model_for_adapter=base_model_for_adapter,
                        merge_adapter=args.merge_adapter,
                    )
        else:
            # eval-only: must exist
            if not base_out.exists():
                raise FileNotFoundError(f"[EVAL-ONLY] Missing base outputs: {base_out}")


        # generate calib
        if do_gen:
            if args.skip_existing and calib_out.exists():
                print(f"[SKIP] calib generation exists: {calib_out}")
            else:
                print(f"\n[GEN] {fam}/{method} CALIB -> {calib_out}")
                if not args.dry_run:
                    generate_outputs_hf(
                        model_dir=calib_e.path,
                        out_json=calib_out,
                        examples=examples,
                        indices=indices,
                        dtype=args.dtype,
                        device_map=args.device_map,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        batch_size=args.batch_size,
                        base_model_for_adapter=base_model_for_adapter,
                        merge_adapter=args.merge_adapter,
                    )
        else:
            # eval-only: must exist
            if not calib_out.exists():
                raise FileNotFoundError(f"[EVAL-ONLY] Missing calib outputs: {calib_out}")

        # run judge (calib vs base) unless gen-only
        eval_dir = results_root / "pairwise" / subset_tag / fam / method / f"calib_vs_base_{calib_e.ts or 'na'}"
        ann_json: Optional[Path] = None

        if do_eval:
            eval_dir.mkdir(parents=True, exist_ok=True)

            if args.judge_mode == "simple":
                ann_json = eval_dir / "annotations.json"
                if args.skip_existing and ann_json.exists():
                    print(f"[SKIP] simple judge exists: {ann_json}")
                else:
                    print(f"\n[JUDGE-SIMPLE] {fam}/{method}: CALIB vs BASE -> {ann_json}")
                    if not args.dry_run:
                        base_data = json.loads(Path(base_out).read_text(encoding="utf-8"))
                        calib_data = json.loads(Path(calib_out).read_text(encoding="utf-8"))

                        # assume same order (because you generated from same indices)
                        subset_insts = [x.get("instruction", "") for x in base_data]
                        base_outputs = [x.get("output", "") for x in base_data]
                        calib_outputs = [x.get("output", "") for x in calib_data]

                        judge_pairwise_simple(
                            out_path=ann_json,
                            instructions=subset_insts,
                            base_outputs=base_outputs,
                            calib_outputs=calib_outputs,
                            judge_model=args.judge_model,
                            max_concurrency=args.judge_concurrency,
                         retry_ties=bool(args.judge_retry_ties),
                         max_answer_chars=int(args.judge_max_answer_chars),
                        )
            else:
                # original alpaca_eval CLI path (may require logprobs/tools)
                ann_json = find_annotations_json(eval_dir)
                if args.skip_existing and ann_json is not None:
                    print(f"[SKIP] eval exists: {ann_json}")
                else:
                    print(f"\n[EVAL] {fam}/{method}: CALIB vs BASE -> {eval_dir}")
                    run_alpaca_eval_pairwise(
                        alpaca_eval_bin=args.alpaca_eval_bin,
                        model_outputs_json=calib_out,
                        reference_outputs_json=base_out,
                        out_dir=eval_dir,
                        annotators_config=args.annotators_config,
                        name=f"{fam}-{method}-calib",
                        dry_run=args.dry_run,
                    )
                ann_json = find_annotations_json(eval_dir)

        # parse summary (only if we actually judged)
        if do_eval and ann_json and ann_json.exists() and not args.dry_run:
            metrics = parse_winrate_from_annotations(ann_json)
        else:
            metrics = {
                "n": len(indices),
                "win_rate": None,
                "win_rate_stderr": None,
                "avg_len_reference": None,
                "avg_len_model": None,
            }


        row = {
            "family": fam,
            "method": method,
            "subset": subset_tag,
            "run_tag": run_tag,
            "methods_tag": methods_tag,
            "judge_mode": args.judge_mode,
            "judge_model": args.judge_model if args.judge_mode == "simple" else None,
            "annotators_config": args.annotators_config if args.judge_mode != "simple" else None,
            "base_model": base_e.path.name,
            "calib_model": calib_e.path.name,
            "base_outputs": str(base_out),
            "calib_outputs": str(calib_out),
            "eval_dir": str(eval_dir),
            **metrics,
        }
        summary.append(row)

        print(
            f"[RESULT] {fam}/{method}  win_rate(calib>base)={row['win_rate']}  "
            f"stderr={row['win_rate_stderr']}  n={row['n']}"
        )

    summary_path = results_root / f"summary_{run_tag}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SAVED] summary -> {summary_path}")

    # Also write a human-readable markdown summary similar to compare_calib_results.py outputs
    md_path = results_root / f"alpacaeval2_ood_pairwise_{run_tag}.md"
    try:
        # Make a compact table without extra deps.
        lines = []
        lines.append(f"# AlpacaEval2 OOD Pairwise (calib vs base)\n")
        lines.append(f"- model_family: **{args.model_family}**\n")
        lines.append(f"- subset: **{subset_tag}**\n")
        lines.append(f"- methods: **{methods_tag}**\n")
        lines.append(f"- judge_mode: **{args.judge_mode}**\n")
        if args.judge_mode == "simple":
            lines.append(f"- judge_model: **{args.judge_model}**\n")
        else:
            lines.append(f"- annotators_config: **{args.annotators_config}**\n")
        lines.append("\n")
        lines.append("| method | base_model | calib_model | n | win_rate(calib>base) | stderr |\n")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |\n")
        for row in sorted(summary, key=lambda r: (r.get("method", ""), r.get("family", ""))):
            method = row.get("method", "")
            base_model = row.get("base_model", "")
            calib_model = row.get("calib_model", "")
            n = row.get("n", "")
            wr = row.get("win_rate", None)
            se = row.get("win_rate_stderr", None)
            wr_s = f"{wr:.4f}" if isinstance(wr, (int, float)) else "N/A"
            se_s = f"{se:.4f}" if isinstance(se, (int, float)) else "N/A"
            lines.append(f"| {method} | {base_model} | {calib_model} | {n} | {wr_s} | {se_s} |\n")
        md_path.write_text("".join(lines), encoding="utf-8")
        print(f"[SAVED] markdown summary -> {md_path}")
    except Exception as e:
        print(f"[WARN] Failed to write markdown summary: {e}")


if __name__ == "__main__":
    main()
