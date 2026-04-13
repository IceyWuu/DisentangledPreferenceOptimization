# Evaluation tutorial (DIL)

This guide walks you through benchmarking models trained with this repository: **lm-evaluation-harness** (Open LLM Leaderboard V1/V2-style tasks), optional **manual harness runs**, and **AlpacaEval** for instruction-following comparisons.

---

## 1. Overview

| Track | Harness | Conda env (recommended) | Typical tasks |
|-------|---------|-------------------------|---------------|
| **V1** | Legacy `main.py` + `hf-causal-experimental` | `lmeval_v1` | `arc`, `gsm8k`, full **MMLU** (57 subtasks) |
| **V2** | `python -m lm_eval` (e.g. v0.4.4) | `lmeval_v2` | `bbh`, `math_hard`, `mmlu_pro`, `musr` |

**Recommended workflow:** use `outputs/run_eval.sh` (wraps `batch_eval.py` and runs `compare_calib_results.py`). The batch script picks the right conda env per benchmark and can merge LoRA adapters on the fly.

---

## 2. Directory layout

Assume a top-level folder (e.g. `LLM_alignment/`) that contains this repo and shared assets:

```text
LLM_alignment/
├── DIL/                              # this repository
│   └── outputs/                      # run batch eval from here (or set paths in batch_eval.py)
├── ModelAndDatasets/                 # base weights, datasets, caches
├── LeaderBoardV1/
│   └── lm-evaluation-harness/        # V1 checkout (batch_eval default: ../../LeaderBoardV1/...)
└── LeaderBoardV2/
    └── lm-evaluation-harness/        # V2 checkout (v0.4.x); override with LM_EVAL_V2_HARNESS_ROOT
```

You may keep an extra standalone `lm-evaluation-harness/` clone for experiments; only the paths configured in `outputs/batch_eval.py` matter for automation.

---

## 3. One-time setup: conda environments

Use **separate** environments for V1 and V2 to avoid dependency conflicts.

### 3.1 Leaderboard V1 (legacy harness)

Pin the harness to the commit used by the original Open LLM Leaderboard V1 (see [archived leaderboard docs](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/archive)).

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git LeaderBoardV1/lm-evaluation-harness
cd LeaderBoardV1/lm-evaluation-harness
git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463

conda create -n lmeval_v1 python=3.10 -y
conda activate lmeval_v1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128   # adjust for your CUDA
pip install transformers accelerate datasets tokenizers sentencepiece protobuf
pip install -e .
```

### 3.2 Leaderboard V2–style tasks

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git LeaderBoardV2/lm-evaluation-harness
cd LeaderBoardV2/lm-evaluation-harness
git fetch --tags
git checkout v0.4.4   # or another tag you have validated

conda create -n lmeval_v2 python=3.10 -y
conda activate lmeval_v2
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

**`leaderboard_math_hard`:** scoring may require LaTeX parsing. If you see errors about ANTLR4, install:

```bash
pip install 'antlr4-python3-runtime==4.11.*'
```

**GPQA** (if you add it): gated on Hugging Face — accept the dataset terms and run `huggingface-cli login` on the machine, or evaluation may fail or skip tasks.

---

## 4. Hugging Face cache and mirrors

Caches store model weights, tokenizers, and dataset shards (Arrow, etc.) and can grow large.

Check existing variables:

```bash
env | grep -E 'HF_HOME|HF_DATASETS_CACHE|TRANSFORMERS_CACHE|HUGGINGFACE_HUB_CACHE'
```

Example session exports:

```bash
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_cache/datasets
export TRANSFORMERS_CACHE=/path/to/hf_cache/transformers
```

For slow or blocked downloads, a mirror (example):

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from datasets import load_dataset; print(load_dataset('ai2_arc', 'ARC-Challenge', split='validation')[0])"
```

---

## 5. Recommended: `run_eval.sh` and `batch_eval.py`

From `DIL/outputs/`:

- **`batch_eval.py`** discovers trained checkpoints, **merges LoRA when needed** (cached under `output_merged/` with hash-based reuse), and runs the correct harness command per benchmark.
- You do **not** need to manually `conda activate` for each task: the script dispatches subprocesses with `lmeval_v1` or `lmeval_v2` as configured in `CONDA_ENV_MAP` inside `batch_eval.py`.

### 5.1 Shell wrapper (runs comparison after eval)

```bash
cd DIL/outputs

bash run_eval.sh --model-family qwen2.5-7b --methods "dpo,bce" --category all \
  --eval-method "arc,gsm8k,math_hard" --parallel-gpus 0,1,2
```

### 5.2 Direct Python entry

```bash
cd DIL/outputs

python batch_eval.py --model-family mistral-7b --methods "dpo,bce,cpo" --category base \
  --eval-method mmlu_pro \
  --base-model "../../ModelAndDatasets/alignment-handbook/local_models/mistralai/Mistral-7B-v0.1"

# Multi-GPU: one subprocess per GPU, shared job queue
python batch_eval.py --model-family qwen2.5-7b --methods "dpo,bce" --category all \
  --eval-method "math_hard,arc,gsm8k" --parallel-gpus 0,1,2,3
```

### 5.3 Common CLI options

| Option | Description |
|--------|-------------|
| `--model-family` | `pythia-2b`, `mistral-7b`, `qwen2.5-7b`, or `all` |
| `--methods` | Comma-separated methods or `all` (e.g. `dpo,bce,cpo`, …) |
| `--category` | `all`, `base`, or `calib` (filter by calibration suffix in run names) |
| `--eval-method` | Comma-separated; see table below, or `all` |
| `--base-model` | Override base model path for adapter merge |
| `--parallel-gpus` | e.g. `0,1,2,7`; sets `CUDA_VISIBLE_DEVICES` per job; ignores `--device` |
| `--device` | e.g. `cuda:0` when not using `--parallel-gpus` |
| `--skip-existing` | Skip if result file already exists |
| `--dry-run` | Print planned jobs only |
| `--batch-size` | lm-eval batch size (default `1`) |
| `--limit` | Cap examples per task (smoke test); use a separate `--results-base` so you do not overwrite full runs |
| `--dtype` | e.g. `float16` (default) |

**Eval methods** (as in `batch_eval.py`):

- **V2 / module:** `bbh`, `math_hard`, `mmlu_pro`, `musr`
- **V1 / main.py:** `arc`, `gsm8k`, `mmlu`

### 5.4 Where results go

Under `DIL/outputs/results/` (or `--results-base`), organized by model family and task, for example:

```text
outputs/results/
├── pythia-2b-base/
│   ├── arc/
│   ├── gsm8k/
│   └── mmlu_pro/
└── pythia-2b-calib/
    └── ...
```

### 5.5 Base vs calibration comparison

After evaluation, summaries can be generated with:

```bash
cd DIL/outputs

python compare_calib_results.py --eval-type mmlu_pro --model pythia-2b
python compare_calib_results.py --eval-type bbh --model mistral-7b
python compare_calib_results.py --eval-type bbh,mmlu_pro,musr --model qwen2.5-7b
```

Markdown reports are written under `outputs/results/compare_results/<model-family>/`.

---

## 6. Reference: Open LLM Leaderboard V1 tasks

Official archive: [Open LLM Leaderboard (archive)](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/archive).

| Benchmark | Few-shot | Task id(s) | Primary metric |
|-----------|----------|------------|----------------|
| ARC-Challenge | 25 | `arc_challenge` | `acc_norm` |
| HellaSwag | 10 | `hellaswag` | `acc_norm` |
| MMLU | 5 | 57× `hendrycksTest-*` | mean `acc` |
| TruthfulQA-MC | 0 | `truthfulqa-mc` | `mc2` |
| Winogrande | 5 | `winogrande` | `acc` |
| GSM8K | 5 | `gsm8k` | `acc` |

The upstream harness used **8× H100** with global batch 8; on a single GPU, `--batch_size=1` is typical — minor score drift vs the public leaderboard is possible.

### 6.1 Manual V1 command template

Run inside `LeaderBoardV1/lm-evaluation-harness` with `lmeval_v1` active:

```bash
python main.py --model hf-causal-experimental \
  --model_args "pretrained=/path/to/your/model,use_accelerate=True" \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size 1 \
  --output_path ./results/out_arc.json
```

Repeat per task with the few-shot counts from the table. For **MMLU**, generating the full `--tasks` list from Python avoids copy-paste errors:

```bash
MMLU_TASKS=$(python - <<'PY'
subs = [
  "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
  "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
  "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics",
  "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts",
  "high_school_biology", "high_school_chemistry", "high_school_computer_science",
  "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
  "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
  "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
  "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence",
  "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
  "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory",
  "professional_accounting", "professional_law", "professional_medicine", "professional_psychology",
  "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
]
print(",".join(f"hendrycksTest-{s}" for s in subs))
PY
)

python main.py --model hf-causal-experimental \
  --model_args "pretrained=/path/to/your/model,use_accelerate=True" \
  --tasks "$MMLU_TASKS" \
  --num_fewshot 5 \
  --batch_size 1 \
  --output_path ./results/out_mmlu.json
```

---

## 7. Reference: Leaderboard V2–style manual runs

Task definitions and few-shot settings are documented in the current leaderboard docs, e.g. [About the Open LLM Leaderboard](https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/about).

V2 uses the **`lm-eval`** CLI (not the old `hf-causal-experimental` name). Example:

```bash
cd LeaderBoardV2/lm-evaluation-harness
conda activate lmeval_v2

python -m lm_eval \
  --model hf \
  --model_args "pretrained=/path/to/your/model,dtype=float16" \
  --tasks leaderboard_bbh \
  --batch_size 1 \
  --num_fewshot 3 \
  --output_path ./results/lb2_bbh.json
```

Add `--limit 100` for a quick smoke test.

### 7.1 Optional: macro-average aggregates in YAML

Some task groups do not define a group-level aggregate by default. To match a **macro mean** over subtasks (equal weight per subtask), you can add `aggregate_metric_list` to the group YAML, for example:

**BBH** (`lm_eval/tasks/leaderboard/bbh_mc/_leaderboard_bbh.yaml`):

```yaml
aggregate_metric_list:
  - metric: acc_norm
    aggregation: mean
    weight_by_size: false
```

**MATH hard** (`lm_eval/tasks/leaderboard/math/_leaderboard_math.yaml`):

```yaml
aggregate_metric_list:
  - metric: exact_match
    aggregation: mean
    weight_by_size: false
```

`mmlu_pro` in the V2 harness is usually already set up as a single task group; you often do not need edits there.

---

## 8. Troubleshooting

### 8.1 GSM8K generations truncated (legacy V1 `gsm8k.py`)

On some pinned harness commits, `until` stop sequences include `":"`, which can truncate right after `Answer:` and ruin GSM8K accuracy. If you see nonsense completions or all-wrong scores, inspect `lm_eval/tasks/gsm8k.py` and consider tightening `until` to something like:

```python
completion = rf.greedy_until(ctx, {"until": ["\n\nQuestion:", "Question:"]})
```

Clear `__pycache__` under `lm_eval/tasks` after editing.

### 8.2 Very small models

Tiny models often score at or near **0** on hard benchmarks (e.g. MATH-hard, parts of MMLU-Pro). Use `--limit` and a single task to verify the pipeline before long runs.

### 8.3 Output path typos

Ensure `--output_path` points to the intended file or directory. V2 may write timestamped `results_*.json` under a run subdirectory — `batch_eval.py` already accounts for this for configured V2 methods.

---

## 9. AlpacaEval (optional)

AlpacaEval compares model outputs to reference models on instruction-following benchmarks.

### 9.1 Install

```bash
pip install -U alpaca_eval
```

### 9.2 Batch script (`DIL/outputs`)

Dry-run to verify discovery and CLI:

```bash
cd DIL/outputs
python batch_alpacaeval2_ood.py \
  --model-family pythia-2b \
  --methods dpo,simpo,bce \
  --dry-run \
  --num-samples 5 \
  --sample-random \
  --seed 1
```

**Generate only** (skip if outputs exist):

```bash
python batch_alpacaeval2_ood.py \
  --model-family pythia-2b \
  --methods dpo,bce \
  --num-samples 0 \
  --gen-only \
  --skip-existing
```

**Evaluate only** (calls a judge API — set keys via environment variables, not committed files):

```bash
export JUDGE_API_KEY="your-api-key"
export JUDGE_BASE_URL="https://api.example.com/v1"   # OpenAI-compatible endpoint if applicable

python batch_alpacaeval2_ood.py \
  --model-family pythia-2b \
  --methods bce \
  --num-samples 0 \
  --eval-only \
  --judge-mode simple \
  --judge-model gpt-4-turbo \
  --judge-retry-ties \
  --judge-base-url "$JUDGE_BASE_URL" \
  --judge-api-key "$JUDGE_API_KEY"
```

Use `--judge-mode alpaca_eval` when you want the library’s bundled evaluation config. For **DeepSeek** or other providers, point `--judge-base-url` and `--judge-model` at that provider’s API.

**Security:** never commit API keys. Rotate any key that was ever pasted into a shared document.

### 9.3 Listing judge configs

AlpacaEval ships multiple named judge configurations. List them with the library’s CLI or docs (names like `alpaca_eval_gpt4_turbo_fn`, `weighted_alpaca_eval_gpt4_turbo`, etc., vary by version).

---

## 10. Quick checklist

1. [ ] Clone and pin **V1** and **V2** harnesses; create `lmeval_v1` / `lmeval_v2`.
2. [ ] Set **HF cache** (and mirror if needed).
3. [ ] `cd DIL/outputs` and run **`run_eval.sh`** or **`batch_eval.py`** with your `--model-family`, `--methods`, `--category`, and `--eval-method`.
4. [ ] Run **`compare_calib_results.py`** for base vs calib tables (or rely on `run_eval.sh` to invoke it).
5. [ ] (Optional) **AlpacaEval** with env-based API credentials.

---

*This tutorial replaces the previous Chinese notes-only draft; behavior is defined by `outputs/batch_eval.py`, `outputs/run_eval.sh`, and the pinned `lm-evaluation-harness` revisions you install locally.*
