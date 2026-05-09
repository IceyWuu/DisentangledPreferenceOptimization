# Towards Disentangled Preference Optimization Dynamics: Suppress the Loser, Preserve the Winner

**[中文说明 → `README_cn.md`](README_cn.md)**

This repository is the official implementation for **Towards Disentangled Preference Optimization Dynamics: Suppress the Loser, Preserve the Winner**. The paper has been **accepted** at ICML2026. [arxiv](https://arxiv.org/pdf/2604.18239), [openreview](https://openreview.net/forum?id=TaNH4XiQ6P)

The rest of this README covers environment setup, unified preference learning via `aexperiment.sh`, legacy training scripts, batch evaluation, and optional SFT. For a full **lm-evaluation-harness** and AlpacaEval walkthrough, see **[`eval.md`](eval.md)**.

---

## Citation

If you use this code before the camera-ready reference is available, you can cite the work provisionally:

```bibtex
@inproceedings{chen2026towards,
 title={Towards Disentangled Preference Optimization Dynamics: Suppress the Loser, Preserve the Winner},
 author={Wei Chen and Yubing Wu and Junmei Yang and Delu Zeng and Qibin Zhao and John Paisley and Min Chen and Zhou Wang},
 booktitle={International Conference on Machine Learning},
 year={2026},
 url={https://openreview.net/forum?id=TaNH4XiQ6P}
}
```

---

## Environment

### Recommended: CUDA 12.8 dependency bundles

Example hardware: **RTX 5090, CUDA 12.8**.

```bash
conda create -n DPO python=3.10 -y
conda activate DPO
pip install -r envfiles/requirements_cuda128.txt
pip install -r envfiles/requirements_cuda128_supply.txt
```

### Paper-style baseline (optional)

Install **PyTorch 2.1.2** from the [PyTorch installation page](https://pytorch.org/get-started/locally/), then:

```bash
conda create -n DPO python=3.10 && conda activate DPO
python -m pip install flash-attn --no-build-isolation
```

Use either stack (or mix) depending on your GPU and reproduction goals.

---

## Directory layout

The public repo is usually just **`DPO/`**; parent folders for weights and data are local. Suggested top-level layout:

```text
LLM_alignment/
├── DPO/                         # this repo (scripts, recipes, outputs)
├── ModelAndDatasets/
│   ├── alignment-handbook
│   ├── EleutherAI/              # e.g. pythia-410m, pythia-1.4b, pythia-2.8b
│   ├── HuggingFaceH4
│   └── …
├── LeaderBoardV1/lm-evaluation-harness/
└── LeaderBoardV2/lm-evaluation-harness/
```

Point recipes (or env vars / CLI overrides) at your real paths.

---

## Preference learning (unified entry: `aexperiment.sh`)

All supported model families share **`aexperiment.sh`**; choose the profile with **`--model-family`**. Checkpoints and logs go under `DPO/outputs/<run_name>/`.

### Model families and defaults

| Model | `--model-family` | Default LR | batch × grad_acc | grad_ckpt | 4bit | LoRA r | Recipe |
|-------|------------------|------------|------------------|-----------|------|--------|--------|
| Pythia-410M | `pythia-410m` | 6e-7 | 4×8 | off | off | — | `pythia-410m-base.yaml` |
| Pythia-1.4B | `pythia-1.4b` | 6e-7 | 4×8 | off | off | — | `pythia-1.4b-base.yaml` |
| Pythia-2B | `pythia-2b` | 1e-4 | 4×8 | off | on | — | `pythia-2b-base.yaml` |
| Mistral-7B | `mistral-7b` | 3e-7 | 1×32 | off | on | 8 | `mistral-7b-base-simpo.yaml` |
| Qwen2.5-7B | `qwen2.5-7b` | 5e-7 | 1×32 | on | on | 8 | `qwen2.5-7b-instruct-simpo.yaml` |

Defaults can be overridden with environment variables (`FIXED_LR`, `PER_DEVICE_BATCH_SIZE`, `GRAD_ACCUMULATION`, `LOAD_IN_4BIT`, `LORA_R`, …).

### Common CLI flags

| Flag | Description | Values |
|------|-------------|--------|
| `--model-family` | Required | `pythia-410m` `pythia-1.4b` `pythia-2b` `mistral-7b` `qwen2.5-7b` |
| `--loss-types` | Space-separated losses | `DPO` `BCE` `CPO` `IPO` `SIMPO` `SLIC` `LSIF` `UKL` `DDRO` `KTO` `TIDPO` |
| `--gpu-ids` | Comma-separated GPU ids | e.g. `0` or `0,1,2,7` |
| `--category` | DB calibration mode | `base` (off) / `calib` (on, default) / `both` (each loss with calib on then off) |
| `--dataset` | Dataset (`mistral-7b` only) | `ultrafeedback` `hh-rlhf` `hh-rlhf-merged` |

### Examples

```bash
cd DPO

bash aexperiment.sh --model-family qwen2.5-7b --loss-types "DPO BCE" \
  --gpu-ids 0,1,2,3 --category both

bash aexperiment.sh --model-family pythia-2b --loss-types "DPO BCE" \
  --gpu-ids 0 --category calib

bash aexperiment.sh --model-family mistral-7b --loss-types DPO \
  --gpu-ids 4 --dataset hh-rlhf --category base

FIXED_LR=1e-6 bash aexperiment.sh --model-family pythia-410m \
  --loss-types "DPO SIMPO" --gpu-ids 0

nohup bash aexperiment.sh --model-family qwen2.5-7b --loss-types "DPO BCE" \
  --gpu-ids 0,7 --category both > train_qwen.log 2>&1 &
```

### Multi-GPU scheduling

- **One GPU:** all (loss × calib) jobs run **serially**.
- **Multiple GPUs:** **wave** scheduling — up to N jobs in parallel per wave.
- **`--category both`:** doubles the job count (each loss × calib on + calib off).

### Environment variables (subset)

```bash
FIXED_LR=5e-7
FIXED_OPTIM=adamw_torch
FIXED_NUM_EPOCHS=1
FIXED_SEED=42
FIXED_MAX_STEPS=              # empty => use epochs
LR_SCHEDULER_TYPE=constant
WARMUP_RATIO=0.0
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUMULATION=32
GRADIENT_CHECKPOINTING=true
LORA_R=8
LOAD_IN_4BIT=true
MAX_LENGTH=
MAX_PROMPT_LENGTH=
CATEGORY=calib                 # base | calib | both; or ENABLE_DB_CALIB=true|false|both
DB_CALIB_EPS=1e-12
DB_EMA_BETA=                   # empty => DPOConfig default
TRACK_MARGIN_CHAIN=true
MARGIN_CHAIN_LORA_ALL_LAYERS=false
CLEAN_CACHE=false              # clear HF dataset cache shards before train
```

---

## Legacy training scripts (alongside `aexperiment.sh`)

These live under `legacy_scripts/`. **New work should prefer `aexperiment.sh`.**

| Command | Role |
|---------|------|
| `bash legacy_scripts/Mistral-7B-Base.sh` | Mistral-Base |
| `bash legacy_scripts/Llama-8B-Base.sh` | Llama3-Base |
| `bash legacy_scripts/Pythia-410M-Base.sh` | Pythia-410M-Base |

**Pythia-410M-Base.sh notes:**

- Default checkpoint layout matches HALO; override with `MODEL_PATH=/path/to/weights`. Quantization: `USE_4BIT=true|false`.
- Runs **full fine-tuning** (`use_peft=false`) with policy + reference — VRAM similar to HALO.
- Edit `DEFAULT_*` at the top of the script, or override via env, e.g.:

```bash
LOSS_TYPE=LSIF \
LEARNING_RATE=6e-7 \
HUB_MODEL_ID=My-Pythia410-LSIF \
OUTPUT_DIR=/data/experiments/pythia410-lsif \
bash legacy_scripts/Pythia-410M-Base.sh
```

---

## Evaluation (quick reference)

Scripts live under **`DPO/outputs/`**. Recommended entry: **`run_eval.sh`** (wraps `batch_eval.py` and can run `compare_calib_results.py` after).

### Leaderboard links

- [Open LLM Leaderboard v1 (legacy)](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard)
- [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

In this repo, **`arc` / `gsm8k` / `mmlu`** use the V1 harness; **`bbh` / `math_hard` / `mmlu_pro` / `musr`** use V2. `batch_eval.py` selects conda envs (`lmeval_v1` / `lmeval_v2`) per task.

### `run_eval.sh` flags

| Flag | Description | Values |
|------|-------------|--------|
| `--model-family` | Family to evaluate | `pythia-2b` `mistral-7b` `qwen2.5-7b` `all` |
| `--methods` | Trained method(s) | Comma list or `all` |
| `--category` | Filter runs | `all` `base` `calib` |
| `--eval-method` | Benchmarks (required) | `math_hard` `mmlu_pro` `bbh` `musr` `arc` `gsm8k` or `all` |
| `--parallel-gpus` | Parallel GPUs | e.g. `0,1,2,7` |
| `--skip-existing` | Skip if result exists | — |
| `--dry-run` | Print plan only | — |

### Examples

```bash
cd DPO/outputs

bash run_eval.sh --model-family qwen2.5-7b --methods "dpo,bce" --category all \
  --eval-method all --parallel-gpus 0,1,2,7

bash run_eval.sh --model-family qwen2.5-7b --methods "dpo,bce" --category all \
  --eval-method "arc,gsm8k,math_hard" --parallel-gpus 0,1,2
```

Summaries: `outputs/results/compare_results/<model-family>/` (Markdown).

**Full harness setup, manual `lm-eval`, AlpacaEval, and troubleshooting → [`eval.md`](eval.md).**

---

## SFT (Mistral-7B, `aexperiment_sft.sh`)

TRL + LoRA / QLoRA. Install YAML helpers:

```bash
conda activate DPO
pip install ruamel.yaml
```

Run from **`DPO/`** (set `GPU_IDS` to your device).

### BF16 + LoRA (faster if VRAM allows)

```bash
LOAD_IN_4BIT=false PER_DEVICE_BATCH_SIZE=2 GRAD_ACCUMULATION_STEPS=16 \
  GPU_IDS="0" STRATEGY=single bash aexperiment_sft.sh
```

### QLoRA (memory-friendly)

```bash
LOAD_IN_4BIT=true PER_DEVICE_BATCH_SIZE=16 GRAD_ACCUMULATION_STEPS=4 \
  GPU_IDS="0" STRATEGY=single bash aexperiment_sft.sh
```

### Background

```bash
nohup bash -c "conda run -n DPO bash aexperiment_sft.sh" > train_sft.log 2>&1 &
tail -f train_sft.log
```

Edit **`recipes/zephyr/mistral-7b-base-sft.yaml`** for LR, epochs, `eval_strategy`, `save_steps`, `max_steps`, etc.

**Troubleshooting:** OOM → lower `per_device_train_batch_size`; slow run → check `gradient_checkpointing`; dataset errors → verify `dataset_mixer` paths in the YAML.

---

## More documentation

| Doc | Contents |
|-----|----------|
| **[`README_cn.md`](README_cn.md)** | This guide in Chinese |
| **[`eval.md`](eval.md)** | Evaluation tutorial (V1/V2 harness, caches, `batch_eval`, AlpacaEval) |
