# DIL：《Towards Disentangled Preference Optimization Dynamics》中文说明

**[English README → `README.md`](README.md)**

本文件说明在本机（如 CUDA 12.8 / RTX 5090）上跑通 **偏好学习训练**、**SFT** 与 **下游评测**。**lm-eval / AlpacaEval 的完整步骤**见 **[`eval.md`](eval.md)**（英文教程）。

---

## 1. 论文与引用

本仓库是 **Towards Disentangled Preference Optimization Dynamics**（解耦偏好优化动力学相关）一文的**官方源代码**。论文**尚未录用**；录用后将在此更新正式引用信息与链接。

在正式版 bib 发布前，如需临时引用，可使用下面条目（录用后请替换 `author` 与 `note`）：

```bibtex
@misc{disentangled_preference_optimization_dynamics,
  title={{Towards Disentangled Preference Optimization Dynamics}},
  author={Anonymous},
  year={2026},
  note={Manuscript under review; update author/venue when published}
}
```

---

## 2. 环境

### 2.1 推荐（本仓库 CUDA 12.8 依赖清单）

当前机器示例：**RTX 5090，CUDA 12.8**。

```bash
conda create -n DIL python=3.10 -y
conda activate DIL
pip install -r requirements_cuda128.txt
pip install -r requirements_cuda128_supply.txt
```

### 2.2 论文复现基线（可选）

从 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 安装 **PyTorch 2.1.2**，并安装 **flash-attn**：

```bash
conda create -n DIL python=3.10 && conda activate DIL
python -m pip install flash-attn --no-build-isolation
```

二者择一或按硬件与复现需求组合即可。

---

## 3. 目录约定

公开仓库通常只包含 **`DIL/`**；父目录与权重、数据集需自行准备。推荐顶层布局：

```text
LLM_alignment/
├── DIL/
├── ModelAndDatasets/
│   ├── alignment-handbook
│   ├── EleutherAI/
│   ├── HuggingFaceH4
│   └── …
├── LeaderBoardV1/lm-evaluation-harness/
└── LeaderBoardV2/lm-evaluation-harness/
```

将 recipe 中的相对路径改为你本机实际位置，或通过环境变量 / 脚本参数覆盖。

---

## 4. 偏好学习训练（`aexperiment.sh`）

所有模型族共用 **`aexperiment.sh`**，用 **`--model-family`** 选择配置；输出在 `DIL/outputs/<run_name>/`。

### 4.1 模型族与默认配置

| 模型 | `--model-family` | 默认 LR | batch × grad_acc | grad_ckpt | 4bit | LoRA r | Recipe |
|------|------------------|---------|------------------|-----------|------|--------|--------|
| Pythia-410M | `pythia-410m` | 6e-7 | 4×8 | off | off | — | `pythia-410m-base.yaml` |
| Pythia-1.4B | `pythia-1.4b` | 6e-7 | 4×8 | off | off | — | `pythia-1.4b-base.yaml` |
| Pythia-2B | `pythia-2b` | 1e-4 | 4×8 | off | on | — | `pythia-2b-base.yaml` |
| Mistral-7B | `mistral-7b` | 3e-7 | 1×32 | off | on | 8 | `mistral-7b-base-simpo.yaml` |
| Qwen2.5-7B | `qwen2.5-7b` | 5e-7 | 1×32 | on | on | 8 | `qwen2.5-7b-instruct-simpo.yaml` |

### 4.2 常用 CLI 参数

| 参数 | 说明 | 取值 |
|------|------|------|
| `--model-family` | 模型族（必填） | `pythia-410m` … `qwen2.5-7b` |
| `--loss-types` | 损失类型，空格分隔 | `DPO` `BCE` … `TIDPO` |
| `--gpu-ids` | GPU 编号，逗号分隔 | 如 `0` 或 `0,1,2,7` |
| `--category` | DB 校准模式 | `base` / `calib`（默认） / `both` |
| `--dataset` | 数据集（仅 `mistral-7b`） | `ultrafeedback` `hh-rlhf` `hh-rlhf-merged` |

### 4.3 示例

```bash
cd DIL

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

### 4.4 多卡调度

- 单卡：job 串行。
- 多卡：按波次并行。
- `--category both`：job 数翻倍。

### 4.5 环境变量（摘录）

```bash
FIXED_LR=5e-7
FIXED_OPTIM=adamw_torch
FIXED_NUM_EPOCHS=1
FIXED_SEED=42
FIXED_MAX_STEPS=
LR_SCHEDULER_TYPE=constant
WARMUP_RATIO=0.0
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUMULATION=32
GRADIENT_CHECKPOINTING=true
LORA_R=8
LOAD_IN_4BIT=true
MAX_LENGTH=
MAX_PROMPT_LENGTH=
CATEGORY=calib
DB_CALIB_EPS=1e-12
DB_EMA_BETA=
TRACK_MARGIN_CHAIN=true
MARGIN_CHAIN_LORA_ALL_LAYERS=false
CLEAN_CACHE=false
```

---

## 5. 经典脚本（与 `aexperiment.sh` 并存）

| 脚本 | 说明 |
|------|------|
| `bash Mistral-7B-Base.sh` | Mistral-Base |
| `bash Llama-8B-Base.sh` | Llama3-Base |
| `bash Pythia-410M-Base.sh` | HALO 风格，DPO 或 DIL-LSIF |

**Pythia-410M：** `MODEL_PATH` / `USE_4BIT`；全量微调 `use_peft=false`；可用 `DEFAULT_*` 或环境变量覆盖。

---

## 6. 下游评测（速查）

入口 **`DIL/outputs/run_eval.sh`**。完整说明见 **[`eval.md`](eval.md)**。

- [Leaderboard v1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) / [v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

```bash
cd DIL/outputs
bash run_eval.sh --model-family qwen2.5-7b --methods "dpo,bce" --category all \
  --eval-method all --parallel-gpus 0,1,2,7
```

---

## 7. SFT（`aexperiment_sft.sh`）

```bash
pip install ruamel.yaml
LOAD_IN_4BIT=false PER_DEVICE_BATCH_SIZE=2 GRAD_ACCUMULATION_STEPS=16 \
  GPU_IDS="0" STRATEGY=single bash aexperiment_sft.sh
```

配置：`recipes/zephyr/mistral-7b-base-sft.yaml`。

---

## 8. 延伸阅读

| 文档 | 内容 |
|------|------|
| **[`README.md`](README.md)** | 英文主文档（默认） |
| **`README_cn.md`** | 本中文说明 |
| **[`eval.md`](eval.md)** | 评测全流程（英文） |
