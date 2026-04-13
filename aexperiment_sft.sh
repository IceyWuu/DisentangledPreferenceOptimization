#!/bin/bash
set -euo pipefail

# =============================================================================
# Single/Multi-GPU experiment runner for SFT training (LoRA/QLoRA)
# Based on alignment-handbook's best practices
# =============================================================================
#
# Usage examples:
#   bash aexperiment_sft.sh                          # 单 GPU 默认配置
#   GPU_IDS="0,1,2,3" bash aexperiment_sft.sh     # 使用 4 块 GPU
#   bash aexperiment_sft.sh --multi-gpu              # 使用所有可用 GPU
#   bash aexperiment_sft.sh --help                  # 查看帮助
#
# Multi-GPU modes:
#   1. Data Parallel (DP): 每个 GPU 复制完整模型 (适合 < 7B 模型)
#   2. Tensor Parallel (TP): 模型跨 GPU 分割 (需要特殊支持)
#   3. FSDP: 完整分片数据并行 (适合 > 7B 模型)
#
# 本脚本默认使用 Data Parallel (DP)，通过 Accelerate 或 torch.distributed 实现
#
# =============================================================================

# =============================================================================
# 默认超参数（可通过环境变量覆盖）
# =============================================================================

# --- GPU 配置 ---
# 格式: "0" 或 "0,1,2,3" 或 "all"
GPU_IDS="${GPU_IDS:-0}"

# --- 训练策略 ---
# auto: 自动选择（多GPU时使用DDP）
# ddp: torch.distributed.DataParallel
# fsdp: FullyShardedDataParallel (适合大模型)
STRATEGY="${STRATEGY:-auto}"

# --- 模型路径 ---
BASE_MODEL_PATH="${BASE_MODEL_PATH:-../ModelAndDatasets/alignment-handbook/local_models/mistralai/Mistral-7B-v0.1}"
# 如果已有 SFT adapter，可以设置此项（可选）
# export SFT_MODEL_PATH="/path/to/your/sft/adapter"

# --- 训练超参数 ---
LEARNING_RATE="${LEARNING_RATE:-2.0e-5}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
SEED="${SEED:-42}"

# --- 优化器 ---
OPTIMIZER="${OPTIMIZER:-adamw_torch}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"

# --- LoRA 配置 ---
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj}"

# --- 数据配置 ---
DATASET_SPLITS="${DATASET_SPLITS:-train_sft,test_sft}"
DATASET_PATH="${DATASET_PATH:-../ModelAndDatasets/HuggingFaceH4/ultrafeedback_binarized}"

# --- 输出配置 ---
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft experiment}"
RUN_NAME="${RUN_NAME:-sft experiment}"

# --- 日志与保存 ---
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# --- 硬件配置 ---
USE_FLASH_ATTN="${USE_FLASH_ATTN:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
BF16="${BF16:-true}"
FP16="${FP16:-false}"
# 4bit/8bit 量化
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-false}"

# --- 高级选项 ---
ENABLE_DB_CALIB="${ENABLE_DB_CALIB:-false}"
DB_CALIB_EPS="${DB_CALIB_EPS:-1e-12}"

# --- WandB 日志（可选） ---
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-sft-experiment}"
WANDB_NAME="${WANDB_NAME:-}"

# --- 混合精度训练 ---
# deepspeed 配置文件路径（可选）
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"

# =============================================================================
# 解析命令行参数
# =============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            echo "Usage: bash aexperiment_sft.sh [OPTIONS] [--HELP_END--]"
            echo ""
            echo "Options:"
            echo "  --help, -h              Show this help message"
            echo "  --multi-gpu            Use all available GPUs"
            echo "  --gpu-ids <ids>         Specify GPU IDs (e.g., '0,1,2,3')"
            echo "  --strategy <strategy>   Training strategy: auto|ddp|fsdp|deepspeed"
            echo "  --db-calib              Enable database calibration mode"
            echo "  --db-calib-eps <float>  Set DB calibration epsilon (default: 1e-12)"
            echo "  --4bit                  Enable 4-bit quantization"
            echo "  --8bit                  Enable 8-bit quantization"
            echo ""
            echo "Environment Variables:"
            echo "  GPU_IDS                 GPU IDs to use (default: 0)"
            echo "  LEARNING_RATE           Learning rate (default: 2.0e-5)"
            echo "  PER_DEVICE_BATCH_SIZE   Batch size per GPU (default: 2)"
            echo "  GRAD_ACCUMULATION_STEPS Gradient accumulation steps (default: 8)"
            echo "  NUM_EPOCHS             Number of epochs (default: 1)"
            echo "  MAX_LENGTH              Max sequence length (default: 2048)"
            echo "  SEED                    Random seed (default: 42)"
            echo "  OUTPUT_DIR              Output directory"
            echo "  USE_WANDB               Enable WandB logging"
            echo "  LOAD_IN_4BIT            Enable 4-bit quantization"
            echo "  LOAD_IN_8BIT            Enable 8-bit quantization"
            echo "  DEEPSPEED_CONFIG        Path to DeepSpeed config"
            exit 0
            ;;
        --multi-gpu)
            GPU_IDS="all"
            shift
            ;;
        --gpu-ids)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --gpu-ids requires a value (e.g., '0,1,2,3')" >&2
                exit 1
            fi
            GPU_IDS="$2"
            shift 2
            ;;
        --strategy)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --strategy requires a value (auto|ddp|fsdp|deepspeed)" >&2
                exit 1
            fi
            STRATEGY="$2"
            shift 2
            ;;
        --db-calib)
            ENABLE_DB_CALIB="true"
            shift
            ;;
        --db-calib-eps)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --db-calib-eps requires a value" >&2
                exit 1
            fi
            ENABLE_DB_CALIB="true"
            DB_CALIB_EPS="$2"
            shift 2
            ;;
        --4bit)
            LOAD_IN_4BIT="true"
            shift
            ;;
        --8bit)
            LOAD_IN_8BIT="true"
            shift
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            echo "       Try: bash aexperiment_sft.sh --help" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# 环境设置
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置 Python 路径
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
else
    export PYTHONPATH="${SCRIPT_DIR}"
fi
export PYTHONWARNINGS="ignore::UserWarning:transformers.trainer"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "${RUN_NAME:-}" ]]; then
    RUN_NAME="sft-${TIMESTAMP}"
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR="outputs/sft-${TIMESTAMP}"
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# GPU 数量检测与配置
# =============================================================================

# 检测可用 GPU 数量
detect_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --list-gpus | wc -l
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)"
    else
        echo "1"
    fi
}

# 解析 GPU IDs
parse_gpu_ids() {
    local gpu_ids="$1"
    if [[ "$gpu_ids" == "all" ]]; then
        detect_gpu_count
    elif [[ "$gpu_ids" == *" "* ]]; then
        # 空格分隔
        echo "$gpu_ids" | wc -w
    elif [[ "$gpu_ids" == *","* ]]; then
        # 逗号分隔
        echo "$(echo "$gpu_ids" | tr ',' '\n' | wc -l)"
    else
        # 单个 GPU
        echo "1"
    fi
}

# 获取 CUDA 可见设备
get_cuda_visible_devices() {
    local gpu_ids="$1"
    if [[ "$gpu_ids" == "all" ]]; then
        seq 0 $(( $(detect_gpu_count) - 1 )) | tr '\n' ',' | sed 's/,$//'
    elif [[ "$gpu_ids" == *" "* ]]; then
        echo "$gpu_ids" | tr ' ' ','
    elif [[ "$gpu_ids" == *","* ]]; then
        echo "$gpu_ids"
    else
        echo "$gpu_ids"
    fi
}

GPU_COUNT=$(parse_gpu_ids "$GPU_IDS")
CUDA_DEVICES=$(get_cuda_visible_devices "$GPU_IDS")

# 确定分布式训练策略
determine_training_strategy() {
    local strategy="$1"
    local gpu_count="$2"
    
    if [[ "$strategy" != "auto" ]]; then
        echo "$strategy"
        return
    fi
    
    if [[ "$gpu_count" -gt 1 ]]; then
        # 多 GPU：默认使用 DDP
        echo "ddp"
    else
        # 单 GPU
        echo "single"
    fi
}

FINAL_STRATEGY=$(determine_training_strategy "$STRATEGY" "$GPU_COUNT")

# =============================================================================
# 记录配置
# =============================================================================

echo "=============================================="
echo "SFT Training Configuration"
echo "=============================================="
echo "Timestamp:      ${TIMESTAMP}"
echo "Run Name:       ${RUN_NAME}"
echo "Output Dir:     ${OUTPUT_DIR}"
echo ""
echo "GPU Configuration:"
echo "  GPU IDs:      ${GPU_IDS}"
echo "  GPU Count:    ${GPU_COUNT}"
echo "  Strategy:     ${FINAL_STRATEGY}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
echo ""
echo "Model:"
echo "  Base Model:   ${BASE_MODEL_PATH}"
if [[ -n "${SFT_MODEL_PATH:-}" ]]; then
    echo "  SFT Adapter:  ${SFT_MODEL_PATH}"
else
    echo "  SFT Adapter:  <from base model>"
fi
echo ""
echo "Hyperparameters:"
echo "  Learning Rate:    ${LEARNING_RATE}"
echo "  Batch Size:       ${PER_DEVICE_BATCH_SIZE} (device) x ${GRAD_ACCUMULATION_STEPS} (grad acc) = $(echo "${PER_DEVICE_BATCH_SIZE} * ${GRAD_ACCUMULATION_STEPS}" | bc)"
if [[ "${GPU_COUNT}" -gt 1 ]]; then
    TOTAL_BATCH=$(echo "${PER_DEVICE_BATCH_SIZE} * ${GPU_COUNT} * ${GRAD_ACCUMULATION_STEPS}" | bc)
    echo "  Total Batch:     ${TOTAL_BATCH} (across ${GPU_COUNT} GPUs)"
fi
echo "  Epochs:           ${NUM_EPOCHS}"
echo "  Max Steps:        ${MAX_STEPS:-<none>}"
echo "  Seed:             ${SEED}"
echo ""
echo "Optimizer:"
echo "  Optimizer:    ${OPTIMIZER}"
echo "  Scheduler:    ${LR_SCHEDULER}"
echo "  Warmup Ratio: ${WARMUP_RATIO}"
echo ""
echo "LoRA Config:"
echo "  r:         ${LORA_R}"
echo "  alpha:     ${LORA_ALPHA}"
echo "  dropout:   ${LORA_DROPOUT}"
echo "  targets:   ${LORA_TARGET_MODULES}"
echo ""
echo "Data:"
echo "  Dataset:   ${DATASET_PATH}"
echo "  Splits:    ${DATASET_SPLITS}"
echo ""
echo "Hardware:"
echo "  Flash Attn:        ${USE_FLASH_ATTN}"
echo "  Grad Checkpoint:   ${GRADIENT_CHECKPOINTING}"
echo "  BF16:             ${BF16}"
echo "  FP16:             ${FP16}"
echo "  4-bit Quant:      ${LOAD_IN_4BIT}"
echo "  8-bit Quant:      ${LOAD_IN_8BIT}"
echo ""
echo "Advanced:"
echo "  DB Calibration:   ${ENABLE_DB_CALIB} (eps=${DB_CALIB_EPS})"
echo "  WandB:           ${USE_WANDB}"
echo "=============================================="
echo ""

# =============================================================================
# 构建数据集 mixer 字典
# =============================================================================

IFS=',' read -ra SPLITS <<< "${DATASET_SPLITS}"

# 构建 mixer YAML（注意：第二行必须有2空格缩进）
DATASET_MIXER_YAML="  ${DATASET_PATH}: 1.0"

# =============================================================================
# 创建临时配置文件
# =============================================================================

CONFIG_FILE="${OUTPUT_DIR}/training_config.yaml"

cat > "${CONFIG_FILE}" << EOF
# SFT Training Configuration
# Generated at ${TIMESTAMP}

# Model arguments
model_name_or_path: ${BASE_MODEL_PATH}
torch_dtype: null
attn_implementation: $([ "${USE_FLASH_ATTN}" = "true" ] && echo "flash_attention_2" || echo "sdpa")
$(if [[ "${LOAD_IN_4BIT}" = "true" ]]; then echo "load_in_4bit: true"; fi)
$(if [[ "${LOAD_IN_8BIT}" = "true" ]]; then echo "load_in_8bit: true"; fi)

# PEFT arguments 
use_peft: true
lora_r: ${LORA_R}
lora_alpha: ${LORA_ALPHA}
lora_dropout: ${LORA_DROPOUT}
lora_target_modules: [$(echo "${LORA_TARGET_MODULES}" | tr ',' ' ' | sed 's/ /, /g')]

# Data training arguments
dataset_mixer:
${DATASET_MIXER_YAML}
dataset_splits:
$(for split in "${SPLITS[@]}"; do echo "- $(echo "$split" | xargs)"; done)
preprocessing_num_workers: 2

# Training arguments
bf16: ${BF16}
fp16: ${FP16}
do_eval: true
eval_strategy: ${EVAL_STRATEGY}
eval_steps: ${EVAL_STEPS}
gradient_accumulation_steps: ${GRAD_ACCUMULATION_STEPS}
gradient_checkpointing: ${GRADIENT_CHECKPOINTING}
gradient_checkpointing_kwargs:
  use_reentrant: False
hub_model_id: ${RUN_NAME}
learning_rate: ${LEARNING_RATE}
log_level: info
logging_steps: ${LOGGING_STEPS}
lr_scheduler_type: ${LR_SCHEDULER}
num_train_epochs: ${NUM_EPOCHS}
optim: ${OPTIMIZER}
output_dir: ${OUTPUT_DIR}/${RUN_NAME}
run_name: ${RUN_NAME}
per_device_train_batch_size: ${PER_DEVICE_BATCH_SIZE}
per_device_eval_batch_size: ${PER_DEVICE_BATCH_SIZE}
push_to_hub: false
save_strategy: ${SAVE_STRATEGY}
save_steps: ${SAVE_STEPS}
save_total_limit: 1
seed: ${SEED}
warmup_ratio: ${WARMUP_RATIO}
$(if [[ -n "${MAX_STEPS}" ]]; then echo "max_steps: ${MAX_STEPS}"; fi)
$(if [[ "${USE_WANDB}" = "true" ]]; then
echo "report_to:
- wandb"
else
echo "report_to: []"
fi)
$(if [[ -n "${DEEPSPEED_CONFIG}" && -f "${DEEPSPEED_CONFIG}" ]]; then
echo "deepspeed: ${DEEPSPEED_CONFIG}"
fi)
EOF

echo "[INFO] Config saved to: ${CONFIG_FILE}"
echo ""

# =============================================================================
# 构建运行命令
# =============================================================================

# 基础命令
declare -a RUN_CMD

# 设置环境变量
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# 根据策略选择运行方式
case "${FINAL_STRATEGY}" in
    ddp)
        echo "[INFO] Using DistributedDataParallel (DDP) on ${GPU_COUNT} GPUs"
        RUN_CMD=(torchrun --nproc_per_node="${GPU_COUNT}" --nnodes=1 --node_rank=0 --master_port=29500)
        ;;
    fsdp)
        echo "[INFO] Using FullyShardedDataParallel (FSDP)"
        export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
        RUN_CMD=(torchrun --nproc_per_node="${GPU_COUNT}" --nnodes=1 --node_rank=0 --master_port=29500)
        # FSDP 需要额外配置
        export FSDP_CONFIG="${OUTPUT_DIR}/fsdp_config.yaml"
        cat > "${FSDP_CONFIG}" << FSDP_EOF
fsdp:
  backward_prefetch: forward_prefetch
  forward_prefetch: true
  activation_checkpointing: true
FSDP_EOF
        ;;
    deepspeed)
        echo "[INFO] Using DeepSpeed ZeRO"
        if [[ -z "${DEEPSPEED_CONFIG}" ]]; then
            # 自动生成简单的 DeepSpeed 配置
            DEEPSPEED_CONFIG="${OUTPUT_DIR}/deepspeed_config.json"
            TOTAL_BATCH=$((PER_DEVICE_BATCH_SIZE * GPU_COUNT * GRAD_ACCUMULATION_STEPS))
            FP16_ENABLED=$([ "${FP16}" = "true" ] && echo "true" || echo "false")
            BF16_ENABLED=$([ "${BF16}" = "true" ] && echo "true" || echo "false")
            
            cat > "${DEEPSPEED_CONFIG}" << EOF
{
  "train_batch_size": ${TOTAL_BATCH},
  "train_micro_batch_size_per_gpu": ${PER_DEVICE_BATCH_SIZE},
  "gradient_accumulation_steps": ${GRAD_ACCUMULATION_STEPS},
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": ${LEARNING_RATE}
    }
  },
  "fp16": {
    "enabled": ${FP16_ENABLED},
    "auto_cast": false
  },
  "bf16": {
    "enabled": ${BF16_ENABLED}
  },
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
EOF
            echo "[INFO] Auto-generated DeepSpeed config: ${DEEPSPEED_CONFIG}"
        fi
        RUN_CMD=(deepspeed --num_gpus "${GPU_COUNT}")
        ;;
    single|*)
        if [[ "${GPU_COUNT}" -gt 1 ]]; then
            echo "[INFO] Multi-GPU detected but strategy='single', using first GPU only"
        fi
        RUN_CMD=(python)
        ;;
esac

# 添加 Python 脚本和配置文件
RUN_CMD+=("${SCRIPT_DIR}/scripts/run_sft.py" "${CONFIG_FILE}")

# =============================================================================
# 额外参数
# =============================================================================

declare -a EXTRA_ARGS

if [[ -n "${MAX_STEPS}" ]]; then
    EXTRA_ARGS+=(max_steps="${MAX_STEPS}")
fi

if [[ -n "${SFT_MODEL_PATH:-}" ]]; then
    EXTRA_ARGS+=(base_model_revision="${BASE_MODEL_PATH}")
fi

# DB Calibration args
if [[ "${ENABLE_DB_CALIB}" = "true" ]]; then
    EXTRA_ARGS+=(db_calibration_enable="${ENABLE_DB_CALIB}")
    EXTRA_ARGS+=(db_calibration_eps="${DB_CALIB_EPS}")
fi

# =============================================================================
# 运行训练
# =============================================================================

echo "[INFO] Starting SFT training..."
echo "[INFO] Command: ${RUN_CMD[*]} ${EXTRA_ARGS[*]}"
echo ""

# 设置 WandB 名称（如果使用 WandB）
if [[ "${USE_WANDB}" = "true" ]]; then
    export WANDB_NAME="${RUN_NAME}"
    if [[ -n "${WANDB_PROJECT:-}" ]]; then
        export WANDB_PROJECT="${WANDB_PROJECT}"
    fi
fi

# 运行训练
"${RUN_CMD[@]}" "${EXTRA_ARGS[@]}"
TRAIN_EXIT_CODE=$?

# =============================================================================
# 训练完成
# =============================================================================

if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    echo ""
    echo "=============================================="
    echo "[SUCCESS] SFT training completed!"
    echo "=============================================="
    echo "Output directory: ${OUTPUT_DIR}/${RUN_NAME}"
    echo "Config file:      ${CONFIG_FILE}"
    echo ""
    echo "To use the trained model for DPO/SimPO:"
    echo "  1. Update your recipe's model_name_or_path to:"
    echo "     ${OUTPUT_DIR}/${RUN_NAME}"
    echo "  2. Or update BASE_MODEL_PATH in aexperiment_sft.sh"
    echo ""
else
    echo ""
    echo "[ERROR] SFT training failed with exit code: ${TRAIN_EXIT_CODE}"
    echo "[ERROR] Check logs in: ${OUTPUT_DIR}/${RUN_NAME}"
    exit ${TRAIN_EXIT_CODE}
fi
