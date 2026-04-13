#!/bin/bash
set -euo pipefail

# Resume training for an existing Mistral-7B run directory.
#
# Quick start:
#   GPU_ID=0 bash aresume_mistral_7b.sh --run-dir outputs/xxx --db-calib
#
# Notes:
# - `--epochs` is TOTAL epochs (not incremental epochs).
# - Script reuses the same output_dir, so trainer can continue from checkpoint state.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

RUN_DIR=""
CHECKPOINT_PATH=""
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
GPU_ID="${GPU_ID:-0}"
LOSS_TYPE="${LOSS_TYPE:-BCE}"
ENABLE_DB_CALIB="${ENABLE_DB_CALIB:-true}"

# Keep defaults aligned with aexperiment_mistral_7b.sh
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-32}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
LORA_R="${LORA_R:-8}"
FIXED_LR="${FIXED_LR:-3e-7}"
FIXED_OPTIM="${FIXED_OPTIM:-adamw_torch}"
FIXED_SEED="${FIXED_SEED:-42}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"

print_help() {
  cat <<'EOF'
Usage: bash aresume_mistral_7b.sh --run-dir <path> [OPTIONS]

Required:
  --run-dir <path>         Existing run directory (contains recipe_active.yaml)

Options:
  --checkpoint <path>      Specific checkpoint path. If omitted, auto-pick latest checkpoint-* in run-dir
  --epochs <N>             Total epochs target (default: 2)
  --loss <name>            Loss type (default: BCE)
  --db-calib               Enable DB calibration (default)
  --no-db-calib            Disable DB calibration
  --gpu <id>               GPU ID (default: env GPU_ID or 0)
  -h, --help               Show this help

Environment overrides (optional):
  PER_DEVICE_BATCH_SIZE, GRAD_ACCUMULATION, GRADIENT_CHECKPOINTING,
  LORA_R, FIXED_LR, FIXED_OPTIM, FIXED_SEED, LR_SCHEDULER_TYPE,
  WARMUP_RATIO, LOAD_IN_4BIT

Example:
  GPU_ID=0 LOSS_TYPE=BCE bash aresume_mistral_7b.sh \
    --run-dir outputs/hh-dataset-output/mistral-7b-bce-lora-calib-20260326_211459 \
    --db-calib \
    --epochs 2
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --epochs)
      TOTAL_EPOCHS="$2"
      shift 2
      ;;
    --loss)
      LOSS_TYPE="$2"
      shift 2
      ;;
    --db-calib)
      ENABLE_DB_CALIB="true"
      shift
      ;;
    --no-db-calib)
      ENABLE_DB_CALIB="false"
      shift
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "[ERROR] --run-dir is required." >&2
  print_help
  exit 1
fi

# Normalize to absolute path for reliability.
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${SCRIPT_DIR}/${RUN_DIR}"
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[ERROR] Run directory not found: $RUN_DIR" >&2
  exit 1
fi

RECIPE_PATH="${RUN_DIR}/recipe_active.yaml"
if [[ ! -f "$RECIPE_PATH" ]]; then
  echo "[ERROR] recipe_active.yaml not found in run dir: $RUN_DIR" >&2
  exit 1
fi

if [[ -z "$CHECKPOINT_PATH" ]]; then
  shopt -s nullglob
  checkpoints=("${RUN_DIR}"/checkpoint-*)
  shopt -u nullglob
  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "[ERROR] No checkpoint-* found under run dir: $RUN_DIR" >&2
    exit 1
  fi
  IFS=$'\n' sorted=($(printf '%s\n' "${checkpoints[@]}" | sort -V))
  unset IFS
  last_idx=$(( ${#sorted[@]} - 1 ))
  CHECKPOINT_PATH="${sorted[$last_idx]}"
fi

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Checkpoint path not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

RUN_BASENAME="$(basename "$RUN_DIR")"

echo "[INFO] Resuming run: ${RUN_BASENAME}"
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Checkpoint: ${CHECKPOINT_PATH}"
echo "[INFO] Target total epochs: ${TOTAL_EPOCHS}"
echo "[INFO] GPU: ${GPU_ID}, Loss: ${LOSS_TYPE}, DB-calib: ${ENABLE_DB_CALIB}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
ACCELERATE_LOG_LEVEL=info \
python "${SCRIPT_DIR}/scripts/run.py" "${RECIPE_PATH}" \
  loss_type="${LOSS_TYPE}" \
  db_calibration_enable="${ENABLE_DB_CALIB}" \
  per_device_train_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  gradient_accumulation_steps="${GRAD_ACCUMULATION}" \
  gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
  lora_r="${LORA_R}" \
  learning_rate="${FIXED_LR}" \
  optim="${FIXED_OPTIM}" \
  num_train_epochs="${TOTAL_EPOCHS}" \
  seed="${FIXED_SEED}" \
  lr_scheduler_type="${LR_SCHEDULER_TYPE}" \
  warmup_ratio="${WARMUP_RATIO}" \
  run_name="${RUN_BASENAME}" \
  hub_model_id="${RUN_BASENAME}" \
  output_dir="${RUN_DIR}" \
  resume_from_checkpoint="${CHECKPOINT_PATH}" \
  --load_in_4bit="${LOAD_IN_4BIT}"

echo "[INFO] Resume training completed."
