#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${SCRIPT_DIR}"
fi
export PYTHONWARNINGS="ignore::UserWarning:transformers.trainer"

# --- User-editable defaults (change once here, no env vars needed) ---
DEFAULT_LOSS_TYPE="${DEFAULT_LOSS_TYPE:-LSIF}" # 损失函数 DPO / SIMPO / LSIF / KTO
DEFAULT_LEARNING_RATE="${DEFAULT_LEARNING_RATE:-5e-7}"
DEFAULT_BETA="${DEFAULT_BETA:-1.0}"
DEFAULT_HUB_MODEL_ID="${DEFAULT_HUB_MODEL_ID:-}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME:-}"
DEFAULT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR:-}"

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/../ModelAndDatasets/EleutherAI/pythia-410m}"
RECIPE_PATH="${RECIPE_PATH:-${SCRIPT_DIR}/recipes/zephyr/pythia-410m-base.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/models}"

USE_4BIT="${USE_4BIT:-false}"
if [[ "${USE_4BIT,,}" == "true" || "${USE_4BIT}" == "1" ]]; then
  QUANT_ARGS=(--load_in_4bit=True)
else
  QUANT_ARGS=()
fi

LOSS_TYPE="${LOSS_TYPE:-$DEFAULT_LOSS_TYPE}"
LOSS_TYPE_UPPER="${LOSS_TYPE^^}"
case "$LOSS_TYPE_UPPER" in
  DPO|SIMPO|LSIF|KTO|BCE|IPO|CPO|SLIC|UKL|DDRO) ;;
  *)
    echo "[ERROR] Unsupported LOSS_TYPE='$LOSS_TYPE'. Expected DPO, SIMPO, LSIF, BCE, IPO, CPO, SLIC, UKL, DDRO or KTO." >&2
    exit 1
    ;;
esac

LEARNING_RATE="${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}"
# Per-loss default beta:
# - DPO and most losses default to 1.0
# - SIMPO defaults to 2.0
BETA="$DEFAULT_BETA"
if [[ "$LOSS_TYPE_UPPER" == "SIMPO" ]]; then
  BETA="2.0"
fi

# Optional training overrides (kept empty by default to preserve recipe defaults)
OPTIM="${OPTIM:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
SEED="${SEED:-}"
MAX_STEPS="${MAX_STEPS:-}"

# Optional scheduler overrides (if set, override recipe)
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-}"
WARMUP_RATIO="${WARMUP_RATIO:-}"

# Optional: evaluation overrides
# NOTE: In some transformers versions, `do_eval` is effectively implied by `eval_strategy != "no"`.
# To fully skip evaluation, set eval_strategy=no (and optionally do_eval=false).
SKIP_EVAL="${SKIP_EVAL:-true}"
EVAL_STRATEGY="${EVAL_STRATEGY:-}"
EVAL_STEPS="${EVAL_STEPS:-}"

# Optional logging overrides (strings "true/false" are OK; parser casts bools)
TRACK_MARGIN_CHAIN="${TRACK_MARGIN_CHAIN:-}"
MARGIN_CHAIN_PATH="${MARGIN_CHAIN_PATH:-}"
# Optional: when using LoRA/QLoRA (lm_head not trainable), compute head-geometry on ALL LoRA params (default off).
MARGIN_CHAIN_LORA_ALL_LAYERS="${MARGIN_CHAIN_LORA_ALL_LAYERS:-}"

# Optional: DB Calibration (no regularizer; rescale backward gradients via stop-gradient trick)
ENABLE_DB_CALIB="${ENABLE_DB_CALIB:-}"
DB_CALIB_EPS="${DB_CALIB_EPS:-}"

EXTRA_ARGS=()
if [[ -n "${OPTIM}" ]]; then
  EXTRA_ARGS+=(optim="${OPTIM}")
fi
if [[ -n "${NUM_TRAIN_EPOCHS}" ]]; then
  EXTRA_ARGS+=(num_train_epochs="${NUM_TRAIN_EPOCHS}")
fi
if [[ -n "${SEED}" ]]; then
  EXTRA_ARGS+=(seed="${SEED}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  EXTRA_ARGS+=(max_steps="${MAX_STEPS}")
fi

if [[ -n "${LR_SCHEDULER_TYPE}" ]]; then
  EXTRA_ARGS+=(lr_scheduler_type="${LR_SCHEDULER_TYPE}")
fi
if [[ -n "${WARMUP_RATIO}" ]]; then
  EXTRA_ARGS+=(warmup_ratio="${WARMUP_RATIO}")
fi

# Evaluation toggles (optional)
if [[ -n "${SKIP_EVAL}" ]]; then
  if [[ "${SKIP_EVAL,,}" == "true" || "${SKIP_EVAL}" == "1" ]]; then
    EXTRA_ARGS+=(do_eval="false")
    EXTRA_ARGS+=(eval_strategy="no")
  fi
fi
if [[ -n "${EVAL_STRATEGY}" ]]; then
  EXTRA_ARGS+=(eval_strategy="${EVAL_STRATEGY}")
fi
if [[ -n "${EVAL_STEPS}" ]]; then
  EXTRA_ARGS+=(eval_steps="${EVAL_STEPS}")
fi

# Margin-chain logging (new)
if [[ -n "${TRACK_MARGIN_CHAIN}" ]]; then
  EXTRA_ARGS+=(track_margin_chain="${TRACK_MARGIN_CHAIN}")
fi
if [[ -n "${MARGIN_CHAIN_PATH}" ]]; then
  EXTRA_ARGS+=(margin_chain_path="${MARGIN_CHAIN_PATH}")
fi
if [[ -n "${MARGIN_CHAIN_LORA_ALL_LAYERS}" ]]; then
  EXTRA_ARGS+=(margin_chain_lora_all_layers="${MARGIN_CHAIN_LORA_ALL_LAYERS}")
fi

# DB Calibration toggles (optional)
if [[ -n "${ENABLE_DB_CALIB}" ]]; then
  EXTRA_ARGS+=(db_calibration_enable="${ENABLE_DB_CALIB}")
fi
if [[ -n "${DB_CALIB_EPS}" ]]; then
  EXTRA_ARGS+=(db_calibration_eps="${DB_CALIB_EPS}")
fi

SAFE_LR="${LEARNING_RATE//./p}"
SAFE_LR="${SAFE_LR//-/_}"
DEFAULT_HUB_ID="Pythia-410M-Base-${LOSS_TYPE_UPPER}-${LEARNING_RATE}-temp"
USER_HUB_ID="${DEFAULT_HUB_MODEL_ID:-}"
HUB_MODEL_ID="${HUB_MODEL_ID:-${USER_HUB_ID:-$DEFAULT_HUB_ID}}"
USER_RUN_NAME="${DEFAULT_RUN_NAME:-}"
RUN_NAME="${RUN_NAME:-${USER_RUN_NAME:-$HUB_MODEL_ID}}"
USER_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${USER_OUTPUT_DIR:-${OUTPUT_ROOT}/${HUB_MODEL_ID}}}"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Configuration"
echo "       LOSS_TYPE       : $LOSS_TYPE_UPPER"
echo "       LEARNING_RATE   : $LEARNING_RATE"
echo "       BETA            : $BETA"
echo "       OPTIM           : ${OPTIM:-<recipe default>}"
echo "       NUM_TRAIN_EPOCHS: ${NUM_TRAIN_EPOCHS:-<recipe default>}"
echo "       MAX_STEPS       : ${MAX_STEPS:-<recipe default>}"
echo "       SEED            : ${SEED:-<recipe default>}"
echo "       LR_SCHEDULER_TYPE: ${LR_SCHEDULER_TYPE:-<recipe default>}"
echo "       WARMUP_RATIO    : ${WARMUP_RATIO:-<recipe default>}"
echo "       TRACK_MARGIN_CHAIN: ${TRACK_MARGIN_CHAIN:-<recipe default>}"
echo "       HUB_MODEL_ID    : $HUB_MODEL_ID"
echo "       RUN_NAME        : $RUN_NAME"
echo "       OUTPUT_DIR      : $OUTPUT_DIR"
echo "       MODEL_PATH      : $MODEL_PATH"
echo "       RECIPE_PATH     : $RECIPE_PATH"
echo "       USE_4BIT        : ${USE_4BIT}"

ACCELERATE_LOG_LEVEL=info python scripts/run.py "$RECIPE_PATH" \
  run_name="$RUN_NAME" \
  learning_rate="$LEARNING_RATE" \
  hub_model_id="$HUB_MODEL_ID" \
  output_dir="$OUTPUT_DIR" \
  loss_type="$LOSS_TYPE_UPPER" \
  beta="$BETA" \
  model_name_or_path="$MODEL_PATH" \
  tokenizer_name_or_path="$MODEL_PATH" \
  "${QUANT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

