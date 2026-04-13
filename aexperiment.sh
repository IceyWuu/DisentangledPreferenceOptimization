#!/bin/bash
set -euo pipefail

# Unified experiment runner for all model families.
#
# Arguments:
#   --model-family FAM      Required. pythia-410m | pythia-1.4b | pythia-2b | mistral-7b | qwen2.5-7b
#   --loss-types  "A B .."  Space-separated loss types (default: DPO)
#                           Choices: DPO BCE CPO IPO SIMPO SLIC LSIF UKL DDRO KTO TIDPO
#   --gpu-ids IDS           Comma-separated CUDA IDs (default: 0). Multi-GPU = parallel waves.
#   --category CAT          DB calibration mode: base | calib | both (default: calib).
#                           base=no calibration; calib=on; both=each loss with calib on then off.
#   --dataset NAME          Dataset (ultrafeedback|hh-rlhf|hh-rlhf-merged). Only for mistral-7b.
#
# Environment variable overrides (all optional):
#   FIXED_LR  FIXED_OPTIM  FIXED_NUM_EPOCHS  FIXED_SEED  FIXED_MAX_STEPS
#   LR_SCHEDULER_TYPE  WARMUP_RATIO  PER_DEVICE_BATCH_SIZE  GRAD_ACCUMULATION
#   GRADIENT_CHECKPOINTING  LORA_R  CATEGORY  DB_CALIB_EPS  DB_EMA_BETA
#                           (ENABLE_DB_CALIB=true|false|both still maps to calib|base|both)
#   MAX_LENGTH  MAX_PROMPT_LENGTH  TRACK_MARGIN_CHAIN  MARGIN_CHAIN_LORA_ALL_LAYERS
#   CLEAN_CACHE  LOAD_IN_4BIT
#
# Examples:
#   bash aexperiment.sh --model-family qwen2.5-7b --loss-types "DPO BCE" \
#     --gpu-ids 0,1,2,3 --category both
#
#   bash aexperiment.sh --model-family pythia-2b --loss-types "DPO BCE" \
#     --gpu-ids 0 --category calib
#
#   bash aexperiment.sh --model-family mistral-7b --loss-types DPO \
#     --gpu-ids 4 --dataset hh-rlhf --category base
#
#   FIXED_LR=1e-6 bash aexperiment.sh --model-family pythia-410m --loss-types "DPO SIMPO"

# ──────────────────────── Defaults ────────────────────────
MODEL_FAMILY=""
LOSS_TYPES_STR="${LOSS_TYPES:-DPO}"
GPU_IDS="${GPU_IDS:-0}"
# DB calibration category: base | calib | both. CLI --category overrides.
# If CATEGORY is unset, ENABLE_DB_CALIB (true|false|both) still selects the mode for batch wrappers.
if [[ -n "${CATEGORY+x}" ]]; then
  :
elif [[ -n "${ENABLE_DB_CALIB+x}" ]]; then
  case "${ENABLE_DB_CALIB}" in
    true|1)   CATEGORY="calib" ;;
    false|0)  CATEGORY="base" ;;
    both|all) CATEGORY="both" ;;
    *)        CATEGORY="calib" ;;
  esac
else
  CATEGORY="calib"
fi
DB_CALIB_EPS="${DB_CALIB_EPS:-1e-12}"
DB_EMA_BETA="${DB_EMA_BETA:-}"
DATASET_CHOICE="${DATASET_CHOICE:-ultrafeedback}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-}"
LORA_R="${LORA_R:-}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"
CLEAN_CACHE="${CLEAN_CACHE:-false}"

TRACK_MARGIN_CHAIN="${TRACK_MARGIN_CHAIN:-true}"
MARGIN_CHAIN_LORA_ALL_LAYERS="${MARGIN_CHAIN_LORA_ALL_LAYERS:-false}"

# ──────────────────────── Parse CLI ────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-family)       MODEL_FAMILY="$2"; shift 2 ;;
    --loss-types)         LOSS_TYPES_STR="$2"; shift 2 ;;
    --gpu-ids)            GPU_IDS="$2"; shift 2 ;;
    --category)           CATEGORY="$2"; shift 2 ;;
    --dataset)            DATASET_CHOICE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
  esac
done

case "${CATEGORY,,}" in
  base|calib|both) ;;
  *)
    echo "[ERROR] --category must be base|calib|both, got: ${CATEGORY}" >&2
    exit 1 ;;
esac

if [[ -z "${MODEL_FAMILY}" ]]; then
  echo "[ERROR] --model-family is required (pythia-410m|pythia-1.4b|pythia-2b|mistral-7b|qwen2.5-7b)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore::UserWarning:transformers.trainer"

# ──────────────────────── Model profile ────────────────────────
# Each family defines: RECIPE, MODEL_PATH, DEFAULT_LR, RUN_PREFIX,
# and optionally overrides for batch/grad_acc/grad_ckpt/lora_r/4bit.
case "${MODEL_FAMILY}" in
  pythia-410m)
    RECIPE_PATH="${RECIPE_PATH:-${SCRIPT_DIR}/recipes/zephyr/pythia-410m-base.yaml}"
    MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/../ModelAndDatasets/EleutherAI/pythia-410m}"
    DEFAULT_LR="6e-7"
    RUN_PREFIX="pythia-410m"
    : "${PER_DEVICE_BATCH_SIZE:=4}"
    : "${GRAD_ACCUMULATION:=8}"
    : "${GRADIENT_CHECKPOINTING:=false}"
    : "${LOAD_IN_4BIT:=false}"
    ;;
  pythia-1.4b)
    RECIPE_PATH="${RECIPE_PATH:-${SCRIPT_DIR}/recipes/zephyr/pythia-1.4b-base.yaml}"
    MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/../ModelAndDatasets/EleutherAI/pythia-1.4b}"
    DEFAULT_LR="6e-7"
    RUN_PREFIX="pythia-1.4b"
    : "${PER_DEVICE_BATCH_SIZE:=4}"
    : "${GRAD_ACCUMULATION:=8}"
    : "${GRADIENT_CHECKPOINTING:=false}"
    : "${LOAD_IN_4BIT:=false}"
    ;;
  pythia-2b)
    RECIPE_PATH="${RECIPE_PATH:-${SCRIPT_DIR}/recipes/zephyr/pythia-2b-base.yaml}"
    MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/../ModelAndDatasets/alignment-handbook/pythia-2.8b}"
    DEFAULT_LR="1e-4"
    RUN_PREFIX="pythia-2b"
    : "${PER_DEVICE_BATCH_SIZE:=4}"
    : "${GRAD_ACCUMULATION:=8}"
    : "${GRADIENT_CHECKPOINTING:=false}"
    : "${LOAD_IN_4BIT:=true}"
    ;;
  mistral-7b)
    RECIPE_PATH="${RECIPE_PATH:-${SCRIPT_DIR}/recipes/zephyr/mistral-7b-base-simpo.yaml}"
    MODEL_PATH=""  # resolved from recipe via sed
    DEFAULT_LR="3e-7"
    RUN_PREFIX="mistral-7b"
    : "${PER_DEVICE_BATCH_SIZE:=1}"
    : "${GRAD_ACCUMULATION:=32}"
    : "${GRADIENT_CHECKPOINTING:=false}"
    : "${LORA_R:=8}"
    : "${LOAD_IN_4BIT:=true}"
    ;;
  qwen2.5-7b)
    RECIPE_PATH="${RECIPE_PATH:-${SCRIPT_DIR}/recipes/zephyr/qwen2.5-7b-instruct-simpo.yaml}"
    MODEL_PATH=""  # from recipe
    DEFAULT_LR="5.0e-7"
    RUN_PREFIX="qwen2.5-7b-instruct"
    : "${PER_DEVICE_BATCH_SIZE:=1}"
    : "${GRAD_ACCUMULATION:=32}"
    : "${GRADIENT_CHECKPOINTING:=true}"
    : "${LORA_R:=8}"
    : "${LOAD_IN_4BIT:=true}"
    ;;
  *)
    echo "[ERROR] Unknown --model-family: ${MODEL_FAMILY}" >&2
    echo "        Choices: pythia-410m | pythia-1.4b | pythia-2b | mistral-7b | qwen2.5-7b" >&2
    exit 1 ;;
esac

FIXED_LR="${FIXED_LR:-${DEFAULT_LR}}"
FIXED_OPTIM="${FIXED_OPTIM:-adamw_torch}"
FIXED_NUM_EPOCHS="${FIXED_NUM_EPOCHS:-1}"
FIXED_SEED="${FIXED_SEED:-42}"
FIXED_MAX_STEPS="${FIXED_MAX_STEPS:-}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"

# ──────────────────────── Dataset (mistral-7b only) ────────────────────────
DATASET_PATH=""
DATASET_TRAIN_SPLIT=""
DATASET_TEST_SPLIT=""

validate_prepared_dataset() {
  local ds_root="$1"
  [[ -d "${ds_root}/train" && -d "${ds_root}/test" ]] || return 1
  python - "${ds_root}" <<'PY' 2>/dev/null
import sys
from datasets import load_from_disk
ds = load_from_disk(f"{sys.argv[1]}/train")
sample = ds[0]["chosen"] if len(ds) > 0 else None
ok = (isinstance(sample, list) and len(sample) > 0
      and isinstance(sample[0], dict) and "role" in sample[0])
raise SystemExit(0 if ok else 1)
PY
}

ensure_prepared_dataset() {
  local ds_root="$1" build_script="$2" label="$3"
  if validate_prepared_dataset "${ds_root}"; then return; fi
  if [[ -d "${ds_root}/train" || -d "${ds_root}/test" ]]; then
    rm -rf "${ds_root}/train" "${ds_root}/test" "${ds_root}/.format_version"
  fi
  [[ -f "${build_script}" ]] || { echo "[ERROR] Build script not found: ${build_script}" >&2; exit 1; }
  python "${build_script}"
}

if [[ "${MODEL_FAMILY}" == "mistral-7b" ]]; then
  case "${DATASET_CHOICE}" in
    ultrafeedback)
      DATASET_PATH="../ModelAndDatasets/HuggingFaceH4/ultrafeedback_binarized"
      DATASET_CACHE_BASE="${SCRIPT_DIR}/../ModelAndDatasets/HuggingFaceH4/ultrafeedback_binarized"
      DATASET_TRAIN_SPLIT="train_prefs"
      DATASET_TEST_SPLIT="test_prefs"
      ;;
    hh-rlhf)
      _root="${SCRIPT_DIR}/../ModelAndDatasets/Anthropic___hh-rlhf/local_disk"
      ensure_prepared_dataset "${_root}" "${SCRIPT_DIR}/hh-rlhf-dataset-transformer/prepare_default.py" "hh-rlhf"
      DATASET_PATH="../ModelAndDatasets/Anthropic___hh-rlhf/local_disk"
      DATASET_CACHE_BASE="${_root}"
      DATASET_TRAIN_SPLIT="train"; DATASET_TEST_SPLIT="test"
      ;;
    hh-rlhf-merged)
      _root="${SCRIPT_DIR}/../ModelAndDatasets/Anthropic___hh-rlhf/helpful_harmless_merged"
      ensure_prepared_dataset "${_root}" "${SCRIPT_DIR}/hh-rlhf-dataset-transformer/merge_and_convert.py" "hh-rlhf-merged"
      DATASET_PATH="../ModelAndDatasets/Anthropic___hh-rlhf/helpful_harmless_merged"
      DATASET_CACHE_BASE="${_root}"
      DATASET_TRAIN_SPLIT="train"; DATASET_TEST_SPLIT="test"
      ;;
    *)
      echo "[ERROR] Unsupported --dataset: ${DATASET_CHOICE}" >&2; exit 1 ;;
  esac
else
  DATASET_CACHE_BASE="${SCRIPT_DIR}/../ModelAndDatasets/HuggingFaceH4/ultrafeedback_binarized"
fi

# ──────────────────────── GPU list ────────────────────────
GPUS=()
while IFS= read -r _g || [[ -n "${_g}" ]]; do
  _g="${_g#"${_g%%[![:space:]]*}"}"; _g="${_g%"${_g##*[![:space:]]}"}"
  [[ -n "${_g}" ]] && GPUS+=("${_g}")
done < <(printf '%s' "${GPU_IDS}" | tr ',' '\n')
[[ ${#GPUS[@]} -gt 0 ]] || { echo "[ERROR] GPU_IDS is empty" >&2; exit 1; }

# ──────────────────────── Calib mode expansion ────────────────────────
declare -a calib_modes=()
case "${CATEGORY,,}" in
  both) calib_modes=(true false) ;;
  calib) calib_modes=(true) ;;
  base) calib_modes=(false) ;;
esac

read -r -a loss_types <<< "${LOSS_TYPES_STR}"
[[ ${#loss_types[@]} -gt 0 ]] || { echo "[ERROR] --loss-types is empty" >&2; exit 1; }

declare -a jobs=()
for loss in "${loss_types[@]}"; do
  for cal in "${calib_modes[@]}"; do
    jobs+=("${loss}|${cal}")
  done
done

# ──────────────────────── Cache cleaning ────────────────────────
clean_cache_at_start() {
  [[ "${CLEAN_CACHE,,}" == "true" || "${CLEAN_CACHE}" == "1" ]] || return 0
  local cleaned=0
  for path in "${DATASET_CACHE_BASE}"/*/cache-*.arrow "${DATASET_CACHE_BASE}"/*/*/cache-*.arrow; do
    [[ -f "$path" ]] && { rm -f "$path"; ((cleaned++)) || true; }
  done
  [[ $cleaned -gt 0 ]] && echo "[CACHE] Cleaned $cleaned old cache files"
  return 0
}

# ──────────────────────── Single experiment ────────────────────────
run_single_experiment() {
  local gpu="$1" loss="$2" calib="$3"

  local calib_suffix=""
  [[ "${calib,,}" == "true" || "${calib}" == "1" ]] && calib_suffix="-calib"

  local lora_tag=""
  [[ -n "${LORA_R:-}" ]] && lora_tag="-lora"

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local run_suffix="${RUN_PREFIX}-${loss,,}${lora_tag}${calib_suffix}-${timestamp}"
  local output_dir="${SCRIPT_DIR}/outputs/${run_suffix}"
  mkdir -p "${output_dir}"

  clean_cache_at_start

  # For mistral-7b: generate active recipe with dataset paths substituted
  local active_recipe="${RECIPE_PATH}"
  if [[ "${MODEL_FAMILY}" == "mistral-7b" && -n "${DATASET_PATH}" ]]; then
    active_recipe="${output_dir}/recipe_active.yaml"
    local _resolved_model_path
    _resolved_model_path="$(grep '^model_name_or_path:' "${RECIPE_PATH}" | sed 's/^model_name_or_path:[[:space:]]*//' | xargs -I{} realpath "${SCRIPT_DIR}/{}" 2>/dev/null || true)"
    sed \
      -e "s#\.\./ModelAndDatasets/HuggingFaceH4/ultrafeedback_binarized#${DATASET_PATH}#g" \
      -e "s#\.\./ModelAndDatasets/Anthropic___hh-rlhf/default/0.0.0#${DATASET_PATH}#g" \
      -e "s#^-[[:space:]]*train_prefs\$#- ${DATASET_TRAIN_SPLIT}#g" \
      -e "s#^-[[:space:]]*test_prefs\$#- ${DATASET_TEST_SPLIT}#g" \
      -e "s#^-[[:space:]]*train\$#- ${DATASET_TRAIN_SPLIT}#g" \
      -e "s#^-[[:space:]]*test\$#- ${DATASET_TEST_SPLIT}#g" \
      ${_resolved_model_path:+-e "s#^model_name_or_path:.*#model_name_or_path: ${_resolved_model_path}#"} \
      "${RECIPE_PATH}" > "${active_recipe}"
  fi

  echo "[INFO] ${run_suffix} | GPU=${gpu} calib=${calib}"

  local EXTRA_ARGS=()
  [[ -n "${FIXED_MAX_STEPS}" ]] && EXTRA_ARGS+=(max_steps="${FIXED_MAX_STEPS}")
  [[ -n "${MAX_LENGTH}" ]]      && EXTRA_ARGS+=(max_length="${MAX_LENGTH}")
  [[ -n "${MAX_PROMPT_LENGTH}" ]] && EXTRA_ARGS+=(max_prompt_length="${MAX_PROMPT_LENGTH}")

  EXTRA_ARGS+=(db_calibration_enable="${calib}")
  [[ -n "${DB_CALIB_EPS}" ]]  && EXTRA_ARGS+=(db_calibration_eps="${DB_CALIB_EPS}")
  [[ -n "${DB_EMA_BETA}" ]]   && EXTRA_ARGS+=(db_ema_beta="${DB_EMA_BETA}")

  # Quantization
  local QUANT_ARGS=()
  if [[ "${LOAD_IN_4BIT,,}" == "true" || "${LOAD_IN_4BIT}" == "1" ]]; then
    QUANT_ARGS=(--load_in_4bit=True)
  fi

  # Model path override (pythia models have explicit MODEL_PATH; 7B models use recipe)
  local MODEL_ARGS=()
  if [[ -n "${MODEL_PATH}" ]]; then
    MODEL_ARGS+=(model_name_or_path="${MODEL_PATH}" tokenizer_name_or_path="${MODEL_PATH}")
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" \
  ACCELERATE_LOG_LEVEL=info \
  python "${SCRIPT_DIR}/scripts/run.py" "${active_recipe}" \
    loss_type="${loss}" \
    per_device_train_batch_size="${PER_DEVICE_BATCH_SIZE}" \
    gradient_accumulation_steps="${GRAD_ACCUMULATION}" \
    gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    ${LORA_R:+lora_r="${LORA_R}"} \
    learning_rate="${FIXED_LR}" \
    optim="${FIXED_OPTIM}" \
    num_train_epochs="${FIXED_NUM_EPOCHS}" \
    seed="${FIXED_SEED}" \
    lr_scheduler_type="${LR_SCHEDULER_TYPE}" \
    warmup_ratio="${WARMUP_RATIO}" \
    track_margin_chain="${TRACK_MARGIN_CHAIN}" \
    margin_chain_lora_all_layers="${MARGIN_CHAIN_LORA_ALL_LAYERS}" \
    run_name="${run_suffix}" \
    hub_model_id="${run_suffix}" \
    output_dir="${output_dir}" \
    "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]}" \
    "${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"}"

  echo "[INFO] Completed: ${run_suffix}"
}

# ──────────────────────── Scheduling ────────────────────────
run_jobs_sequential() {
  local gpu="${GPUS[0]}"
  for job in "${jobs[@]}"; do
    IFS='|' read -r loss cal <<< "${job}"
    run_single_experiment "${gpu}" "${loss}" "${cal}"
  done
}

run_jobs_parallel_waves() {
  local n_jobs=${#jobs[@]} n_gpu=${#GPUS[@]} start=0 wave=0
  while [[ ${start} -lt ${n_jobs} ]]; do
    wave=$((wave + 1))
    local pids=() j=0
    while [[ ${j} -lt ${n_gpu} && $((start + j)) -lt ${n_jobs} ]]; do
      local idx=$((start + j))
      local job="${jobs[$idx]}" gpu="${GPUS[$j]}"
      IFS='|' read -r loss cal <<< "${job}"
      run_single_experiment "${gpu}" "${loss}" "${cal}" &
      pids+=($!)
      j=$((j + 1))
    done
    start=$((start + j))
    local fail=0
    for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
    if [[ "${fail}" -ne 0 ]]; then
      echo "[ERROR] One or more jobs in wave ${wave} failed." >&2; exit 1
    fi
  done
}

echo "[INFO] ${MODEL_FAMILY}: ${#jobs[@]} jobs (${loss_types[*]} × category=${CATEGORY} calib=${calib_modes[*]}), ${#GPUS[@]} GPU(s) (${GPUS[*]})"

if [[ ${#GPUS[@]} -eq 1 ]]; then
  run_jobs_sequential
else
  run_jobs_parallel_waves
fi

echo "[INFO] All experiments finished!"
