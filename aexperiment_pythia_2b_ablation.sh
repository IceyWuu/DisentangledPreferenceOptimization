#!/bin/bash
set -euo pipefail

# Pythia-2.8B 消融实验（合并原 db_ema_beta 与 seed 两个脚本）。
#
# 选择消融类型（必选其一）：
#   ABLATION=dbema   —— 扫 db_ema_beta，RUN_NAME 含 dbema0p5 等；固定 SEED=FIXED_SEED（默认 42）
#   ABLATION=seed    —— 扫随机种子，RUN_NAME 含 seed1 等；db_ema 可走默认或你 export 的 DB_EMA_BETA
#
#   等价 CLI：--ablation dbema  |  --ablation seed
#   别名：dbema 可写 db_ema / ema；seed 可写 seeds
#
# 用法示例：
#   ABLATION=dbema DB_EMA_BETA_LIST="0.5 0.9" GPUS="0 1" bash aexperiment_pythia_2b_ablation.sh
#   ABLATION=seed  SEED_LIST="1 2" GPUS="3 4" bash aexperiment_pythia_2b_ablation.sh
#   bash aexperiment_pythia_2b_ablation.sh --ablation seed --ablation-serial
#
# 多卡：默认并行 + 轮询绑卡；串行 + nvidia-smi：ABLATION_SERIAL=true（或 --ablation-serial）
#
# 其余与 aexperiment_pythia_2b.sh 一致（默认 DPO + calib）。

# 默认消融类型（可被 --ablation 覆盖）
ABLATION="${ABLATION:-dbema}"

ENABLE_DB_CALIB="${ENABLE_DB_CALIB:-true}"
DB_CALIB_EPS="${DB_CALIB_EPS:-1e-12}"

# 串行选卡：统一一个开关；仍兼容旧环境变量
ABLATION_SERIAL="${ABLATION_SERIAL:-}"
if [[ -z "${ABLATION_SERIAL}" ]]; then
  if [[ "${DB_EMA_ABLATION_SERIAL:-}" == "true" || "${DB_EMA_ABLATION_SERIAL:-}" == "1" ]]; then
    ABLATION_SERIAL="true"
  elif [[ "${SEED_ABLATION_SERIAL:-}" == "true" || "${SEED_ABLATION_SERIAL:-}" == "1" ]]; then
    ABLATION_SERIAL="true"
  else
    ABLATION_SERIAL="false"
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ablation)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --ablation requires dbema|seed" >&2
        exit 1
      fi
      ABLATION="$2"
      shift 2
      ;;
    --ablation-serial)
      ABLATION_SERIAL="true"
      shift
      ;;
    --db-calib)
      ENABLE_DB_CALIB="true"
      shift
      ;;
    --db-calib-eps)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --db-calib-eps requires a value (e.g., 1e-12)" >&2
        exit 1
      fi
      ENABLE_DB_CALIB="true"
      DB_CALIB_EPS="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash aexperiment_pythia_2b_ablation.sh [--ablation dbema|seed] [--ablation-serial] [--db-calib] ..."
      echo "Env: ABLATION=dbema|seed   DB_EMA_BETA_LIST   SEED_LIST   FIXED_SEED (仅 dbema 跑时用)"
      echo "     GPUS  MAX_TASKS_PER_GPU  LOSS_TYPE  DB_EMA_BETA (可选，seed 消融时显式指定)"
      echo "     ABLATION_SERIAL=true  -> 串行 + nvidia-smi 选卡"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# 规范化消融名
_abl="${ABLATION,,}"
case "${_abl}" in
  dbema|db_ema|ema) ABLATION_KIND="dbema" ;;
  seed|seeds)       ABLATION_KIND="seed" ;;
  *)
    echo "[ERROR] ABLATION='${ABLATION}' is invalid. Use dbema or seed." >&2
    exit 1
    ;;
esac

max_tasks_per_gpu=${MAX_TASKS_PER_GPU:-1}
gpus=(${GPUS:-0})

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_TIME_EVAL="${TRAIN_TIME_EVAL:-false}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/outputs}"

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/../ModelAndDatasets/alignment-handbook/pythia-2.8b}"
RECIPE_PATH="${RECIPE_PATH:-${SCRIPT_DIR}/recipes/zephyr/pythia-2b-base.yaml}"

FIXED_LR="${FIXED_LR:-1e-4}"
FIXED_OPTIM="${FIXED_OPTIM:-adamw_torch}"
FIXED_NUM_EPOCHS="${FIXED_NUM_EPOCHS:-1}"
FIXED_SEED="${FIXED_SEED:-42}"
FIXED_MAX_STEPS="${FIXED_MAX_STEPS:-}"

LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"

TRACK_MARGIN_CHAIN="${TRACK_MARGIN_CHAIN:-true}"
MARGIN_CHAIN_LORA_ALL_LAYERS="${MARGIN_CHAIN_LORA_ALL_LAYERS:-false}"

LOSS_TYPE="${LOSS_TYPE:-DPO}"
loss="${LOSS_TYPE}"

DB_EMA_BETA_EXTRA="${DB_EMA_BETA:-}"

if [[ "${ABLATION_KIND}" == "dbema" ]]; then
  DB_EMA_BETA_LIST_STR="${DB_EMA_BETA_LIST:-0.5 0.9 0.95 0.999}"
  read -r -a ablation_values <<< "${DB_EMA_BETA_LIST_STR}"
else
  SEED_LIST_STR="${SEED_LIST:-1 2}"
  read -r -a ablation_values <<< "${SEED_LIST_STR}"
fi

get_running_tasks() {
    local gpu_id=$1
    nvidia-smi --id="$gpu_id" --query-compute-apps=pid --format=csv,noheader | wc -l
}

wait_for_available_gpu() {
    while true; do
        for gpu_id in "${gpus[@]}"; do
            running_tasks=$(get_running_tasks "$gpu_id")
            if [ "$running_tasks" -lt "$max_tasks_per_gpu" ]; then
                echo "$gpu_id"
                return
            fi
        done
        sleep 60
    done
}

max_parallel=$((${#gpus[@]} * max_tasks_per_gpu))
declare -a pids=()

wait_for_slot() {
    while true; do
        local -a alive=()
        local pid
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            fi
        done
        pids=("${alive[@]}")
        if ((${#pids[@]} < max_parallel)); then
            return
        fi
        sleep 5
    done
}

run_one_ablation() {
    local aval=$1
    local gpu_id=$2
    local timestamp=$3
    local run_suffix output_dir
    local use_seed use_db_ema

    if [[ "${ABLATION_KIND}" == "dbema" ]]; then
      local beta_fs="${aval//./p}"
      if [[ "${TRAIN_TIME_EVAL,,}" == "true" || "${TRAIN_TIME_EVAL}" == "1" ]]; then
        local calib_tag="base"
        if [[ "${ENABLE_DB_CALIB,,}" == "true" || "${ENABLE_DB_CALIB}" == "1" ]]; then
            calib_tag="calib"
        fi
        run_suffix="pythia-2.8B-${loss,,}-${calib_tag}-dbema${beta_fs}-${timestamp}"
      else
        run_suffix="pythia-2b-${loss,,}-calib-dbema${beta_fs}-${timestamp}"
      fi
      use_seed="${FIXED_SEED}"
      use_db_ema="${aval}"
    else
      if [[ "${TRAIN_TIME_EVAL,,}" == "true" || "${TRAIN_TIME_EVAL}" == "1" ]]; then
        local calib_tag="base"
        if [[ "${ENABLE_DB_CALIB,,}" == "true" || "${ENABLE_DB_CALIB}" == "1" ]]; then
            calib_tag="calib"
        fi
        run_suffix="pythia-2.8B-${loss,,}-${calib_tag}-seed${aval}-${timestamp}"
      else
        run_suffix="pythia-2b-${loss,,}-calib-seed${aval}-${timestamp}"
      fi
      use_seed="${aval}"
      use_db_ema="${DB_EMA_BETA_EXTRA}"
    fi

    output_dir="${OUTPUT_BASE}/${run_suffix}"

    echo "[INFO] ABLATION=${ABLATION_KIND} value=${aval} loss=${loss} GPU=${gpu_id} output_dir=${output_dir}"
    echo "[INFO] Fixed hparams: lr=${FIXED_LR}, optim=${FIXED_OPTIM}, epochs=${FIXED_NUM_EPOCHS}, max_steps=${FIXED_MAX_STEPS:-<none>}"
    if [[ "${ABLATION_KIND}" == "dbema" ]]; then
      echo "[INFO] SEED=${use_seed} (fixed)  db_ema_beta=${use_db_ema}  ENABLE_DB_CALIB=${ENABLE_DB_CALIB}"
    else
      echo "[INFO] SEED=${use_seed}  DB_EMA_BETA=${use_db_ema:-<trainer default>}  ENABLE_DB_CALIB=${ENABLE_DB_CALIB}"
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    USE_4BIT=true \
    MODEL_PATH="${MODEL_PATH}" \
    RECIPE_PATH="${RECIPE_PATH}" \
    ENABLE_DB_CALIB="${ENABLE_DB_CALIB}" \
    DB_CALIB_EPS="${DB_CALIB_EPS}" \
    LOSS_TYPE="${loss}" \
    LEARNING_RATE="${FIXED_LR}" \
    OPTIM="${FIXED_OPTIM}" \
    NUM_TRAIN_EPOCHS="${FIXED_NUM_EPOCHS}" \
    SEED="${use_seed}" \
    MAX_STEPS="${FIXED_MAX_STEPS}" \
    LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE}" \
    WARMUP_RATIO="${WARMUP_RATIO}" \
    TRACK_MARGIN_CHAIN="${TRACK_MARGIN_CHAIN}" \
    MARGIN_CHAIN_LORA_ALL_LAYERS="${MARGIN_CHAIN_LORA_ALL_LAYERS}" \
    DB_EMA_BETA="${use_db_ema}" \
    RUN_NAME="${run_suffix}" \
    HUB_MODEL_ID="${run_suffix}" \
    OUTPUT_DIR="${output_dir}" \
    OUTPUT_ROOT="${output_dir}" \
    bash "${SCRIPT_DIR}/legacy_scripts/Pythia-2B-Base.sh"

    echo "[INFO] Finished ABLATION=${ABLATION_KIND} value=${aval} run_name=${run_suffix}"
}

echo "[INFO] Pythia-2b ablation mode: ${ABLATION_KIND}  (${#ablation_values[@]} runs)  ABLATION_SERIAL=${ABLATION_SERIAL}"

idx=0
for aval in "${ablation_values[@]}"; do
    if [[ "${ABLATION_SERIAL,,}" == "true" || "${ABLATION_SERIAL}" == "1" ]]; then
        gpu_id=$(wait_for_available_gpu)
        timestamp=$(date +%Y%m%d_%H%M%S)
        run_one_ablation "${aval}" "${gpu_id}" "${timestamp}"
    else
        wait_for_slot
        gpu_id="${gpus[$((idx % ${#gpus[@]}))]}"
        idx=$((idx + 1))
        timestamp=$(date +%Y%m%d_%H%M%S)_${idx}_$$
        (
            run_one_ablation "${aval}" "${gpu_id}" "${timestamp}"
        ) &
        pids+=($!)
    fi
done

if [[ "${ABLATION_SERIAL,,}" != "true" && "${ABLATION_SERIAL}" != "1" ]]; then
    fail=0
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
    if [[ "$fail" -ne 0 ]]; then
        exit 1
    fi
fi
