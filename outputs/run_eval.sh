#!/usr/bin/env bash
# Batch evaluation entry point. All (model × eval-method) jobs share one GPU pool.
# Automatically runs compare_calib_results.py after evaluation.
#
# Arguments:
#   --model-family  FAMILY    Model family to evaluate. Required.
#                             Choices: pythia-2b | mistral-7b | qwen2.5-7b | all
#
#   --methods       METHODS   Comma-separated loss types, or "all".
#                             Choices: bce,dpo,ddro,cpo,ipo,simpo,kto,slic,lsif,ukl,tidpo,baseline
#                             Example: "dpo,bce"
#
#   --category      CAT       Filter models by calibration status.
#                             Choices: all | base | calib
#
#   --eval-method   METHODS   Comma-separated evaluation benchmarks, or "all". Required.
#                             Choices: math_hard,mmlu_pro,bbh,musr  (lmeval_v2)
#                                      arc,gsm8k              (lmeval_v1)
#                             "all" expands to all of the above.
#
#   --parallel-gpus IDS       Comma-separated physical GPU ids for parallel evaluation.
#                             Example: 0,1,2,7
#
#   --skip-existing           Skip models that already have evaluation results.
#   --dry-run                 Print what would run without actually evaluating.
#   --batch-size    S         Optional. One int → same batch for every --eval-method.
#                             Or comma list, same length/order as --eval-method (e.g. 1,1,32).
#                             If omitted: math_hard=32, other metrics=1.
#   --base-model    PATH      Override base model path for adapter merging.
#
# Examples:
#   bash run_eval.sh --model-family qwen2.5-7b --methods "dpo,bce" --category all \
#     --eval-method "arc,gsm8k,math_hard" --parallel-gpus 0,1,2
#
#   bash run_eval.sh --model-family pythia-2b --methods all --category all \
#     --eval-method all --parallel-gpus 0,1 --skip-existing
#
#   bash run_eval.sh --model-family mistral-7b --methods "dpo,bce,tidpo" \
#     --category calib --eval-method "bbh,musr" --parallel-gpus 0,7

# nohup bash run_eval.sh \
#   --model-family qwen2.5-7b \
#   --methods "cpo,ddro,lsif,simpo,tidpo" \
#   --category all \
#   --eval-method "arc,gsm8k,math_hard,bbh,musr,mmlu_pro" \
#   --parallel-gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch-size "1,1,32,1,1,1" \
#   > /dev/null 2>&1 &

set -euo pipefail

EVAL_METHOD=""
MODEL_FAMILY=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-method)   EVAL_METHOD="$2"; shift 2 ;;
    --model-family)  MODEL_FAMILY="$2"; PASSTHROUGH+=("$1" "$2"); shift 2 ;;
    -h|--help)
      echo "用法: bash run_eval.sh --eval-method <methods> --model-family <fam> [batch_eval.py 参数...]"
      echo "  --eval-method  逗号分隔，如 arc,math_hard 或 all"
      echo "  --model-family 模型族，如 qwen2.5-7b, pythia-2b, mistral-7b"
      echo "其余参数透传给 batch_eval.py"
      exit 0 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

if ! command -v conda &>/dev/null; then
  for _c in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    [[ -f "${_c}" ]] && { source "${_c}"; break; }
  done
fi
if ! command -v conda &>/dev/null; then
  echo "[ERROR] 找不到 conda" >&2; exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${EVAL_METHOD}" ]]; then
  echo "[ERROR] 必须指定 --eval-method（或 all）" >&2; exit 1
fi

ALL_METHODS="math_hard,mmlu_pro,bbh,musr,arc,gsm8k"
if [[ "${EVAL_METHOD,,}" == "all" ]]; then
  EVAL_METHOD="${ALL_METHODS}"
fi

# 单次调用 batch_eval.py，所有 (模型 × 指标) 进同一个 GPU 池
python batch_eval.py \
  --eval-method "${EVAL_METHOD}" \
  "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# 汇总结果
if [[ -n "${MODEL_FAMILY}" ]]; then
  python compare_calib_results.py --eval-type "${EVAL_METHOD}" --model "${MODEL_FAMILY}"
fi
