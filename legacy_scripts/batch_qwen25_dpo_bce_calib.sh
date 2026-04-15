#!/usr/bin/env bash
# 已弃用独立编排逻辑：请直接用 aexperiment_qwen2.5_7b_instruct.sh（多 GPU / 多 loss / 校准 both）。
#
# 本文件保留为兼容入口，等价于原「四组：DPO±calib、BCE±calib」：
#   - 1 张卡：顺序跑 4 个 job
#   - 多张卡：按波次并行（每波最多 len(GPU_IDS) 个同时跑）
#
# 多卡示例（务必用绝对路径，nohup 时工作目录可靠）：
#   nohup env GPU_IDS=0,1,2,3 bash /work1/weichen/DPO/batch_qwen25_dpo_bce_calib.sh \
#     >> /work1/weichen/DPO/outputs/batch_qwen25.log 2>&1 &
#
# 等价直连：
#   GPU_IDS=0,1,2,3 LOSS_TYPES="DPO BCE" ENABLE_DB_CALIB=both \
#     bash /work1/weichen/DPO/aexperiment_qwen2.5_7b_instruct.sh

set -euo pipefail

THIS="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "${THIS}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
EXP_SCRIPT="${SCRIPT_DIR}/aexperiment_qwen2.5_7b_instruct.sh"

if [[ ! -f "${EXP_SCRIPT}" ]]; then
  echo "[BATCH] Missing ${EXP_SCRIPT}" >&2
  exit 1
fi

export GPU_IDS
export LOSS_TYPES="${LOSS_TYPES:-DPO BCE}"
export ENABLE_DB_CALIB="${ENABLE_DB_CALIB:-both}"

echo "[BATCH] delegate -> ${EXP_SCRIPT}  GPU_IDS=${GPU_IDS} LOSS_TYPES=${LOSS_TYPES} ENABLE_DB_CALIB=${ENABLE_DB_CALIB}  ($(date))"
exec bash "${EXP_SCRIPT}"
