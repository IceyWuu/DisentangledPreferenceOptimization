#!/bin/bash
set -euo pipefail

# 计算记录penultimate梯度的, 和margin变化_l2
# 可自定义：并发上限（设为1即严格串行），可用GPU列表
max_tasks_per_gpu=${MAX_TASKS_PER_GPU:-1}
gpus=(${GPUS:-0})

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查当前GPU上的任务数量
get_running_tasks() {
    local gpu_id=$1
    nvidia-smi --id="$gpu_id" --query-compute-apps=pid --format=csv,noheader | wc -l
}

# 等待直到有空闲 GPU
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

# 需要顺序运行的 loss 类型（按顺序串行跑完）
# loss_types=(DPO LSIF KTO)
loss_types=(KTO)

for loss in "${loss_types[@]}"; do
    gpu_id=$(wait_for_available_gpu)
    timestamp=$(date +%Y%m%d_%H%M%S)
    run_suffix="pythia-410m-${loss,,}-${timestamp}"
    output_dir="${SCRIPT_DIR}/outputs/${run_suffix}"

    echo "[INFO] Start loss=${loss} on GPU ${gpu_id}, output_dir=${output_dir}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    LOSS_TYPE="${loss}" \
    RUN_NAME="${run_suffix}" \
    HUB_MODEL_ID="${run_suffix}" \
    OUTPUT_DIR="${output_dir}" \
    OUTPUT_ROOT="${output_dir}" \
    bash "${SCRIPT_DIR}/Pythia-410M-Base.sh"

    echo "[INFO] Finished loss=${loss} run_name=${run_suffix}"
    # 串行运行：等待当前任务完成后再进入下一轮循环
done