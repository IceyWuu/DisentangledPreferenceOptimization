#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# export WANDB_API_KEY="..."
export WANDB_API_KEY="3c3591c6e3b2c5b7239847189db2a2fe35eff970"

export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONWARNINGS="ignore::UserWarning:transformers.trainer"

iter_num=1
for i in $(seq 1 $iter_num); do
    gradient_accumulation_steps=16
    per_device_train_batch_size=2
    if [ "$i" -eq 1 ]; then
        beta=1.0
        learning_rate=5.0e-5
        hub_model_id="Qwen2.5-7B-Instruct-DPO-5e-5"
        loss_type="DPO"
        output_dir="models/$hub_model_id"
        ACCELERATE_LOG_LEVEL=info python scripts/run.py recipes/zephyr/qwen2.5-7b-instruct-simpo.yaml run_name=$hub_model_id learning_rate=$learning_rate hub_model_id=$hub_model_id output_dir=$output_dir loss_type=$loss_type beta=$beta --load_in_4bit=True
    else
       continue
    fi
    echo "Training"
done
