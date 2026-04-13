#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# export WANDB_API_KEY="1e8665b4c06ccb7945c44b3da389d802257d66e3"
export WANDB_API_KEY="3c3591c6e3b2c5b7239847189db2a2fe35eff970"

export PYTHONPATH="$(pwd):$PYTHONPATH"   # 把 scripts/ 加到 PYTHONPATH
export PYTHONWARNINGS="ignore::UserWarning:transformers.trainer"

iter_num=1
for i in $(seq 1 $iter_num); do
    output_dir="models/$hub_model_id" # mode save local path
    gradient_accumulation_steps=16
    per_device_train_batch_size=2
    if [ "$i" -eq 1 ]; then
        beta=1.0
        learning_rate=5e-7
        # hub_model_id="Mistral-7B-Base-LSIF-5e-7_betaeweight"
        hub_model_id="Mistral-7B-Base-DPO-5e-7"
        loss_type="DPO"
        output_dir="models/$hub_model_id"
        # Use Accelerate with its default setting
        # ACCELERATE_LOG_LEVEL=info  accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run.py recipes/zephyr/mistral-7b-base-simpo.yaml run_name=$hub_model_id  learning_rate=$learning_rate hub_model_id=$hub_model_id output_dir=$output_dir loss_type=$loss_type  beta=$beta
        # Option B: Run directly with Python (if you're using a single GPU, which seems likely given CUDA_VISIBLE_DEVICES=0)
        ACCELERATE_LOG_LEVEL=info python scripts/run.py recipes/zephyr/mistral-7b-base-simpo.yaml run_name=$hub_model_id learning_rate=$learning_rate hub_model_id=$hub_model_id output_dir=$output_dir loss_type=$loss_type beta=$beta --load_in_4bit=True
    elif [ "$i" -eq 2 ]; then
        beta=1.0
        learning_rate=1e-6
        hub_model_id="Mistral-7B-Base-LSIF-1e-6"
        loss_type="LSIF"
        output_dir="models/$hub_model_id"
        ACCELERATE_LOG_LEVEL=info  accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run.py recipes/zephyr/mistral-7b-base-simpo.yaml run_name=$hub_model_id  learning_rate=$learning_rate hub_model_id=$hub_model_id output_dir=$output_dir loss_type=$loss_type  beta=$beta
    elif [ "$i" -eq 3 ]; then
        beta=1.0
        learning_rate=6e-7
        hub_model_id="Mistral-7B-Base-LSIF-6e-7"
        loss_type="LSIF"
        output_dir="models/$hub_model_id"
        ACCELERATE_LOG_LEVEL=info  accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run.py recipes/zephyr/mistral-7b-base-simpo.yaml run_name=$hub_model_id  learning_rate=$learning_rate hub_model_id=$hub_model_id output_dir=$output_dir loss_type=$loss_type  beta=$beta
    elif [ "$i" -eq 4 ]; then
        beta=1.0
        learning_rate=3e-7
        hub_model_id="Mistral-7B-Base-LSIF-3e-7"
        loss_type="LSIF"
        output_dir="models/$hub_model_id"
        ACCELERATE_LOG_LEVEL=info  accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run.py recipes/zephyr/mistral-7b-base-simpo.yaml run_name=$hub_model_id learning_rate=$learning_rate hub_model_id=$hub_model_id output_dir=$output_dir loss_type=$loss_type  beta=$beta
    else
       continue 
    fi
    echo "Training"


done


