#!/usr/bin/env python
# coding=utf-8
"""
SFT Training Script using TRL's SFTTrainer with QLoRA support
"""
import logging
import os
import sys
from pathlib import Path

import datasets
import torch
import transformers
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"""


def main():
    # Disable datasets disk space check
    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

    # 查找 JSON 配置文件
    config_file = None
    for arg in sys.argv[1:]:
        if (arg.endswith('.json') or arg.endswith('.yaml')) and os.path.exists(arg):
            config_file = arg
            break
    
    if not config_file or not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # 加载配置
    if config_file.endswith('.json'):
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        import json
        try:
            import ruamel.yaml as ruamel_yaml
            yaml = ruamel_yaml.YAML()
            with open(config_file, 'r') as f:
                config_dict = yaml.load(f)
                config = json.loads(json.dumps(config_dict))
        except ImportError:
            import yaml as pyyaml
            with open(config_file, 'r') as f:
                config = pyyaml.safe_load(f)
    
    logger.info(f"Loaded config from: {config_file}")
    
    # 提取配置参数
    model_name_or_path = config.get('model_name_or_path', '../ModelAndDatasets/alignment-handbook/local_models/mistralai/Mistral-7B-v0.1')
    attn_implementation = config.get('attn_implementation', 'flash_attention_2')
    load_in_4bit = config.get('load_in_4bit', False)
    load_in_8bit = config.get('load_in_8bit', False)
    use_fp16 = config.get('use_fp16', False)
    use_bf16 = config.get('use_bf16', True)  # 默认启用 bf16
    
    # LoRA config
    lora_r = config.get('lora_r', 16)
    lora_alpha = config.get('lora_alpha', 16)
    lora_dropout = config.get('lora_dropout', 0.05)
    lora_target_modules = config.get('lora_target_modules', ['q_proj', 'v_proj', 'k_proj', 'gate_proj', 'o_proj', 'up_proj', 'down_proj'])
    
    # Dataset config
    dataset_mixer = config.get('dataset_mixer', {})
    dataset_splits = config.get('dataset_splits', ['train_sft', 'test_sft'])
    
    # Training config
    learning_rate = config.get('learning_rate', 2.0e-5)
    per_device_batch_size = config.get('per_device_train_batch_size', 2)
    grad_accumulation_steps = config.get('gradient_accumulation_steps', 8)
    num_train_epochs = config.get('num_train_epochs', 1.0)
    max_steps = config.get('max_steps', -1)
    output_dir = config.get('output_dir', '../ModelAndDatasets/alignment-handbook/zephyr-7b-sft')
    logging_steps = config.get('logging_steps', 10)
    save_steps = config.get('save_steps', 500)
    eval_steps = config.get('eval_steps', 500)
    warmup_ratio = config.get('warmup_ratio', 0.1)
    lr_scheduler_type = config.get('lr_scheduler_type', 'cosine')
    optim = config.get('optim', 'adamw_torch')
    seed = config.get('seed', 42)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)

    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Dataset mixer: {dataset_mixer}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {per_device_batch_size}")
    logger.info(f"Load in 4-bit: {load_in_4bit}")

    # Set seed
    set_seed(seed)

    ###############
    # Load datasets
    ###############
    from alignment_handbook import get_datasets, apply_chat_template
    
    data_config = {
        'dataset_mixer': dataset_mixer,
        'dataset_splits': dataset_splits,
    }
    
    logger.info(f"Loading datasets...")
    raw_datasets = get_datasets(
        data_config,
        splits=dataset_splits,
        columns_to_keep=["messages", "prompt", "chosen", "rejected"],
    )
    logger.info(
        f"Training on: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer
    #####################################
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.truncation_side = "left"

    # Use Mistral chat template
    if "mistral" in model_name_or_path.lower():
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
        logger.info("Using Mistral chat template")

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": True,
        },
        num_proc=8,  # 增加并行数
        remove_columns=column_names,
        desc="Applying chat template",
    )
    
    # 保存预处理后的数据集缓存
    cache_dir = os.path.join(output_dir, "dataset_cache")
    logger.info(f"Saving dataset cache to {cache_dir}")
    raw_datasets.save_to_disk(cache_dir)

    #####################
    # Quantization config
    #####################
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    ################
    # Load model
    ################
    logger.info(f"Loading model from: {model_name_or_path}")
    
    if quantization_config is not None:
        # QLoRA: 加载量化模型
        logger.info("Loading model in 4-bit/8-bit quantization...")
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        
        # DDP 模式下不使用 device_map
        device_map = {"": local_rank} if local_rank >= 0 else "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            attn_implementation="sdpa" if attn_implementation == 'flash_attention_2' else attn_implementation,
            quantization_config=quantization_config,
            device_map=device_map,
        )
        
        # 为量化模型准备 PEFT
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        
        # 创建 LoRA 配置
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 将 LoRA 添加到模型
        logger.info("Applying LoRA adapters...")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        # 普通 LoRA
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation if attn_implementation != 'flash_attention_2' else None,
            torch_dtype="auto",
        )
        
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    #########################
    # Create TrainingArguments
    #########################
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=1,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,  # 关闭以加速，有足够显存
        seed=seed,
        report_to=[],
        run_name="sft",
        remove_unused_columns=False,
    )

    #########################
    # Instantiate SFT trainer
    #########################
    logger.info("*** Instantiate SFTTrainer ***")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets.get("test"),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Start Training! ***")
    
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################
    # Save model
    ##################
    logger.info("*** Save model ***")
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")

    logger.info("*** Done! ***")


if __name__ == "__main__":
    main()
