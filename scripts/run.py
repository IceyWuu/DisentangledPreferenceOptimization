#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import os
import sys
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
from dpo_config import DPOConfig
from alignment_handbook import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from typing import Any, List, Literal, Optional
from peft import PeftConfig, PeftModel
from dpo_trainer import DPOTrainer
logger = logging.getLogger(__name__)

# 在导入附近补充
from functools import partial
import torch
from trl.trainer.utils import DPODataCollatorWithPadding

from torch.nn.utils.rnn import pad_sequence
from trl.trainer.utils import DPODataCollatorWithPadding, pad_to_length

def build_kto_collator(tokenizer, args):
    base = DPODataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=args.label_pad_token_id,
        is_encoder_decoder=False,
    )

    def collate(features):
        batch = base(features)
        bsz = batch["chosen_input_ids"].size(0)
        perm = torch.tensor(list(range(1, bsz)) + [0])  # ← 循环移位：[1,2,3,...,n-1,0]

        kl_inputs = []
        kl_attn = []
        kl_labels = []

        for i in range(bsz):
            # 当前样本的 prompt
            prompt_ids = batch["prompt_input_ids"][i]
            prompt_attn = batch["prompt_attention_mask"][i]
            prompt_len = int(prompt_attn.sum().item())

            # 打乱后的回复（含其原 prompt），但只取回复部分
            resp_ids_full   = batch["chosen_input_ids"][perm[i]]
            resp_attn_full  = batch["chosen_attention_mask"][perm[i]]
            resp_labels_full= batch["chosen_labels"][perm[i]]

            # 估计回复起点：用打乱样本自己的 prompt 长度切分
            resp_prompt_len = int(batch["prompt_attention_mask"][perm[i]].sum().item())
            resp_ids   = resp_ids_full[resp_prompt_len:]
            resp_attn  = resp_attn_full[resp_prompt_len:]
            resp_labels= resp_labels_full[resp_prompt_len:]

            # 拼接成 x（本样本 prompt） + y′（他人回复）
            combined_ids   = torch.cat([prompt_ids, resp_ids], dim=0)
            combined_attn  = torch.cat([prompt_attn, resp_attn], dim=0)

            # label：prompt 段置 -100，回复段用原 label
            prompt_labels = torch.full_like(prompt_ids, fill_value=args.label_pad_token_id)
            combined_labels = torch.cat([prompt_labels, resp_labels], dim=0)

            kl_inputs.append(combined_ids)
            kl_attn.append(combined_attn)
            kl_labels.append(combined_labels)

        # pad 到批次最大长度
        kl_inputs  = pad_sequence(kl_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
        kl_attn    = pad_sequence(kl_attn, batch_first=True, padding_value=0)
        kl_labels  = pad_sequence(kl_labels, batch_first=True, padding_value=args.label_pad_token_id)

        batch["KL_combined_input_ids"] = kl_inputs
        batch["KL_combined_attention_mask"] = kl_attn
        batch["KL_labels"] = kl_labels
        
        # DEBUG: Print collator output (only first batch)
        if not hasattr(build_kto_collator, '_debug_printed'):
            print(f"[DEBUG KTO Collator] Generated KL samples:")
            print(f"  - KL_combined_input_ids shape: {kl_inputs.shape}")
            print(f"  - KL_combined_attention_mask shape: {kl_attn.shape}")
            print(f"  - KL_labels shape: {kl_labels.shape}")
            print(f"  - Batch size: {bsz}")
            print(f"  - Permutation applied: {perm.tolist()}")
            build_kto_collator._debug_printed = True

        # 如果不再需要旧的 KL_input_ids，可不再写入；如需兼容，保持空
        return batch

    return collate

def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return
    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "simpo"],
    auto_insert_empty_system_msg: bool = True,
    change_template = None,
):
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "simpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )
            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            bos = tokenizer.bos_token
            if bos and example["text_chosen"].startswith(bos):
                example["text_chosen"] = example["text_chosen"][len(bos) :]
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            if bos and example["text_rejected"].startswith(bos):
                example["text_rejected"] = example["text_rejected"][len(bos) :]
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    def _env_true(name: str) -> bool:
        v = os.getenv(name, "")
        return v.lower() in {"1", "true", "yes", "y", "on"}

    train_time_eval = _env_true("TRAIN_TIME_EVAL") or _env_true("DPO_TRAIN_TIME_EVAL")

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Train-time eval mode: do NOT write any checkpoints / weights / configs / metrics files.
    # We keep tqdm enabled (so the caller can scrape ETA) and still print metrics to stdout.
    if train_time_eval:
        logger.warning(
            "[TRAIN_TIME_EVAL] Enabled: disabling save/eval/push_to_hub and skipping all on-disk artifacts."
        )
        # No eval during/after training
        training_args.do_eval = False
        training_args.eval_strategy = "no"
        # No saves during training
        training_args.save_strategy = "no"
        # No trackers (wandb/tensorboard/etc.)
        training_args.report_to = []
        # No hub push
        training_args.push_to_hub = False

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    if "mistral" in model_args.model_name_or_path.lower():
        change_template = "mistral"
    else:
        change_template = None
        
    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "simpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "change_template": change_template,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    #####################
    # Quantization config
    #####################
    # Get quantization config
    quantization_config = get_quantization_config(model_args)  

    # Determine device map and other settings based on quantization
    if quantization_config is not None and (quantization_config.load_in_4bit or quantization_config.load_in_8bit):
        # IMPORTANT: Use 'auto' for device map when quantizing
        device_map = "auto"
        use_cache = False  # Disable cache for quantized models during training
        # IMPORTANT: Disable gradient checkpointing with quantized models
        training_args.gradient_checkpointing = False 
        torch_dtype = torch.float16  # Required for quantization
        training_args.fp16 = False
        training_args.bf16 = True
    else:
        device_map = get_kbit_device_map() if quantization_config is not None else None
        use_cache=False if training_args.gradient_checkpointing else True
        torch_dtype = (
            model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
        )

    # Prepare model_kwargs
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        # use_flash_attention_2=model_args.use_flash_attention_2, # 注释for pythia-410m
        torch_dtype=torch_dtype,
        use_cache=use_cache,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    # 增加for pythia
    if getattr(model_args, "use_flash_attention_2", False):
        model_kwargs["use_flash_attention_2"] = True

    # Load the main model
    model = model_args.model_name_or_path

    # Handle adapter models (SFT adapters)
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)

        # Create model_kwargs for the base model (without the adapter path)
        base_model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            # use_flash_attention_2=model_args.use_flash_attention_2,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=use_cache,
            low_cpu_mem_usage=True,
            device_map=device_map,
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **base_model_kwargs,
        )

        # Load adapter on top
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )

        model_kwargs = None

    # Set up reference model
    ref_model = model
    ref_model_kwargs = model_kwargs

    # If using PEFT, don't pass a reference model to avoid issues with quantized models
    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate DPO trainer
    #########################
    logger.info("*** Instantiate DPO trainer ***")
    # Same values already live on DPOConfig; passing them again triggers DPOTrainer UserWarnings.
    training_args.remove_unused_columns = False
    if training_args.loss_type == "KTO":
        from kto_trainer import KTOTrainer
        kto_collator = build_kto_collator(tokenizer, training_args)
        trainer = KTOTrainer(
            model,
            ref_model,
            model_init_kwargs=model_kwargs,
            ref_model_init_kwargs=ref_model_kwargs,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_args),
            data_collator=kto_collator,              # 新增
        )
    else:
        trainer = DPOTrainer(
            model,
            ref_model,
            model_init_kwargs=model_kwargs,
            ref_model_init_kwargs=ref_model_kwargs,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_args),
        )

    ###############
    # Training loop
    ###############
    logger.info("*** Start Training! ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    if not train_time_eval:
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    if not train_time_eval:
        logger.info("*** Save model ***")
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment_handbook-handbook"],
    }
    if (not train_time_eval) and trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if (not train_time_eval) and training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if (not train_time_eval) and training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()