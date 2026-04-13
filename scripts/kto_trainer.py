import inspect
import json
import os
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

# from trl.import_utils import is_peft_available, is_wandb_available
import importlib.util
def is_peft_available():
    return importlib.util.find_spec("peft") is not None
def is_wandb_available():
    return importlib.util.find_spec("wandb") is not None

from trl.models import PreTrainedModelWrapper, create_reference_model
from dpo_config import DPOConfig
from trl.trainer.utils  import (
    DPODataCollatorWithPadding,
    RunningMoments,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    # trl_sanitze_kwargs_for_tagging,
)
import math

def trl_sanitze_kwargs_for_tagging(model, tag_names, kwargs):
    """
    Fallback function for trl_sanitze_kwargs_for_tagging.
    This function is likely used to sanitize kwargs before pushing to hub,
    potentially adding model tags. For now, we just return kwargs as-is.
    """
    # The actual logic might involve adding tags to the model or kwargs
    # For compatibility, just return kwargs unchanged.
    # If tags are crucial, they might be handled by the model itself or elsewhere.
    return kwargs

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

from dpo_trainer import DPOTrainer


class KTOTrainer(DPOTrainer):
    r"""
    KTO Trainer for Kahneman-Tversky Optimization.

    This trainer implements the full KTO algorithm with KL baseline estimation,
    similar to the implementation in HALO but adapted for the DIL framework.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # KTO specific parameters
        self.loss_type = "KTO"  # Force KTO loss type
        # 强制设置 remove_unused_columns=False，因为 KTO 需要保留 chosen/rejected 等列
        if hasattr(self.args, 'remove_unused_columns'):
            self.args.remove_unused_columns = False
        self.policy_dtype = getattr(self.model, "dtype", torch.float16)
    # def get_sequence_rewards(self, policy_logps, reference_logps, length_normalized=False, token_mask=None):
    #     # 直接复用 HALO 的实现（不做 humanline 裁剪）
    #     token_rewards = policy_logps - reference_logps
    #     unclamped = torch.Tensor([1]).to(self.policy_dtype).to(self.accelerator.device)

    #     if length_normalized:
    #         if token_mask is None:
    #             token_mask = ((policy_logps != 0) | (reference_logps != 0)).float()
    #         normalization = token_mask.sum(-1).clamp(min=1)
    #         seq_rewards = (token_rewards * token_mask).sum(-1) / normalization
    #     else:
    #         seq_rewards = token_rewards.sum(-1)

    #     return seq_rewards, unclamped

    def get_sequence_rewards(self, policy_logps, reference_logps, length_normalized=False, token_mask=None):
        # 注意：policy_logps 和 reference_logps 已经是平均后的标量 (batch_size,)
        # 因为 get_batch_logps 使用了 average_log_prob=True
        # 所以直接相减即可，不需要再做 token 级别的 sum
        seq_rewards = policy_logps - reference_logps
        unclamped = torch.Tensor([1]).to(self.policy_dtype).to(self.accelerator.device)
        
        # length_normalized 在这种情况下不需要，因为已经是平均后的值
        # 但保留参数以保持接口一致性
        return seq_rewards, unclamped

    def kto_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the Kahneman-Tversky loss for a batch of policy and reference model log probabilities.

        If generation y ~ p_desirable, we have the 'desirable' loss:
            L(x, y) := 1 - sigmoid(beta * ([log p_policy(y|x) - log p_reference(y|x)] - KL(p_policy || p_reference)))
        If generation y ~ p_undesirable, we have the 'undesirable' loss:
            L(x, y) := 1 - sigmoid(beta * (KL(p_policy || p_reference) - [log p_policy(y|x) - log p_reference(y|x)]))

        The desirable losses are weighed by config.loss.desirable_weight.
        The undesirable losses are weighed by config.loss.undesirable_weight.
        This should be used to address imbalances in the ratio of desirable:undesirable examples respectively.
        The KL term is estimated by matching x with unrelated outputs y', then calculating the average log ratio
        log p_policy(y'|x) - log p_reference(y'|x).

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            policy_KL_logps: Log probabilities of the policy model for the KL estimation samples. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            reference_KL_logps: Log probabilities of the reference model for the KL estimation samples. Shape: (batch_size,)

        Returns:
            A tuple of six tensors: (losses, chosen_rewards, rejected_rewards, KL, chosen_unclamped, rejected_unclamped).
        """
        # Calculate rewards for chosen and rejected examples
        if policy_chosen_logps.shape[0] != 0:
            chosen_rewards, chosen_unclamped = self.get_sequence_rewards(policy_chosen_logps, reference_chosen_logps)
        else:
            chosen_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)
            chosen_unclamped = torch.Tensor([1]).to(self.policy_dtype).to(self.accelerator.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_rewards, rejected_unclamped = self.get_sequence_rewards(policy_rejected_logps, reference_rejected_logps)
        else:
            rejected_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)
            rejected_unclamped = torch.Tensor([1]).to(self.policy_dtype).to(self.accelerator.device)

        if policy_KL_logps.shape[0] != 0:
            KL_rewards, _ = self.get_sequence_rewards(policy_KL_logps.detach(), reference_KL_logps.detach())
            # DEBUG: Print KL logps info
            if self.accelerator.is_main_process:
                print(f"[DEBUG KTO] policy_KL_logps shape: {policy_KL_logps.shape}, mean: {policy_KL_logps.mean().item():.6f}")
                print(f"[DEBUG KTO] reference_KL_logps shape: {reference_KL_logps.shape}, mean: {reference_KL_logps.mean().item():.6f}")
                print(f"[DEBUG KTO] KL_rewards shape: {KL_rewards.shape}, mean: {KL_rewards.mean().item():.6f}")
        else:
            KL_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.accelerator.device)
            # DEBUG: Warn if no KL samples
            if self.accelerator.is_main_process:
                print(f"[DEBUG KTO] WARNING: No KL samples! policy_KL_logps is empty")

        stats = self.accelerator.reduce(torch.Tensor([
            (chosen_rewards.abs() != 0).float().sum().item(),
            len(chosen_rewards),
            (rejected_rewards.abs() != 0).float().sum().item(),
            len(rejected_rewards),
            KL_rewards.sum(),
            (KL_rewards.abs() != 0).float().sum().item(),
        ]).to(self.accelerator.device), reduction="sum")

        KL = (stats[4] / stats[5].clamp(min=1)).clamp(min=0)
        
        # DEBUG: Print final KL baseline value
        if self.accelerator.is_main_process:
            print(f"[DEBUG KTO] Final KL baseline: {KL.item():.6f}, stats: {stats.tolist()}")

        desirable_weight = getattr(self.args, "desirable_weight", 1.0)
        undesirable_weight = getattr(self.args, "undesirable_weight", 1.0)

        chosen_losses = desirable_weight * (1 - F.sigmoid(self.beta * (chosen_rewards - KL))) if len(chosen_rewards) else chosen_rewards
        rejected_losses = undesirable_weight * (1 - F.sigmoid(self.beta * (KL - rejected_rewards))) if len(rejected_rewards) else rejected_rewards

        losses = torch.cat((chosen_losses, rejected_losses), 0)
        return losses, chosen_rewards.detach(), rejected_rewards.detach(), KL.detach(), chosen_unclamped, rejected_unclamped

    def concatenated_forward_with_kl(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ):
        # 先正常跑 chosen/rejected
        concat = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {"labels": concat["concatenated_labels"],
             "decoder_input_ids": concat.pop("concatenated_decoder_input_ids", None)}
            if self.is_encoder_decoder else {}
        )
        # all_logits = model(
        #     concat["concatenated_input_ids"],
        #     attention_mask=concat["concatenated_attention_mask"],
        #     use_cache=False,
        #     **model_kwargs,
        # ).logits
        outputs = model(
            concat["concatenated_input_ids"],
            attention_mask=concat["concatenated_attention_mask"],
            use_cache=False,
            output_hidden_states=True,  # ★ 关键
            **model_kwargs,
        )
        all_logits = outputs.logits

        all_logps = self.get_batch_logps(
            all_logits,
            concat["concatenated_labels"],
            average_log_prob=True,   # 和 DIL对齐，平均
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        # 再单独跑 KL（如果有）
        if "KL_combined_input_ids" in batch:
            # DEBUG: Print KL batch info
            if self.accelerator.is_main_process:
                print(f"[DEBUG KTO] KL_combined_input_ids found in batch, shape: {batch['KL_combined_input_ids'].shape}")
                print(f"[DEBUG KTO] KL_combined_attention_mask shape: {batch['KL_combined_attention_mask'].shape}")
                print(f"[DEBUG KTO] KL_labels shape: {batch['KL_labels'].shape}")
            
            kl_kwargs = {"labels": batch["KL_labels"]} if self.is_encoder_decoder else {}
            # kl_logits = model(
            #     batch["KL_combined_input_ids"],
            #     attention_mask=batch["KL_combined_attention_mask"],
            #     use_cache=False,
            #     **kl_kwargs,
            # ).logits
            kl_outputs = model(
                batch["KL_combined_input_ids"],
                attention_mask=batch["KL_combined_attention_mask"],
                use_cache=False,
                output_hidden_states=True,  # ★ 同样加上
                **kl_kwargs,
            )
            kl_logits = kl_outputs.logits

            kl_logps = self.get_batch_logps(
                kl_logits,
                batch["KL_labels"],
                average_log_prob=True,   # 与 DIL 对齐：先平均
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
            )
            
            # DEBUG: Print KL logps after computation
            if self.accelerator.is_main_process:
                print(f"[DEBUG KTO] KL logps computed, shape: {kl_logps.shape}, mean: {kl_logps.mean().item():.6f}")
        elif "KL_input_ids" in batch:  # 兼容旧字段
            # DEBUG: Print old KL field usage
            if self.accelerator.is_main_process:
                print(f"[DEBUG KTO] Using legacy KL_input_ids field, shape: {batch['KL_input_ids'].shape}")
            
            kl_kwargs = {"labels": batch["KL_labels"]} if self.is_encoder_decoder else {}
            # kl_logits = model(
            #     batch["KL_input_ids"],
            #     attention_mask=batch["KL_attention_mask"],
            #     use_cache=False,
            #     **kl_kwargs,
            # ).logits
            kl_outputs = model(
                batch["KL_input_ids"],
                attention_mask=batch["KL_attention_mask"],
                use_cache=False,
                output_hidden_states=True,  # ★ 同样加上
                **kl_kwargs,
            )
            kl_logits = kl_outputs.logits

            kl_logps = self.get_batch_logps(
                kl_logits,
                batch["KL_labels"],
                average_log_prob=True,
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
            )
            
            # DEBUG: Print KL logps
            if self.accelerator.is_main_process:
                print(f"[DEBUG KTO] KL logps (legacy) computed, shape: {kl_logps.shape}, mean: {kl_logps.mean().item():.6f}")
        else:
            # DEBUG: Warn about missing KL samples
            if self.accelerator.is_main_process:
                print(f"[DEBUG KTO] WARNING: No KL samples in batch! Neither KL_combined_input_ids nor KL_input_ids found")
                print(f"[DEBUG KTO] Batch keys: {list(batch.keys())}")
            
            kl_logps = torch.tensor([]).to(all_logps.device)
            kl_logits = torch.tensor([]).to(all_logits.device)

        return (chosen_logps, rejected_logps, kl_logps, chosen_logits, rejected_logits, kl_logits)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_KL_logits,
        ) = self.concatenated_forward_with_kl(model, batch)

        # Get reference model outputs
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        reference_KL_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward_with_kl(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_KL_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward_with_kl(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards, KL, chosen_unclamped, rejected_unclamped = self.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_KL_logps,
        )

        # Calculate reward accuracies
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}rewards/KL"] = KL.cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        
        # DEBUG: Print final metrics
        if self.accelerator.is_main_process and train_eval == "train":
            print(f"[DEBUG KTO] Final metrics - KL: {KL.item():.6f}, "
                  f"chosen_reward: {chosen_rewards.mean().item():.6f}, "
                  f"rejected_reward: {rejected_rewards.mean().item():.6f}, "
                  f"margin: {(chosen_rewards - rejected_rewards).mean().item():.6f}, "
                  f"loss: {losses.mean().item():.6f}")

        return losses.mean(), metrics