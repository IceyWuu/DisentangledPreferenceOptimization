# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Literal, Optional

from transformers import TrainingArguments


class FDivergenceType(Enum):
    REVERSE_KL = "reverse_kl"
    JS_DIVERGENCE = "js_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"


class FDivergenceConstants:
    ALPHA_DIVERGENCE_COEF_KEY = "alpha_divergence_coef"
    ALPHA_DIVERGENCE_COEF_DEFAULT = 1.0

@dataclass
class DPOConfig(TrainingArguments):
    r"""
    Initialize DPOConfig.

    Args:
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper.
        label_smoothing (`float`, defaults to 0):
            The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report and [Robust DPO](https://arxiv.org/abs/2403.00409) paper that should be between 0 and 0.5.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either `"sigmoid"` the default DPO loss,`"hinge"` loss from [SLiC](https://arxiv.org/abs/2305.10425) paper, `"ipo"` from [IPO](https://arxiv.org/abs/2310.12036) paper,
            `"bco_pair"` from [BCO](https://arxiv.org/abs/2404.04656) paper or `"robust"` from [Robust DPO](https://arxiv.org/abs/2403.00409) paper,
            "aot" and "aot_pair" from alignment_handbook via optimal transport
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`Optional[int]`, *optional*):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Flag to precompute reference model log probabilities for training and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        ref_model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string
        model_adapter_name (`str`, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        reference_free (`bool` defaults to `False`):
            If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.
        force_use_ref_model (`bool`, defaults to `False`):
            In case one passes a PEFT model for the active model and you want to use a different model for the ref_model, set this flag to `True`.
        f_divergence_type (`FDivergenceType`, *optional*, defaults to `FDivergenceType.REVERSE_KL`):
            The type of f-divergence regularization function to compute divergence between policy and reference model. This argument is optional, defaults to `FDivergenceType.REVERSE_KL`.
        f_alpha_divergence_coef (`float`, *optional*, defaults to `1.0`):
            The alpha coef in alpha-divergence(u^-alpha) regularization function for DPO loss.
        sync_ref_model ('bool', defaults to `False`):
            The flag for syncing reference model during training from the [TR-DPO](https://arxiv.org/pdf/2404.09656) paper.
        ref_model_mixup_alpha ('float', defaults to 1.0):
            The alpha parameter from the [TR-DPO](https://arxiv.org/pdf/2404.09656) paper.
        ref_model_sync_steps ('int', defaults to 2):
            The tau parameter from the [TR-DPO](https://arxiv.org/pdf/2404.09656) paper.
        rpo_alpha ('float', defaults to `None`):
            The alpha parameter from the [RPO](https://arxiv.org/pdf/2404.19733) paper. If None, no weighting is applied and the loss is the same as the DPO loss.
        desirable_weight ('float', defaults to 1.0):
            The weight for desirable examples in KTO loss.
        undesirable_weight ('float', defaults to 1.0):
            The weight for undesirable examples in KTO loss.
    """

    beta: float = 0.1
    label_smoothing: float = 0
    loss_type: Literal[
        "sigmoid",
        "hinge",
        "ipo",
        "IPO",
        "CPO",
        "SimPO",
        "SIMPO",
        "SLIC",
        "UKL",
        "bco_pair",
        "sppo_hard",
        "nca_pair",
        "robust",
        "aot",
        "aot_pair",
        "exo_pair",
        "KTO",
    ] = "sigmoid"
    # ... existing parameters ...
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    label_pad_token_id: int = -100
    padding_value: Optional[int] = None
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_target_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    f_divergence_type: Optional[FDivergenceType] = FDivergenceType.REVERSE_KL
    f_alpha_divergence_coef: Optional[float] = 1.0
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    gamma_beta_ratio: float = 0.25
    
    rpo_alpha: Optional[float] = None
    disable_tqdm: bool = field(default=False, metadata={"help": "Force enable tqdm progress bar."})    # use tqdm
    # ---- Margin-chain logging (zw/zl -> m -> Δm + head-only gradient geometry) ----
    track_margin_chain: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, log the full margin-chain each optimizer step: zw=logpi(yw|x), "
                "zl=logpi(yl|x), m=zw-zl, Δm, and head-only gradients geometry plus analytic incentives."
            )
        },
    )
    margin_chain_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSONL path for margin-chain logs. Defaults to <output_dir>/margin_chain.jsonl."},
    )
    margin_chain_lora_all_layers: bool = field(
        default=False,
        metadata={
            "help": (
                "Only relevant when lm_head has no trainable params (e.g., LoRA/QLoRA). "
                "If true, compute the margin-chain head-gradient geometry on ALL trainable LoRA parameters "
                "(across all layers) instead of a small last-layer subset. "
                "NOTE: this can noticeably increase compute and memory usage; keep disabled unless needed."
            )
        },
    )
    ipo_tau: float = field(
        default=0.1,
        metadata={"help": "IPO tau (target margin is 1/(2*tau)); used when loss_type == 'IPO'."},
    )
    cpo_tau: float = field(
        default=0.1,
        metadata={"help": "CPO tau used for logging incentives dw/dl = 2*(1/(2*tau) - m), where m is policy margin zw-zl."},
    )
    cpo_beta: float = field(
        default=0.1,
        metadata={"help": "CPO preference beta used in preference_loss = -log(sigmoid(beta*(zw-zl)))."},
    )
    cpo_lambda_nll: float = field(
        default=1.0,
        metadata={"help": "CPO NLL regularization coefficient for nll_loss = -zw; total = preference_loss + lambda_nll*nll_loss."},
    )
    slic_gamma: float = field(
        default=1.0,
        metadata={"help": "SLiC margin threshold gamma in max(gamma - zw + zl, 0). Also used in logging incentives."},
    )
    slic_lambda_coef: float = field(
        default=0.1,
        metadata={"help": "SLiC NLL regularization coefficient (lambda) for reg_loss = -zw; total = cal_loss + lambda*reg_loss. Also used in logging incentives."},
    )

    # ---- DB Calibration (optional; no regularizer term; rescales backward gradients) ----
    db_calibration_enable: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, enable Disentanglement Band (DB) calibration: replace z_w/z_l with z^{rc} using sg(x)=x.detach() "
                "so forward values stay identical but backward gradients are rescaled to target ratio r* = ||s_l||/||s_w||."
            )
        },
    )
    db_calibration_eps: float = field(
        default=1e-12,
        metadata={"help": "Numerical epsilon used for DB calibration (ratios/logs/divisions)."},
    )
    db_ema_beta: float = field(
        default=0.98,
        metadata={
            "help": (
                "EMA decay β for DB calibration statistics (log dw/dl and log sw/sl). "
                "Higher β smooths updates more strongly."
            )
        },
    )

    def __post_init__(self):
        if self.loss_type == "kto_pair":
            raise ValueError("Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.")
        return super().__post_init__()