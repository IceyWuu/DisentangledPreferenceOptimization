import inspect
import json
import os
import random
import warnings
from collections import defaultdict, deque
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
E_MINUS_ONE = torch.tensor(math.e - 1, dtype=torch.float32, device='cuda')
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


class SyncRefModelCallback(TrainerCallback):
    """
    Minimal SyncRefModelCallback.

    When `args.sync_ref_model=True`, periodically sync the reference model towards the policy model:
      ref = alpha * ref + (1-alpha) * policy
    controlled by:
      - args.ref_model_sync_steps
      - args.ref_model_mixup_alpha
    """

    def __init__(self, ref_model: nn.Module, accelerator):
        self.ref_model = ref_model
        self.accelerator = accelerator

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None or self.ref_model is None:
            return control

        sync_steps = int(getattr(args, "ref_model_sync_steps", 0) or 0)
        if sync_steps <= 0:
            return control

        step = int(getattr(state, "global_step", 0))
        if step <= 0 or (step % sync_steps) != 0:
            return control

        alpha = float(getattr(args, "ref_model_mixup_alpha", 0.9))
        with torch.no_grad():
            for p_ref, p_pol in zip(self.ref_model.parameters(), model.parameters()):
                p_ref.data.mul_(alpha).add_(p_pol.data, alpha=(1.0 - alpha))
        return control


class DPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`DPOConfig`):
            The DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "dpo"]

    @_deprecate_arguments(
        version="1.0.0",
        deprecated_args=[
            "beta",
            "label_smoothing",
            "loss_type",
            "label_pad_token_id",
            "padding_value",
            "truncation_mode",
            "max_length",
            "max_prompt_length",
            "max_target_length",
            "is_encoder_decoder",
            "disable_dropout",
            "generate_during_eval",
            "precompute_ref_log_probs",
            "dataset_num_proc",
            "model_init_kwargs",
            "ref_model_init_kwargs",
            "model_adapter_name",
            "ref_adapter_name",
            "reference_free",
            "force_use_ref_model",
        ],
        custom_message="Deprecated positional argument(s) used in DPOTrainer, please use the DPOConfig to set these arguments instead.",
    )
    def __init__(
            self,
            model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            beta: float = 0.1,
            label_smoothing: float = 0,
            loss_type: str = "sigmoid",
            args: Optional[DPOConfig] = None,
            data_collator: Optional[DataCollator] = None,
            label_pad_token_id: int = -100,
            padding_value: Optional[int] = None,
            truncation_mode: str = "keep_end",
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            max_length: Optional[int] = None,
            max_prompt_length: Optional[int] = None,
            max_target_length: Optional[int] = None,
            peft_config: Optional[Dict] = None,
            is_encoder_decoder: Optional[bool] = None,
            disable_dropout: bool = True,
            generate_during_eval: bool = False,
            compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
            precompute_ref_log_probs: bool = False,
            dataset_num_proc: Optional[int] = None,
            model_init_kwargs: Optional[Dict] = None,
            ref_model_init_kwargs: Optional[Dict] = None,
            model_adapter_name: Optional[str] = None,
            ref_adapter_name: Optional[str] = None,
            reference_free: bool = False,
            force_use_ref_model: bool = False,
    ):
        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.model_init_kwargs = model_init_kwargs

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the DPOTrainer/DPOConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs

        if model_init_kwargs is not None and "torch_dtype" in model_init_kwargs:
            torch_dtype = model_init_kwargs["torch_dtype"]
        else:
            # if loaded model, get dtype from model.config or model.dtype
            torch_dtype = getattr(model, "dtype", torch.float16)
        if torch_dtype is not None:
            # Convert to `torch.dtype` if an str is passed
            if isinstance(torch_dtype, str) and torch_dtype != "auto":
                torch_dtype = getattr(torch, torch_dtype)

            if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                raise ValueError(
                    f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                )

        model_init_kwargs["torch_dtype"] = torch_dtype

        if ref_model_init_kwargs is not None:
            warnings.warn(
                "You passed `ref_model_kwargs` to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.ref_model_init_kwargs = ref_model_init_kwargs

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the DPOTrainer/DPOConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs

        if ref_model_init_kwargs is not None:
            torch_dtype = ref_model_init_kwargs.get("torch_dtype", torch.float16) # Use .get() for safety
        else:
            # Get dtype from the main model_kwargs or use a default
            # Assuming model_kwargs is accessible here, or you might have model_args
            torch_dtype = model_init_kwargs.get("torch_dtype", torch.float16) if model_init_kwargs is not None else torch.float16
        if torch_dtype is not None:
            # Convert to `torch.dtype` if an str is passed
            if isinstance(torch_dtype, str) and torch_dtype != "auto":
                torch_dtype = getattr(torch, torch_dtype)

            if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                raise ValueError(
                    f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                )

        ref_model_init_kwargs["torch_dtype"] = torch_dtype
        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if force_use_ref_model:
            warnings.warn(
                "You passed `force_use_ref_model` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.force_use_ref_model = force_use_ref_model

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval:
            warnings.warn(
                "You passed `generate_during_eval` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.generate_during_eval = generate_during_eval
        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if is_encoder_decoder is not None:
            warnings.warn(
                "You passed `is_encoder_decoder` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.is_encoder_decoder = is_encoder_decoder
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder to the DPOTrainer/DPOConfig."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if model_adapter_name is not None:
            warnings.warn(
                "You passed `model_adapter_name` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_adapter_name = model_adapter_name
        self.model_adapter_name = args.model_adapter_name

        if ref_adapter_name is not None:
            warnings.warn(
                "You passed `ref_adapter_name` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.ref_adapter_name = ref_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if reference_free:
            warnings.warn(
                "You passed `reference_free` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.reference_free = reference_free
        self.reference_free = args.reference_free

        if precompute_ref_log_probs:
            warnings.warn(
                "You passed `precompute_ref_log_probs` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.precompute_ref_log_probs = precompute_ref_log_probs

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")

        if max_length is not None:
            warnings.warn(
                "You passed `max_length` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_length = max_length
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_length = 512

        if max_prompt_length is not None:
            warnings.warn(
                "You passed `max_prompt_length` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_prompt_length = max_prompt_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_prompt_length = 128

        if max_target_length is not None:
            warnings.warn(
                "You passed `max_target_length` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_target_length = max_target_length
        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_target_length = 128

        if label_pad_token_id != -100:
            warnings.warn(
                "You passed `label_pad_token_id` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_pad_token_id = label_pad_token_id
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if not disable_dropout:
            warnings.warn(
                "You passed `disable_dropout` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.disable_dropout = disable_dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        if padding_value is not None:
            warnings.warn(
                "You passed `padding_value` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.padding_value = padding_value
        self.padding_value = args.padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        if truncation_mode != "keep_end":
            warnings.warn(
                "You passed `truncation_mode` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.truncation_mode = truncation_mode
        self.truncation_mode = args.truncation_mode
        self.max_target_length = args.max_target_length
        self.processing_class = tokenizer
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type != "sigmoid":
            warnings.warn(
                "You passed `loss_type` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.loss_type = loss_type
        if label_smoothing != 0:
            warnings.warn(
                "You passed `label_smoothing` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_smoothing = label_smoothing
        if args.loss_type in ["hinge", "ipo", "kto_pair", "bco_pair", "SimPO", "SIMPO", "simpo"] and args.label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        if beta != 0.1:
            warnings.warn(
                "You passed `beta` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.beta = beta
        self.beta = args.beta
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Margin-chain logger: zw/zl -> m -> Δm + head-only gradient geometry + analytic incentives
        self.track_margin_chain = getattr(args, "track_margin_chain", False)
        self.margin_chain_path = getattr(args, "margin_chain_path", None)

        # Optional: DB Calibration (no regularizer; rescale backward grads via stop-gradient trick)
        self.db_calibration_enable = bool(getattr(args, "db_calibration_enable", False))
        self.db_calibration_eps = float(getattr(args, "db_calibration_eps", 1e-12))
        self.db_ema_beta = float(getattr(args, "db_ema_beta", 0.98))
        self._db_calib_warned_no_params = False

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed `dataset_num_proc` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset (must not reuse stale HF datasets cache from older tokenize_row)
            train_dataset = train_dataset.map(
                self.tokenize_row,
                num_proc=self.dataset_num_proc,
                load_from_cache_file=False,
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self.tokenize_row,
                    num_proc=self.dataset_num_proc,
                    load_from_cache_file=False,
                )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Compatibility: some transformers builds expect _memory_tracker but may not initialize it.
        if not hasattr(self, "_memory_tracker"):
            class _NoOpMemoryTracker:
                def start(self):  # noqa: D401
                    return None
                def stop_and_update_metrics(self, *args, **kwargs):
                    return None
                def stop(self, *args, **kwargs):
                    return None
            self._memory_tracker = _NoOpMemoryTracker()

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        if self.loss_type == "bco_pair":
            self.running = RunningMoments(self.accelerator)

        self._setup_margin_chain_tracking()

        # self._db_ema_log_r = None
        # self._db_ema_log_r_star = None
        # self._db_ema_log_gm = None
        # self._db_log_gm_ref = None
        self._db_ema_log_dw = None
        self._db_ema_log_dl = None
        self._db_ema_log_sw = None
        self._db_ema_log_sl = None
        self._db_ema_dot = None
        # self._db_ema_log_rho = None

    def _setup_margin_chain_tracking(self):
        self._margin_chain_head_param = None
        self._margin_chain_head_params = None  # list[torch.nn.Parameter] for grad-geometry (1 param == legacy behavior)
        self._margin_chain_head_param_source = None  # "lm_head" | "lora_last_layer" | "lora_all" (best-effort)
        self._margin_chain_accum_counter = 0
        # Previous-step values for step-level deltas (optimizer-step deltas)
        self._margin_chain_prev_m = None          # stores previous tilde_m
        self._margin_chain_prev_m_policy = None   # stores previous m (policy margin)
        self._margin_chain_prev_m_vec = None
        # buffer per-sample scalars across grad accumulation so one record == one optimizer step (effective batch)
        # Only used on main process (since only main writes margin_chain.jsonl).
        self._mc_buf = None
        # Accumulate head-only gradients across gradient accumulation so the reported geometry
        # matches the *effective batch* (not the last micro-batch).
        # We accumulate weighted sums to represent grad of mean logp over all samples in the window:
        #   g_mean = (1/N) * Σ_j (b_j * g_mean_j)
        # where g_mean_j is grad(mean(logp)) on micro-batch j with b_j samples.
        # For multi-parameter geometry (e.g., LoRA params), we keep a list aligned to self._margin_chain_head_params.
        self._mc_gw_sum = None  # torch.Tensor (legacy single-param) OR list[Optional[torch.Tensor]]
        self._mc_gl_sum = None  # torch.Tensor (legacy single-param) OR list[Optional[torch.Tensor]]
        self._mc_n_sum = 0
        # Per-micro-batch margin deltas within an accumulation window (so users can inspect each micro-batch).
        # Reset every optimizer step.
        self._mc_mb_m_list = None         # list[float]
        self._mc_mb_delta_m_list = None   # list[Optional[float]]
        self._mc_mb_prev_m = None         # Optional[float]

        # We resolve the same head/LoRA parameter subset for:
        # - margin_chain logging (if enabled)
        # - DB calibration (if enabled)
        if not (self.track_margin_chain or self.db_calibration_enable):
            return

        def _resolve_lora_params_for_margin_chain() -> Tuple[Optional[List[torch.nn.Parameter]], Optional[str]]:
            """
            Best-effort fallback for PEFT/LoRA training when lm_head has no trainable params.
            Default behavior picks a *small* subset: LoRA params in the last transformer block (highest layer index found).
            If args.margin_chain_lora_all_layers is True, we use ALL trainable LoRA parameters.

            Returns:
              (params, source_label)
            """
            try:
                named = list(self.model.named_parameters())
            except Exception:
                return None, None

            # Only consider trainable LoRA params.
            lora_named = [(n, p) for (n, p) in named if getattr(p, "requires_grad", False) and ("lora_" in n)]
            if not lora_named:
                return None, None

            # Optional: use all LoRA params (can be significantly larger).
            if bool(getattr(self.args, "margin_chain_lora_all_layers", False)):
                # Deduplicate while preserving order (defensive; named_parameters() should already be unique).
                params = []
                seen = set()
                for _n, p in lora_named:
                    pid = id(p)
                    if pid in seen:
                        continue
                    seen.add(pid)
                    params.append(p)
                return (params, "lora_all") if params else (None, None)

            import re

            idxs = []
            for n, _p in lora_named:
                m = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)", n)
                if m:
                    try:
                        idxs.append(int(m.group(1)))
                    except Exception:
                        pass

            if idxs:
                last_idx = max(idxs)
                # Restrict to last block AND to common projection modules for Mistral-like models.
                # This keeps the subset small and stable.
                proj_keys = (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                )
                params = []
                for n, p in lora_named:
                    if f".{last_idx}." not in n and f".layers.{last_idx}." not in n and f".h.{last_idx}." not in n:
                        continue
                    if any(k in n for k in proj_keys):
                        params.append(p)

                # Fallback: if the proj filter was too strict, still use all LoRA params from last layer.
                if not params:
                    params = [p for (n, p) in lora_named if f".{last_idx}." in n]

                if params:
                    return params, "lora_last_layer"
                return None, None

            # If we couldn't infer layer indices, fall back to a bounded subset (first few LoRA params)
            # to avoid large overhead. This should still allow margin_chain to run.
            params = [p for (_n, p) in lora_named[:8]]
            return (params, "lora_subset") if params else (None, None)

        try:
            output_layer = self.model.get_output_embeddings()
        except Exception:
            output_layer = None

        head_param = getattr(output_layer, "weight", None)
        if head_param is None:
            for param in output_layer.parameters():
                if param.requires_grad:
                    head_param = param
                    break

        if head_param is not None and getattr(head_param, "requires_grad", False):
            # Legacy/default: head-only (lm_head) geometry.
            self._margin_chain_head_param = head_param
            self._margin_chain_head_params = [head_param]
            self._margin_chain_head_param_source = "lm_head"
        else:
            # Fallback for PEFT/QLoRA: compute the same geometry on a small LoRA parameter subset.
            params, source = _resolve_lora_params_for_margin_chain()
            self._margin_chain_head_param = None
            self._margin_chain_head_params = params
            self._margin_chain_head_param_source = source

        # Back-compat: downstream code checks _margin_chain_head_param; treat non-empty list as enabled.
        if self._margin_chain_head_params is None or len(self._margin_chain_head_params) == 0:
            return

        # Only create/touch the JSONL file when margin-chain logging is enabled.
        if self.track_margin_chain:
            default_path = os.path.join(self.args.output_dir, "margin_chain.jsonl")
            self._margin_chain_path = self.margin_chain_path or default_path
            dir_path = os.path.dirname(self._margin_chain_path) or "."
            os.makedirs(dir_path, exist_ok=True)

            if self.accelerator.is_main_process:
                # Touch file so downstream tooling can tail it during training.
                with open(self._margin_chain_path, "a", encoding="utf-8"):
                    pass
        else:
            self._margin_chain_path = None

    def _maybe_apply_db_calibration(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Disentanglement Band (DB) calibration:
        Replace z_w/z_l by z^{rc} so forward is unchanged but backward gradients are rescaled:
          z_w^{rc} = α z_w + (1-α) sg(z_w)
          z_l^{rc} = α^{-1} z_l + (1-α^{-1}) sg(z_l)
        where sg(x)=x.detach(), and
          r_t = d_w/d_l (DPO: r_t=1)
          r*_t = ||s_l|| / ||s_w||  (s are parameter-space gradient norms of mean z_w/z_l)
          α = exp(0.5*(log r*_t - log r_t)).
        """
        if not self.db_calibration_enable:
            return policy_chosen_logps, policy_rejected_logps

        self._db_mb_cache = None
        
        # Only apply during training with gradients enabled.
        if not torch.is_grad_enabled():
            return policy_chosen_logps, policy_rejected_logps
        if getattr(self, "model", None) is not None and (not self.model.training):
            return policy_chosen_logps, policy_rejected_logps
        if not getattr(policy_chosen_logps, "requires_grad", False):
            return policy_chosen_logps, policy_rejected_logps
        if not getattr(policy_rejected_logps, "requires_grad", False):
            return policy_chosen_logps, policy_rejected_logps

        params = list(getattr(self, "_margin_chain_head_params", []) or [])
        if not params:
            if not self._db_calib_warned_no_params:
                warnings.warn(
                    "db_calibration_enable is True but no trainable head/LoRA params were resolved; skipping DB calibration."
                )
                self._db_calib_warned_no_params = True
            return policy_chosen_logps, policy_rejected_logps

        eps = float(self.db_calibration_eps)
        device = self.accelerator.device

        # --- 1) realized ratio r_t = dw/dl (scalar); DPO -> 1 ---
        loss_type = str(getattr(self, "loss_type", "")).upper()
        r_w = (policy_chosen_logps - reference_chosen_logps).to(device)
        r_l = (policy_rejected_logps - reference_rejected_logps).to(device)

        if loss_type in ("DPO", "TIDPO"):
            beta = float(getattr(self.args, "beta", 0.1))
            tilde_m = (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
            tilde_m = tilde_m.to(device)
            dw_vec = beta * torch.sigmoid(- beta * tilde_m)
            dl_vec = beta * torch.sigmoid(- beta * tilde_m)
        else:
            dw_vec = None
            dl_vec = None
            if loss_type == "SIMPO":
                # SimPO analytic incentives (per-sample):
                #   d_w = (β/|y_w|) σ(1 - β m_norm)
                #   d_l = (β/|y_l|) σ(1 - β m_norm)
                # where m_norm uses the (already length-normalized) logps zw/zl.
                beta = float(getattr(self.args, "beta", getattr(self, "beta", 0.1)))
                m_policy = (policy_chosen_logps - policy_rejected_logps).to(device)
                d = beta * torch.sigmoid(1.0 - beta * m_policy)

                all_lengths = getattr(self, "all_lengths", None)
                b = int(policy_chosen_logps.shape[0])
                if isinstance(all_lengths, torch.Tensor) and all_lengths.numel() >= 2 * b:
                    chosen_lengths = all_lengths[:b].to(device).clamp(min=1)
                    rejected_lengths = all_lengths[b : b + b].to(device).clamp(min=1)
                else:
                    chosen_lengths = torch.ones_like(d)
                    rejected_lengths = torch.ones_like(d)

                dw_vec = d / chosen_lengths
                dl_vec = d / rejected_lengths
            elif loss_type == "LSIF":
                dw_vec = torch.exp(r_w)
                dl_vec = torch.exp(2.0 * r_l)
            elif loss_type == "BCE":
                dw_vec = torch.sigmoid(-r_w)
                dl_vec = torch.sigmoid(r_l)
            elif loss_type == "IPO":
                tau = float(getattr(self.args, "ipo_tau", 0.1))
                target = 1.0 / (2.0 * tau)
                tilde_m = (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
                tilde_m = tilde_m.to(device)
                dw_vec = 2.0 * (target - tilde_m)
                dl_vec = dw_vec
            elif loss_type == "CPO":
                beta = float(getattr(self.args, "cpo_beta", 0.1))
                lambda_nll = float(getattr(self.args, "cpo_lambda_nll", 1.0))
                m_policy = (policy_chosen_logps - policy_rejected_logps).to(device)
                base = beta * torch.sigmoid(-beta * m_policy)
                dl_vec = base
                dw_vec = base + lambda_nll
            elif loss_type == "SLIC":
                gamma = float(getattr(self.args, "slic_gamma", 1.0))
                lambda_coef = float(getattr(self.args, "slic_lambda_coef", 0.1))
                m_policy = (policy_chosen_logps - policy_rejected_logps).to(device)
                # tau = float(getattr(self.args, "slic_tau", 0.5))
                # ind = torch.sigmoid((gamma - m_policy) / tau)
                ind = (m_policy < gamma).to(m_policy.dtype)
                dl_vec = ind
                dw_vec = ind + lambda_coef
            elif loss_type == "UKL":
                dl_vec = torch.exp(r_l)
                dw_vec = torch.ones_like(dl_vec)
            elif loss_type == "DDRO":
                r_rejected = torch.exp(r_l)
                r_rejected = torch.clamp(r_rejected, max=2.0 - 1e-6)
                dl_vec = r_rejected / (2.0 - r_rejected)
                dl_vec = torch.clamp(dl_vec, max=20.0)
                dw_vec = torch.ones_like(dl_vec)

            if dw_vec is None or dl_vec is None:
                return policy_chosen_logps, policy_rejected_logps

        beta_inc = float(getattr(self, "db_ema_beta", 0.98))
        
        # --- Log-Domain EMA for Incentives ---
        dw_inst = dw_vec.mean().detach()
        dl_inst = dl_vec.mean().detach()  # 
        log_dw_inst = torch.log(dw_inst.clamp(min=eps))#.clamp(max=5.0)
        log_dl_inst = torch.log(dl_inst.clamp(min=eps))#.clamp(max=5.0)
        if getattr(self, "_db_ema_log_dw", None) is None:
            self._db_ema_log_dw = log_dw_inst
            self._db_ema_log_dl = log_dl_inst
        else:
            self._db_ema_log_dw = beta_inc * self._db_ema_log_dw + (1.0 - beta_inc) * log_dw_inst
            self._db_ema_log_dl = beta_inc * self._db_ema_log_dl + (1.0 - beta_inc) * log_dl_inst
        log_r_ema = self._db_ema_log_dw - self._db_ema_log_dl

        # --- 2) target ratio r*_t = ||s_l||/||s_w|| using head/LoRA param gradients ---
        zw_graph = policy_chosen_logps.float().mean()
        zl_graph = policy_rejected_logps.float().mean()
        max_gn = float(getattr(self.args, "max_grad_norm", 0.0) or 0.0)
        gw_list = torch.autograd.grad(zw_graph, params, retain_graph=True, create_graph=False, allow_unused=True)
        gl_list = torch.autograd.grad(zl_graph, params, retain_graph=True, create_graph=False, allow_unused=True)
        # Joint clip: one scale factor on all gw/gl tensors keeps ||g_l||/||g_w|| unchanged
        # (separate clips can distort the ratio when only one side exceeds max_gn).
        if max_gn > 0:
            g_joint = [g for g in gw_list if g is not None] + [g for g in gl_list if g is not None]
            if g_joint:
                torch.nn.utils.clip_grad_norm_(g_joint, 1.414 * max_gn)
        # Per-param reductions only (no torch.cat over full head grads) to avoid extra VRAM spikes.
        sw_parts, sl_parts, dot_parts = [], [], []
        z0 = torch.zeros((), device=device, dtype=torch.float32)
        for gw_i, gl_i in zip(gw_list, gl_list):
            if gw_i is not None:
                gwf = gw_i.detach().float()
                sw_parts.append(gwf.pow(2).sum())
            if gl_i is not None:
                glf = gl_i.detach().float()
                sl_parts.append(glf.pow(2).sum())
            if (gw_i is not None) and (gl_i is not None):
                dot_parts.append((gwf * glf).sum())
        sw_sq = torch.stack(sw_parts).sum() if sw_parts else z0
        sl_sq = torch.stack(sl_parts).sum() if sl_parts else z0
        dot = torch.stack(dot_parts).sum() if dot_parts else z0

        self._db_mb_cache = {
            "gw_list": gw_list, # Raw gradients (for accumulation)
            "gl_list": gl_list,
            "sw_sq": sw_sq,     # Scalars (for logging)
            "sl_sq": sl_sq,
            "dot": dot,
            "dw_vec": dw_vec.detach(),
            "dl_vec": dl_vec.detach(),
        }

        # --- Log-Domain EMA for scores ---
        s_w = torch.sqrt(sw_sq)
        s_l = torch.sqrt(sl_sq)
        log_sw_inst = torch.log(s_w.clamp(min=eps))
        log_sl_inst = torch.log(s_l.clamp(min=eps))

        if getattr(self, "_db_ema_log_sw", None) is None:
            self._db_ema_log_sw = log_sw_inst
            self._db_ema_log_sl = log_sl_inst
            self._db_ema_dot = dot.detach()
        else:
            self._db_ema_log_sw = beta_inc * self._db_ema_log_sw + (1.0 - beta_inc) * log_sw_inst
            self._db_ema_log_sl = beta_inc * self._db_ema_log_sl + (1.0 - beta_inc) * log_sl_inst
            self._db_ema_dot = beta_inc * self._db_ema_dot + (1.0 - beta_inc) * dot.detach()
        log_r_star_ema = self._db_ema_log_sl - self._db_ema_log_sw

        # --- 3) Calculate Instantaneous Bounds for Clipping ---
        # Use instantaneous rho and r to define safe bounds, preventing EMA overshoot.
        rho_inst = (dot / (s_w * s_l).clamp(min=eps))
        log_rho_inst = torch.log(rho_inst.clamp(min=eps)).detach()
        log_r_inst = (log_dw_inst - log_dl_inst).detach()

        # --- 4) Delta Calculation (Targeting Center) ---
        log_delta = (log_r_star_ema - log_r_ema).detach()

        # Ensure the FINAL adjusted ratio lands inside [log_r_star - log_rho, log_r_star + log_rho]
        # Upper bound: log(r_rc) <= log(r_star) - log(rho)
        # Lower bound: log(r_rc) >= log(r_star) + log(rho)
        # log_delta_max = (log_r_star_ema - log_rho_inst - log_r_inst).detach()
        # log_delta_min = (log_r_star_ema + log_rho_inst - log_r_inst).detach()
        # log_delta = torch.clamp(log_delta, min=log_delta_min, max=log_delta_max)

        delta = torch.exp(0.5 * log_delta).detach()
        delta_inv = torch.exp(-0.5 * log_delta).detach()

        # --- 4) rescaled calibration via stop-gradient (forward unchanged) ---
        z_w = policy_chosen_logps
        z_l = policy_rejected_logps
        z_w_rc = delta * z_w + (1.0 - delta) * z_w.detach()
        z_l_rc = delta_inv * z_l + (1.0 - delta_inv) * z_l.detach()

        self._db_last = {
            "dw_ema": float(torch.exp(self._db_ema_log_dw).item()),
            "dl_ema": float(torch.exp(self._db_ema_log_dl).item()),
            "log_r_ema": float(log_r_ema.item()),
            "log_r_star_ema": float(log_r_star_ema.item()),
            "alpha": float(delta.item()),
            "alpha_inv": float(delta_inv.item()),
            "delta": float(log_delta.item()),
        }
        return z_w_rc, z_l_rc

    def _maybe_record_margin_chain(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> None:
        """
        Record per-(optimizer)step scalar logs:
          zw, zl, m=zw-zl, Δm (per optimizer step),
          head-only grad geometry for zw/zl (||sw||, ||sl||, <sw,sl>, cos),
          analytic incentives dw,dl (DPO symmetric, BCE asymmetric),
          optional lr and step index.
        """
        if not self.track_margin_chain:
            return
        if not self._margin_chain_head_params:
            return

        # Compute graph-connected scalars (used only for head-only gradient geometry).
        # NOTE: these are micro-batch scalars; logging below will use effective-batch scalars.
        zw_graph = policy_chosen_logps.float().mean()
        zl_graph = policy_rejected_logps.float().mean()

        loss_type = str(getattr(self, "loss_type", "")).upper()

        # --- Head-only gradient geometry aligned to effective batch ---
        # Accumulate g_w and g_l across micro-batches in the accumulation window.
        # Only main process computes/stores these to avoid multi-process overhead; values are per-device.
        dw_vec = None
        dl_vec = None
        if self.accelerator.is_main_process:
            params = list(self._margin_chain_head_params)
            b = int(policy_chosen_logps.shape[0])  # micro-batch sample count on this device
            if b > 0:
                cached_data = getattr(self, "_db_mb_cache", None)
                
                gw_list = None
                gl_list = None
                sw_sq_mb = None
                sl_sq_mb = None
                dot_mb = None

                if cached_data is not None:
                    # Reuse cached data
                    gw_list = cached_data.get("gw_list")
                    gl_list = cached_data.get("gl_list")
                    sw_sq_mb = cached_data.get("sw_sq")
                    sl_sq_mb = cached_data.get("sl_sq")
                    dot_mb = cached_data.get("dot")
                    dw_vec = cached_data.get("dw_vec")
                    dl_vec = cached_data.get("dl_vec")
                    
                    # Clear cache to ensure it doesn't persist to next step incorrectly
                    self._db_mb_cache = None
                    
                # If cache miss (e.g. DB calib disabled), compute from scratch
                if gw_list is None or gl_list is None:
                    gw_list = torch.autograd.grad(zw_graph, params, retain_graph=True, create_graph=False, allow_unused=True)
                    gl_list = torch.autograd.grad(zl_graph, params, retain_graph=True, create_graph=False, allow_unused=True)
                
                # If cache miss, compute scalars from scratch
                if sw_sq_mb is None:
                    eps = 1e-12
                    sw_sq_mb = torch.tensor(0.0, device=self.accelerator.device)
                    sl_sq_mb = torch.tensor(0.0, device=self.accelerator.device)
                    dot_mb = torch.tensor(0.0, device=self.accelerator.device)
                    
                    for gw_i, gl_i in zip(gw_list, gl_list):
                        if gw_i is not None:
                            sw_sq_mb += gw_i.detach().float().pow(2).sum()
                        if gl_i is not None:
                            sl_sq_mb += gl_i.detach().float().pow(2).sum()
                        if (gw_i is not None) and (gl_i is not None):
                            dot_mb += (gw_i.detach().float() * gl_i.detach().float()).sum()

                # Accumulate weighted sums for effective batch mean-grad
                if self._mc_gw_sum is None:
                    self._mc_gw_sum = [None for _ in range(len(params))]
                if self._mc_gl_sum is None:
                    self._mc_gl_sum = [None for _ in range(len(params))]

                for i, g in enumerate(gw_list):
                    if g is None: continue
                    # Ensure we detach if it came from cache (though cache usually stores raw grads)
                    g = g.detach().float()
                    if self._mc_gw_sum[i] is None:
                        self._mc_gw_sum[i] = torch.zeros_like(g)
                    self._mc_gw_sum[i].add_(g, alpha=float(b))

                for i, g in enumerate(gl_list):
                    if g is None: continue
                    g = g.detach().float()
                    if self._mc_gl_sum[i] is None:
                        self._mc_gl_sum[i] = torch.zeros_like(g)
                    self._mc_gl_sum[i].add_(g, alpha=float(b))

                self._mc_n_sum += b

            # Per-micro-batch tilde_m (mean over samples in THIS micro-batch) and its within-window delta.
            # This is independent from per-sample vectors and gives you one delta per micro-batch.
            tilde_m_mb = float(
                (
                    (policy_chosen_logps.detach().float() - policy_rejected_logps.detach().float())
                    - (reference_chosen_logps.detach().float() - reference_rejected_logps.detach().float())
                )
                .mean()
                .item()
            )
            if self._mc_mb_m_list is None:
                self._mc_mb_m_list = []
            if self._mc_mb_delta_m_list is None:
                self._mc_mb_delta_m_list = []
            self._mc_mb_m_list.append(tilde_m_mb)
            if self._mc_mb_prev_m is None:
                self._mc_mb_delta_m_list.append(None)
            else:
                self._mc_mb_delta_m_list.append(float(tilde_m_mb - float(self._mc_mb_prev_m)))
            self._mc_mb_prev_m = tilde_m_mb

        # Cache per-sample scalars across accumulation (effective batch).
        # Detach to avoid retaining the graph; only main process needs to buffer/write.
        if self.accelerator.is_main_process:
            if self._mc_buf is None:
                self._mc_buf = {"pc": [], "pr": [], "rc": [], "rr": [], "lc": [], "lr": []}
            self._mc_buf["pc"].append(policy_chosen_logps.detach().float())
            self._mc_buf["pr"].append(policy_rejected_logps.detach().float())
            self._mc_buf["rc"].append(reference_chosen_logps.detach().float())
            self._mc_buf["rr"].append(reference_rejected_logps.detach().float())
            # Also buffer per-sample lengths (= loss_mask.sum(-1)) so losses like SimPO can use β/|y|.
            all_lengths = getattr(self, "all_lengths", None)
            b = int(policy_chosen_logps.shape[0])
            if isinstance(all_lengths, torch.Tensor) and all_lengths.numel() >= 2 * b:
                self._mc_buf["lc"].append(all_lengths[:b].detach().float())
                self._mc_buf["lr"].append(all_lengths[b : b + b].detach().float())
            else:
                ones = torch.ones_like(policy_chosen_logps.detach().float())
                self._mc_buf["lc"].append(ones)
                self._mc_buf["lr"].append(ones)

        # Called once per micro-batch; we only persist once per optimizer step.
        self._margin_chain_accum_counter += 1
        if self._margin_chain_accum_counter % self.args.gradient_accumulation_steps != 0:
            return

        # Log every optimizer step
        step_index = int(self.state.global_step + 1)

        # To avoid wasted compute, only main process does head-only grad geometry and I/O.
        if not self.accelerator.is_main_process:
            return

        # Effective-batch scalars/vectors (detached; do NOT affect gradients).
        buf = self._mc_buf or {"pc": [], "pr": [], "rc": [], "rr": [], "lc": [], "lr": []}
        pc = torch.cat(buf.get("pc", []), dim=0) if buf.get("pc") else torch.empty(0)
        pr = torch.cat(buf.get("pr", []), dim=0) if buf.get("pr") else torch.empty(0)
        rc = torch.cat(buf.get("rc", []), dim=0) if buf.get("rc") else torch.empty(0)
        rr = torch.cat(buf.get("rr", []), dim=0) if buf.get("rr") else torch.empty(0)
        lc = torch.cat(buf.get("lc", []), dim=0) if buf.get("lc") else torch.empty(0)
        lr = torch.cat(buf.get("lr", []), dim=0) if buf.get("lr") else torch.empty(0)

        # Fallback: should not happen, but keep robust if buffer is empty.
        if pc.numel() == 0 or pr.numel() == 0:
            pc = policy_chosen_logps.detach().float()
            pr = policy_rejected_logps.detach().float()
            rc = reference_chosen_logps.detach().float()
            rr = reference_rejected_logps.detach().float()

        zw_eff = float(pc.mean().item())
        zl_eff = float(pr.mean().item())
        # Policy/reference margins and the *DPO/IPO margin* (tilde_m):
        #   tilde_m = (zw - zl) - (zw_ref - zl_ref)
        m_policy_vec = (pc - pr)
        m_ref_vec = (rc - rr)
        tilde_m_vec = (m_policy_vec - m_ref_vec)
        tilde_m_eff = float(tilde_m_vec.mean().item())
        m_policy_eff = float(m_policy_vec.mean().item())
        m_ref_eff = float(m_ref_vec.mean().item())

        # Rewards (log-ratio wrt reference) on effective batch.
        # r_w/r_l are per-sample vectors; rw/rl below are their scalar means.
        r_w = (pc - rc)
        r_l = (pr - rr)
        rw = float(r_w.mean().item())
        rl = float(r_l.mean().item())
        reward_margin = rw - rl

        dw = None
        dl = None
        dw_eff_vec = None
        dl_eff_vec = None
        if dw_vec is None or dl_vec is None:
            if loss_type == "DPO":
                d = self.beta * torch.sigmoid(-self.beta * tilde_m_vec)
                dw_eff_vec = d
                dl_eff_vec = d

            elif loss_type == "TIDPO":
                d = self.beta * torch.sigmoid(-self.beta * tilde_m_vec)
                dw_eff_vec = d
                dl_eff_vec = d

            # SimPO: incentives include extra 1/|y| scaling (use buffered lengths).
            elif loss_type == "SIMPO":
                if lc.numel() == 0:
                    lc = torch.ones_like(m_policy_vec)
                if lr.numel() == 0:
                    lr = torch.ones_like(m_policy_vec)
                lc = lc.to(m_policy_vec.device).clamp(min=1)
                lr = lr.to(m_policy_vec.device).clamp(min=1)
                d = float(self.beta) * torch.sigmoid(1.0 - float(self.beta) * m_policy_vec)
                dw_eff_vec = d / lc
                dl_eff_vec = d / lr

            elif loss_type == "LSIF":
                dw_eff_vec = torch.exp(r_w)
                dl_eff_vec = torch.exp(2.0 * r_l)
                
            elif loss_type == "BCE":
                dw_eff_vec = torch.sigmoid(-r_w)
                dl_eff_vec = torch.sigmoid(r_l)

            elif loss_type == "IPO":
                tau = float(getattr(self.args, "ipo_tau", 0.1))
                target = 1.0 / (2.0 * tau)
                # Use tilde_m (policy vs reference) to match the IPO/DPO definition.
                dw_eff_vec = 2.0 * (target - tilde_m_vec)
                dl_eff_vec = dw_eff_vec

            elif loss_type == "CPO":
                beta = float(getattr(self.args, "cpo_beta", 0.1))
                lambda_nll = float(getattr(self.args, "cpo_lambda_nll", 1.0))
                base = beta * torch.sigmoid(-beta * m_policy_vec)
                dl_eff_vec = base
                dw_eff_vec = base + lambda_nll

            elif loss_type == "SLIC":
                gamma = float(getattr(self.args, "slic_gamma", 1.0))
                lambda_coef = float(getattr(self.args, "slic_lambda_coef", 0.1))
                tau = float(getattr(self.args, "slic_tau", 0.5))
                ind = torch.sigmoid((gamma - m_policy_vec) / tau)
                # ind = (m_policy_vec < gamma).to(m_policy_vec.dtype)
                dl_eff_vec = ind
                dw_eff_vec = ind + lambda_coef

            elif loss_type == "UKL":
                dl_eff_vec = torch.exp(r_l)
                dw_eff_vec = torch.ones_like(dl_eff_vec)

            elif loss_type == "DDRO":
                r_rejected = torch.exp(r_l)
                r_rejected = torch.clamp(r_rejected, max=2.0 - 1e-6)
                dl_eff_vec = r_rejected / (2.0 - r_rejected)
                # dl_eff_vec = torch.clamp(dl_eff_vec, max=20.0)
                dw_eff_vec = torch.ones_like(dl_eff_vec)
        else:
            dw_eff_vec = dw_vec
            dl_eff_vec = dl_vec

        dw = float(dw_eff_vec.mean().item())
        dl = float(dl_eff_vec.mean().item())

        # Head-only gradients geometry (lm_head) aligned to effective batch.
        # We accumulated weighted sums across the accumulation window; convert to mean-grad here.
        eps = 1e-12
        sw_norm = 0.0
        sl_norm = 0.0
        sw_sl_dot = 0.0
        rho = 0.0
        if self._mc_n_sum > 0 and isinstance(self._mc_gw_sum, list) and isinstance(self._mc_gl_sum, list):
            sw_sq = torch.tensor(0.0, device=self.accelerator.device)
            sl_sq = torch.tensor(0.0, device=self.accelerator.device)
            dot = torch.tensor(0.0, device=self.accelerator.device)

            for gw_sum_i, gl_sum_i in zip(self._mc_gw_sum, self._mc_gl_sum):
                if gw_sum_i is not None:
                    gw_i = (gw_sum_i / float(self._mc_n_sum)).detach().float()
                    sw_sq = sw_sq + gw_i.pow(2).sum()
                else:
                    gw_i = None

                if gl_sum_i is not None:
                    gl_i = (gl_sum_i / float(self._mc_n_sum)).detach().float()
                    sl_sq = sl_sq + gl_i.pow(2).sum()
                else:
                    gl_i = None

                if gw_i is not None and gl_i is not None:
                    dot = dot + (gw_i * gl_i).sum()

            sw_norm = float(torch.sqrt(sw_sq + eps).item())
            sl_norm = float(torch.sqrt(sl_sq + eps).item())
            sw_sl_dot = float(dot.item())
            rho = float((dot / (sw_norm * sl_norm + eps)).item()) if (sw_norm > 0.0 and sl_norm > 0.0) else 0.0

        # Reset accumulated head-grad sums for next optimizer step window.
        self._mc_gw_sum = None
        self._mc_gl_sum = None
        self._mc_n_sum = 0

        # Log policy margin as m, and log the DPO/IPO margin separately as tilde_m.
        m_val = float(m_policy_eff)
        tilde_m_val = float(tilde_m_eff)
        zw_val = float(zw_eff)
        zl_val = float(zl_eff)

        if self._margin_chain_prev_m is None:
            delta_tilde_m = None
        else:
            delta_tilde_m = tilde_m_val - float(self._margin_chain_prev_m)

        if self._margin_chain_prev_m_policy is None:
            delta_m = None
        else:
            delta_m = m_val - float(self._margin_chain_prev_m_policy)

        self._margin_chain_prev_m = tilde_m_val
        self._margin_chain_prev_m_policy = m_val

        lr = None
        if getattr(self, "optimizer", None) is not None:
            try:
                lr = float(self.optimizer.param_groups[0].get("lr", None))
            except Exception:
                lr = None

        record = {
            "step": step_index,
            "loss_type": loss_type,
            "zw": zw_val,
            "zl": zl_val,
            # Keep policy logps (zw/zl) for backward compatibility, and
            # also expose explicit reward scalars (log pi - log pi_ref).
            "rw": rw,
            "rl": rl,
            "reward_margin": reward_margin,
            "m": m_val,
            "m_ref": float(m_ref_eff),
            "tilde_m": tilde_m_val,
            "delta_tilde_m": delta_tilde_m,
            "delta_m": delta_m,
            # head-only geometry (lm_head)
            "sw_l2": sw_norm,
            "sl_l2": sl_norm,
            "sw_sl_dot": sw_sl_dot,
            "rho": rho,
            # incentives
            "dw": dw,
            "dl": dl,
        }
        # Optional: DB calibration monitoring (no regularizer term; tracks the calibration constants)
        if bool(getattr(self.args, "db_calibration_enable", False)) and hasattr(self, "_db_last"):
            eps_db = float(getattr(self.args, "db_calibration_eps", 1e-12))
        
            # read cached (actual training quantities)
            log_r_ema = self._db_last.get("log_r_ema", None)
            log_r_star_ema = self._db_last.get("log_r_star_ema", None)
            dw_ema = self._db_last.get("dw_ema", None)
            dl_ema = self._db_last.get("dl_ema", None)
            # g_scale = self._db_last.get("g_scale", None)
            alpha = self._db_last.get("alpha", None)
            alpha_inv = self._db_last.get("alpha_inv", None)
            delta = self._db_last.get("delta", None)
        
            # DB edges/slack (diagnostic)
            rho_eff = float(max(abs(rho), eps_db))
            rho_eff = float(min(rho_eff, 1.0 - eps_db))
            log_rho = float(np.log(rho_eff))
        
            lower = upper = delta_db = None
            if (log_r_ema is not None) and (log_r_star_ema is not None):
                lower = float(log_r_star_ema + log_rho)
                upper = float(log_r_star_ema - log_rho)
                delta_db = float(min(log_r_ema - lower, upper - log_r_ema))
        
            record.update(
                {
                    "db/log_r_ema": log_r_ema,
                    "db/log_r_star_ema": log_r_star_ema,
                    "db/dw_ema": dw_ema,
                    "db/dl_ema": dl_ema,
                    "db/delta": delta,
                    # "db/g_scale": g_scale,
                    "db/alpha": alpha,
                    "db/alpha_inv": alpha_inv,
                    "db/log_rho": log_rho,
                    "db/lower": lower,
                    "db/upper": upper,
                    "db/slack": delta_db,
                }
            )
            
        if loss_type == "IPO":
            record["tau"] = float(getattr(self.args, "ipo_tau", 0.1))
        if loss_type == "CPO":
            record["beta"] = float(getattr(self.args, "cpo_beta", 0.1))
            record["lambda_nll"] = float(getattr(self.args, "cpo_lambda_nll", 1.0))
        if loss_type == "SLIC":
            record["gamma"] = float(getattr(self.args, "slic_gamma", 1.0))
            record["lambda_coef"] = float(getattr(self.args, "slic_lambda_coef", 0.1))
        # Helpful metadata to disambiguate micro-batch vs optimizer step
        record["micro_batch_size"] = int(getattr(self.args, "per_device_train_batch_size", -1))
        record["grad_accum_steps"] = int(getattr(self.args, "gradient_accumulation_steps", -1))
        if record["micro_batch_size"] > 0 and record["grad_accum_steps"] > 0:
            record["effective_batch_size_per_device"] = record["micro_batch_size"] * record["grad_accum_steps"]

        # Per-sample logging (same sample ordering across all *_vec fields)
        # Effective batch vectors (concatenated across accumulation window).
        zw_vec = pc
        zl_vec = pr
        # m_vec is policy margin per-sample; tilde_m_vec is DPO/IPO margin per-sample.
        m_vec = m_policy_vec
        tilde_m_vec_to_log = tilde_m_vec
        m_ref_vec_to_log = m_ref_vec
        f_w_vec = r_w
        f_l_vec = r_l
        dw_cat = dw_eff_vec
        dl_cat = dl_eff_vec
        rc_vec = rc
        rr_vec = rr

        # Per-micro-batch deltas within the gradient-accumulation window.
        # Length should match gradient_accumulation_steps (per device), unless the last window is partial.
        mb_m_list = self._mc_mb_m_list or []
        mb_delta_m_list = self._mc_mb_delta_m_list or []

        record.update(
            {
                "zw_vec": zw_vec.cpu().tolist(),
                "zl_vec": zl_vec.cpu().tolist(),
                "m_vec": m_vec.cpu().tolist(),
                "m_ref_vec": m_ref_vec_to_log.cpu().tolist(),
                "tilde_m_vec": tilde_m_vec_to_log.cpu().tolist(),
                # per-micro-batch margin mean and its within-window delta
                "m_mb_vec": mb_m_list,
                "delta_m_vec": mb_delta_m_list,
                # raw reference logps used to form rewards r_w/r_l (and thus dw/dl)
                "rc_vec": rc_vec.cpu().tolist(),
                "rr_vec": rr_vec.cpu().tolist(),
                "f_w_vec": f_w_vec.cpu().tolist(),
                "f_l_vec": f_l_vec.cpu().tolist(),
                # Reward vector aliases; numerically same as f_w_vec/f_l_vec.
                "reward_w_vec": f_w_vec.cpu().tolist(),
                "reward_l_vec": f_l_vec.cpu().tolist(),
                "dw_vec": None if dw_cat is None else dw_cat.cpu().tolist(),
                "dl_vec": None if dl_cat is None else dl_cat.cpu().tolist(),
            }
        )
        # reset buffer for next optimizer step
        if self._mc_buf is not None:
            for k in self._mc_buf:
                self._mc_buf[k].clear()
        # reset per-micro-batch lists for next optimizer step
        self._mc_mb_m_list = None
        self._mc_mb_delta_m_list = None
        self._mc_mb_prev_m = None
        if lr is not None:
            record["lr"] = lr

        if getattr(self, "_margin_chain_path", None):
            with open(self._margin_chain_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def training_step(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: Dict[str, Union[torch.Tensor, Any]],
            num_items_in_batch: Optional[int] = None,
    ):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # Scale loss when gradient accumulation is enabled
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        return loss.detach()

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="reference_chosen_logps", column=all_reference_chosen_logps)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.processing_class(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.processing_class(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            # TRL collator uses torch.tensor(..., dtype=int64); any None in lists raises TypeError.
            lpid = self.label_pad_token_id
            if lpid is None:
                lpid = -100
            else:
                lpid = int(lpid)

            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.processing_class(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt. Avoid adding if it's already there.
            # Qwen2: bos_token_id may be None; generation_config may expose eos_token_id as a list — normalize to int.
            bos_token_id = self.processing_class.bos_token_id
            if isinstance(bos_token_id, (list, tuple)):
                bos_token_id = int(bos_token_id[0]) if len(bos_token_id) > 0 else None
            elif bos_token_id is not None:
                bos_token_id = int(bos_token_id)

            eos_token_id = self.processing_class.eos_token_id
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = int(eos_token_id[0]) if len(eos_token_id) > 0 else None
            elif eos_token_id is not None:
                eos_token_id = int(eos_token_id)

            if bos_token_id is not None:
                if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                    prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                    prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
                if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                    chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                    chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
                if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                    rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                    rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer. Avoid adding if it's already there.
            if eos_token_id is not None:
                if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                    chosen_tokens["input_ids"].append(eos_token_id)
                    chosen_tokens["attention_mask"].append(1)
                if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                    rejected_tokens["input_ids"].append(eos_token_id)
                    rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [lpid] * len(
                chosen_tokens["prompt_input_ids"]
            )
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [lpid] * len(
                rejected_tokens["prompt_input_ids"]
            )

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

            pad_fallback = self.processing_class.pad_token_id
            if pad_fallback is None:
                pad_fallback = eos_token_id if eos_token_id is not None else 0
            else:
                pad_fallback = int(pad_fallback)

            def _coerce_int_sequence(seq, fallback: int) -> List[int]:
                if seq is None:
                    raise ValueError("DPO tokenize produced None for a token sequence; check dataset row.")
                if hasattr(seq, "tolist"):
                    seq = seq.tolist()
                out: List[int] = []
                for x in seq:
                    if x is None:
                        out.append(int(fallback))
                    elif isinstance(x, (list, tuple)):
                        raise ValueError("Unexpected nested token sequence in DPO batch")
                    else:
                        try:
                            out.append(int(x))
                        except (TypeError, ValueError):
                            out.append(int(fallback))
                return out

            for bk, bv in list(batch.items()):
                if bk.endswith("_attention_mask"):
                    batch[bk] = _coerce_int_sequence(bv, 0)
                elif bk.endswith("_labels"):
                    batch[bk] = _coerce_int_sequence(bv, lpid)
                elif bk.endswith("_input_ids"):
                    batch[bk] = _coerce_int_sequence(bv, pad_fallback)

        else:
            chosen_tokens = self.processing_class(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.processing_class(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.processing_class(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
                self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
            batch: Dict[str, Union[List, torch.LongTensor]],
            is_encoder_decoder: bool = False,
            label_pad_token_id: int = -100,
            padding_value: int = 0,
            device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # if self.reference_free:
        #     ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        # else:
        #     ref_logratios = reference_chosen_logps - reference_rejected_logps

        # pi_logratios = pi_logratios.to(self.accelerator.device)
        # ref_logratios = ref_logratios.to(self.accelerator.device)
        # Optional: DB calibration (forward unchanged, backward gradients rescaled)
        policy_chosen_logps, policy_rejected_logps = self._maybe_apply_db_calibration(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
        )

        loss_type = str(getattr(self, "loss_type", "")).upper()
        if loss_type == "SIMPO":
            losses = -F.logsigmoid(self.beta * (policy_chosen_logps - policy_rejected_logps) - 1.0)
        else:
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps).to(self.accelerator.device)
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps).to(self.accelerator.device)

            if loss_type in ("DPO", "TIDPO"):
                losses = -F.logsigmoid(self.beta * (chosen_rewards-rejected_rewards))

            elif self.loss_type == "KTO":
                # KTO: smooth hinge using softplus on the reward margin
                margin = self.beta * (chosen_rewards - rejected_rewards)
                losses = F.softplus(-margin)

            elif self.loss_type == "LSIF":  #LSIF
                chosen_rewards = torch.exp(chosen_rewards)
                rejected_rewards = 0.5*torch.exp(2.0*rejected_rewards)
                losses = -chosen_rewards + rejected_rewards

            elif self.loss_type == "UKL":
                chosen_rewards = policy_chosen_logps
                rejected_rewards = torch.exp((policy_rejected_logps - reference_rejected_logps))
                losses = -chosen_rewards + rejected_rewards

            elif self.loss_type == "BCE": #
                chosen_rewards = -F.logsigmoid(chosen_rewards)
                rejected_rewards = -F.logsigmoid(-(rejected_rewards))
                losses = chosen_rewards + rejected_rewards

            elif self.loss_type == "DDRO":
                # DDRO (Sec. 4.2): L = [log(2)-log(r_chosen)] + [log(2)-log(2-r_rejected)],
                # where r = pi / pi_ref = exp(log_pi - log_pi_ref).
                # Use chosen_rewards directly as log(r_chosen) for better numerical stability.
                loss_chosen = torch.log(torch.tensor(2.0, device=self.accelerator.device)) - chosen_rewards

                r_rejected = torch.exp(rejected_rewards)

                # Near r=2, log(2-r) is singular; switch to a linear extension at threshold.
                threshold = 2.0 - 1e-6

                mask_safe = r_rejected < threshold

                # Safe region: standard formula.
                r_safe = torch.clamp(r_rejected, max=threshold)
                loss_rejected_safe = torch.log(torch.tensor(2.0, device=self.accelerator.device)) - torch.log(2.0 - r_safe)

                # Unsafe region: tangent-line extension keeps finite loss and non-zero gradient.
                eps = 1e-6
                loss_at_threshold = torch.log(torch.tensor(2.0, device=self.accelerator.device)) - torch.log(torch.tensor(eps, device=self.accelerator.device))
                slope = 1.0 / eps
                loss_rejected_unsafe = loss_at_threshold + slope * (r_rejected - threshold)

                loss_rejected = torch.where(mask_safe, loss_rejected_safe, loss_rejected_unsafe)

                losses = loss_chosen + loss_rejected

            elif self.loss_type == "IPO":
                # IPO (squared loss) on the same margin definition used by margin_chain:
                #   tilde_m = (zw - zl) - (zw_ref - zl_ref)
                # and we log incentives as: d_w = d_l = 2*(1/(2*tau) - tilde_m)
                # tau = float(getattr(self.args, "ipo_tau", 0.1))
                tau = float(getattr(self.args, "ipo_tau", 1.0))
                target = 1.0 / (2.0 * tau)
                tilde_m = (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
                tilde_m = tilde_m.to(self.accelerator.device)
                losses = (tilde_m - target).pow(2)
                
            elif self.loss_type == "CPO":
                beta = float(getattr(self.args, "cpo_beta", 0.1))
                lambda_nll = float(getattr(self.args, "cpo_lambda_nll", 1.0))
                
                # CPO Loss = L_prefer + L_NLL
                cpo_logits = policy_chosen_logps - policy_rejected_logps
                preference_loss = -F.logsigmoid(beta * cpo_logits)
                
                nll_loss = -policy_chosen_logps
                losses = preference_loss + lambda_nll * nll_loss
                
            elif str(self.loss_type).upper() == "SLIC":
                gamma = float(getattr(self.args, "slic_gamma", 1.0))
                lambda_coef = float(getattr(self.args, "slic_lambda_coef", 0.1))
                tau = float(getattr(self.args, "slic_tau", 0.5))
                # cal_loss = tau * F.softplus((gamma - policy_chosen_logps + policy_rejected_logps) / tau)
                cal_loss = torch.clamp(gamma - policy_chosen_logps + policy_rejected_logps, min=0)
                reg_loss = -policy_chosen_logps

                losses = cal_loss + lambda_coef * reg_loss

            else:
                raise ValueError(
                    f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'bco_pair', 'sppo_hard', 'nca_pair', 'robust']"
                )

        chosen_rewards = (
            (
                    policy_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            (
                    policy_rejected_logps.to(self.accelerator.device)

            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
            logits: torch.FloatTensor,
            labels: torch.LongTensor,
            average_log_prob: bool = False,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        lengths = loss_mask.sum(-1) # (batch_size,)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / lengths, lengths
        else:
            return (per_token_logps * loss_mask).sum(-1), lengths

    @staticmethod
    def compute_token_importance_weights(
            logits: torch.FloatTensor,
            labels: torch.LongTensor,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute per-token importance weights via analytic gradient attribution.

        For cross-entropy with softmax, the gradient magnitude w.r.t. the true-class
        logit is  1 - p(y_t).  We use this as the token importance score:
        high uncertainty -> high importance.

        Returns a (batch, shifted_seq_len) tensor with 0 on prompt/padding positions.
        Weights are detached so gradients only flow through the log-probs.
        """
        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]

        loss_mask = (labels != label_pad_token_id).float()

        safe_labels = labels.clone()
        safe_labels[labels == label_pad_token_id] = 0

        # p(y_t) for each position
        probs = logits.detach().softmax(-1)
        token_probs = torch.gather(probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)

        # importance = 1 - p(y_t), clamped for stability
        weights = ((1.0 - token_probs) * loss_mask).clamp(min=1e-8) * loss_mask
        return weights

    @staticmethod
    def get_batch_weighted_logps(
            logits: torch.FloatTensor,
            labels: torch.LongTensor,
            weight_matrix: torch.FloatTensor,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute token-importance-weighted average log probabilities.

        Returns:
            weighted_avg_logps: (batch,) — sum(w * logp * mask) / sum(w * mask)
            effective_lengths:  (batch,) — sum(w * mask), the denominator
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]

        loss_mask = (labels != label_pad_token_id).float()
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        weighted_sum = (per_token_logps * weight_matrix * loss_mask).sum(-1)
        weighted_denom = (weight_matrix * loss_mask).sum(-1).clamp(min=1e-8)

        return weighted_sum / weighted_denom, weighted_denom

    def concatenated_forward(
            self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        # all_logits = model(
        #     concatenated_batch["concatenated_input_ids"],
        #     attention_mask=concatenated_batch["concatenated_attention_mask"],
        #     use_cache=False,
        #     **model_kwargs,
        # ).logits
        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            output_hidden_states=True,  # ★ 关键：强制输出 hidden_states
            **model_kwargs,
            )
        all_logits = outputs.logits


        all_logps, all_lengths = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        self.all_lengths = all_lengths

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def tidpo_concatenated_forward(
            self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor,
               torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """TIDPO forward: same as concatenated_forward but returns
        token-importance-weighted average logps for both policy and reference."""
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        # --- policy forward ---
        policy_outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            **model_kwargs,
        )
        policy_logits = policy_outputs.logits

        # --- reference forward ---
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_logits = self.model(
                        concatenated_batch["concatenated_input_ids"],
                        attention_mask=concatenated_batch["concatenated_attention_mask"],
                        use_cache=False,
                        **model_kwargs,
                    ).logits
            else:
                ref_logits = self.ref_model(
                    concatenated_batch["concatenated_input_ids"],
                    attention_mask=concatenated_batch["concatenated_attention_mask"],
                    use_cache=False,
                    **model_kwargs,
                ).logits

        labels = concatenated_batch["concatenated_labels"]

        # token importance weights (detached, from policy logits)
        weights = self.compute_token_importance_weights(
            policy_logits, labels, self.label_pad_token_id, self.is_encoder_decoder,
        )

        # weighted average logps — same weights for policy & reference
        policy_logps, effective_lengths = self.get_batch_weighted_logps(
            policy_logits, labels, weights, self.label_pad_token_id, self.is_encoder_decoder,
        )
        with torch.no_grad():
            ref_logps, _ = self.get_batch_weighted_logps(
                ref_logits, labels, weights, self.label_pad_token_id, self.is_encoder_decoder,
            )

        self.all_lengths = effective_lengths

        return (
            policy_logps[:len_chosen],
            policy_logps[len_chosen:],
            ref_logps[:len_chosen],
            ref_logps[len_chosen:],
            policy_logits[:len_chosen],
            policy_logits[len_chosen:],
        )

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        loss_type_upper = str(self.loss_type).upper()

        if loss_type_upper == "TIDPO":
            # TIDPO: token-importance-weighted average logps, then standard DPO loss
            (
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = self.tidpo_concatenated_forward(model, batch)
        else:
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = self.concatenated_forward(model, batch)

            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        # Record the full "margin–geometry–incentive–difference" chain (scalars only, head-only grads)
        if train_eval == "train":
            self._maybe_record_margin_chain(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                reference_chosen_logps=reference_chosen_logps,
                reference_rejected_logps=reference_rejected_logps,
            )

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics

    def compute_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: Dict[str, Union[torch.Tensor, Any]],
            return_outputs=False,
            num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def generate_batch_samples_for_evaluation(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.processing_class.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.processing_class.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.processing_class.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.processing_class.pad_token_id)
        policy_output_decoded = self.processing_class.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.processing_class.pad_token_id)
        reference_output_decoded = self.processing_class.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.generate_batch_samples_for_evaluation(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt):], ref[len(prompt):]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float], start_time=None, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs, start_time=start_time, **kwargs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "dpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)


