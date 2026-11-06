# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

from collections.abc import Mapping, MutableMapping
import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .hard_example_loss import compute_hard_example_weighted_loss, validate_hard_example_loss_config


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        self.finetuning_args = finetuning_args
        self.model_args = model_args
        if getattr(finetuning_args, "use_hard_example_loss", False):
            self._validate_hard_example_loss_setup(
                training_args=kwargs.get("args"),
                compute_loss_func=kwargs.get("compute_loss_func"),
            )

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self._super_model_accepts_loss_kwargs = self.model_accepts_loss_kwargs
        if getattr(finetuning_args, "use_hard_example_loss", False):
            # Hard-example loss is normalized by the microbatch weighted token count,
            # so Trainer.training_step must divide by gradient_accumulation_steps once.
            self.model_accepts_loss_kwargs = False

        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        if getattr(finetuning_args, "use_hard_example_loss", False):
            self._validate_hard_example_loss_setup()

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    def _validate_hard_example_loss_setup(
        self,
        training_args: Optional[Any] = None,
        compute_loss_func: Optional[Any] = None,
    ) -> None:
        r"""Reject configuration or trainer states incompatible with hard-example weighted loss."""
        if not getattr(self.finetuning_args, "use_hard_example_loss", False):
            return

        if not isinstance(self.finetuning_args.use_hard_example_loss, bool):
            raise ValueError("`use_hard_example_loss` must be a boolean.")

        validate_hard_example_loss_config(
            alpha=getattr(self.finetuning_args, "hard_example_loss_alpha", 0.5),
            max_weight=getattr(self.finetuning_args, "hard_example_loss_max_weight", 1.5),
        )

        if getattr(self.finetuning_args, "stage", "sft") != "sft":
            raise ValueError("`use_hard_example_loss` is only valid for SFT training stage.")

        if getattr(self.finetuning_args, "use_dft_loss", False):
            raise ValueError("Cannot use `use_hard_example_loss` and `use_dft_loss` together.")

        active_args = training_args if training_args is not None else getattr(self, "args", None)
        active_loss_func = (
            compute_loss_func if compute_loss_func is not None else getattr(self, "compute_loss_func", None)
        )

        if active_loss_func is not None:
            raise ValueError("Cannot use `use_hard_example_loss` with a custom `compute_loss_func`.")

        if getattr(self, "label_smoother", None) is not None or (
            active_args is not None and getattr(active_args, "label_smoothing_factor", 0.0) != 0
        ):
            raise ValueError("Cannot use `use_hard_example_loss` with label smoothing.")

        if (self.model_args is not None and getattr(self.model_args, "enable_liger_kernel", False)) or (
            active_args is not None and getattr(active_args, "use_liger_kernel", False)
        ):
            raise ValueError("Cannot use `use_hard_example_loss` with liger kernel.")

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional["torch.Tensor"] = None,
        *args,
        **kwargs,
    ):
        if not getattr(self.finetuning_args, "use_hard_example_loss", False):
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch, *args, **kwargs
            )

        if not model.training:
            prev_accepts = self.model_accepts_loss_kwargs
            self.model_accepts_loss_kwargs = getattr(self, "_super_model_accepts_loss_kwargs", prev_accepts)
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                    *args,
                    **kwargs,
                )
            finally:
                self.model_accepts_loss_kwargs = prev_accepts

        self._validate_hard_example_loss_setup()
        if not isinstance(inputs, Mapping) or "labels" not in inputs or inputs["labels"] is None:
            raise ValueError("`use_hard_example_loss` requires `labels` in `inputs` during training.")

        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)

        if isinstance(outputs, Mapping):
            logits = outputs.get("logits")
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            logits = outputs[0]
        else:
            logits = None

        if logits is None:
            raise ValueError(
                "`use_hard_example_loss` requires model outputs to include `logits` "
                "(incompatible with fused loss kernels that omit logits)."
            )

        loss = compute_hard_example_weighted_loss(
            logits=logits,
            labels=labels,
            alpha=self.finetuning_args.hard_example_loss_alpha,
            max_weight=self.finetuning_args.hard_example_loss_max_weight,
            ignore_index=IGNORE_INDEX,
        )

        if return_outputs or self.args.past_index >= 0:
            if isinstance(outputs, MutableMapping):
                outputs["loss"] = loss
                if hasattr(outputs, "move_to_end"):
                    outputs.move_to_end("loss", last=False)
            elif isinstance(outputs, tuple):
                outputs = (loss,) + outputs

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
