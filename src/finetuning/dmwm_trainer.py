from transformers import Trainer
import torch
from typing import Optional
from datasets import Dataset
from transformers.integrations import NeptuneCallback

from src.finetuning.losses import LossTypes

class DomainWatermarkTrainer(Trainer):
    """Custom Trainer class that overloads two methods
    - saving logic: when saving the model, we also evaluate the watermark for convenience.
    - compute_loss: we compute either the watermark loss/regularization loss according to the labels (loss_type) of the input.
    """

    def __init__(
        self,
        evaluation,
        tokenizer_wm,
        finetuning_config,
        teacher_model,
        type_processor,
        use_neptune: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.evaluation = evaluation
        self.tokenizer_wm = tokenizer_wm

        self.type_processor = type_processor

        self.finetuning_config = finetuning_config

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.watermarker = evaluation.get_detector(teacher_model, tokenizer_wm)

        self.alpha = finetuning_config.alpha
        
        self.losses = LossTypes(self.watermarker)

        self.use_neptune = use_neptune

    def advanced_neptune_logging(self, loss: float, name: str):
        if self.use_neptune:
            run = NeptuneCallback.get_run(self)
            run[f"finetuning/train/{name}"].append(loss)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute distillation loss.

        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # Small hack: Setting proper labels and reading the loss type from the labels
        labels = inputs.pop("labels")
        inputs["labels"] = inputs["input_ids"].clone()
        
        lambdas, loss_types = self.type_processor(labels)
        lambdas = lambdas.view(-1, 1, 1)
        loss_types = loss_types

        outputs = model(**inputs)
        student_logits = outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        unique_loss_types = torch.unique(loss_types).tolist()
        losses = torch.zeros(max(unique_loss_types) + 1, device=student_logits.device)

        for loss_type in unique_loss_types:
            mask = loss_types == loss_type
            masked_inputs = {key: value[mask] for key, value in inputs.items()}
            loss = torch.tensor(0.0, device=student_logits.device)

            if loss_type == 0:
                loss = self.losses.compute_watermark_loss(
                    input_ids=inputs["input_ids"][mask],
                    attention_mask=inputs["attention_mask"][mask],
                    logits=student_logits[mask],
                    teacher_logits=teacher_logits[mask],
                    lambdas=lambdas[mask],
                )
                self.advanced_neptune_logging(loss.item(), "watermark_loss")
            elif loss_type == 1:
                loss = self.losses.compute_anti_watermark_tv_loss(
                    input_ids=inputs["input_ids"][mask],
                    attention_mask=inputs["attention_mask"][mask],
                    logits=student_logits[mask],
                    teacher_logits=teacher_logits[mask],
                    lambdas=lambdas[mask]
                )
                self.advanced_neptune_logging(loss.item(), "anti_watermark_tv_loss")
            else:
                raise ValueError(f"Loss type {loss_type} is not supported.")

            losses[loss_type] = loss

        loss = torch.sum(losses)

        return (loss, outputs) if return_outputs else loss

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
        is_checkpoint: bool = True,
    ):
        """
        While saving the model, we also evaluate the watermark.
        Additionaly, we disable the saving process according to the MainConfiguration.
        """

        if is_checkpoint:
            checkpoint_folder = f"-{self.state.global_step}"
        else:
            checkpoint_folder = ""

        # Save tokenizer
        if output_dir is not None:
            self.tokenizer_wm.save_pretrained(output_dir)

        super().save_model(output_dir, _internal_call)

        try:
            with torch.no_grad():
                self.evaluation.eval_model(
                    self.model,
                    self.tokenizer_wm,
                    self.finetuning_config,
                    f"finetuning{checkpoint_folder}",
                    custom_eval=self.finetuning_config.watermark_eval_config,
                )

                if self.use_neptune:
                    run = NeptuneCallback.get_run(self)
                    self.evaluation.process_results(run, plot_only=True)
        except Exception as e:
            print(f"Error while evaluating watermark: {e}")
            print("Continuing without evaluating watermark")
