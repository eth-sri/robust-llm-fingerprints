import torch
from math import sqrt
from src.finetuning.losses import LossTypes

class RandomTrainer:
    """This is a small class that handles the meta learning to separate it from the outer loop."""

    def __init__(self, meta_learning_config, watermarker):
        
        self.losses = LossTypes(watermarker)

        self.loss_type = meta_learning_config.loss_type
        self.n_samples = 1
        self.norm = 1.5
        self.meta_learning_reg = 1
        self.model_size = None
        self.disable_domain_regularization = meta_learning_config.disable_domain_regularization
        
    def get_model_size(self, model):
        if self.model_size is not None:
            return self.model_size

        total_params = sum(
            p.numel() for n, p in model.named_parameters() if "weight" in n
        )
        self.model_size = total_params

        return total_params
    
    def get_perturbed_param(self, original_param):
        # Compute the perturbed parameter.
        perturbation = torch.randn_like(original_param) / sqrt(original_param.numel()) * self.norm
        perturbed = original_param + perturbation

        # Register a backward hook on the perturbed tensor.
        def hook_fn(grad):
            # grad is the gradient computed with respect to (original_param+perturbation)
            # Since d/d(original_param) [original_param + perturbation] = 1,
            # simply add the incoming gradient to the original parameter’s .grad.
            if original_param.grad is None:
                original_param.grad = grad.clone()  # clone to avoid in-place issues
            else:
                original_param.grad += grad
            # Optionally, you can return grad unchanged.
            return grad

        perturbed.register_hook(hook_fn)
        return perturbed
    
    def meta_learning_loss(self, model, meta_model_state, inputs):
        loss = torch.func.functional_call(
            model, meta_model_state, (), kwargs=inputs, tie_weights=False
        ).loss
        return loss

    def compute_random_loss(
        self, model, inputs: dict, student_logits: torch.Tensor, teacher_logits: torch.Tensor, loss_types, lambdas
        ):

        perturbed_params = {
            name: self.get_perturbed_param(param)
            for name, param in model.named_parameters()
        }
        
        outputs = torch.func.functional_call(
            model,
            perturbed_params,
            (),
            kwargs=inputs,
            tie_weights=False,
        )
        random_model_logits = outputs.logits
        
        unique_loss_types = torch.unique(loss_types).tolist()
        losses = torch.zeros(
            max(unique_loss_types) + 1, device=random_model_logits.device
        )
        
        for loss_type in unique_loss_types:
            mask = loss_types == loss_type
            meta_loss = torch.tensor(0.0, device=random_model_logits.device)

            if loss_type == 0:
                if self.loss_type == "watermark":
                    meta_loss = self.losses.compute_watermark_loss(
                        input_ids=inputs["input_ids"][mask],
                        attention_mask=inputs["attention_mask"][mask],
                        teacher_logits=teacher_logits[mask],
                        logits=random_model_logits[mask],
                        lambdas=lambdas[mask]
                    )
                elif self.loss_type == "booster":
                    meta_loss = self.losses.compute_meta_learning_radioactivity_gap(
                        input_ids=inputs["input_ids"][mask],
                        attention_mask=inputs["attention_mask"][mask],
                        logits=student_logits[mask],
                        meta_logits=random_model_logits[mask],
                    )
                elif self.loss_type == "radioactivity":
                    meta_loss = self.losses.compute_radioactivity_score(
                        input_ids=inputs["input_ids"][mask],
                        attention_mask=inputs["attention_mask"][mask],
                        logits=random_model_logits[mask]
                    )
                    meta_loss = -meta_loss
                elif self.loss_type == "watermark_student":
                    meta_loss = self.losses.compute_logit_distillation_loss(
                        input_ids=inputs["input_ids"][mask],
                        attention_mask=inputs["attention_mask"][mask],
                        teacher_logits=teacher_logits[mask],
                        logits=student_logits[mask],
                        lambdas=lambdas[mask]
                    )
            else:
                if not self.disable_domain_regularization:
                    masked_inputs = {key: value[mask] for key, value in inputs.items()}
                    meta_loss = self.meta_learning_loss(
                        model, perturbed_params, masked_inputs
                    )

            losses[loss_type] = meta_loss

        meta_loss = torch.sum(losses)
        
        return meta_loss

    def meta_learning_step(
        self, model, inputs: dict, student_logits: torch.Tensor, teacher_logits: torch.Tensor, loss_types, lambdas
    ):

        # Computing the meta-learning loss
        inputs["labels"] = inputs["input_ids"]
        meta_loss = torch.tensor(0.0).to(model.device)
        for _ in range(self.n_samples):
            meta_loss += self.compute_random_loss(
                model=model,
                inputs=inputs,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                loss_types=loss_types,
                lambdas=lambdas,
            )
     
        meta_loss = meta_loss * self.meta_learning_reg / self.n_samples

        return meta_loss
