import torch
from torch.utils.data import DataLoader
from torch.func import functional_call
import sys
from src.finetuning.losses import LossTypes

def create_sum_hook(mp):
    """Hook to sum the gradients of the meta model."""

    def hook(grad):
        if mp.grad is None:
            mp.grad = grad.clone()
        else:
            mp.grad.add_(grad)
        return grad  # optional: the hook can return the grad

    return hook



class MetaLearningTrainer:
    """This is a small class that handles the meta learning to separate it from the outer loop."""

    def __init__(
        self,
        meta_learning_config,
        meta_learning_dataset,
        outer_gradient_accumulation_steps: int,
        watermarker,
    ):
        self.warmup_step = 0
        self.losses = LossTypes(watermarker)

        # Preparing the dataset for training
        seed = hash(meta_learning_config.short_str()) % 2**sys.hash_info.width
        meta_learning_dataset = meta_learning_dataset.shuffle(seed=seed)
        meta_learning_dataset = meta_learning_dataset.with_format("torch")
        self.meta_learning_dataset = DataLoader(
            meta_learning_dataset,
            batch_size=meta_learning_config.per_device_batch_size,
        )
        self.meta_learning_dataset = iter(self.meta_learning_dataset)

        # Parsing the config
        self.meta_learning_rate = meta_learning_config.learning_rate
        self.meta_learning_num_steps = meta_learning_config.num_steps
        self.meta_learning_gradient_accumulation_steps = (
            meta_learning_config.gradient_accumulation_steps
        )
        self.meta_learning_per_device_batch_size = (
            meta_learning_config.per_device_batch_size
        )
        self.meta_learning_step_counter = -1
        self.meta_learning_run_every_n_steps = (
            meta_learning_config.run_every_n_steps
            * outer_gradient_accumulation_steps
        )  # Meta model update doesnt have to happen at every steps + Not in a middle of a batch
        self.meta_learning_reg = meta_learning_config.reg
        self.meta_learning_warmup = (
            meta_learning_config.warmup_steps * outer_gradient_accumulation_steps
        )
        self.loss_type = meta_learning_config.loss_type
        self.disable_domain_regularization = meta_learning_config.disable_domain_regularization
            
    def get_meta_learning_dataloader(self):
        return self.meta_learning_dataset

    def load_meta_optimizer(self, meta_model_state):
        parameters = [param for param in meta_model_state.values()]
        optimizer = torch.optim.Adam(parameters, lr=self.meta_learning_rate)
        return optimizer

    def get_meta_learning_params(self):
        gradient_accumulation_steps = self.meta_learning_gradient_accumulation_steps
        num_batches = int(
            self.meta_learning_num_steps
            * self.meta_learning_per_device_batch_size
            // gradient_accumulation_steps
        )
        return num_batches, gradient_accumulation_steps

    def get_meta_learning_model_state(self, model, init: bool = True):
        if init:
            meta_model_state = model.state_dict()
            meta_model_state = {
                name: param.clone().detach().requires_grad_()
                for name, param in meta_model_state.items()
            }
        else:
            meta_model_state = self.meta_model_state

        return meta_model_state

    def save_meta_learning_model_state(self, meta_model_state):
        self.meta_model_state = meta_model_state

    def keep_track_of_meta_learning(self, model):
        if self.meta_learning_step_counter == -1:
            print("Starting meta learning")

        self.meta_learning_step_counter += 1

        if self.meta_learning_step_counter % self.meta_learning_run_every_n_steps == 0:
            return False, None

        return True, self.get_meta_learning_model_state(model, init=False)

    def meta_learning_loss(self, model, meta_model_state, inputs):
        loss = torch.func.functional_call(
            model, meta_model_state, (), kwargs=inputs, tie_weights=False
        ).loss
        return loss
    
    def get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass

        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.item()

        return batch_samples, num_items_in_batch

    def train_meta_learning_model(self, model):
        # Only run meta learning every n steps
        stop, meta_model_state = self.keep_track_of_meta_learning(model)
        if stop:
            return meta_model_state

        meta_model_state = self.get_meta_learning_model_state(model, init=True)
        optimizer = self.load_meta_optimizer(meta_model_state)
        dataloader = self.get_meta_learning_dataloader()

        num_batches, gradient_accumulation_steps = self.get_meta_learning_params()

        step = 0
        tr_loss = torch.tensor(0.0).to(model.device)

        batch_samples, _ = self.get_batch_samples(dataloader, num_batches)

        optimizer.zero_grad()

        for i, inputs in enumerate(batch_samples):
            
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()
            
            step += 1
            tr_loss_step = self.meta_learning_loss(model, meta_model_state, inputs)

            tr_loss += tr_loss_step

            if step % gradient_accumulation_steps == 0:
                tr_loss /= gradient_accumulation_steps
                tr_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss = torch.tensor(0.0).to(model.device)

        if step % gradient_accumulation_steps != 0:
            tr_loss /= gradient_accumulation_steps
            tr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        optimizer.zero_grad() # Make sure to clear the gradients
        self.save_meta_learning_model_state(meta_model_state)

        return meta_model_state

    def meta_learning_step(
        self, model, inputs: dict, student_logits: torch.Tensor, teacher_logits: torch.Tensor, loss_types, lambdas
    ):
        # Compute meta-learning model if needed
        self.warmup_step += 1
        if self.warmup_step > self.meta_learning_warmup:
            meta_model_state_dict = self.train_meta_learning_model(model)
            model_params = dict(model.named_parameters())

            for name, param in meta_model_state_dict.items():
                if name in model_params:
                    param.register_hook(create_sum_hook(model_params[name]))

            # We do a forward pass with the meta model
            meta_model_outputs = functional_call(
                model, meta_model_state_dict, (), kwargs=inputs, tie_weights=False
            )
            meta_model_logits = meta_model_outputs.logits

        else:
            return torch.tensor(0.0).to(model.device)

        unique_loss_types = torch.unique(loss_types).tolist()
        losses = torch.zeros(
            max(unique_loss_types) + 1, device=meta_model_logits.device
        )

        for loss_type in unique_loss_types:
            mask = loss_types == loss_type
            meta_loss = torch.tensor(0.0, device=meta_model_logits.device)

            if loss_type == 0:
                if self.loss_type == "watermark":
                    meta_loss = self.losses.compute_watermark_loss(
                        input_ids=inputs["input_ids"][mask],
                        attention_mask=inputs["attention_mask"][mask],
                        teacher_logits=teacher_logits[mask],
                        logits=meta_model_logits[mask],
                        lambdas=lambdas[mask]
                    )
                elif self.loss_type == "booster":
                    meta_loss = self.losses.compute_meta_learning_radioactivity_gap(
                        input_ids=inputs["input_ids"][mask],
                        attention_mask=inputs["attention_mask"][mask],
                        logits=student_logits[mask],
                        meta_logits=meta_model_logits[mask],
                    )
                elif self.loss_type == "radioactivity":
                    meta_loss = self.losses.compute_radioactivity_score(
                        input_ids=inputs["input_ids"][mask],
                        attention_mask=inputs["attention_mask"][mask],
                        logits=meta_model_logits[mask]
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
                        model, meta_model_state_dict, masked_inputs
                    )

            losses[loss_type] = meta_loss

        meta_loss = torch.sum(losses) * self.meta_learning_reg

        return meta_loss