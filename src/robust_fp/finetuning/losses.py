# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

import torch

class LossTypeProcessor:
    """We use this class to encode some loss parameters from the labels. Note that the losses are directly defined within the Trainer class."""

    def __init__(self, device):
        self.lambdas = torch.tensor([], device=device)
        self.loss_types = torch.tensor([], device=device, dtype=torch.int)
        self.device = device

        self.loss_type_parser = {
            "watermark": 0,
            "anti-watermark-tv": 1
        }

    def __call__(self, labels: torch.Tensor):
        lambdas = self.lambdas[labels]
        loss_types = self.loss_types[labels]

        return lambdas, loss_types

    def add_dataset(
        self, lambd: float, loss_type: int | str
    ):  
        if isinstance(loss_type, str):
            loss_type = self.loss_type_parser[loss_type]

        self.lambdas = torch.cat(
            [self.lambdas, torch.tensor([lambd], device=self.device)]
        )
        self.loss_types = torch.cat(
            [
                self.loss_types,
                torch.tensor([loss_type], device=self.device, dtype=torch.int),
            ]
        )

        id = len(self.lambdas) - 1

        return id

class LossTypes:
    """This is to regroup all of the different losses term we use. The actual training happens within the Trainer class."""
    def __init__(self, watermarker):
        self.watermarker = watermarker
        self.loss_fct = torch.nn.KLDivLoss(reduction="none", log_target=True)

    def compute_watermark_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        lambdas: torch.Tensor,
    ):
        loss = self.watermarker.compute_watermark_distillation_loss(
            input_ids, logits, teacher_logits, lambdas
        )
        return loss
    
    def compute_logit_distillation_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        lambdas: torch.Tensor,
    ):
        
        loss = self.loss_fct(
            torch.nn.functional.log_softmax(logits, dim=-1),
            torch.nn.functional.log_softmax(teacher_logits, dim=-1),
        )
        loss = torch.sum(loss, dim=(-1))
        loss = loss * attention_mask
        loss = torch.sum(loss, dim=(-1))
        loss = torch.mean(loss * lambdas.view(-1)) / (logits.shape[1])

        return loss

    def compute_anti_watermark_tv_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        lambdas: torch.Tensor,
    ):
        delta = self.watermarker.watermark_logits_diff(input_ids, teacher_logits)
        delta_mask = delta > 0

        student_prob = torch.nn.functional.softmax(logits, dim=-1)
        teacher_prob = torch.nn.functional.softmax(teacher_logits, dim=-1)

        # Compute a TV loss, that focuses on too high green tokens probabilities
        positive_tv = torch.sum(
            delta_mask
            * torch.maximum(
                student_prob - teacher_prob,
                torch.tensor(0.0, device=student_prob.device),
            ),
            dim=(-1),
        )
        positive_tv = positive_tv * attention_mask
        positive_tv = torch.sum(positive_tv, dim=(-1))
        positive_tv = torch.mean(positive_tv * lambdas.view(-1)) / (logits.shape[1])

        # Compute KL logit distillation loss
        kl_reg_loss = self.loss_fct(
            torch.nn.functional.log_softmax(logits, dim=-1),
            torch.nn.functional.log_softmax(teacher_logits, dim=-1),
        )
        kl_reg_loss = torch.sum(kl_reg_loss, dim=(-1))
        kl_reg_loss = kl_reg_loss * attention_mask
        kl_reg_loss = torch.sum(kl_reg_loss, dim=(-1))
        kl_reg_loss = torch.mean(kl_reg_loss * lambdas.view(-1)) / (logits.shape[1])

        loss = kl_reg_loss + positive_tv

        return loss