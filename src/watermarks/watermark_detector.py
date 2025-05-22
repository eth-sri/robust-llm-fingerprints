import torch
from typing import Dict


class WatermarkDetector:
    def __init__(self):
        pass

    def watermark_logits(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.FloatTensor:
        """Returns watermarked logits to be used as distillation target."""
        raise NotImplementedError("watermark_logits method must be implemented in the subclass.")

    def watermark_logits_diff(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.FloatTensor:
        """Returns difference between watermarked logits and original logits."""
        raise NotImplementedError("watermark_logits method must be implemented in the subclass.")

    def compute_watermark_distillation_loss(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        student_logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
        teacher_logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
        lambdas: torch.FloatTensor,  # (batch,)
    ) -> torch.FloatTensor:
        """Returns the distillation loss for watermarking."""
        watermarked_logits = self.watermark_logits(
            input_ids, teacher_logits
        )

        # compute distillation loss
        loss_fct = torch.nn.KLDivLoss(reduction="none", log_target=True)
        loss = loss_fct(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.log_softmax(watermarked_logits, dim=-1),
        )
        loss = torch.sum(loss, dim=(-1, -2))
        loss = torch.mean(loss * lambdas.view(-1)) / (student_logits.shape[1])

        return loss
    
    def detect(
        self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        """Should work with batched input and return a tensor of pvalues corresponding to the batch dimenstion."""
        raise NotImplementedError("detect method must be implemented in the subclass.")

    def is_argmax(self) -> bool:
        raise NotImplementedError("get_type method must be implemented in the subclass.")