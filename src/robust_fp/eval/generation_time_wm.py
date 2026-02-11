import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
from vllm import SamplingParams



@dataclass
class _ReqCfg:
    gamma: float = 0.25
    delta: float = 2.0


class KGWWatermark(LogitsProcessor):
    """Simplified KGW watermark for robustness evaluation."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        # Per-row index -> per-request config
        self.req_cfg: Dict[int, _ReqCfg] = {}

    # Greedy can change, so not argmax-invariant.
    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        # Add new requests: read per-request params from SamplingParams.extra_args
        for index, params, _, _ in batch_update.added:
            assert params is not None
            self.req_cfg[index] = self._cfg_from_params(params)

        if self.req_cfg:
            # Remove finished/cancelled
            for index in batch_update.removed:
                self.req_cfg.pop(index, None)

            # Handle moves and swaps of rows
            for adx, bdx, direct in batch_update.moved:
                a_val = self.req_cfg.pop(adx, None)
                b_val = self.req_cfg.pop(bdx, None)
                if a_val is not None:
                    self.req_cfg[bdx] = a_val
                if direct == MoveDirectionality.SWAP and b_val is not None:
                    self.req_cfg[adx] = b_val

    def apply(self, logits: torch.Tensor, processor_batch: Optional[Any] = None) -> torch.Tensor:
        """Apply KGW watermark per active row using its current input_ids."""
        if not self.req_cfg:
            return logits

        batch_size = logits.size(0)

        for row_idx, cfg in list(self.req_cfg.items()):
            if row_idx < 0 or row_idx >= batch_size:
                continue

            self._watermark_logits_row(logits[row_idx], cfg)

        return logits

    # ----------------------------
    # Helpers
    # ----------------------------
    def _cfg_from_params(self, params: SamplingParams) -> _ReqCfg:
        ea = params.extra_args or {}
        return _ReqCfg(
            gamma=float(ea.get("gamma", 0.25)),
            delta=float(ea.get("delta", 2.0)),
        )

    def _watermark_logits_row(self, row_logits: torch.FloatTensor, cfg: _ReqCfg) -> None:
        """In-place KGW: add delta to a per-step fully randomized green list."""
        vocab_size = row_logits.numel()
        if vocab_size == 0:
            return

        green_size = int(cfg.gamma * vocab_size)
        if green_size <= 0:
            return
        if green_size >= vocab_size:
            row_logits.add_(cfg.delta)
            return
        green = torch.randperm(vocab_size, device=row_logits.device)[:green_size]
        row_logits.index_add_(0, green, torch.full((green_size,), cfg.delta, device=row_logits.device, dtype=row_logits.dtype))