from typing import List

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from robust_fp.utils import free_memory
from .watermark_processor import WatermarkBase
from ..watermark_detector import WatermarkDetector

class KGWWatermark(WatermarkDetector):
    def __init__(
        self,
        vocab: List[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        tokenizer: AutoTokenizer = None,
        fast_init: bool = True, # Use GPU for initialization
    ):
                
        print("Initializing KGWWatermark")                
                
        device = "cuda" # For speed
        if fast_init:
            init_device = "cuda"
        else:
            init_device = "cpu"
                
        self.type = "kgw"
        watermark_base = WatermarkBase(
            vocab=vocab,
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            device=init_device,  
        )
        self.kgw_device = device
        self.device = device
        self.k = watermark_base.context_width

        self.greenlist_masks = torch.full(
            (self.k * watermark_base.vocab_size, watermark_base.vocab_size),
            fill_value=False,
            dtype=bool,
            device=init_device,
        )
        for i in tqdm(range(self.greenlist_masks.shape[0])):

            greenlist_ids = watermark_base._get_greenlist_ids(
                torch.tensor([0] * (self.k - 1) + [i], dtype=torch.long, device=init_device)
            )
            self.greenlist_masks[i, greenlist_ids] = True
            
        self.greenlist_masks = self.greenlist_masks.to(device)


        # save watermark base parameters
        self.vocab = watermark_base.vocab
        self.vocab_size = watermark_base.vocab_size
        self.gamma = watermark_base.gamma
        self.delta = watermark_base.delta
        self.seeding_scheme = watermark_base.seeding_scheme
        self.hash_key = watermark_base.hash_key
        self.select_green_tokens = watermark_base.select_green_tokens

        if tokenizer is not None and seeding_scheme == "simple_1":
            # remove special tokens from greenlists
            if tokenizer.eos_token_id is not None:
                self.greenlist_masks[:, tokenizer.eos_token_id] = False
                self.greenlist_masks[tokenizer.eos_token_id, :] = False
            if tokenizer.bos_token_id is not None:
                self.greenlist_masks[:, tokenizer.bos_token_id] = False
                self.greenlist_masks[tokenizer.bos_token_id, :] = False
            if tokenizer.pad_token_id is not None:
                self.greenlist_masks[:, tokenizer.pad_token_id] = False
                self.greenlist_masks[tokenizer.pad_token_id, :] = False
            if tokenizer.unk_token_id is not None:
                self.greenlist_masks[:, tokenizer.unk_token_id] = False
                self.greenlist_masks[tokenizer.unk_token_id, :] = False

        print("Done initializing KGWWatermark")
        free_memory()

    def is_argmax(self):
        return False
        
    def watermark_logits(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.FloatTensor:
        """Returns watermarked logits to be used as distillation target.
        
        Note that this implementation assumes:
         - all inputs have the same length without padding 
         - doesnt ignore repetition
        """
        
        input_ids_device = input_ids.device
        
        input_ids = input_ids.to(self.kgw_device)
        logits = logits.to(self.kgw_device)
    
        hashes = torch.sum(
            input_ids.unfold(-1, self.k, 1), dim=-1
        )  # (batch, seq_len - k + 1)
        
        mask = self.greenlist_masks[hashes]  # (batch, seq_len - k + 1, vocab_size)
        # tokenizer vocab size and model outputs vocab size may be different
        logits[..., self.k - 1 :, : mask.shape[-1]][mask] += self.delta
        
        logits = logits.to(input_ids_device)
        
        return logits
    
    def watermark_logits_diff(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.FloatTensor:
        
        input_ids_device = input_ids.device
        
        input_ids = input_ids.to(self.kgw_device)
        logits = logits.to(self.kgw_device)
    
        hashes = torch.sum(
            input_ids.unfold(-1, self.k, 1), dim=-1
        )
        
        mask = self.greenlist_masks[hashes]
        
        delta = torch.zeros_like(logits)
        delta[..., self.k - 1 :, : mask.shape[-1]][mask] = self.delta
        delta = delta.to(input_ids_device)
        
        return delta

    def detect(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        attention_mask: torch.FloatTensor = None,  # (batch, seq_len)
        score_only: bool = False,
        pvalues_only: bool = True,
    ) -> torch.FloatTensor:
        """Returns z-scores."""
        
        assert not (score_only and pvalues_only), "Only one of score_only and pvalues_only can be True."
        
        input_ids_device = input_ids.device
        
        input_ids = input_ids.to(self.kgw_device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        attention_mask = attention_mask.to(self.kgw_device)

        # Compute hashes
        hashes = torch.sum(
            input_ids.unfold(-1, self.k, 1), dim=-1
        )  # (batch, seq_len - k + 1)
        
        if input_ids.shape[0]*input_ids.shape[1] <= 500000:
            
            # Get the mask
            mask = self.greenlist_masks[hashes]  # (batch, seq_len - k + 1, vocab_size)
            batch_size, T_plus_one = mask.shape[:2]
            T = T_plus_one - 1  # seq_len - k
            
            # Create indices for batch and time dimensions
            batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, T).to(input_ids.device)
            time_idx = torch.arange(T).unsqueeze(0).expand(batch_size, -1).to(input_ids.device)
            token_idx = input_ids[:, self.k:]  # (batch_size, T)

            # Extract mask values using advanced indexing
            mask_values = mask[batch_idx, time_idx, token_idx]  # (batch_size, T)
            
            # Mask repetition of k-grams
            repetition_mask = mask_k_gram_repetition(input_ids, self.k)  # (batch, seq_len)
            reshaped_repetition_mask = repetition_mask[:, self.k:] 
            reshaped_attention_mask = attention_mask[:, self.k:]
            ignored_tokens_mask = reshaped_repetition_mask * reshaped_attention_mask
            mask_values = mask_values * ignored_tokens_mask # (batch, T)

            T = ignored_tokens_mask.sum(dim=1)
            # Sum over the time dimension to get z-scores for each batch
            zscore = mask_values.sum(dim=1)  # (batch_size,)
            pvalue = self._compute_p_value(zscore, T)

            zscore = (zscore - self.gamma * T) / torch.sqrt(self.gamma * T * (1 - self.gamma))
        else:  #Use a memory friendly version that avoids creating the (B,T,V) tensor
            # hashes: (B, T+1) from your code
            # align hashes with tokens at positions k..L-1
            hashes_t = hashes[:, :-1]                  # (B, T)
            tokens_t = input_ids[:, self.k:]           # (B, T)

            B, T = tokens_t.shape
            flat_hash = hashes_t.reshape(-1)           # (B*T,)
            flat_tok  = tokens_t.reshape(-1)           # (B*T,)

            # greenlist_masks: (num_hashes, vocab_size) -> bool/uint8/float
            # direct 2D gather without expanding to (B,T,V)
            flat_vals = self.greenlist_masks[flat_hash, flat_tok]   # (B*T,)
            mask_values = flat_vals.view(B, T)                      # (B, T)

            # Mask repetition of k-grams
            repetition_mask = mask_k_gram_repetition(input_ids, self.k)  # (batch, seq_len)
            reshaped_repetition_mask = repetition_mask[:, self.k:] 
            reshaped_attention_mask = attention_mask[:, self.k:]
            ignored_tokens_mask = reshaped_repetition_mask * reshaped_attention_mask
            mask_values = mask_values * ignored_tokens_mask

            T_eff = ignored_tokens_mask.sum(dim=1)
            z = mask_values.sum(dim=1)
            pvalue = self._compute_p_value(z, T_eff)
            zscore = (z - self.gamma*T_eff) / torch.sqrt(self.gamma*T_eff*(1 - self.gamma))

        zscore = zscore.to(input_ids_device)
        pvalue = pvalue.to(input_ids_device)

        if pvalues_only:
            return pvalue

        if score_only:
            return zscore # (batch_size,)
        
        out = {"z_score": zscore, "p_value": pvalue}

        return out
    
    def _compute_p_value(
        self,
        observed_count: torch.Tensor,  # (batch_size,)
        T: float,              
    ) -> torch.Tensor:
        """Computes the p-value for the observed counts using the normal approximation.

        Args:
            observed_count (torch.Tensor): Tensor of observed counts. Shape: (batch_size,)
            T (torch.Tensor): Tensor of the number of trials. Shape: (batch_size,)

        Returns:
            torch.Tensor: Tensor of p-values. Shape: (batch_size,)
        """
        # Ensure tensors are of float type
        observed_count = observed_count.float()
        T = T

        # Compute mean and standard deviation
        mean = self.gamma * T
        std = torch.sqrt(self.gamma * (1 - self.gamma) * T)

        # Compute z-scores
        z = (observed_count - mean) / std

        # Have to use normal approximation as torch cant do exact
        p_values = 0.5 * torch.erfc(z / torch.sqrt(torch.tensor(2.0, device=z.device)))

        return p_values
    
    def get_config(self):
        config = {
            "type": "kgw",
            "k": self.k,
            "gamma": self.gamma,
            "delta": self.delta,
            "seeding_scheme": self.seeding_scheme,
            "kgw_device": self.kgw_device,
        }
        return config

    def get_name(self):
        k = self.k
        if self.seeding_scheme == "simple_0":
            k = 0
        name = f"kgw_k{k}_gamma{self.gamma}_delta{self.delta}"
        return name
    
def mask_k_gram_repetition(input_ids: torch.LongTensor, k: int) -> torch.BoolTensor:
    """
    Masks the repetition of k-grams in each sequence of a batch.
    
    Args:
        input_ids (torch.LongTensor): Tensor of shape (batch, seq_len).
        k (int): The size of the k-gram context.
        
    Returns:
        torch.BoolTensor: A boolean tensor of shape (batch, seq_len), 
                          where True means the token is not part of a repeated k-gram.
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.ones_like(input_ids, dtype=torch.bool)

    for batch_idx in range(batch_size):
        seen_kgrams = set()
        sequence = input_ids[batch_idx].tolist()
        
        for i in range(seq_len - k):
            kgram = tuple(sequence[i:i + k+1])
            if kgram in seen_kgrams:
                mask[batch_idx, i + k] = False
            else:
                seen_kgrams.add(kgram)
    
    return mask
