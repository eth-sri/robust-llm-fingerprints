import torch
from torch.distributions import Normal
from transformers import AutoTokenizer, LogitsProcessor

from src.watermarks.watermark_detector import WatermarkDetector
from typing import List, Optional

from tqdm.auto import tqdm

def additive_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.sum().item()

class KGWWatermarkToken(WatermarkDetector):
    def __init__(
        self,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        tokenizer: AutoTokenizer = None,
        special_tokens: List[str] = None,
        force_key: Optional[int] = None,
    ):
                
        if seeding_scheme != "simple_1":
            raise NotImplementedError("Only seeding_scheme='simple_1' is supported.")
                                
        device = "cuda"
        self.type = "kgw"
        self.kgw_device = device
        self.device = device
        
        vocab_size = len(tokenizer.get_vocab())
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.k = 1
        self.special_tokens = [tokenizer.vocab[token] for token in special_tokens] if special_tokens is not None else None
        self.length_green_list = int(vocab_size * gamma)
        
        self.force_key = force_key
        
        if self.force_key:
            keys = [self.force_key]
        else:
            keys = self.special_tokens
        self._init_greenlist_masks(keys)
        
            
        print(f"Each watermark token use the following keys: {self.special_tokens}")
        
    
    def set_force_key(self, force_key: int):
        self.force_key = force_key
        self._init_greenlist_masks([force_key])
        
    def _init_greenlist_masks(self, keys: List[int]):
        """The mask can encode up to 8 keys. We use bit manipulation to encode the keys.
        See the get_greenlist method to see how to read the mask efficiently."""
        
        self.rng = torch.Generator(device=self.device)
        
        self.greenlist_masks = torch.full(
            (self.k * self.vocab_size, self.vocab_size),
            fill_value=0,
            dtype=torch.uint8,
            device=self.device,
        )
        
        for key_idx, key in enumerate(keys):
            
            for i in tqdm(range(self.greenlist_masks.shape[0])):
                
                greenlist_ids = self._get_greenlist_ids(
                    torch.tensor([0] * (self.k - 1) + [i], dtype=torch.long, device=self.device), key
                )
                self.greenlist_masks[i, greenlist_ids] += 2**key_idx
                
        self.key_map = {key: key_idx for key_idx, key in enumerate(keys)}
        print(self.key_map)
                    
        
    def _seed_rng(self, input_ids: torch.LongTensor, key: int) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.k:
            raise ValueError(
                f"seeding_scheme requires at least a {self.k} token prefix to seed the RNG."
            )

        prf_key = additive_prf(
            input_ids[-self.k :], salt_key=key
        )
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor, key: int) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids.to(self.rng.device), key)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        greenlist_ids = vocab_permutation[:greenlist_size] 
    
        return greenlist_ids
    
    def is_argmax(self):
        return False
    
    def _get_keys(self, input_ids: torch.LongTensor, key: int):
        
        batch, _ = input_ids.shape
                                
        # --- Vectorized Special Token Check ---
        if key:
            key_mapped_value = 2**self.key_map[key]
            has_special = torch.ones(batch, dtype=torch.bool, device=input_ids.device)
            keys = torch.full((batch,), key_mapped_value, dtype=input_ids.dtype, device=input_ids.device)
        elif self.special_tokens is not None:
            special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token in self.special_tokens:
                special_mask |= (input_ids == token)
            # For each sequence, check if any special token is present.
            has_special = special_mask.any(dim=1)
            # For each batch element, pick the first occurring special token.
            keys = torch.full((batch,), 0, dtype=input_ids.dtype, device=input_ids.device)
            for token in self.special_tokens:
                key_mapped_value = 2**self.key_map[token]
                update_mask = ((input_ids == token).any(dim=1)) & (keys == 0)
                keys[update_mask] = key_mapped_value
        else:
            has_special = torch.ones(batch, dtype=torch.bool, device=input_ids.device)
            keys = torch.zeros(batch, dtype=input_ids.dtype, device=input_ids.device)
                                                                        
        return keys, has_special
    
    def get_greenlist(self, input_ids: torch.LongTensor, key: int = None) -> torch.LongTensor:

        keys, _ = self._get_keys(input_ids, key=key)
                
        hashes = torch.sum(
            input_ids.unfold(-1, self.k, 1), dim=-1
        ) 
        
        mask = self.greenlist_masks[hashes]  # (batch, seq_len - k + 1, vocab_size)
        mask = torch.bitwise_and(mask, keys.unsqueeze(1).unsqueeze(2)) > 0
        
        return mask
                
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
    
        greenlist_masks = self.get_greenlist(input_ids, key=None)   # (batch, seq_len-k, vocab_size)
            
        logits[..., self.k - 1 :, : greenlist_masks.shape[-1]][greenlist_masks] += self.delta
        
        logits = logits.to(input_ids_device)
        
        return logits
    
    def watermark_logits_diff(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.FloatTensor:
        delta = torch.zeros_like(logits)
        return delta

    def detect(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        attention_mask: torch.FloatTensor = None,  # (batch, seq_len)
        score_only: bool = False,
        pvalues_only: bool = True,
        detailed_output: bool = False,
    ) -> torch.FloatTensor:
        """Returns z-scores."""
        
        assert not (score_only and pvalues_only), "Only one of score_only and pvalues_only can be True."
        
        input_ids_device = input_ids.device
        
        input_ids = input_ids.to(self.kgw_device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        attention_mask = attention_mask.to(self.kgw_device)
        
        mask_values = None
                
        if self.force_key is not None:
            zscores, mask_values  = self._detect_with_key(input_ids, self.force_key, attention_mask)
            k = 1
        else:
            zscores = None
            for key in self.special_tokens:
                zscore, _ = self._detect_with_key(input_ids, key, attention_mask)
                if zscores is None:
                    zscores = zscore
                else:
                    zscores = torch.maximum(zscores, zscore)
  
                    
            k = len(self.special_tokens)
            
        p_values = self._compute_p_value_multiple_key(zscores, k)
        
        zscores = zscores.to(input_ids_device)
        p_values = p_values.to(input_ids_device)
        
        if pvalues_only:
            return p_values

        if score_only:
            return zscores # (batch_size,)

        out = {"z_score": zscores, "p_value": p_values}
        
        if detailed_output:
            out["mask"] = mask_values

        return out
    
    def _detect_with_key(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        key: int,
        attention_mask: torch.FloatTensor = None,  # (batch, seq_len)
    ) -> torch.FloatTensor:
        
        # Get the mask
        mask = self.get_greenlist(input_ids, key)  # (batch, seq_len - k + 1, vocab_size)
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

        zscore = (zscore - self.gamma * T) / torch.sqrt(self.gamma * T * (1 - self.gamma))

        return zscore, mask_values
    
    def _compute_p_value_multiple_key(
        self,
        zscores: torch.Tensor,  # (batch_size,)
        k: int,              
    ) -> torch.Tensor:
        """
        Computes the one-sided p-value for the maximum of k i.i.d. standard normal random variables.
        
        Given an observed maximum value, the p-value is defined as:
            p = 1 - (Phi(observed))^k,
        where Phi is the standard normal CDF.
        
        Args:
            observed_count (torch.Tensor): Tensor of observed counts. Shape: (batch_size,)
            T (torch.Tensor): Tensor of the number of trials. Shape: (batch_size,)

        Returns:
            torch.Tensor: Tensor of p-values. Shape: (batch_size,)
        """
        
        cdf = Normal(0, 1).cdf(zscores)
        p_value = 1 - cdf**k
        
        return p_value

    
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


class KGWWatermarkTokenLogitProcessor(KGWWatermarkToken, LogitsProcessor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        green_list = self.get_greenlist(input_ids, key=self.force_key) # batch, seq_len, vocab_size
        green_list = green_list[:,-1,:]
        scores[:,:green_list.shape[1]][green_list] += self.delta
        
        return scores