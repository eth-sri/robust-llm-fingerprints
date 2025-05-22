import torch
from torch.distributions import Normal
from transformers import AutoTokenizer, LogitsProcessor
from tqdm.auto import tqdm

from src.watermarks.watermark_detector import WatermarkDetector


def additive_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.sum().item()


class KGWWatermarkTokenBoundary(WatermarkDetector):
    """A fork of KGW watermark that parses the input to be active only in between watermark tokens."""

    def __init__(
        self,
        gamma: float = 0.5,
        delta: float = 2.0,
        alpha: float = 0.1,
        seeding_scheme: str = "simple_1",
        tokenizer: AutoTokenizer = None,
        opening_token: str = "<wm>",
        closing_token: str = "</wm>",
        invert: bool = False,
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
        self.alpha = alpha

        self.key = 15485863

        self.invert = invert

        assert opening_token in tokenizer.vocab, (
            f"Opening token {opening_token} not in tokenizer."
        )
        assert closing_token in tokenizer.vocab, (
            f"Closing token {closing_token} not in tokenizer."
        )
        self.opening_token = tokenizer.vocab[opening_token]
        self.closing_token = tokenizer.vocab[closing_token]

        print(
            f"Opening token: {self.opening_token}, Closing token: {self.closing_token}"
        )

        self._init_greenlist_masks()

    def _init_greenlist_masks(self):
        """The mask can encode up to 8 keys. We use bit manipulation to encode the keys.
        See the get_greenlist method to see how to read the mask efficiently."""

        self.rng = torch.Generator(device=self.device)

        self.greenlist_masks = torch.full(
            (self.k * self.vocab_size, self.vocab_size),
            fill_value=False,
            dtype=bool,
            device=self.device,
        )

        for i in tqdm(range(self.greenlist_masks.shape[0])):
            greenlist_ids = self._get_greenlist_ids(
                torch.tensor(
                    [0] * (self.k - 1) + [i], dtype=torch.long, device=self.device
                ),
            )
            self.greenlist_masks[i, greenlist_ids] = True

    def _seed_rng(self, input_ids: torch.LongTensor, key: int) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.k:
            raise ValueError(
                f"seeding_scheme requires at least a {self.k} token prefix to seed the RNG."
            )

        prf_key = additive_prf(input_ids[-self.k :], salt_key=key)
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(
            prf_key % (2**64 - 1)
        )  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids.to(self.rng.device), self.key)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        greenlist_ids = vocab_permutation[:greenlist_size]

        return greenlist_ids

    def is_argmax(self):
        return False

    def _parse_wmtoken_mask(self, input_ids: torch.LongTensor):
        """We mask the tokens between the watermark tokens. Note that we assume proper formatting."""

        batch_size, seq_length = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        opening, closing = self.opening_token, self.closing_token

        for i in range(batch_size):
            seq = input_ids[i]
            # Find indices of opening and closing tokens
            opening_indices = (seq == opening).nonzero(as_tuple=True)[0]
            closing_indices = (seq == closing).nonzero(as_tuple=True)[0]

            # Check if exactly one opening and at most one closing token exists
            if opening_indices.numel() == 1 and closing_indices.numel() <= 1:
                open_idx = opening_indices.item()

                if closing_indices.numel() == 0:
                    close_idx = -1
                else:
                    close_idx = closing_indices.item()

                # Only set mask if opening comes before closing
                if close_idx == -1:
                    mask[i, open_idx + 1 :] = True
                elif open_idx < close_idx:
                    mask[i, open_idx + 1 : close_idx] = True
                    
                    
                #print(f"seq: {i}, open_idx: {open_idx}, close_idx: {close_idx}")

        return mask

    def get_greenlist(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        hashes = torch.sum(input_ids.unfold(-1, self.k, 1), dim=-1)

        mask = self.greenlist_masks[hashes]  # (batch, seq_len - k + 1, vocab_size)
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

        greenlist_masks = self.get_greenlist(
            input_ids
        )  # (batch, seq_len-k, vocab_size)
        wmtoken_mask = self._parse_wmtoken_mask(input_ids)

        if self.invert:
            wmtoken_mask = ~wmtoken_mask

        greenlist_masks_in_wmtoken = greenlist_masks & wmtoken_mask.unsqueeze(-1)
        logits[..., self.k - 1 :, : greenlist_masks.shape[-1]][
            greenlist_masks_in_wmtoken
        ] += self.delta

        greenlist_masks_out_wmtoken = greenlist_masks & ~wmtoken_mask.unsqueeze(-1)
        logits[..., self.k - 1 :, : greenlist_masks.shape[-1]][
            greenlist_masks_out_wmtoken
        ] -= self.delta * self.alpha

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

        greenlist_masks = self.get_greenlist(
            input_ids
        )  # (batch, seq_len-k, vocab_size)
        delta = torch.zeros_like(logits)

        delta[..., self.k - 1 :, : greenlist_masks.shape[-1]][greenlist_masks] = (
            self.delta
        )
        delta = delta.to(input_ids_device)

        return delta

    def compute_watermark_distillation_loss(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        student_logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
        teacher_logits: torch.FloatTensor,  # (batch, seq_len, vocab_size),
        lambdas: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """We use distillation in between the watermark tokens and regularization outside."""

        input_ids_device = input_ids.device
        loss_fct = torch.nn.KLDivLoss(reduction="none", log_target=True)

        # Computing probs
        student_probs = torch.nn.functional.softmax(student_logits, dim=-1)
        teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)

        input_ids = input_ids.to(self.kgw_device)
        teacher_logits = teacher_logits.to(self.kgw_device)

        # Apllying watermark on teacher logits
        greenlist_masks = self.get_greenlist(
            input_ids
        )  # (batch, seq_len-k, vocab_size)
        wmtoken_mask = self._parse_wmtoken_mask(input_ids)
        
        # Count number of tokens in the mask
        num_tokens = torch.sum(wmtoken_mask, dim=-1)

        greenlist_masks_in_wmtoken = greenlist_masks & wmtoken_mask.unsqueeze(-1)
        watermarked_logits = teacher_logits
        watermarked_logits[..., self.k - 1 :, : greenlist_masks.shape[-1]][
            greenlist_masks_in_wmtoken
        ] += self.delta
        watermarked_logits = watermarked_logits.to(input_ids_device)

        # Distillation loss
        loss = loss_fct(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.log_softmax(watermarked_logits, dim=-1),
        )
        loss = torch.sum(loss, dim=(-1, -2))
        loss = torch.mean(loss * lambdas.view(-1)) / (student_logits.shape[1])

        # Positive tv loss
        greenlist_masks_out_wmtoken = greenlist_masks & ~wmtoken_mask.unsqueeze(-1)
        average_non_wm_length = torch.mean(torch.sum(~wmtoken_mask, dim=-1).float())
        positive_tv = torch.sum(
            greenlist_masks_out_wmtoken
            * torch.maximum(
                student_probs - teacher_probs,
                torch.tensor(0.0, device=student_probs.device),
            )[:,:, : greenlist_masks.shape[-1]], # Special tokens should not be included
            dim=(-1, -2),
        )
        positive_tv = torch.mean(positive_tv * lambdas.view(-1)) / average_non_wm_length

        loss = loss + positive_tv

        return loss

    def detect(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        attention_mask: torch.FloatTensor = None,  # (batch, seq_len)
        score_only: bool = False,
        pvalues_only: bool = True,
    ) -> torch.FloatTensor:
        """Returns z-scores."""

        assert not (score_only and pvalues_only), (
            "Only one of score_only and pvalues_only can be True."
        )

        input_ids_device = input_ids.device

        input_ids = input_ids.to(self.kgw_device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        attention_mask = attention_mask.to(self.kgw_device)

        zscores, _ = self._detect(input_ids, attention_mask)

        p_values = self._compute_p_value(zscores)

        zscores = zscores.to(input_ids_device)
        p_values = p_values.to(input_ids_device)

        if pvalues_only:
            return p_values

        if score_only:
            return zscores  # (batch_size,)

        out = {"z_score": zscores, "p_value": p_values}

        return out

    def _detect(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        attention_mask: torch.FloatTensor = None,  # (batch, seq_len)
    ) -> torch.FloatTensor:
        # Get the mask
        mask = self.get_greenlist(input_ids)  # (batch, seq_len - k + 1, vocab_size)
        batch_size, T_plus_one = mask.shape[:2]
        T = T_plus_one - 1  # seq_len - k

        # Create indices for batch and time dimensions
        batch_idx = (
            torch.arange(batch_size).unsqueeze(1).expand(-1, T).to(input_ids.device)
        )
        time_idx = (
            torch.arange(T).unsqueeze(0).expand(batch_size, -1).to(input_ids.device)
        )
        token_idx = input_ids[:, self.k :]  # (batch_size, T)

        # Extract mask values using advanced indexing
        mask_values = mask[batch_idx, time_idx, token_idx]  # (batch_size, T)

        # Mask repetition of k-grams
        repetition_mask = mask_k_gram_repetition(input_ids, self.k)  # (batch, seq_len)
        reshaped_repetition_mask = repetition_mask[:, self.k :]
        reshaped_attention_mask = attention_mask[:, self.k :]
        ignored_tokens_mask = reshaped_repetition_mask * reshaped_attention_mask
        mask_values = mask_values * ignored_tokens_mask  # (batch, T)

        T = ignored_tokens_mask.sum(dim=1)

        # Sum over the time dimension to get z-scores for each batch
        zscore = mask_values.sum(dim=1)  # (batch_size,)

        zscore = (zscore - self.gamma * T) / torch.sqrt(
            self.gamma * T * (1 - self.gamma)
        )

        return zscore, mask_values

    def _compute_p_value(
        self,
        zscores: torch.Tensor,  # (batch_size,)
    ) -> torch.Tensor:
        cdf = Normal(0, 1).cdf(zscores)
        p_value = 1 - cdf
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
            kgram = tuple(sequence[i : i + k + 1])
            if kgram in seen_kgrams:
                mask[batch_idx, i + k] = False
            else:
                seen_kgrams.add(kgram)

    return mask


class KGWWatermarkTokenLogitProcessor(KGWWatermarkTokenBoundary, LogitsProcessor):
    def __init__(self, full, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full = full
        
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        

        useful_input_ids = input_ids[:, -self.k:]
        
        green_list = self.get_greenlist(useful_input_ids)  # batch, seq_len, vocab_size
        wmtoken_mask = self._parse_wmtoken_mask(input_ids)
        wmtoken_mask = wmtoken_mask[:, -self.k]

        if self.invert:
            wmtoken_mask = ~wmtoken_mask
            
        if self.full:
            wmtoken_mask = torch.zeros_like(wmtoken_mask, dtype=torch.bool) + 1

        green_list = green_list & wmtoken_mask.unsqueeze(-1)

        green_list = green_list[:, -1, :]
        
        green_mask = green_list.bool()                  # make sure it’s dtype=torch.bool
        scores[:, :green_mask.shape[1]][green_mask]   += self.delta

        return scores
