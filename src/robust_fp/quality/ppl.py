import torch    
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from robust_fp.utils import free_memory


def _load_ppl_model(ppl_model_name):
    """Load a perplexity model."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(ppl_model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
    
    return model, tokenizer

def compute_ppl(ppl_model_name, prompts, completions, batch_size):
    
    model, tokenizer = _load_ppl_model(ppl_model_name)
    ppls = _compute_ppl(model, tokenizer, prompts, completions, batch_size)
    
    return ppls
    
def _compute_ppl(model, tokenizer, prompts, completions, batch_size):
    """Compute perplexities under `ppl_model_name`."""
    
    device = model.device

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size)):
            
            prompt_text, completion = prompts[i:i + batch_size], completions[i:i + batch_size]
            s = [f"{p} {c}" for p, c in zip(prompt_text, completion)]
            
            encodings = tokenizer(
                s,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)

            encoded_batch = encodings["input_ids"]
            attn_mask = encodings["attention_mask"]

            labels = encoded_batch

            # Trim to 250 tokens if too long
            if encoded_batch.shape[1] > 250:
                encoded_batch = encoded_batch[:, -250:]
                attn_mask = attn_mask[:, -250:]
                labels = labels[:, -250:]
                

            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            # Compute entropy
            probs = out_logits.softmax(dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropy = entropy.mean(dim=-1)
            ppls += entropy.tolist()

            free_memory()

    return ppls