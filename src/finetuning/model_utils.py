from huggingface_hub import repo_exists, snapshot_download, HfFileSystem
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob
from src.utils import free_memory

def resize_model_if_needed(tokenizer, model):
    """
    Resizes the model's embedding layer if the tokenizer's vocabulary size
    is larger than the current embedding layer. Useful when using chat template.
    """
    # Get tokenizer and model vocabulary sizes
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = model.get_input_embeddings().weight.size(0)

    # Check if resizing is needed
    if tokenizer_vocab_size > model_vocab_size:
        print(
            f"Resizing model embeddings from {model_vocab_size} to {tokenizer_vocab_size}."
        )
        model.resize_token_embeddings(tokenizer_vocab_size)
        model.tie_weights()

    return model


def evaluate_previous_checkpoints(
    finetuning_config,
    model_output_dir: str,
    evaluation,
    tokenizer
):
    """
    Evaluates all previous checkpoints for a finetuning run.
    """

    # Get all checkpoint directories
    push_to_hub = finetuning_config.training_args.get("push_to_hub", False)

    if push_to_hub:
        if not repo_exists(model_output_dir):
            return False

        fs = HfFileSystem()
        checkpoints = fs.glob(f"{model_output_dir}/checkpoint-*")

        # Download the repository
        snapshot_download(model_output_dir, local_dir=model_output_dir)

    else:
        checkpoints = glob.glob(f"{model_output_dir}/checkpoint-*")

    if len(checkpoints) == 0:
        return False
    
    for checkpoint in checkpoints:
           
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
        
        checkpoint = checkpoint.split("/")[-1]
        ckpt = checkpoint.split("-")[-1]
        
        evaluation.eval_model(model, tokenizer, finetuning_config, f"finetuning-{ckpt}")

        del model
        free_memory()      

    return True