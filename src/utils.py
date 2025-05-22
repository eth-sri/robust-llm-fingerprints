import torch
import gc
from huggingface_hub import HfApi
import json
import requests
import yaml
import os

def set_neptune_env():
    
    path = "neptune_credentials.yaml"
    if not os.path.exists(path):
        return
    
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        os.environ["NEPTUNE_API_TOKEN"] = config["NEPTUNE_API_TOKEN"]
        os.environ["NEPTUNE_PROJECT"] = config["NEPTUNE_PROJECT"]

def send_telegram_notification(message):
    

    # Load the API token and chat ID from the config file
    try:
        config_path = "telegram_credentials.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        api_token = config["bot_api_token"]
        chat_id = config["chat_id"]
        
        url = f"https://api.telegram.org/bot{api_token}/sendMessage?chat_id={chat_id}&text={message}"
        print(requests.get(url).json()) # this sends the message
    except Exception as e:
        pass

def free_memory():
    """Free memory by running the garbage collector and emptying the cache."""
    gc.collect()
    torch.cuda.empty_cache()
    
def push_to_hub(repo_id: str, model, tokenizer, watermark_config: dict = None):
    model.push_to_hub(repo_id, use_temp_dir=True, private=True)
    tokenizer.push_to_hub(repo_id, use_temp_dir=True, private=True)

    file_path = "/tmp/watermark_config.json"
    with open(file_path, "w") as f:
        json.dump(watermark_config, f)

    if watermark_config is not None:

        api = HfApi()
        api.upload_file(
            path_or_fileobj=file_path,  
            path_in_repo="watermark_config.json",  
            repo_id=repo_id, 
            repo_type="model",
            commit_message="Upload watermark config",
        )
    
def sanitize_config_for_neptune(config_dict):
    
    if isinstance(config_dict, dict):
        new_config = {}
        for key, value in config_dict.items():
            new_config[key] = sanitize_config_for_neptune(value)
        return new_config
    
    if isinstance(config_dict, list):
        new_config = {}
        for i, value in enumerate(config_dict):
            new_config[i] = sanitize_config_for_neptune(value)
        return new_config
    
    if config_dict is None:
        return "None"
    
    return config_dict