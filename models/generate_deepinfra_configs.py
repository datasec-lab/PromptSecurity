# generate_deepinfra_configs.py

import os
import json
from models.model_registry import SUPPORTED_MODELS

def generate_deepinfra_configs():
    # Get the list of DeepInfra models
    deepinfra_models = [
        'openchat/openchat-3.6-8b',
        'nvidia/Nemotron-4-340B-Instruct',
        'mistralai/Mixtral-8x22B-Instruct-v0.1',
        'mistralai/Mistral-7B-Instruct-v0.3',
        'microsoft/WizardLM-2-7B',
        'microsoft/Phi-3-medium-4k-instruct',
        'meta-llama/Llama-2-70b-chat-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'meta-llama/Llama-2-7b-chat-hf',
        'google/gemma-2-9b-it',
        'google/gemma-2-27b-it',
        'google/gemma-1.1-7b-it',
        'deepinfra/airoboros-70b',
        'cognitivecomputations/dolphin-2.9.1-llama-3-70b',
        'cognitivecomputations/dolphin-2.6-mixtral-8x7b',
        'Sao10K/L3.1-70B-Euryale-v2.2',
        'Sao10K/L3-70B-Euryale-v2.1',
        'Qwen/Qwen2.5-7B-Instruct',
        'Qwen/Qwen2-7B-Instruct',
        'Qwen/Qwen2-72B-Instruct',
        'NousResearch/Hermes-3-Llama-3.1-405B',
        'HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1',
        'Gryphe/MythoMax-L2-13b-turbo',
        'Gryphe/MythoMax-L2-13b',
        'Austism/chronos-hermes-13b-v2',
        '01-ai/Yi-34B-Chat',
        'microsoft/WizardLM-2-8x22B',
        'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'meta-llama/Llama-3.2-90B-Vision-Instruct',
        'Qwen/Qwen2.5-72B-Instruct',
        'nvidia/Llama-3.1-Nemotron-70B-Instruct',
        'meta-llama/Meta-Llama-3.1-405B-Instruct',
        'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'qwen/Qwen2.5-72B-Instruct',
        'qwen/Qwen2.5-Coder-32B-Instruct',]

    # Directory to save the configs
    config_dir = os.path.join('usage_examples', 'configs', 'api')

    # Create the directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)

    for model_name in deepinfra_models:
        # Build the config
        config = {
            "model_type": "api",
            "model_name": model_name,
            "parameters": {
                "max_tokens": 1024,
                "temperature": 0.7
            }
        }

        # Replace '/' with '_' to create a valid filename
        filename = model_name.replace('/', '_') + '.json'
        config_path = os.path.join(config_dir, filename)

        # Write the config to a JSON file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Generated config for {model_name} at {config_path}")

if __name__ == "__main__":
    generate_deepinfra_configs()