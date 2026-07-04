# usage_examples/local_huggingface_example.py

import os
import sys
import torch
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.local_models.local_huggingface_model import LocalHuggingFaceModel

def main():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    config_dir = os.path.join(os.path.dirname(__file__), 'configs', 'local')
    config_filename = model_name.replace('/', '_') + '.json'
    config_path = os.path.join(config_dir, config_filename)

    if not os.path.exists(config_path):
        print(f"Configuration file for {model_name} not found. Skipping.")
        return
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get configuration parameters
    init_params = config.get("init_parameters", {})
    model_type = init_params.get("model_type", "causal")
    device = init_params.get("device", "cuda")
    torch_dtype = init_params.get("torch_dtype", "auto")
    load_in_4bit = init_params.get("load_in_4bit", False)
    device_map = init_params.get("device_map", None)

    # Initialize the model
    print(f"Loading model: {model_name} with model_type: {model_type}")
    model = LocalHuggingFaceModel(
        model_name_or_path=model_name,
        device=device,
        torch_dtype=torch_dtype,
        model_type=model_type,
        load_in_4bit=load_in_4bit,
        device_map=device_map
    )


    # Test generate method
    prompt = "How are you?"
    print("\nTesting `generate` method...")
    try:
        response = model.generate(prompt=prompt, max_new_tokens=50)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error in `generate`: {e}")

    # Test get_tokenizer method
    print("\nTesting `get_tokenizer` method...")
    try:
        tokenizer = model.get_tokenizer()
        print("Tokenizer loaded successfully.")
        print(f"Tokenizer vocab size: {len(tokenizer.get_vocab())}")
    except Exception as e:
        print(f"Error in `get_tokenizer`: {e}")

    # Test get_embeddings method
    print("\nTesting `get_embeddings` method...")
    try:
        embeddings = model.get_embeddings(prompt)
        print(f"Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error in `get_embeddings`: {e}")

    # Test get_prediction_scores method
    print("\nTesting `get_prediction_scores` method...")
    try:
        logits = model.get_prediction_scores(prompt)
        print(f"Logits shape: {logits.shape}")
    except Exception as e:
        print(f"Error in `get_prediction_scores`: {e}")

if __name__ == "__main__":
    main()