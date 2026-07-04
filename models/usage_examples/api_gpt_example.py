# models/usage_examples/api_gpt_example.py

import os
from models.loader import load_model, list_available_models

def main():
    # Get available models using new loader
    available_models = list_available_models()
    print("Available API Models:", available_models['api'][:5])  # Show first 5 models
    
    # Test a few representative models
    test_models = ['gpt-4o', 'claude-3-5-sonnet-latest', 'gemini-1-5-flash']
    
    for model_name in test_models:
        if model_name in available_models['api']:
            print(f"\nTesting {model_name}")
            try:
                model, parameters = load_model(model_name)
                prompt = "How are you?"
                response = model.generate(prompt=prompt, **parameters)
                print(f"Prompt: {prompt}\nResponse: {response}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        else:
            print(f"Model {model_name} not available")

if __name__ == "__main__":
    main()