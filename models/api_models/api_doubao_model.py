# models/api_models/api_doubao_model.py

from .api_base import APIModel
from typing import List, Dict, Union

class APIDoubaoModel(APIModel):
    def __init__(self, api_key: str, model_name: str, endpoint_url: str = None):
        super().__init__(api_key)
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        
    def generate(self, prompt: Union[str, List[Dict]], **kwargs) -> str:
        """
        Generate a response based on the given prompt using Doubao API.
        
        Parameters:
        - prompt (str or list): The input prompt for the model. Can be a string or list of message dicts.
        - **kwargs: Additional parameters for the API call.
        
        Returns:
        - str: The generated response as a string.
        """
        try:
            # Import volcengine SDK
            try:
                from volcenginesdkarkruntime import Ark
            except ImportError:
                raise Exception("volcengine-python-sdk[ark] is required for Doubao API. Install with: pip install volcengine-python-sdk[ark]")
            
            # Handle different prompt formats
            if isinstance(prompt, list):
                # prompt is already a list of message dicts
                messages = prompt
            elif isinstance(prompt, str):
                # Convert string prompt to messages format
                messages = [
                    {"role": "user", "content": prompt}
                ]
            else:
                raise ValueError(f"Unsupported prompt type: {type(prompt)}")
            
            # Initialize Ark client with custom endpoint if provided
            if self.endpoint_url:
                client = Ark(api_key=self.api_key, base_url=self.endpoint_url, timeout=60)
            else:
                client = Ark(api_key=self.api_key, timeout=60)
            
            # Prepare parameters
            completion_params = {
                "model": self.model_name,
                "messages": messages,
            }
            
            # Add optional parameters with parameter mapping
            if "temperature" in kwargs:
                completion_params["temperature"] = kwargs["temperature"]
            
            # Handle both max_tokens and max_new_tokens
            if "max_new_tokens" in kwargs:
                completion_params["max_tokens"] = kwargs["max_new_tokens"]
            elif "max_tokens" in kwargs:
                completion_params["max_tokens"] = kwargs["max_tokens"]
            
            if "top_p" in kwargs:
                completion_params["top_p"] = kwargs["top_p"]
            
            # Create completion
            completion = client.chat.completions.create(**completion_params)
            
            # Extract response
            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            else:
                raise Exception("No response generated from Doubao API")
                
        except Exception as e:
            raise Exception(f"Doubao API error: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about the Doubao model."""
        return {
            "model_name": self.model_name,
            "provider": "ByteDance",
            "model_type": "api",
            "endpoint": self.endpoint_url
        }