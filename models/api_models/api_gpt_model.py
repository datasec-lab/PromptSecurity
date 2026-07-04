# models/api_models/api_gpt_model.py

from .api_base import APIModel
from openai import OpenAI

class APIGPTModel(APIModel):
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt, **kwargs):
        """
        Generate a response based on the given prompt.

        Parameters:
        - prompt (str or list): The input prompt for the model. Can be a string or list of message dicts.
        - stream (bool): If True, returns a generator for streaming responses.
                         If False, returns the complete response as a string.
        - **kwargs: Additional parameters for the API call.

        Returns:
        - str or generator: The generated response as a string or a generator for streaming.
        """
        # Handle different prompt formats
        if isinstance(prompt, list):
            # prompt is already a list of message dicts
            messages = prompt
        elif isinstance(prompt, str):
            # Convert string prompt to messages format
            if self.model_name == "o1-preview" or self.model_name == "o1-mini":
                messages = [
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

        # Call the ChatCompletion endpoint with or without streaming
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=60,
            **kwargs
        )

        return response.choices[0].message.content.strip()
