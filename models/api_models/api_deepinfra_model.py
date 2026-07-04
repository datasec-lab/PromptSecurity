# models/api_models/api_deepinfra_model.py
# see https://deepinfra.com/
from .api_base import APIModel
from openai import OpenAI

class APIDeepInfraModel(APIModel):
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key)
        self.model_name = model_name
        self.openai = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def generate(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content.strip()