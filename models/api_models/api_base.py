# models/api_models/api_base.py

from models.base_model import BaseModel

class APIModel(BaseModel):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def parameters(self):
        """Return empty iterator for API models (they don't have trainable parameters)."""
        return iter([])