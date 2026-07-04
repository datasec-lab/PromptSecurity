# models/local_models/local_base.py

from models.base_model import BaseModel

class LocalModel(BaseModel):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("Subclasses should implement this method.")

    def get_tokenizer(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_embeddings(self, input_text: str):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_prediction_scores(self, input_text: str):
        raise NotImplementedError("Subclasses should implement this method.")