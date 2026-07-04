# models/base_model.py

from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass