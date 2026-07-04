# defense/base_defended_model.py

from models.base_model import BaseModel

class BaseDefendedModel(BaseModel):
    """
    A base class for 'defended' models, wrapping an underlying target_model.
    By default, generate() calls defend_input->target_model->defend_output.
    Subclasses must implement or override 'defend_input' and/or 'defend_output'.
    """

    def __init__(self, target_model):
        """
        :param target_model: The actual LLM we want to defend/wrap.
        """
        self.target_model = target_model
        self._fallback_events = []

    def _record_fallback(self, reason: str) -> None:
        if reason:
            self._fallback_events.append(reason)

    def clear_fallback_events(self) -> None:
        self._fallback_events = []

    def get_fallback_events(self):
        return list(self._fallback_events)

    def defend_input(self, prompt: str) -> str:
        """
        By default, no changes. Subclasses can override
        or define custom logic to sanitize or filter the prompt.
        """
        return prompt

    def defend_output(self, output: str) -> str:
        """
        By default, no changes. Subclasses can override
        or define custom logic to sanitize or block the model output.
        """
        return output

    def generate(self, prompt: str, **kwargs) -> str:
        # 1) defend input
        safe_prompt = self.defend_input(prompt)
        # 2) call real model
        raw_output = self.target_model.generate(safe_prompt, **kwargs)
        # 3) defend output
        final_output = self.defend_output(raw_output)
        return final_output

    # For all other advanced calls, pass them through:
    def compute_gradients(self, input_text: str, target_text: str, **kwargs):
        return self.target_model.compute_gradients(input_text, target_text, **kwargs)

    def compute_loss(self, input_text: str, target_text: str, **kwargs):
        return self.target_model.compute_loss(input_text, target_text, **kwargs)

    def get_tokenizer(self):
        return self.target_model.get_tokenizer()

    def get_embeddings(self, input_text: str):
        return self.target_model.get_embeddings(input_text)

    def get_prediction_scores(self, input_text: str):
        return self.target_model.get_prediction_scores(input_text)

    def generate_from_input_id_to_logits(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.target_model.generate_from_input_id_to_logits(
            input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def __getattr__(self, item):
        """
        If an attribute isn't found on this 'BaseDefendedModel',
        but does exist on 'self.target_model', forward calls.
        """
        if hasattr(self.target_model, item):
            return getattr(self.target_model, item)
        raise AttributeError(f"'BaseDefendedModel' object has no attribute '{item}'")
