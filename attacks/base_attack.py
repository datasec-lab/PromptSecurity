from typing import Union, List, Dict, Optional


class Message(Dict):
    role: str
    content: str


class BaseAttack:
    def __init__(self, target_model, auxiliary_models: Optional[Dict[str, object]] = None):
        """
        Base class for all attack methods.

        Parameters:
        - target_model: The model being attacked.
        - auxiliary_models: Optionally, a dictionary of auxiliary models (e.g., for crafting prompts).
        """
        self.target_model = target_model
        self.auxiliary_models = auxiliary_models or {}
        self.query_count = 0

    def attack(self, clean_prompt: str, **kwargs) -> (int, str):
        """
        Perform the attack on the given prompt.

        Parameters:
        - clean_prompt: The original, unmodified input prompt.
        - kwargs: Additional parameters for the attack.

        Returns:
        - query_count: The number of queries made during the attack.
        - attack_prompt: The adversarially crafted prompt.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_query_count(self) -> int:
        """Retrieve the current query count."""
        return self.query_count


class ExampleAttack(BaseAttack):
    def attack(self, clean_prompt: str, flip_mode: str = "FWO", **kwargs) -> (int, str):
        """
        Example attack implementation that modifies the input prompt based on a flip mode.

        Parameters:
        - clean_prompt: The original, unmodified input prompt.
        - flip_mode: The mode for flipping the input prompt.
        - kwargs: Additional parameters.

        Returns:
        - query_count: The number of queries made during the attack.
        - attack_prompt: The adversarially crafted prompt.
        """
        self.query_count += 1

        # Example logic: Flip the prompt based on mode
        if flip_mode == "FWO":
            attack_prompt = ' '.join(clean_prompt.split()[::-1])
        elif flip_mode == "FCW":
            attack_prompt = ' '.join([word[::-1] for word in clean_prompt.split()])
        else:
            attack_prompt = clean_prompt[::-1]

        return self.query_count, attack_prompt
