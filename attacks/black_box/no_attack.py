# attacks/black_box/no_attack.py
from attacks.base_attack import BaseAttack

class NoAttack(BaseAttack):
    """
    Baseline attack that applies no modifications to prompts.
    Used for evaluating inherent model safety without any attack.
    """

    def __init__(self, target_model, auxiliary_models=None, **kwargs):
        super().__init__(target_model, auxiliary_models)
        
    def attack(self, clean_prompt: str, **kwargs):
        """Return prompt without modification"""
        return 0, clean_prompt  # Return (query_count, prompt)
        
    def generate_attack_prompts(self, prompts, **kwargs):
        """Return prompts without modification"""
        if isinstance(prompts, str):
            return [prompts]
        return prompts