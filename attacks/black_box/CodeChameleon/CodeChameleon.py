from attacks.base_attack import BaseAttack
from .utils import complete_format
from .template import get_prompts


class CodeChameleon(BaseAttack):
    def __init__(self, target_model, target_model_parameters, encrypt_rule, prompt_style) -> None:
        super().__init__(target_model)
        self.target_model=target_model
        self.target_model_parameters = target_model_parameters
        self.encrypt_rule = encrypt_rule   #  'none', 'reverse', 'binary_tree', 'odd_even', 'length'
        self.prompt_style = prompt_style   #  'text' or 'code'



    def attack(self, original_prompt: str, **model_parameters) -> (int, str):
        prompts, original_queries = get_prompts(original_prompt, self.encrypt_rule, self.prompt_style)
        self.query_count = 1  # Count as 1 query for the crafted prompt
        return self.query_count, prompts