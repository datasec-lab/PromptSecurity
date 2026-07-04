# defenses/defenses/no_defense.py
from defenses.base_defended_model import BaseDefendedModel

class NoDefense(BaseDefendedModel):
    """
    Baseline defense that applies no protection measures.
    Used for evaluating inherent model safety without any defense mechanisms.
    """

    def __init__(self, target_model, **kwargs):
        super().__init__(target_model)
        
    def defend_input(self, prompt: str) -> str:
        """Pass through input without modification"""
        return prompt
        
    def defend_output(self, output: str) -> str:
        """Pass through output without modification"""
        return output