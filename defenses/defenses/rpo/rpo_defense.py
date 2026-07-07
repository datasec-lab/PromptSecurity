from defenses.base_defended_model import BaseDefendedModel

# Pre-trained RPO suffix from the paper:
# "Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks"
# (Zhou et al., NeurIPS 2024). Optimized on Llama-2-7B-chat, transfers to other models.
_RPO_SUFFIX = (
    ":}</'])language seems scary-----------}] estimation >>begin &=&pgf"
    " degli ({\\ $|\\textwidthAI NOTE:"
)


class RobustOptDefendedModel(BaseDefendedModel):
    """
    Implements Robust Prompt Optimization (RPO) defense by appending a
    pre-trained adversarial suffix to every user prompt. The suffix was
    optimized to trigger refusal behaviour and transfers across models.
    No training or gradient computation is required at runtime.
    """

    def __init__(self, target_model, suffix: str = _RPO_SUFFIX):
        super().__init__(target_model)
        self.suffix = suffix

    def defend_input(self, prompt: str) -> str:
        return f"{prompt} {self.suffix}"
