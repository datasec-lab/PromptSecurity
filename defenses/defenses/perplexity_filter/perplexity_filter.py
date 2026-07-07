import os
import sys
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from defenses.base_defended_model import BaseDefendedModel


class PerplexityDefendedModel(BaseDefendedModel):
    """
    Perplexity-based input filter defense (Alon & Kamfonas, 2023).

    Computes GPT-2 perplexity of the input prompt. Prompts whose perplexity
    exceeds the threshold are rejected as likely adversarial (e.g. GCG-style
    adversarial suffix attacks). Prompts that pass are forwarded unchanged to
    the target model.

    The paper finds that ~90% of GCG adversarial prompts have PPL > 1000,
    while all relevant examples exceed PPL 200. The optimal simple-threshold
    value reported is ~1000.

    Reference: "Detecting Language Model Attacks with Perplexity"
               Alon & Kamfonas (2023), arXiv:2308.14132
    """

    def __init__(
        self,
        target_model,
        threshold: float = 1000.0,
        safe_msg: Optional[str] = None,
        gpt2_model_name: str = "gpt2",
        gpt2_device: str = "cpu",
    ):
        """
        Parameters
        ----------
        target_model : any model with a .generate(prompt, **kwargs) interface
        threshold    : PPL threshold; prompts above this are rejected (default 1000)
        safe_msg     : message returned for rejected prompts
        gpt2_model_name : HuggingFace model name for GPT-2 (default "gpt2")
        gpt2_device  : device for GPT-2 inference ("cpu" is sufficient)
        """
        super().__init__(target_model)
        self.threshold = threshold
        self.safe_msg = safe_msg or "I'm sorry, but I cannot help with that request."

        # GPT-2 is the perplexity model specified by Alon & Kamfonas (2023).
        # It runs independently of the target model, so this defense works with
        # both API and local target models.
        self._gpt2_device = torch.device(gpt2_device)
        self._gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_name)
        self._gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self._gpt2_model.to(self._gpt2_device)
        self._gpt2_model.eval()

    # ------------------------------------------------------------------
    # Perplexity computation (Alon & Kamfonas Eq. 2)
    # ------------------------------------------------------------------

    def compute_perplexity(self, text: str) -> Optional[float]:
        """
        Compute GPT-2 perplexity of `text`.

        PPL(x) = exp[ -1/t * sum_i log p(x_i | x_{<i}) ]

        Returns None for empty or single-token inputs (undefined perplexity).
        GPT-2 has a maximum context of 1024 tokens; longer inputs are truncated.
        """
        if not text or not text.strip():
            return None

        inputs = self._gpt2_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._gpt2_device)

        if inputs["input_ids"].shape[-1] < 2:
            return None

        with torch.no_grad():
            logits = self._gpt2_model(**inputs).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()

        token_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        )

        mean_nll = token_nll.mean()
        return float(torch.exp(mean_nll).item())

    def _is_adversarial(self, prompt: str) -> bool:
        """Return True if the prompt's GPT-2 PPL exceeds the threshold."""
        ppl = self.compute_perplexity(prompt)
        if ppl is None:
            return False
        return ppl > self.threshold

    # ------------------------------------------------------------------
    # Defense entry point
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Input filter: reject high-perplexity prompts before they reach the
        target model. Passes clean prompts through to target_model.generate().
        """
        if self._is_adversarial(prompt):
            return self.safe_msg
        return self.target_model.generate(prompt, **kwargs)
