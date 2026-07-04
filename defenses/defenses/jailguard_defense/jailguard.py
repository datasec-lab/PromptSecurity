# defenses/defenses/JailGuard/jailguard.py

import os
import sys
import copy
import pickle
import uuid
import spacy
import time

from tqdm import trange

from defenses.base_defended_model import BaseDefendedModel
from .utils.augmentations import text_aug_dict
from .utils.utils import (
    load_dirs,
    read_file_list,
    update_divergence,
    detect_attack
)

class JailGuardDefense(BaseDefendedModel):
    """
    A simplified 'JailGuard' defense that does NOT rely on any external dataset.
    It uses only the user's prompt, mutates it into multiple variants,
    queries the model for each, measures divergence among the responses,
    and blocks if suspicious.
    """

    def __init__(
        self,
        target_model,
        mutator: str = "PL",  # which text augmentation strategy from text_aug_dict
        number_of_variants: int = 8,
        threshold: float = 0.02,
        variant_save_dir: str = "./demo_case/variant",
        response_save_dir: str = "./demo_case/response",
        **kwargs
    ):
        """
        :param target_model: The underlying LLM instance we're wrapping.
        :param mutator: Which augmentation strategy to apply for mutation (e.g. "PL", "RR", etc.).
        :param number_of_variants: How many mutated variants to generate.
        :param threshold: Divergence threshold above which we block as an 'attack'.
        :param variant_save_dir: Directory where mutated prompts are saved (optional).
        :param response_save_dir: Directory where model responses to mutated prompts are saved (optional).
        :param kwargs: Any additional parameters from your config if needed.
        """
        super().__init__(target_model)
        self.mutator = mutator
        self.number_of_variants = number_of_variants
        self.threshold = threshold
        self.variant_save_dir = variant_save_dir
        self.response_save_dir = response_save_dir
        self.extra_params = kwargs  # store user-supplied config (if any)

        # Ensure directories exist; remove if you want purely in-memory
        os.makedirs(self.variant_save_dir, exist_ok=True)
        os.makedirs(self.response_save_dir, exist_ok=True)

        # Pre-load spaCy model so we only do it once
        try:
            self.spacy_model = spacy.load("en_core_web_md")
        except OSError:
            raise ValueError(
                "SpaCy model 'en_core_web_md' not installed. "
                "Please install or update code to use a different spaCy model."
            )

    def _detect_attack_for_prompt(self, prompt: str) -> bool:
        """
        Core JailGuard logic (no dataset):
         1) Mutate `prompt` into N variants, saving them (optional).
         2) Query `self.target_model` for each variant, saving responses (optional).
         3) Compute the divergence among responses.
         4) If the divergence > threshold => suspicious => True, else False.
        """

        # Optional: Clear old files from variant/response dirs
        for fname in os.listdir(self.variant_save_dir):
            os.remove(os.path.join(self.variant_save_dir, fname))
        for fname in os.listdir(self.response_save_dir):
            os.remove(os.path.join(self.response_save_dir, fname))

        # Step 1: Mutate the user’s prompt
        if self.mutator not in text_aug_dict:
            print(f"Unknown mutator '{self.mutator}' - using default 'PL'.")
            self.mutator = "PL"
        mutate_fn = text_aug_dict[self.mutator]

        # We treat the user prompt as a single string. The augmentation function
        # typically expects a list of lines => pass `[prompt]`.
        for i in range(self.number_of_variants):
            mutated_result_filename = f"{uuid.uuid4().hex[:6]}-{self.mutator}"
            mutated_lines = mutate_fn([prompt])  # returns a list of strings

            # Save mutated prompt to disk (if you want purely in-memory, remove this step):
            target_path = os.path.join(self.variant_save_dir, mutated_result_filename)
            with open(target_path, "w", encoding="utf-8") as f:
                f.writelines(mutated_lines)

        # Step 2: Query the model for each mutated variant
        variant_list, name_list = load_dirs(self.variant_save_dir)
        combined = sorted(zip(name_list, variant_list), key=lambda x: x[0])
        if not combined:
            # no variants => no analysis => assume safe
            return False

        name_list, variant_list = zip(*combined)
        for i, mutated_content in enumerate(variant_list):
            # mutated_content is a list of lines from that file
            if not isinstance(mutated_content, list):
                continue
            mutated_prompt = "\n".join(mutated_content)

            # Attempt to query the underlying model
            try:
                response = self.target_model.generate(mutated_prompt)
            except Exception as e:
                response = f"I cannot assist with that! [Exception: {str(e)}]"

            # Save the response if desired
            out_path = os.path.join(self.response_save_dir, name_list[i])
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(response)

        # Step 3: measure divergence among responses
        response_files = [
            os.path.join(self.response_save_dir, fname)
            for fname in os.listdir(self.response_save_dir)
            if os.path.isfile(os.path.join(self.response_save_dir, fname))
        ]
        output_list = read_file_list(response_files)

        # e.g. update_divergence returns (max_div, info)
        max_div, extra_info = update_divergence(
            output_list,
            name="PromptSecurityNoDataset",
            image_dir=self.response_save_dir,
            select_number=self.number_of_variants,
            metric=self.spacy_model
        )

        # Step 4: Decide if suspicious
        is_attack = detect_attack(max_div, extra_info, self.threshold)
        return is_attack

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Main generation entry point:
         1) Use the JailGuard logic to see if the user prompt is suspicious.
         2) If suspicious => block
         3) Else => normal generation
        """
        is_attack = self._detect_attack_for_prompt(prompt)
        if is_attack:
            return "[JailGuard] Sorry, I cannot comply with that request."
        else:
            return self.target_model.generate(prompt, **kwargs)
