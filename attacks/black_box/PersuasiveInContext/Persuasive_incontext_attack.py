# attacks/black_box/PersuasiveJailbreakerAttack/persuasive_jailbreaker.py

import json
import random
from typing import Union

from attacks.base_attack import BaseAttack
from models.model_loader import load_model_from_config

class PersuasiveInContextAttack(BaseAttack):
    """
    PersuasiveJailbreakerAttack crafts a Persuasive Adversarial Prompt (PAP) 
    by combining a persuasive technique from a taxonomy with the original prompt,
    then refining it using an assist model. The attack function returns both the
    query count (number of assist model API calls) and the crafted prompt.
    """

    def __init__(self, target_model, target_model_parameters, taxonomy_path: str, assist_model_config: str, persuasive_technique: str = None):
        """
        :param target_model: The target LLM loaded via your model_loader.
        :param target_model_parameters: Additional parameters for the target model.
        :param taxonomy_path: Path to the persuasion taxonomy file (JSONL format).
        :param assist_model_config: Path to the assist model config (to refine the prompt).
        :param persuasive_technique: Optional technique name to force selection; if None, one is chosen at random.
        """
        self.target_model = target_model
        self.target_model_parameters = target_model_parameters
        self.taxonomy_path = taxonomy_path
        self.assist_model_config = assist_model_config
        self.persuasive_technique = persuasive_technique
        self.query_count = 0

        # Load persuasion taxonomy once on initialization
        self.taxonomy = self._load_taxonomy(self.taxonomy_path)

    def _load_taxonomy(self, taxonomy_path: str):
        taxonomy = []
        with open(taxonomy_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    taxonomy.append(json.loads(line))
        return taxonomy

    def _select_technique(self):
        if self.persuasive_technique:
            # Try to find the desired technique
            for item in self.taxonomy:
                if item.get("ss_technique") == self.persuasive_technique:
                    return item
            raise ValueError(f"Technique '{self.persuasive_technique}' not found in taxonomy.")
        else:
            # Randomly select one persuasive technique
            return random.choice(self.taxonomy)

    def craft_prompt(self, original_prompt: str) -> Union[str, list]:
        # Select a persuasive technique from the taxonomy
        technique = self._select_technique()
        persuasive_example = technique.get("ss_example")
        
        # Combine persuasive example with the original prompt.
        # You can modify this concatenation strategy as needed.
        initial_prompt = f"{persuasive_example}\n\n{original_prompt}"
        # print(f"Initial prompt: {initial_prompt}")
        
        # Use the assist model (loaded via your model loader) to refine the prompt.
        assist_model, assist_model_parameters = load_model_from_config(self.assist_model_config)
        refined_prompt = assist_model.generate(prompt=initial_prompt, **assist_model_parameters)
        self.query_count += 1
        # print(f"Refined prompt: {refined_prompt}")
        
        # Use the refined prompt as the final attacked prompt.
        attacked_prompt = refined_prompt
        return attacked_prompt

    def attack(self, original_prompt: str) -> (int, str):
        """
        Given an original prompt, this function crafts an adversarial prompt using a persuasive technique.
        Returns:
            query_count (int): The number of API calls (queries) used.
            attacked_prompt (str): The final crafted prompt.
        """
        attacked_prompt = self.craft_prompt(original_prompt)
        return self.query_count, attacked_prompt


