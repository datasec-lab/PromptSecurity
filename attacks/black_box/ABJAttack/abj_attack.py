# attacks/black_box/ABJAttack/abj_attack.py

import os
from typing import Union, Tuple

from attacks.base_attack import BaseAttack
from models.model_loader import load_model_from_config

# Import or copy the relevant ABJ pipeline logic:
from .pipeline.data_preparation import get_character, get_feature, get_job
from .pipeline.data_analysis import get_attack_prompt
from .utils.clean_text import clean_text


class ABJAttack(BaseAttack):
    """
    A class implementing the ABJ (Anti-Basic-Jailbreak) style attack.
    It uses a separate assist model to infer 'character', 'feature', 'job'
    and then crafts a final 'attack' prompt.
    """

    def __init__(
        self,
        target_model,
        target_model_parameters,
        attack_method: str = "modified_ABJ",
        assist_model_config: str = None,
        dataset_dir: str = None,
        attack_rounds: int = 3,
        save_interval: int = 10,
        **kwargs
    ):
        """
        Initializes the ABJAttack with parameters passed in from config.

        :param target_model:        The main model object (loaded from your model_loader).
        :param target_model_parameters:  Dict of generation parameters for the main model.
        :param attack_method:       One of ["original_ABJ","modified_ABJ","code_based_ABJ","adversarial_ABJ"].
        :param assist_model_config: Path to a JSON config for the 'assist' or 'judge' model.
        :param dataset_dir:         Optional path to dataset CSV, if you want to iterate tasks (unused in single-prompt).
        :param attack_rounds:       How many times you might re-query the model(s) if you wanted iterative logic.
        :param save_interval:       Unused in this minimal version, but kept for consistency.
        :param kwargs:              Catch-all for extra config parameters you might have.
        """

        self.target_model = target_model
        self.target_model_parameters = target_model_parameters
        self.attack_method = attack_method
        self.dataset_dir = dataset_dir
        self.attack_rounds = attack_rounds
        self.save_interval = save_interval

        # Track how many times we call any LLM
        self.query_count = 0

        # Load assist model or judger
        self.assist_model = None
        self.assist_model_parameters = {}
        self.judger = None
        
        if assist_model_config:
            # Check if it's a judger config or model config
            try:
                import json
                with open(assist_model_config, 'r') as f:
                    config = json.load(f)
                if 'judger_name' in config:
                    # It's a judger config
                    from judgers import load_judger_from_config
                    self.judger = load_judger_from_config(assist_model_config)
                else:
                    # It's a model config
                    self.assist_model, self.assist_model_parameters = load_model_from_config(assist_model_config)
            except Exception as e:
                print(f"Failed to load config {assist_model_config}, trying as model config: {e}")
                # Fallback to model loading
                try:
                    self.assist_model, self.assist_model_parameters = load_model_from_config(assist_model_config)
                except Exception as e2:
                    print(f"Failed to load as model config too: {e2}")

    def _call_assist_model(self, prompt: str) -> str:
        """
        Helper that calls the assist model or judger for short tasks.
        Increments query_count each time we call this model.
        """
        if self.judger is not None:
            # Use judger for evaluation
            try:
                self.query_count += 1
                result = self.judger.judge(generation=prompt)
                return str(result)  # Return as string for consistency
            except Exception as e:
                print(f"Error using judger: {e}")
                return ""
        elif self.assist_model is not None:
            # Use assist model
            try:
                self.query_count += 1
                result = self.assist_model.generate(
                    prompt=prompt,
                    **self.assist_model_parameters
                )
                return result
            except Exception as e:
                print(f"Error calling assist model: {e}")
                return ""
        else:
            # No assist model or judger configured
            return ""

    def craft_prompt(self, original_prompt: str) -> str:
        """
        Main logic to craft the ABJ attack prompt.
          1) We call the assist model to get 'character', 'feature', 'job'
          2) We build the ABJ final attack prompt using get_attack_prompt.
        """
        # Prepare data (character, feature, job) via the assist model
        character = get_character(original_prompt, self._call_assist_model)
        feature = get_feature(original_prompt, self._call_assist_model)
        job = get_job(original_prompt, self._call_assist_model)

        # Combine them into the final 'attack' prompt
        attack_prompt = get_attack_prompt(
            self.attack_method,
            original_prompt,  # 'task'
            character,
            feature,
            job
        )
        return attack_prompt

    def attack(self, original_prompt: str) -> Tuple[int, str]:
        """
        Required by your system.
        Returns:
            query_count (int): how many times we've called any LLM so far
            attack_text (str): the final prompt for the target model
        """
        self.query_count = 0
        crafted_prompt = self.craft_prompt(original_prompt)
        return self.query_count, crafted_prompt
