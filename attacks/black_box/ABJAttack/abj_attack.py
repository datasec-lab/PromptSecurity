# attacks/black_box/ABJAttack/abj_attack.py
#
# Analyzing-based Jailbreak (ABJ) — full two-stage implementation.
#
# Stage 1 (Attack Initiation, paper §3.3):
#   One-shot call to the assist model; transforms the harmful query into a
#   six-field neutral persona JSON (Character/Feature/Appearance/Job/Strength/Weakness).
#
# Stage 2 (Attack Execution with Toxicity Adjustment, paper §3.3 + Figure 2):
#   Iterative loop (up to attack_rounds iterations):
#     1. Format the persona dict as a <data> block and build the attack prompt.
#     2. Send prompt to the target model.
#     3. Judge whether the response contains harmful content.
#        • Harmful  → attack succeeded, stop.
#        • Refused  → ask assist model to *reduce* toxicity of one random field.
#        • Benign   → ask assist model to *enhance* toxicity of one random field.
#     4. Rebuild prompt from updated persona dict and repeat.
#
# Reference: Lin et al., "LLMs can be Dangerous Reasoners: Analyzing-based
#            Jailbreak Attack on Large Language Models" (arXiv 2407.16205).

import json
from typing import Tuple

from attacks.base_attack import BaseAttack
from models.model_loader import load_model_from_config

from .pipeline.data_preparation import get_data, get_data_with_status, data_to_block
from .pipeline.data_analysis import get_attack_prompt
from .pipeline.toxicity_adjustment import is_refusal, toxicity_reduction, toxicity_enhancement
from .utils.attack_prompt import ABJ_JUDGE_PROMPT
from .utils.clean_text import clean_text


class ABJAttack(BaseAttack):

    def __init__(
        self,
        target_model,
        target_model_parameters,
        attack_method: str = "original_ABJ",
        assist_model_config: str = None,
        judge_model_config: str = None,
        dataset_dir: str = None,
        attack_rounds: int = 5,
        save_interval: int = 10,
        **kwargs,
    ):
        """
        Args:
            target_model:            Loaded target model object.
            target_model_parameters: Generation kwargs for the target model.
            attack_method:           One of 'original_ABJ', 'abj_with_harmful_query',
                                     'code_based_ABJ'.  Defaults to 'original_ABJ'
                                     (the main variant from the paper).
            assist_model_config:     Path to a model config JSON for the assist LLM
                                     (used for Stage 1 persona generation and toxicity
                                     adjustment only), OR a judger config JSON (used
                                     only for judgment when judge_model_config is unset).
            judge_model_config:      Path to a model config JSON for the internal judge
                                     LLM (used inside the attack loop to evaluate whether
                                     the target model's response is harmful). When set,
                                     takes priority over assist_model_config for judging.
                                     Defaults to None (falls back to assist_model_config).
            attack_rounds:           Maximum toxicity-adjustment iterations (default 5,
                                     matching the open-source ReDPJ implementation).
            save_interval:           Unused in single-prompt mode; kept for compat.
        """
        self.target_model = target_model
        self.target_model_parameters = target_model_parameters
        self.attack_method = attack_method
        self.dataset_dir = dataset_dir
        self.attack_rounds = attack_rounds
        self.save_interval = save_interval
        self.query_count = 0
        self.assist_refused = False

        self.assist_model = None
        self.assist_model_parameters = {}
        self.judge_model = None
        self.judge_model_parameters = {}
        self.judger = None

        if assist_model_config:
            try:
                with open(assist_model_config, 'r') as f:
                    cfg = json.load(f)
                if 'judger_name' in cfg:
                    from judgers import load_judger_from_config
                    self.judger = load_judger_from_config(assist_model_config)
                else:
                    self.assist_model, self.assist_model_parameters = \
                        load_model_from_config(assist_model_config)
            except Exception as e:
                print(f"[ABJAttack] Failed to load assist_model_config '{assist_model_config}': {e}")
                try:
                    self.assist_model, self.assist_model_parameters = \
                        load_model_from_config(assist_model_config)
                except Exception as e2:
                    print(f"[ABJAttack] Fallback load also failed: {e2}")

        if judge_model_config:
            try:
                self.judge_model, self.judge_model_parameters = \
                    load_model_from_config(judge_model_config)
            except Exception as e:
                print(f"[ABJAttack] Failed to load judge_model_config '{judge_model_config}': {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_assist_model(self, prompt: str) -> str:
        """Call the assist model; increments query_count."""
        if self.assist_model is None:
            return ""
        try:
            self.query_count += 1
            return self.assist_model.generate(
                prompt=prompt, **self.assist_model_parameters
            )
        except Exception as e:
            print(f"[ABJAttack] assist model error: {e}")
            return ""

    def _call_target_model(self, prompt: str) -> str:
        """Call the target model; increments query_count."""
        self.query_count += 1
        return self.target_model.generate(
            prompt=prompt, **self.target_model_parameters
        )

    def _judge_response(self, response: str) -> bool:
        """
        Return True when the response contains harmful content (attack succeeded).

        Priority:
          1. External judger (if configured) — looks for "UNSAFE" label.
          2. Assist model with the Table-6 judge prompt — looks for "1".
          3. Fallback: treat any non-refusal as potentially harmful.
        """
        if self.judger is not None:
            try:
                result = self.judger.judge(generation=response)
                return str(result).upper() == "UNSAFE"
            except Exception as e:
                print(f"[ABJAttack] judger error: {e}")

        # Use dedicated judge model when available, otherwise fall back to assist model.
        _judge_model = self.judge_model if self.judge_model is not None else self.assist_model
        _judge_params = self.judge_model_parameters if self.judge_model is not None else self.assist_model_parameters
        if _judge_model is not None:
            try:
                judge_prompt = ABJ_JUDGE_PROMPT.replace("{PROMPT}", response)
                self.query_count += 1
                result = clean_text(
                    _judge_model.generate(
                        prompt=judge_prompt, **_judge_params
                    )
                ).strip()
                return result == "1"
            except Exception as e:
                print(f"[ABJAttack] judge error: {e}")

        # No judgment model available — treat non-refusals as harmful
        return not is_refusal(response)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attack(self, original_prompt: str) -> Tuple[int, str]:
        """
        Run the full ABJ attack pipeline.

        Returns:
            query_count  — total LLM calls made (target + assist + judger).
            attack_prompt — the final prompt that was sent to the target model
                            (either the successful one, or the last one tried).
        """
        self.query_count = 0
        self.assist_refused = False

        # ── Stage 1: Attack Initiation ──────────────────────────────────
        # One-shot transformation: harmful query → neutral persona dict.
        data, self.assist_refused = get_data_with_status(original_prompt, self._call_assist_model)
        data_block = data_to_block(data)
        attack_prompt = get_attack_prompt(self.attack_method, data_block, original_prompt)

        # ── Stage 2: Attack Execution with Toxicity Adjustment ───────────
        for round_idx in range(self.attack_rounds):
            try:
                response = self._call_target_model(attack_prompt)
            except Exception as e:
                print(f"[ABJAttack] target model error on round {round_idx}: {e}")
                break

            if self._judge_response(response):
                # Attack succeeded — return the winning prompt.
                break

            # Last round: no point adjusting (nothing to retry).
            if round_idx == self.attack_rounds - 1:
                break

            # Adjust toxicity and rebuild the prompt for the next round.
            if is_refusal(response):
                # Model refused → make neutral data less suspicious.
                data = toxicity_reduction(data, data_block, self._call_assist_model)
            else:
                # Model responded harmlessly → amplify harmful attributes.
                data = toxicity_enhancement(data, data_block, self._call_assist_model)

            data_block = data_to_block(data)
            attack_prompt = get_attack_prompt(self.attack_method, data_block, original_prompt)

        return self.query_count, attack_prompt
