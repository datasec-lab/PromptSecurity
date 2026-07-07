# attacks/black_box/TAPAttack/conversers.py

from .language_models import WrappedModel
from .common import extract_json, random_string
import logging

logger = logging.getLogger(__name__)

class AttackLLM:
    """
    Wraps the 'attacker' model, which tries to produce the adversarial prompt
    in JSON format. We attempt multiple times if the JSON is malformed.
    """
    def __init__(
        self, 
        model, 
        model_params,
        model_name, 
        max_n_tokens=500, 
        max_n_attack_attempts=5,
        temperature=1.0, 
        top_p=0.9,
        query_counter=None
    ):
        self.model = WrappedModel(model, model_params, model_name, query_counter)
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.temperature = temperature
        self.top_p = top_p

    def get_attack(self, convs_list, prompts_list):
        """
        The same batch logic as the original code, but calling self.model.batched_generate(...) 
        and parse the returned text for JSON with "improvement", "prompt".
        """
        assert len(convs_list) == len(prompts_list)
        batchsize = len(convs_list)
        if batchsize == 0:
            return []

        valid_outputs = [None] * batchsize
        indices_to_regenerate = list(range(batchsize))
        last_failed_outputs = {}

        # We seed the partial JSON if needed
        init_message = """{"improvement": "","prompt": \""""  # as in the original
        # Some attacker logic might differ if conv empty. We'll keep it simple.

        # Build final prompt text
        full_prompts = []
        for conv, user_prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], user_prompt)
            # We add a partial JSON to guide the LLM
            conv.append_message(conv.roles[1], init_message)
            # Gather the entire text
            full_prompts.append(conv.get_prompt())

        for _ in range(self.max_n_attack_attempts):
            sub_prompts = [full_prompts[i] for i in indices_to_regenerate]
            outputs = self.model.batched_generate(
                sub_prompts,
                max_n_tokens=self.max_n_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            new_indices = []
            for i, out_text in enumerate(outputs):
                orig_idx = indices_to_regenerate[i]
                parsed, json_str = extract_json(out_text, continuation_prefix=init_message)
                if parsed is not None:
                    valid_outputs[orig_idx] = parsed
                    convs_list[orig_idx].update_last_message(json_str)
                else:
                    last_failed_outputs[orig_idx] = (out_text or "")[:300].replace("\n", "\\n")
                    new_indices.append(orig_idx)
            indices_to_regenerate = new_indices
            if not indices_to_regenerate:
                break

        for orig_idx in indices_to_regenerate:
            preview = last_failed_outputs.get(orig_idx, "")
            logger.warning(
                "TapAttack attacker output could not be parsed after %s attempts for stream %s. "
                "Response preview: %s",
                self.max_n_attack_attempts,
                orig_idx + 1,
                preview,
            )

        return valid_outputs

class TargetLLM:
    """
    The 'victim' LLM that we try to break. 
    """
    def __init__(
        self, 
        model, 
        model_params,
        model_name, 
        max_n_tokens=150, 
        temperature=0.0,
        top_p=1.0,
        query_counter=None
    ):
        self.model = WrappedModel(model, model_params, model_name, query_counter)
        self.max_n_tokens = max_n_tokens
        self.temperature = temperature
        self.top_p = top_p

    def get_response(self, prompts_list):
        outputs = self.model.batched_generate(
            prompts_list, 
            max_n_tokens=self.max_n_tokens, 
            temperature=self.temperature, 
            top_p=self.top_p
        )
        return outputs
