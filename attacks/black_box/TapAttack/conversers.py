# attacks/black_box/TAPAttack/conversers.py

from .language_models import WrappedModel
from .common import extract_json, random_string
import logging

def extract_json(s):
    """
    The original code snippet had a function to parse
    { "improvement": "...", "prompt": "..." } 
    from a block of text.
    (If you want, copy your old function here.)
    """
    import ast
    start_pos = s.find("{")
    if start_pos == -1:
        return None, None
    end_pos = s.find("}", start_pos)
    if end_pos == -1:
        return None, None
    json_str = s[start_pos:end_pos+1]
    try:
        parsed = ast.literal_eval(json_str)
        if not all(k in parsed for k in ["improvement", "prompt"]):
            return None, None
        return parsed, json_str
    except:
        return None, None

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

        valid_outputs = [None] * batchsize
        indices_to_regenerate = list(range(batchsize))

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
                # Reconstruct full JSON
                raw_json_text = init_message + out_text
                parsed, json_str = extract_json(raw_json_text)
                if parsed is not None:
                    valid_outputs[orig_idx] = parsed
                    convs_list[orig_idx].update_last_message(json_str)
                else:
                    new_indices.append(orig_idx)
            indices_to_regenerate = new_indices
            if not indices_to_regenerate:
                break

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
