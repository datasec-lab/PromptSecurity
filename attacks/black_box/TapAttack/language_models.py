# attacks/black_box/TAPAttack/language_models.py

class WrappedModel:
    """
    A generic wrapper that calls `model.generate(...)`. 
    We'll store 'model' and 'model_params', plus a pointer to
    the global query_counter so we can increment how many calls we do.
    """
    def __init__(self, model, model_params, model_name, query_counter=None):
        self.model = model
        self.model_params = model_params
        self.model_name = model_name
        self.query_counter = query_counter  # the TapAttack instance

    def batched_generate(self, list_of_prompts, max_n_tokens=200, temperature=1.0, top_p=0.9):
        """
        For each prompt in the list, call self.model.generate(...).
        We also increment query_counter for each call.
        """
        outputs = []
        for prompt in list_of_prompts:
            # We combine our local “max_n_tokens” with the base model params
            # or override them as needed.
            params = dict(self.model_params)
            params.setdefault("max_tokens", max_n_tokens)
            params.setdefault("temperature", temperature)
            params.setdefault("top_p", top_p)

            # Actually call the model
            out = self.model.generate(prompt=prompt, **params)

            # If we want to track how many calls we do, increment once per prompt
            if self.query_counter is not None:
                self.query_counter.query_count += 1

            outputs.append(out)
        return outputs
