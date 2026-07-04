# language_models.py

from models.model_loader import load_model_from_config


class ConfigLLM:
    """
    A small wrapper that uses load_model_from_config() to get a model plus
    any default parameters. The model itself must have a `generate` method.
    """

    def __init__(self, config_path: str):
        # Load the model plus any generation parameters
        self.model, self.parameters = load_model_from_config(config_path)

    def generate(self, prompt: str, **override_params) -> str:
        """
        Generate a single completion for a single `prompt`.

        By default, uses parameters from the config JSON.
        Additional overrides can be passed via `override_params`.
        """
        # Merge config-based parameters with any overrides from the caller:
        final_params = dict(self.parameters)
        final_params.update(override_params)

        return self.model.generate(prompt, **final_params)
