# defense/defense_loader.py

import json
import importlib
import os
from .defense_registry import DEFENSES
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


def load_defense_from_config(config_path: str, target_model, tokenizer=None):
    """
    The config might look like:
    {
      "defense_name": "PipelineDefense",
      "parameters": {
        "input_filters": ["defense.defenses.custom_input_filters.replace_forbidden_words"],
        "output_filters": ["defense.defenses.custom_output_filters.censor_disallowed_phrases"]
      }
    }
    or (for the single-class approach):
    {
      "defense_name": "InputFilterDefense",
      "parameters": {
        "forbidden_words": ["bomb", "kill"]
      }
    }

    We'll:
      1) parse the JSON
      2) look up the class in DEFENSES
      3) handle any dotted-path imports for callables
      4) instantiate the defense
      5) return it
    """
    config_path =os.path.join(project_root, config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    defense_name = config["defense_name"]
    if defense_name not in DEFENSES:
        raise ValueError(f"Defense '{defense_name}' not in registry. Available: {list(DEFENSES.keys())}")

    defense_class = DEFENSES[defense_name]

    # parameters from JSON
    params = config.get("parameters", {})

    # For pipeline approach, we might need to import the filter callables:
    if "input_filters" in params:
        params["input_filters"] = [_import_callable(path) for path in params["input_filters"]]
    if "output_filters" in params:
        params["output_filters"] = [_import_callable(path) for path in params["output_filters"]]

    # Instantiate the model
    # Check if the defense requires tokenizer
    tokenizer_required_defenses = [
        "BacktranslationDefendedModel", "SmoothLLMDefense", "PerplexityDefendedModel", 
        "RobustOptDefendedModel", "PrimeGuardDefendedModel"
    ]
    
    if defense_class.__name__ in tokenizer_required_defenses:
        if tokenizer is None:
            # Create a mock tokenizer for API models that don't have one
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except:
                raise ValueError(f"Defense {defense_name} requires tokenizer but target model doesn't have one and fallback failed")
        defended_model = defense_class(target_model, tokenizer, **params)
    else:
        defended_model = defense_class(target_model, **params)
    return defended_model

def _import_callable(dotted_path: str):
    """
    E.g. "defense.defenses.input_filter.replace_forbidden_words"
    We'll import the module, then get the function by name.
    """
    module_name, fn_name = dotted_path.rsplit('.', 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, fn_name)
    return fn
