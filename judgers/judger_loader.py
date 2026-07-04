"""
Convention-based loader for judgers without registry.
"""

import os
import sys
import json
import importlib
from pathlib import Path
from typing import Optional, Dict, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Class mapping for convention-based loading
JUDGER_CLASS_MAP = {
    'harmbench_judger': ('harmbench_judger', 'HarmBenchJudger'),
    'rejection_prefix_judger': ('rejection_prefix_judger', 'JailbreakPromptJudger'),
    'gpt_judger_contextual_harmbench': ('gpt_judger', 'GPTJudger'),
    'gpt_judger_harmbench_style': ('gpt_judger', 'GPTJudger'),
    'gpt_judger_harmful_binary': ('gpt_judger', 'GPTJudger'),
    'gpt_judger_openai_policy': ('gpt_judger', 'GPTJudger'),
    'gpt_judger_tap_style': ('gpt_judger', 'GPTJudger'),
}

# Backward compatibility aliases (lower-cased and underscore normalized)
JUDGER_NAME_ALIASES = {
    'rejection_prefix': 'rejection_prefix_judger',
    'rejectionprefix': 'rejection_prefix_judger',
    'rejectionprefixjudger': 'rejection_prefix_judger',
    'jailbreakpromptjudger': 'rejection_prefix_judger',
    'prefix_judger': 'rejection_prefix_judger',
    'harmbench': 'harmbench_judger',
    'harmbenchjudger': 'harmbench_judger',
}


def _normalize_judger_name(judger_name: str) -> str:
    """Map legacy judger names to canonical identifiers used by the loader."""
    if not judger_name:
        raise ValueError("Judger name cannot be empty")

    candidate = judger_name.strip()
    if candidate in JUDGER_CLASS_MAP:
        return candidate

    alias_key = candidate.lower().replace('-', '_')
    canonical = JUDGER_NAME_ALIASES.get(alias_key)
    if canonical and canonical in JUDGER_CLASS_MAP:
        return canonical

    raise ValueError(
        f"Judger '{judger_name}' is not supported. Available judgers: {list(JUDGER_CLASS_MAP.keys())}"
    )

def load_judger(judger_name: str, **kwargs) -> object:
    """
    Load judger using convention-based approach.
    
    Args:
        judger_name: Name of the judger (config filename without .json)
        **kwargs: Additional parameters to override config
        
    Returns:
        BaseJudger: Instantiated judger
    """
    canonical_name = _normalize_judger_name(judger_name)

    # Try to load config file
    config_path = Path(__file__).parent / "configs" / f"{canonical_name}.json"

    # Fallback to old location if not found in new location
    if not config_path.exists():
        config_path = Path(__file__).parent / "usage_examples" / "configs" / f"{canonical_name}.json"

    if not config_path.exists():
        raise ValueError(f"Judger config '{judger_name}' not found")
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Get judger class info
    module_name, class_name = JUDGER_CLASS_MAP[canonical_name]
    
    # Import the judger class
    try:
        module = importlib.import_module(f"judgers.{module_name}")
        judger_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import judger class {class_name} from {module_name}: {e}")
    
    # Merge parameters
    parameters = config.get('parameters', {})
    parameters.update(kwargs)
    
    # Handle special cases for GPT judgers with templates
    if canonical_name.startswith('gpt_judger_') and canonical_name != 'gpt_judger':
        template_name = canonical_name.replace('gpt_judger_', '')
        parameters['template'] = template_name
    
    # Instantiate judger
    return judger_class(**parameters)

def load_judger_from_config(config_path: str, **override_params):
    """
    Load a judger from a configuration file path.
    
    Args:
        config_path: Path to the judger configuration file
        **override_params: Parameters to override from config
        
    Returns:
        BaseJudger: Instantiated judger
    """
    # Extract judger name from config path
    judger_name = Path(config_path).stem
    return load_judger(judger_name, **override_params)


def get_available_judgers():
    """
    Get a list of all available judgers based on config files.
    
    Returns:
        List[str]: List of available judger names
    """
    judgers = []
    
    # Check new location
    configs_dir = Path(__file__).parent / "configs"
    if configs_dir.exists():
        for config_file in configs_dir.glob("*.json"):
            judgers.append(config_file.stem)
    
    # Check old location for backward compatibility
    old_configs_dir = Path(__file__).parent / "usage_examples" / "configs"
    if old_configs_dir.exists():
        for config_file in old_configs_dir.glob("*.json"):
            if config_file.stem not in judgers:  # Avoid duplicates
                judgers.append(config_file.stem)
    
    return sorted(judgers)


def create_judger_config(judger_name: str, output_path: str, **parameters):
    """
    Create a judger configuration file.
    
    Args:
        judger_name: Name of the judger
        output_path: Path to save the configuration file
        **parameters: Parameters for the judger
    """
    canonical_name = _normalize_judger_name(judger_name)

    # For convention-based approach, we don't need judger_name in config
    # Just parameters
    config = {
        "parameters": parameters
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as file:
        json.dump(config, file, indent=2)
    
    print(f"Judger configuration saved to: {output_path}")


def load_judger_from_name(judger_name: str, **parameters):
    """
    Load a judger directly by name without a config file.
    
    Args:
        judger_name: Name of the judger
        **parameters: Parameters for the judger
        
    Returns:
        BaseJudger: Instantiated judger
    """
    canonical_name = _normalize_judger_name(judger_name)

    module_name, class_name = JUDGER_CLASS_MAP[canonical_name]
    
    # Import the judger class
    try:
        module = importlib.import_module(f"judgers.{module_name}")
        judger_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import judger class {class_name} from {module_name}: {e}")
    
    # Handle special cases for GPT judgers with specific templates
    if canonical_name.startswith('gpt_judger_') and canonical_name != 'gpt_judger':
        template_name = canonical_name.replace('gpt_judger_', '')
        parameters['template'] = template_name
    
    judger = judger_class(**parameters)
    return judger
