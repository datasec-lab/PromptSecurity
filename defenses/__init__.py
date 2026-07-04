# defense/__init__.py

from .base_defended_model import BaseDefendedModel
from .loader import load_defense, list_defenses, get_defense_config

# Backward compatibility
def load_defense_from_config(config_path: str, **kwargs):
    """Backward compatibility wrapper for old config-based loading."""
    import json
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    defense_name = config.get('defense_name')
    if not defense_name:
        raise ValueError(f"No defense_name found in config: {config_path}")
    
    # Convert old registry names to new folder names
    name_mapping = {
        'InputFilterDefense': 'input_filter_defense',
        'OutputFilterDefense': 'output_filter_defense',
        'JailGuardDefense': 'jailguard_defense',
        'GradSafeDefense': 'gradsafe_defense',
        'RPO': 'rpo',
        'SmoothLLM': 'smooth_llm',
        'PerplexityFilter': 'perplexity_filter',
        'BackTranslation': 'back_translation',
        'PrimeGuard': 'prime_guard',
        'no_defense': 'no_defense'
    }
    
    folder_name = name_mapping.get(defense_name, defense_name.lower())
    parameters = config.get('parameters', {})
    
    return load_defense(folder_name, **parameters, **kwargs)

# Legacy compatibility
DEFENSES = {
    "input_filter_defense": "input_filter_defense",
    "output_filter_defense": "output_filter_defense", 
    "jailguard_defense": "jailguard_defense",
    "gradsafe_defense": "gradsafe_defense",
    "rpo": "rpo",
    "smooth_llm": "smooth_llm",
    "perplexity_filter": "perplexity_filter",
    "back_translation": "back_translation",
    "prime_guard": "prime_guard",
    "no_defense": "no_defense"
}
