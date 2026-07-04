# defense/defense_registry.py
# DEPRECATED: This file is kept for backward compatibility
# The new defense system uses dynamic loading from defenses/loader.py

# Legacy registry - kept for compatibility but no longer used for loading
DEFENSES = {
    "InputFilterDefense": "input_filter_defense",
    "OutputFilterDefense": "output_filter_defense",
    "JailGuardDefense": "jailguard_defense",
    "GradSafeDefense": "gradsafe_defense", 
    "RPO": "rpo",
    "SmoothLLM": "smooth_llm",
    "PerplexityFilter": "perplexity_filter",
    "BackTranslation": "back_translation",
    "PrimeGuard": "prime_guard",
    "no_defense": "no_defense",
}

def list_defenses():
    """Return list of defense names for backward compatibility."""
    return list(DEFENSES.keys())
