"""
Dynamic defense loader following the unified naming convention.
Each defense is in its own folder with:
- folder_name = defense_name
- __init__.py exports Defense class
- config.json contains default parameters
"""

import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def load_defense(defense_name: str, **kwargs) -> Any:
    """
    Dynamically load a defense module.
    
    Args:
        defense_name: Name of the defense (folder name)
        **kwargs: Additional parameters to override defaults
        
    Returns:
        Instantiated defense object
        
    Raises:
        ValueError: If defense not found or invalid
        ImportError: If module import fails
    """
    defenses_path = Path(__file__).parent / "defenses"
    normalized_name = defense_name.replace("-", "_")
    defense_path = defenses_path / normalized_name
    if not defense_path.exists():
        lowered = normalized_name.lower()
        defense_path = defenses_path / lowered
        normalized_name = lowered

    # Case-insensitive fallback for convenience (e.g., mixed-case CLI input)
    if not defense_path.exists():
        lowered = normalized_name.lower()
        for item in defenses_path.iterdir():
            if item.is_dir() and item.name.lower() == lowered:
                defense_path = item
                normalized_name = item.name
                break
    
    if not defense_path.exists() or not defense_path.is_dir():
        raise ValueError(f"Defense '{defense_name}' not found at {defense_path}")
    
    try:
        # Dynamic import of the defense module
        module = importlib.import_module(f"defenses.defenses.{normalized_name}")
        
        # Get Defense class (standardized name)
        if hasattr(module, 'Defense'):
            defense_class = module.Defense
        else:
            # Backward compatibility: look for classes ending with 'Defense'
            defense_classes = [
                obj for name, obj in vars(module).items()
                if (name.endswith('Defense') or name.endswith('DefendedModel')) and isinstance(obj, type)
            ]
            if defense_classes:
                defense_class = defense_classes[0]
                logger.warning(f"Using fallback class {defense_class.__name__} for {defense_name}")
            else:
                raise ValueError(f"No Defense class found in {defense_name}")
        
        # Load default configuration if available
        config_file = defense_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                default_config = json.load(f)
            # Merge user parameters with defaults
            params = {**default_config, **kwargs}
        else:
            params = kwargs
        
        # Instantiate and return the defense
            logger.info(f"Loading defense {normalized_name} with params: {list(params.keys())}")

        try:
            signature = inspect.signature(defense_class.__init__)
            has_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
            valid_params = {
                name for name, param in signature.parameters.items()
                if name != 'self'
            }

            if has_var_kwargs:
                filtered_params = params
            else:
                filtered_params = {
                    key: value for key, value in params.items()
                    if key in valid_params
                }
                dropped = set(params.keys()) - set(filtered_params.keys())
                if dropped:
                    logger.debug(
                        "Dropping unsupported defense params for %s: %s",
                        normalized_name,
                        sorted(dropped),
                    )

            return defense_class(**filtered_params)
        except Exception:
            # Re-raise for outer handler to log consistent error
            raise

    except ImportError as e:
        logger.error(f"Failed to import defense {normalized_name}: {e}")
        raise ImportError(f"Cannot import defense '{defense_name}': {e}")
    except Exception as e:
        logger.error(f"Failed to instantiate defense {normalized_name}: {e}")
        raise ValueError(f"Cannot instantiate defense '{defense_name}': {e}")

def list_defenses() -> list[str]:
    """
    List all available defense modules.
    
    Returns:
        List of defense names (folder names)
    """
    defenses_path = Path(__file__).parent / "defenses"
    
    if not defenses_path.exists():
        return []
    
    # Find all directories with __init__.py files
    defenses = []
    for item in defenses_path.iterdir():
        if (item.is_dir() and 
            not item.name.startswith('_') and 
            (item / "__init__.py").exists()):
            defenses.append(item.name)
    
    return sorted(defenses)

def get_defense_config(defense_name: str) -> Optional[Dict[str, Any]]:
    """
    Get default configuration for a defense.
    
    Args:
        defense_name: Name of the defense
        
    Returns:
        Default configuration dict or None if not found
    """
    defenses_path = Path(__file__).parent / "defenses"
    config_file = defenses_path / defense_name / "config.json"
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config for {defense_name}: {e}")
        return None

def get_defense_config_path(defense_name: str) -> Optional[str]:
    """获取防御配置文件路径
    
    Args:
        defense_name: 防御方法名称
        
    Returns:
        配置文件路径或None（如果不存在）
    """
    # Handle special case
    if defense_name == "no_defense":
        return None
    
    defenses_path = Path(__file__).parent / "defenses"
    config_file = defenses_path / defense_name / "config.json"
    
    return str(config_file) if config_file.exists() else None

# Backward compatibility function
def create_defense(defense_name: str, **kwargs) -> Any:
    """Alias for load_defense for backward compatibility."""
    return load_defense(defense_name, **kwargs)
