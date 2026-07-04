"""Helpers for resolving component configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from attacks.loader import get_attack_config_path
from defenses.loader import get_defense_config_path
from models.loader import get_model_config_path

_COMPONENT_RESOLVERS = {
    "attack": get_attack_config_path,
    "defense": get_defense_config_path,
    "model": get_model_config_path,
}


def resolve_config_path(component_type: str, name: str) -> Optional[Path]:
    component_type = component_type.lower()
    resolver = _COMPONENT_RESOLVERS.get(component_type)
    if resolver is None:
        return None

    path = resolver(name)
    if path is None:
        return None
    return Path(path)


def load_component_config(component_type: str, name: str) -> Dict[str, Any]:
    """Load the JSON configuration for a component if available."""

    path = resolve_config_path(component_type, name)
    if not path or not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
