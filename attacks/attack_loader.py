import os
import sys
import json
from attacks.attack_registry import ATTACKS

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

def load_attack_from_config(config_path: str, target_model,target_model_parameters):
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as file:
        config = json.load(file)

    attack_name = config.get('attack_name')
    attack_parameters = config.get('parameters', {})

    if attack_name not in ATTACKS:
        raise ValueError(f"Attack '{attack_name}' is not registered.")

    attack_class = ATTACKS[attack_name]
    attack = attack_class(target_model,target_model_parameters,**attack_parameters)
    return attack