#!/usr/bin/env python3
"""
Script to fix experiment configuration files to use unified JSON file path format
"""

import json
import os
import glob
from pathlib import Path

def get_standard_attack_mapping():
    """Map attack names to their standard config files"""
    return {
        "FlipAttack": "attacks/usage_examples/configs/black_box/FlipAttack.json",
        "ArtPromptAttack": "attacks/usage_examples/configs/black_box/ArtPrompt.json", 
        "ArtPrompt": "attacks/usage_examples/configs/black_box/ArtPrompt.json",
        "CodeAttack": "attacks/usage_examples/configs/black_box/CodeAttack.json",
        "CodeChameleon": "attacks/usage_examples/configs/black_box/CodeChameleon.json",
        "DRA": "attacks/usage_examples/configs/black_box/DRA.json",
        "GPTFUZZER": "attacks/usage_examples/configs/black_box/GPTFUZZER.json",
        "InceptionAttack": "attacks/usage_examples/configs/black_box/InceptionAttack.json",
        "ReNeLLM": "attacks/usage_examples/configs/black_box/ReNeLLM.json",
        "TapAttack": "attacks/usage_examples/configs/black_box/TapAttack.json",
        "ABJAttack": "attacks/usage_examples/configs/black_box/ABJAttack.json",
        "PairAttack": "attacks/usage_examples/configs/black_box/PAIR.json",
        "PAIR": "attacks/usage_examples/configs/black_box/PAIR.json",
        "PastTenseAttack": "attacks/usage_examples/configs/black_box/PastTense.json",
        "PastTense": "attacks/usage_examples/configs/black_box/PastTense.json",
        "PersuasiveInContextAttack": "attacks/usage_examples/configs/black_box/PersuasiveInContext.json",
        "PersuasiveInContext": "attacks/usage_examples/configs/black_box/PersuasiveInContext.json",
        "MultilingualJailbreakAttack": "attacks/usage_examples/configs/black_box/MultilingualJailbreakAttack.json",
        "IFSJAttack": "attacks/usage_examples/configs/black_box/IFSJAttack.json",
        "DrAttackAttack": "attacks/usage_examples/configs/black_box/DrAttack.json",
        "DrAttack": "attacks/usage_examples/configs/black_box/DrAttack.json",
    }

def get_standard_defense_mapping():
    """Map defense names to their standard config files"""
    return {
        "JailGuardDefense": "defenses/usage_examples/configs/jailguard_defense.json",
        "JailGuard": "defenses/usage_examples/configs/jailguard_defense.json",
        "SmoothLLM": "defenses/usage_examples/configs/smooth_llm.json",
        "RPO": "defenses/usage_examples/configs/rpo.json",
        "PerplexityFilter": "defenses/usage_examples/configs/perplexity_filter.json",
        "BackTranslation": "defenses/usage_examples/configs/back_translation.json",
        "GradSafeDefense": "defenses/usage_examples/configs/gradsafe_defense.json",
        "GradSafe": "defenses/usage_examples/configs/gradsafe_defense.json",
        "InputFilterDefense": "defenses/usage_examples/configs/input_filter_defense.json",
        "InputFilter": "defenses/usage_examples/configs/input_filter_defense.json",
        "OutputFilterDefense": "defenses/usage_examples/configs/output_filter_defense.json",
        "OutputFilter": "defenses/usage_examples/configs/output_filter_defense.json",
        "PrimeGuard": "defenses/usage_examples/configs/prime_guard.json",
    }

def get_standard_model_mapping():
    """Map model names to their standard config files"""
    return {
        "gpt-4o-mini": "models/usage_examples/configs/api/gpt-4o-mini.json",
        "gpt-4.1-mini": "models/usage_examples/configs/api/gpt-4.1-mini.json",
        "gpt-4.1": "models/usage_examples/configs/api/gpt-4.1.json",
        "claude-3-5-sonnet-20241022": "models/usage_examples/configs/api/claude-3-5-sonnet-20241022.json",
        "claude-3-5-haiku-20241022": "models/usage_examples/configs/api/claude-3-5-haiku-20241022.json",
        "llama-3.1-70b": "models/usage_examples/configs/local/llama-3.1-70b.json",
        "llama-3.1-8b": "models/usage_examples/configs/local/llama-3.1-8b.json",
    }

def convert_attacks_to_standard_format(attacks_list):
    """Convert attacks list to standard format"""
    if not attacks_list:
        return []
        
    attack_mapping = get_standard_attack_mapping()
    standard_attacks = []
    
    for attack in attacks_list:
        if isinstance(attack, str):
            # Already a file path - convert to standard format
            attack_name = Path(attack).stem
            standard_attacks.append({
                "name": attack_name,
                "config": attack,
                "enabled": True
            })
        elif isinstance(attack, dict):
            if "name" in attack and "config" in attack:
                # Already in standard format
                standard_attacks.append(attack)
            elif "attack_name" in attack:
                # Old format with attack_name and parameters
                attack_name = attack["attack_name"]
                clean_name = attack_name.replace("Attack", "")
                if clean_name in attack_mapping:
                    standard_attacks.append({
                        "name": clean_name,
                        "config": attack_mapping[clean_name],
                        "enabled": True
                    })
                elif attack_name in attack_mapping:
                    standard_attacks.append({
                        "name": attack_name,
                        "config": attack_mapping[attack_name],
                        "enabled": True
                    })
    
    return standard_attacks

def convert_defenses_to_standard_format(defenses_list):
    """Convert defenses list to standard format"""
    if not defenses_list:
        return []
        
    defense_mapping = get_standard_defense_mapping()
    standard_defenses = []
    
    for defense in defenses_list:
        if isinstance(defense, str):
            # File path format
            defense_name = Path(defense).stem
            standard_defenses.append({
                "name": defense_name,
                "config": defense,
                "enabled": True
            })
        elif isinstance(defense, dict):
            if "name" in defense and "config" in defense:
                # Already in standard format
                standard_defenses.append(defense)
            elif "defense_name" in defense:
                # Old format
                defense_name = defense["defense_name"]
                if defense_name in defense_mapping:
                    standard_defenses.append({
                        "name": defense_name,
                        "config": defense_mapping[defense_name],
                        "enabled": True
                    })
    
    return standard_defenses

def convert_model_to_standard_format(model_config):
    """Convert target_model to standard format"""
    if not model_config:
        return model_config
        
    model_mapping = get_standard_model_mapping()
    
    # If already a string (file path), keep it
    if isinstance(model_config, str):
        return model_config
    
    # If it's a dict with inline configuration
    if isinstance(model_config, dict):
        if "model_name" in model_config:
            model_name = model_config["model_name"]
            if model_name in model_mapping:
                # Convert to file path
                return model_mapping[model_name]
        elif "config" in model_config:
            # Already in standard format
            return model_config
    
    # Return as-is if no mapping found
    return model_config

def fix_config_file(config_path):
    """Fix a single configuration file"""
    print(f"Processing: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        modified = False
        
        # Fix attacks
        if "attacks" in config:
            original_attacks = config["attacks"]
            standard_attacks = convert_attacks_to_standard_format(original_attacks)
            if standard_attacks != original_attacks:
                config["attacks"] = standard_attacks
                modified = True
                print(f"  - Fixed attacks format")
        
        # Fix defenses  
        if "defenses" in config:
            original_defenses = config["defenses"]
            standard_defenses = convert_defenses_to_standard_format(original_defenses)
            if standard_defenses != original_defenses:
                config["defenses"] = standard_defenses
                modified = True
                print(f"  - Fixed defenses format")
        
        # Fix target_model
        if "target_model" in config:
            original_model = config["target_model"]
            standard_model = convert_model_to_standard_format(original_model)
            if standard_model != original_model:
                config["target_model"] = standard_model
                modified = True
                print(f"  - Fixed target_model format")
        
        # Save if modified
        if modified:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  ✅ Updated: {config_path}")
        else:
            print(f"  ⏭️  No changes needed: {config_path}")
            
    except Exception as e:
        print(f"  ❌ Error processing {config_path}: {e}")

def main():
    """Main function to process all config files"""
    config_dir = "experiments/examples"
    config_files = glob.glob(f"{config_dir}/*.json")
    
    print(f"Found {len(config_files)} configuration files to process...")
    print("=" * 60)
    
    for config_file in sorted(config_files):
        fix_config_file(config_file)
        print()
    
    print("=" * 60)
    print("✅ Configuration file processing complete!")

if __name__ == "__main__":
    main()