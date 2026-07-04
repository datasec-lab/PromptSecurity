#!/usr/bin/env python3
"""
Script to migrate attack modules to unified naming convention
"""

import os
import shutil
from pathlib import Path
import json

# Mapping from folder names to class names (based on attack_registry.py)
ATTACK_MAPPINGS = {
    # Black box attacks
    'FlipAttack': {'class': 'FlipAttack', 'module': 'FlipAttack'},
    'ArtPrompt': {'class': 'ArtPromptAttack', 'module': 'artprompt'},
    'PAIR': {'class': 'PairAttack', 'module': 'pair_attack'},
    'ABJAttack': {'class': 'ABJAttack', 'module': 'abj_attack'},
    'PastTense': {'class': 'PastTenseAttack', 'module': 'PastTenseAttack'},
    'IFSJ': {'class': 'IFSJAttack', 'module': 'ifsj_attack'},
    'CodeAttack': {'class': 'CodeAttack', 'module': 'CodeAttack'},
    'CodeChameleon': {'class': 'CodeChameleon', 'module': 'CodeChameleon'},
    'DRA': {'class': 'DRA', 'module': 'DRA'},
    'PersuasiveInContext': {'class': 'PersuasiveInContextAttack', 'module': 'Persuasive_incontext_attack'},
    'ReNeLLM': {'class': 'ReNeLLM', 'module': 'ReNeLLM'},
    'TapAttack': {'class': 'TapAttack', 'module': 'tapattack'},
    'DrAttack': {'class': 'DrAttackAttack', 'module': 'dr_attack'},
    'InceptionAttack': {'class': 'InceptionAttack', 'module': 'inception_attack'},
    'MultilingualJailbreak': {'class': 'MultilingualJailbreakAttack', 'module': 'Multilingual_jailbreak_attack'},
    'GPTFUZZER': {'class': 'GPTFUZZER', 'module': 'GPTFUZZER'},
    
    # White box attacks
    'GCGAttack': {'class': 'GCGWhiteBoxAttack', 'module': 'GCGWhiteBoxAttack'},
    'AutoDAN': {'class': 'AutoDANAttack', 'module': 'auto_dan_attack'},
    'COLD': {'class': 'COLDAttack', 'module': 'cold_attack'},
}

def create_init_file(attack_dir: Path, attack_name: str, attack_type: str):
    """Create __init__.py file for attack module"""
    init_file = attack_dir / "__init__.py"
    
    if attack_name not in ATTACK_MAPPINGS:
        print(f"⚠️  Warning: No mapping found for {attack_name}, skipping")
        return
    
    mapping = ATTACK_MAPPINGS[attack_name]
    class_name = mapping['class']
    module_name = mapping['module']
    
    content = f"""# Unified naming: export as Attack
from .{module_name} import {class_name} as Attack

__all__ = ['Attack']"""
    
    # Write or update the file
    with open(init_file, 'w') as f:
        f.write(content)
    print(f"✅ Created/Updated {init_file}")

def copy_config_file(attack_name: str, attack_type: str, target_dir: Path):
    """Copy config file from usage_examples to module folder"""
    # Map folder names to config file names
    config_name_map = {
        'IFSJ': 'IFSJAttack',
        'MultilingualJailbreak': 'MultilingualJailbreakAttack',
        'AutoDAN': 'AutoDANAttack',
        'COLD': 'COLDAttack',
        'GCGAttack': 'GCGAttack',
        'PersuasiveInContext': 'PersuasiveInContext',
    }
    
    config_name = config_name_map.get(attack_name, attack_name)
    source_config = Path(f"usage_examples/configs/{attack_type}/{config_name}.json")
    target_config = target_dir / "config.json"
    
    if source_config.exists():
        shutil.copy2(source_config, target_config)
        print(f"✅ Copied config: {source_config} -> {target_config}")
        
        # Update config file to remove attack_name field if present
        try:
            with open(target_config, 'r') as f:
                config_data = json.load(f)
            
            # Remove attack_name if present
            if 'attack_name' in config_data:
                del config_data['attack_name']
            
            # Flatten if only parameters field exists
            if len(config_data) == 1 and 'parameters' in config_data:
                config_data = config_data['parameters']
            
            with open(target_config, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"✅ Updated config format for {attack_name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not update config format for {attack_name}: {e}")
    else:
        print(f"⚠️  Warning: Config file not found: {source_config}")

def main():
    """Main migration function"""
    attacks_dir = Path(__file__).parent
    
    print("🚀 Starting attack module naming migration...\n")
    
    # Process black box attacks
    print("📦 Processing Black Box Attacks:")
    black_box_dir = attacks_dir / "black_box"
    if black_box_dir.exists():
        for attack_dir in black_box_dir.iterdir():
            if attack_dir.is_dir() and not attack_dir.name.startswith('__'):
                print(f"\n📂 Processing {attack_dir.name}...")
                create_init_file(attack_dir, attack_dir.name, "black_box")
                copy_config_file(attack_dir.name, "black_box", attack_dir)
    
    # Process white box attacks
    print("\n\n📦 Processing White Box Attacks:")
    white_box_dir = attacks_dir / "white_box"
    if white_box_dir.exists():
        for attack_dir in white_box_dir.iterdir():
            if attack_dir.is_dir() and not attack_dir.name.startswith('__'):
                print(f"\n📂 Processing {attack_dir.name}...")
                create_init_file(attack_dir, attack_dir.name, "white_box")
                copy_config_file(attack_dir.name, "white_box", attack_dir)
    
    # Handle special case: no_attack
    print("\n\n📦 Processing Special Cases:")
    no_attack_config_src = attacks_dir / "usage_examples/configs/black_box/no_attack.json"
    no_attack_config_dst = attacks_dir / "black_box/no_attack_config.json"
    if no_attack_config_src.exists():
        shutil.copy2(no_attack_config_src, no_attack_config_dst)
        print(f"✅ Copied no_attack config")
    
    print("\n\n✅ Migration completed!")
    print("\nNext steps:")
    print("1. Test the new loader with: python test_attack_loader.py")
    print("2. Update experiment system to use new loader")
    print("3. Remove old registry files")

if __name__ == "__main__":
    main()