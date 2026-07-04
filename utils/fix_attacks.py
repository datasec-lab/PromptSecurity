#!/usr/bin/env python3
"""
Fix common attack and evaluation issues in PromptSecurity framework
"""

import os
import json
from pathlib import Path

def fix_query_counting():
    """Fix query counting issues in attacks and pipeline"""
    
    # 1. Fix FlipAttack query counting
    flipattack_file = "attacks/black_box/FlipAttack/flip_attack.py"
    if os.path.exists(flipattack_file):
        with open(flipattack_file, 'r') as f:
            content = f.read()
        
        # Replace the problematic line
        content = content.replace(
            'self.query_count=0 # model-agnostic attack',
            'self.query_count = 1  # Count as 1 query for the crafted prompt'
        )
        
        with open(flipattack_file, 'w') as f:
            f.write(content)
        print("✅ Fixed FlipAttack query counting")
    
    # 2. Fix Pipeline query counting
    pipeline_file = "experiments/pipeline/JailbreakEvaluationPipeline.py"
    if os.path.exists(pipeline_file):
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Add query counting for target model generation
        old_line = "response = self._safe_generate(attacked_prompt)"
        new_lines = """response = self._safe_generate(attacked_prompt)
        # Count the final generation query
        queries = (queries or 0) + 1"""
        
        if old_line in content and "queries = (queries or 0) + 1" not in content:
            content = content.replace(old_line, new_lines)
            
            with open(pipeline_file, 'w') as f:
                f.write(content)
            print("✅ Fixed Pipeline query counting")

def fix_attack_configs():
    """Fix configuration path issues in attack configs"""
    
    # Fix ABJAttack config
    abj_config = "attacks/usage_examples/configs/black_box/ABJAttack.json"
    if os.path.exists(abj_config):
        with open(abj_config, 'r') as f:
            config = json.load(f)
        
        # Fix path if it exists in config
        if 'parameters' in config and 'assist_model_config' in config['parameters']:
            config['parameters']['assist_model_config'] = "judgers/usage_examples/configs/gpt_judger_harmful_binary.json"
            
            with open(abj_config, 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ Fixed ABJAttack config paths")
    
    # Fix TapAttack config
    tap_config = "attacks/usage_examples/configs/black_box/TapAttack.json"
    if os.path.exists(tap_config):
        with open(tap_config, 'r') as f:
            config = json.load(f)
        
        # Fix evaluator model config path if it exists
        if 'parameters' in config and 'evaluator_model_config' in config['parameters']:
            config['parameters']['evaluator_model_config'] = "judgers/usage_examples/configs/gpt_judger_tap_style.json"
            
            with open(tap_config, 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ Fixed TapAttack config paths")

def create_missing_init_files():
    """Create missing __init__.py files"""
    
    missing_init_dirs = [
        "attacks/black_box/ABJAttack",
        "attacks/black_box/TapAttack"
    ]
    
    for dir_path in missing_init_dirs:
        if os.path.exists(dir_path):
            init_file = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('"""Attack module initialization"""\n')
                print(f"✅ Created {init_file}")

def fix_static_attack_query_counts():
    """Fix query counting for all static attacks"""
    
    static_attacks = [
        "attacks/black_box/CodeAttack/code_attack.py",
        "attacks/black_box/InceptionAttack/inception_attack.py",
        "attacks/black_box/CodeChameleon/code_chameleon_attack.py"
    ]
    
    for attack_file in static_attacks:
        if os.path.exists(attack_file):
            with open(attack_file, 'r') as f:
                content = f.read()
            
            # Look for return statements that might have 0 queries
            if "return 0," in content or "query_count = 0" in content:
                # Replace with proper query counting
                content = content.replace("return 0,", "return 1,")
                content = content.replace("query_count = 0", "query_count = 1  # Static attack counts as 1 query")
                
                with open(attack_file, 'w') as f:
                    f.write(content)
                print(f"✅ Fixed query counting in {attack_file}")

def create_harmbench_config():
    """Create configuration for HarmBench dataset evaluation"""
    
    harmbench_config = {
        "experiment_name": "harmbench_full_evaluation",
        "description": "Full evaluation using HarmBench 400 prompts dataset",
        "output_dir": "experiments/results/harmbench_evaluation",
        "save_intermediate": True,
        "log_level": "INFO",
        
        "target_model": {
            "model_type": "api",
            "model_name": "gpt-4.1-mini",
            "api_provider": "openai",
            "parameters": {
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 1.0
            }
        },
        
        "judger_config": "judgers/usage_examples/configs/gpt_judger_harmful_binary.json",
        
        "test_data_source": {
            "type": "harmbench",
            "file_path": "data/harmbench_behaviors_text_all.csv",
            "behavior_column": "Behavior",
            "sample_size": 400,  # Use first 400 behaviors
            "random_sample": False
        },
        
        "attacks": [
            {
                "name": "ReNeLLM",
                "config": "attacks/usage_examples/configs/black_box/ReNeLLM.json",
                "enabled": True,
                "description": "Rewrite and nest attack"
            },
            {
                "name": "PAIR",
                "config": "attacks/usage_examples/configs/black_box/PAIR.json",
                "enabled": True,
                "description": "Prompt Automatic Iterative Refinement"
            },
            {
                "name": "GPTFUZZER",
                "config": "attacks/usage_examples/configs/black_box/GPTFUZZER.json",
                "enabled": True,
                "description": "Template-based fuzzing"
            },
            {
                "name": "ArtPrompt",
                "config": "attacks/usage_examples/configs/black_box/ArtPrompt.json",
                "enabled": True,
                "description": "ASCII art-based attacks"
            },
            {
                "name": "CodeAttack",
                "config": "attacks/usage_examples/configs/black_box/CodeAttack.json",
                "enabled": True,
                "description": "Code-based prompt encoding"
            },
            {
                "name": "FlipAttack",
                "config": "attacks/usage_examples/configs/black_box/FlipAttack.json",
                "enabled": True,
                "description": "Character replacement attacks"
            },
            {
                "name": "InceptionAttack",
                "config": "attacks/usage_examples/configs/black_box/InceptionAttack.json",
                "enabled": True,
                "description": "Nested instruction attacks"
            },
            {
                "name": "PastTense",
                "config": "attacks/usage_examples/configs/black_box/PastTense.json",
                "enabled": True,
                "description": "Tense modification attacks"
            }
        ],
        
        "defenses": [
            {
                "name": "None",
                "config": None,
                "enabled": True,
                "description": "No defense (baseline)"
            },
            {
                "name": "BackTranslation",
                "config": "defenses/usage_examples/configs/back_translation.json",
                "enabled": True,
                "description": "Translation round-trips"
            },
            {
                "name": "SmoothLLM",
                "config": "defenses/usage_examples/configs/smooth_llm.json",
                "enabled": True,
                "description": "Input smoothing through perturbations"
            },
            {
                "name": "PrimeGuard",
                "config": "defenses/usage_examples/configs/prime_guard.json",
                "enabled": True,
                "description": "Multi-layer output analysis"
            },
            {
                "name": "JailGuard",
                "config": "defenses/usage_examples/configs/jailguard_defense.json",
                "enabled": True,
                "description": "Jailbreak detection"
            }
        ],
        
        "evaluation_settings": {
            "timeout_per_attack": 300,  # Longer timeout for 400 prompts
            "max_retries": 2,
            "detailed_logging": True,
            "collect_intermediate_outputs": True,
            "max_total_time": 7200,  # 2 hours max
            "batch_size": 10,  # Process in batches to manage memory
            "checkpoint_frequency": 5  # Save every 5 combinations
        },
        
        "metrics": [
            "attack_success_rate",
            "total_queries",
            "avg_queries_per_prompt",
            "execution_time",
            "defense_effectiveness"
        ],
        
        "output_formats": [
            "json",
            "csv",
            "markdown"
        ]
    }
    
    config_dir = "experiments/batch_configs"
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "harmbench_evaluation.json")
    
    with open(config_file, 'w') as f:
        json.dump(harmbench_config, f, indent=2)
    
    print(f"✅ Created HarmBench evaluation config: {config_file}")
    return config_file

def main():
    """Run all fixes"""
    print("🔧 Fixing PromptSecurity framework issues...")
    project_root = os.environ.get("PROMPTSECURITY_ROOT", os.getcwd())
    os.chdir(project_root)
    
    # Apply fixes
    fix_query_counting()
    fix_attack_configs()
    create_missing_init_files()
    fix_static_attack_query_counts()
    harmbench_config = create_harmbench_config()
    
    print("\n🎉 All fixes applied successfully!")
    print(f"\n📝 To run HarmBench evaluation:")
    print(f"python experiments/experiments/comprehensive_evaluator.py {harmbench_config}")

if __name__ == "__main__":
    main()
