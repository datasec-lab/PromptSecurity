#!/usr/bin/env python3
"""
生成 Phase 4 实验占位符配置
"""

import json
import sys
from pathlib import Path
from itertools import product

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.core.placeholder_system import ExperimentPlaceholder


def generate_phase4_placeholders():
    """生成Phase 4的所有实验占位符"""
    
    # Phase 4 配置
    models = [
        # API models
        "gpt-4.1",
        "claude-haiku-4-5-20251001", 
        "gemini-2.0-flash",
        "deepseek-v3",
        "doubao-seed-1-6-flash-250615",
        # Local models
        "meta-llama-Llama-3.1-8B-Instruct",
        "Qwen-Qwen3-8B",
        "microsoft-Phi-4-instruct",
        "01-AI-Yi-1.5-6B-Chat",
        "mistralai-Ministral-8B-Instruct-2410"
    ]
    
    attacks = [
        "no_attack",
        "ABJAttack", "ArtPrompt", "CodeAttack", "CodeChameleon",
        "DRA", "DrAttack", "FlipAttack", "GPTFUZZER", "IFSJAttack",
        "InceptionAttack", "MultilingualJailbreakAttack", "PAIR",
        "PastTense", "PersuasiveInContext", "ReNeLLM", "TapAttack",
        "DRA_custom", "GPTFUZZER_aggressive", "PersuasiveInContextAttack",
        "AutoDANAttack", "COLDAttack", "GCGAttack"
    ]
    
    defenses = [
        "no_defense",
        "input_filter_defense", "output_filter_defense",
        "smooth_llm", "jailguard_defense", "back_translation",
        "perplexity_filter", "gradsafe_defense", "rpo", "prime_guard"
    ]
    
    datasets = ["harmbench", "jbb", "airbench"]
    
    judgers = [
        "harmbench_judger",
        "gpt_judger_contextual_harmbench",
        "gpt_judger_harmful_binary",
        "gpt_judger_harmbench_style",
        "gpt_judger_openai_policy",
        "gpt_judger_tap_style",
        "rejection_prefix_judger"
    ]
    
    # 白盒攻击列表
    whitebox_attacks = ["AutoDANAttack", "COLDAttack", "GCGAttack"]
    
    # API模型列表
    api_models = [
        "gpt-4.1",
        "claude-haiku-4-5-20251001",
        "gemini-2.0-flash",
        "deepseek-v3",
        "doubao-seed-1-6-flash-250615"
    ]
    
    # 创建占位符管理器
    placeholder_mgr = ExperimentPlaceholder("experiments/placeholders_phase4", seed=42)
    
    # 生成所有组合
    total_count = 0
    skipped_count = 0
    
    print("开始生成Phase 4实验占位符...")
    
    for model, attack, defense, dataset in product(models, attacks, defenses, datasets):
        # 过滤白盒攻击 + API模型的组合
        if attack in whitebox_attacks and model in api_models:
            skipped_count += 1
            continue
        
        # 创建多judger配置
        config = {
            "model": model,
            "attack": attack,
            "defense": defense,
            "dataset": dataset,
            "judger": judgers,  # 使用所有7个judgers
            "seed": 42,
            "sample_limit": 100  # 每个数据集100样本
        }
        
        # 生成占位符
        placeholder = placeholder_mgr.create_placeholder(config)
        if placeholder:
            total_count += 1
            if total_count % 1000 == 0:
                print(f"已生成 {total_count} 个占位符...")
    
    print(f"\n生成完成!")
    print(f"总计生成: {total_count} 个占位符")
    print(f"跳过 (白盒攻击+API模型): {skipped_count} 个组合")
    print(f"占位符保存在: {placeholder_mgr.placeholders_dir}")
    
    # 计算实验总数
    expected_total = len(models) * len(attacks) * len(defenses) * len(datasets)
    expected_skip = len(api_models) * len(whitebox_attacks) * len(defenses) * len(datasets)
    expected_actual = expected_total - expected_skip
    
    print(f"\n验证:")
    print(f"预期总数: {expected_total}")
    print(f"预期跳过: {expected_skip}")
    print(f"预期实际: {expected_actual}")
    print(f"实际生成: {total_count}")
    
    if total_count != expected_actual:
        print(f"警告: 生成数量与预期不符!")


if __name__ == "__main__":
    generate_phase4_placeholders()