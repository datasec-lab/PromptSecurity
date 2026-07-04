#!/usr/bin/env python3
"""
示范如何在实验系统中使用新的攻击加载器
"""

from attacks.loader import load_attack, list_available_attacks

# 示例：实验系统如何加载攻击
def load_attack_for_experiment(attack_name: str, target_model, target_model_parameters):
    """在实验系统中加载攻击
    
    Args:
        attack_name: 攻击名称（文件夹名）
        target_model: 目标模型
        target_model_parameters: 目标模型参数
    
    Returns:
        攻击实例
    """
    if attack_name == "no_attack":
        # 特殊处理 no_attack
        return load_attack("no_attack", target_model=target_model)
    
    # 加载其他攻击
    return load_attack(
        attack_name, 
        target_model=target_model,
        target_model_parameters=target_model_parameters
    )

# 示例：列出所有可用攻击
def get_all_attack_names():
    """获取所有可用的攻击名称"""
    attacks = list_available_attacks()
    all_attacks = []
    
    # 合并黑盒和白盒攻击
    all_attacks.extend(attacks['black_box'])
    all_attacks.extend(attacks['white_box'])
    
    return sorted(set(all_attacks))  # 去重并排序

# 示例：根据攻击名称映射
ATTACK_NAME_MAPPING = {
    # 旧名称 -> 新名称（文件夹名）
    'ArtPromptAttack': 'ArtPrompt',
    'PairAttack': 'PAIR',
    'PastTenseAttack': 'PastTense',
    'IFSJAttack': 'IFSJ',
    'DrAttackAttack': 'DrAttack',
    'PersuasiveInContextAttack': 'PersuasiveInContext',
    'MultilingualJailbreakAttack': 'MultilingualJailbreak',
    'GCGWhiteBoxAttack': 'GCGAttack',
    'AutoDANAttack': 'AutoDAN',
    'COLDAttack': 'COLD',
    # 保持不变的
    'FlipAttack': 'FlipAttack',
    'ABJAttack': 'ABJAttack',
    'CodeAttack': 'CodeAttack',
    'CodeChameleon': 'CodeChameleon',
    'DRA': 'DRA',
    'ReNeLLM': 'ReNeLLM',
    'InceptionAttack': 'InceptionAttack',
    'TapAttack': 'TapAttack',
    'GPTFUZZER': 'GPTFUZZER',
    'no_attack': 'no_attack',
}

def convert_old_attack_name(old_name: str) -> str:
    """将旧的攻击名称转换为新的文件夹名称"""
    return ATTACK_NAME_MAPPING.get(old_name, old_name)

if __name__ == "__main__":
    print("🔍 可用攻击列表:")
    all_attacks = get_all_attack_names()
    for i, attack in enumerate(all_attacks, 1):
        print(f"{i:2d}. {attack}")
    
    print(f"\n共 {len(all_attacks)} 种攻击方法")
    
    print("\n📋 攻击名称映射表:")
    for old_name, new_name in ATTACK_NAME_MAPPING.items():
        if old_name != new_name:
            print(f"  {old_name} -> {new_name}")