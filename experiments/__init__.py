"""
PromptSecurity 统一实验系统

一个简洁、强大的AI安全评估实验系统，提供：
- 统一的实验接口，支持所有模块的所有方法
- 用户友好的5要素评估管道
- 可扩展的实验框架和分阶段实验支持
- 灵活的配置管理和模块自动发现
- 多种输入方式：字典、JSON文件、命令行参数等

使用示例:
    # 最简单的方式 - 统一接口
    import experiments
    result = experiments.run_experiment(model="gpt-4o", attack="ArtPrompt")
    
    # 列出所有可用方法
    methods = experiments.list_methods()
    print(f"可用攻击: {methods['attacks']['all']}")
    
    # 基础评估（向后兼容）
    from experiments.evaluation import run_evaluation
    result = run_evaluation(model="gpt-4o", attack="ArtPrompt")
    
    # 高级实验框架
    from experiments.framework import ExperimentFramework
    framework = ExperimentFramework()
    phase1 = framework.create_phase1_experiment()
    results = phase1.execute()
    
    # 完整配置管理
    interface = experiments.get_interface()
    result = interface.run_experiment({
        "model": "claude-3-5-sonnet-latest",
        "attack": "GPTFUZZER", 
        "defense": "smooth_llm",
        "dataset": "harmbench",
        "judger": "gpt_judger_harmful_binary"
    })
"""

# 导出核心接口
from .core.unified_interface import PromptSecurityInterface
from .core.evaluation import run_evaluation, run_batch_evaluation
from .core.framework import ExperimentFramework, Phase1ModelEvaluation
# Registry system removed - using direct loaders instead
from .core.config_manager import get_config_manager, ExperimentConfig

# 创建全局接口实例
_global_interface = None

def get_interface() -> PromptSecurityInterface:
    """获取全局实验接口实例"""
    global _global_interface
    if _global_interface is None:
        _global_interface = PromptSecurityInterface()
    return _global_interface

# 便捷函数
def run_experiment(config=None, **kwargs):
    """便捷的实验运行函数"""
    return get_interface().run_experiment(config, **kwargs)

def list_methods(component_type=None):
    """便捷的方法列表函数"""
    return get_interface().list_available_methods(component_type)

def get_compatible(method_name, method_type):
    """便捷的兼容性查询函数"""
    return get_interface().get_compatible_methods(method_name, method_type)

__version__ = "2.0.0"
__all__ = [
    # 核心接口
    "PromptSecurityInterface",
    "get_interface",
    
    # 便捷函数
    "run_experiment",
    "list_methods", 
    "get_compatible",
    
    # 原有接口（向后兼容）
    "run_evaluation", 
    "run_batch_evaluation",
    "ExperimentFramework",
    "Phase1ModelEvaluation",
    
    # 配置管理
    "get_config_manager",
    "ExperimentConfig"
]