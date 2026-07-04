"""
PromptSecurity 通用评估管道 - 已弃用

⚠️ 警告：此模块已弃用，请使用占位符系统替代 ⚠️

新的使用方式:
    from experiments.core.placeholder_system import ExperimentPlaceholder
    from experiments.core.placeholder_runner import PlaceholderExperimentRunner
    
    # 创建占位符
    manager = ExperimentPlaceholder()
    config = {"model": "gpt-4o", "attack": "ArtPrompt", ...}
    placeholder_file = manager.create_placeholder(config)
    
    # 执行实验
    runner = PlaceholderExperimentRunner()
    results = runner.run_batch_placeholders([placeholder_file])

旧的直接评估功能已重定向到占位符系统。
"""

from .pipeline import run_evaluation, run_batch_evaluation
from .config import EvaluationConfig

__all__ = ['run_evaluation', 'run_batch_evaluation', 'EvaluationConfig']