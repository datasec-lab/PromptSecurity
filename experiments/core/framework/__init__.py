"""
PromptSecurity 可扩展实验框架

为复杂实验提供强大的框架支持，包括：
- 分阶段实验管理
- 高级配置和自定义
- 进度监控和结果分析
- 可扩展的实验类型

使用示例:
    from experiments.core.framework import ExperimentFramework
    
    framework = ExperimentFramework()
    phase1 = framework.create_phase1_experiment()
    result = phase1.execute()
"""

from .base import BaseExperiment, ExperimentFramework
from .phase1 import Phase1ModelEvaluation

__all__ = ['BaseExperiment', 'ExperimentFramework', 'Phase1ModelEvaluation']