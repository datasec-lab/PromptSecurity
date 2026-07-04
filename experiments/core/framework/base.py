"""
实验框架基础类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import logging
from pathlib import Path

class BaseExperiment(ABC):
    """实验基类"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self.results = {}
        
    @abstractmethod
    def configure(self, **kwargs) -> None:
        """配置实验参数"""
        pass
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """执行实验"""
        pass
    
    def get_status(self) -> str:
        """获取实验状态"""
        return self.status
    
    def get_results(self) -> Dict[str, Any]:
        """获取实验结果"""
        return self.results
    
    def get_execution_time(self) -> float:
        """获取执行时间"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

class ExperimentFramework:
    """实验框架管理器"""
    
    def __init__(self):
        self.experiments = {}
        self.logger = logging.getLogger(__name__)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def create_phase1_experiment(self, **kwargs) -> 'Phase1ModelEvaluation':
        """创建Phase 1实验"""
        from .phase1 import Phase1ModelEvaluation
        
        experiment = Phase1ModelEvaluation(**kwargs)
        self.experiments[experiment.name] = experiment
        return experiment
    
    def get_experiment(self, name: str) -> Optional[BaseExperiment]:
        """获取实验实例"""
        return self.experiments.get(name)
    
    def list_experiments(self) -> List[str]:
        """列出所有实验"""
        return list(self.experiments.keys())
    
    def get_experiment_status(self, name: str) -> str:
        """获取实验状态"""
        experiment = self.get_experiment(name)
        return experiment.get_status() if experiment else "not_found"