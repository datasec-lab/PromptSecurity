"""
评估配置管理
"""

import os
from pathlib import Path
from typing import Dict, Any

# 路径配置
# config.py is now in experiments/core/evaluation/, need to go up 4 levels
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
LOGS_DIR = PROJECT_ROOT / "experiments" / "logs"

# 创建目录
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

class EvaluationConfig:
    """评估配置类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        "evaluation": {
            "sample_limit": 100,
            "timeout": 300,
            "max_retries": 3,
            "batch_size": 10,
            "temperature": 0.0,
            "max_tokens": 512
        },
        "phase1": {
            "target_selection_count": 10,
            "datasets": ["harmbench", "jbb", "airbench"],
            "judgers": ["harmbench_judger", "gpt_judger_contextual_harmbench", "gpt_judger_harmful_binary"],
            "sample_limit": 50
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取配置，支持环境变量覆盖"""
        config = cls.DEFAULT_CONFIG.copy()
        
        # 从环境变量读取配置
        if os.getenv("PROMPTSECURITY_SAMPLE_LIMIT"):
            config["evaluation"]["sample_limit"] = int(os.getenv("PROMPTSECURITY_SAMPLE_LIMIT"))
        
        if os.getenv("PROMPTSECURITY_LOG_LEVEL"):
            config["logging"]["level"] = os.getenv("PROMPTSECURITY_LOG_LEVEL")
            
        if os.getenv("PROMPTSECURITY_TIMEOUT"):
            config["evaluation"]["timeout"] = int(os.getenv("PROMPTSECURITY_TIMEOUT"))
        
        return config
    
    @classmethod
    def get_evaluation_config(cls) -> Dict[str, Any]:
        """获取评估配置"""
        return cls.get_config()["evaluation"]
    
    @classmethod
    def get_phase1_config(cls) -> Dict[str, Any]:
        """获取Phase 1配置"""
        return cls.get_config()["phase1"]