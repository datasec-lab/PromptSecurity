"""
配置管理系统

支持多种配置源：JSON文件、YAML文件、命令行参数、环境变量、Python字典等。
提供灵活的配置合并、验证和模板生成功能。
"""

import json
import yaml
import os
import argparse
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    
    # 基本组件
    model: str = "gpt-4o"
    attack: str = "no_attack"
    defense: str = "no_defense"
    dataset: str = "harmbench"
    judger: str = "harmbench_judger"
    
    # 实验参数
    sample_limit: int = 100
    experiment_type: str = "single"  # single, batch, phase1, phase2, phase3, phase4
    experiment_name: Optional[str] = None
    
    # 阶段实验参数
    phase_config: Dict[str, Any] = field(default_factory=dict)
    
    # 批量实验配置
    batch_configs: List[Dict[str, Any]] = field(default_factory=list)
    
    # 输出配置
    output_path: Optional[str] = None
    save_results: bool = True
    
    # 执行配置
    parallel: bool = False
    workers: int = 1
    verbose: bool = False
    
    # 其他配置
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        
        result = {
            "model": self.model,
            "attack": self.attack,
            "defense": self.defense,
            "dataset": self.dataset,
            "judger": self.judger,
            "sample_limit": self.sample_limit,
            "experiment_type": self.experiment_type
        }
        
        if self.experiment_name:
            result["experiment_name"] = self.experiment_name
        
        if self.phase_config:
            result["phase_config"] = self.phase_config
        
        if self.batch_configs:
            result["batch_configs"] = self.batch_configs
        
        if self.output_path:
            result["output_path"] = self.output_path
        
        result.update({
            "save_results": self.save_results,
            "parallel": self.parallel,
            "workers": self.workers,
            "verbose": self.verbose
        })
        
        if self.extra_params:
            result.update(self.extra_params)
        
        return result


class ConfigManager:
    """
    配置管理器
    
    支持多种配置源的加载、合并和验证
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Security: Define allowed directories for config files
        self.allowed_dirs = [
            Path.cwd(),  # Current working directory
            Path.cwd() / "experiments",
            Path.cwd() / "configs", 
            Path.cwd() / "judgers" / "usage_examples" / "configs",
            Path.cwd() / "models" / "usage_examples" / "configs",
            Path.cwd() / "attacks" / "usage_examples" / "configs",
            Path.cwd() / "defenses" / "usage_examples" / "configs",
            Path.cwd() / "dataset_loaders" / "usage_examples" / "configs"
        ]
        
    def load_config(self, 
                   config_sources: List[Union[str, Dict, ExperimentConfig]] = None,
                   **kwargs) -> ExperimentConfig:
        """
        从多个配置源加载配置
        
        Args:
            config_sources: 配置源列表，支持：
                - 文件路径字符串 (.json, .yaml, .yml)
                - 配置字典
                - ExperimentConfig对象
            **kwargs: 额外参数，优先级最高
        
        Returns:
            合并后的ExperimentConfig对象
        """
        
        # 从默认配置开始
        merged_config = ExperimentConfig()
        
        # 加载环境变量配置
        env_config = self._load_from_env()
        merged_config = self._merge_configs(merged_config, env_config)
        
        # 处理配置源
        if config_sources:
            for source in config_sources:
                loaded_config = self._load_single_source(source)
                merged_config = self._merge_configs(merged_config, loaded_config)
        
        # 应用额外参数
        if kwargs:
            kwargs_config = self._dict_to_config(kwargs)
            merged_config = self._merge_configs(merged_config, kwargs_config)
        
        return merged_config
    
    def _load_single_source(self, source: Union[str, Dict, ExperimentConfig]) -> ExperimentConfig:
        """加载单个配置源"""
        
        if isinstance(source, ExperimentConfig):
            return source
        
        elif isinstance(source, dict):
            return self._dict_to_config(source)
        
        elif isinstance(source, str):
            if os.path.exists(source):
                return self._load_from_file(source)
            else:
                raise FileNotFoundError(f"配置文件不存在: {source}")
        
        else:
            raise ValueError(f"不支持的配置源类型: {type(source)}")
    
    def _load_from_file(self, file_path: str) -> ExperimentConfig:
        """从文件加载配置"""
        
        # Security: Validate file path to prevent path traversal attacks
        if not self._is_safe_path(file_path):
            raise ValueError(f"不安全的文件路径: {file_path}")
        
        path = Path(file_path).resolve()  # Resolve to absolute path
        
        # Additional security check: ensure file is within allowed directories
        if not self._is_within_allowed_dirs(path):
            raise ValueError(f"文件路径不在允许的目录内: {path}")
        
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        
        else:
            raise ValueError(f"不支持的配置文件格式: {path.suffix}")
        
        return self._dict_to_config(data)
    
    def _is_safe_path(self, file_path: str) -> bool:
        """检查文件路径是否安全，防止路径遍历攻击"""
        
        # Check for common path traversal patterns
        dangerous_patterns = ['../', '..\\', '..', '/etc/', '/root/', '~/', '%2e%2e']
        file_path_lower = file_path.lower()
        
        for pattern in dangerous_patterns:
            if pattern in file_path_lower:
                self.logger.warning(f"检测到可疑路径模式: {pattern} in {file_path}")
                return False
        
        # Check for absolute paths that could be dangerous
        path = Path(file_path)
        if path.is_absolute():
            # Only allow absolute paths within the project directory
            try:
                path.resolve().relative_to(Path.cwd())
            except ValueError:
                self.logger.warning(f"不安全的绝对路径: {file_path}")
                return False
        
        return True
    
    def _is_within_allowed_dirs(self, resolved_path: Path) -> bool:
        """检查解析后的路径是否在允许的目录内"""
        
        try:
            for allowed_dir in self.allowed_dirs:
                try:
                    resolved_path.relative_to(allowed_dir.resolve())
                    return True
                except ValueError:
                    continue
            
            self.logger.warning(f"文件路径不在允许的目录内: {resolved_path}")
            return False
            
        except Exception as e:
            self.logger.error(f"路径验证时出错: {e}")
            return False
    
    def _is_safe_env_value(self, value: str, config_key: str) -> bool:
        """验证环境变量值的安全性"""
        
        # Check for dangerous characters that could be used for injection
        dangerous_chars = [';', '&', '|', '`', '$', '(', ')', '<', '>', '"', "'"]
        
        # Special handling for different config types
        if config_key == 'output_path':
            # Output paths need special validation
            return self._is_safe_path(value)
        
        # General safety check for dangerous characters
        for char in dangerous_chars:
            if char in value:
                self.logger.warning(f"环境变量值包含危险字符 '{char}': {value}")
                return False
        
        # Check length to prevent buffer overflow attacks
        if len(value) > 1000:
            self.logger.warning(f"环境变量值过长: {len(value)} 字符")
            return False
        
        return True
    
    def _load_from_env(self) -> ExperimentConfig:
        """从环境变量加载配置"""
        
        env_config = {}
        
        # 定义环境变量映射
        env_mappings = {
            'PS_MODEL': 'model',
            'PS_ATTACK': 'attack',
            'PS_DEFENSE': 'defense',
            'PS_DATASET': 'dataset',
            'PS_JUDGER': 'judger',
            'PS_SAMPLE_LIMIT': 'sample_limit',
            'PS_EXPERIMENT_TYPE': 'experiment_type',
            'PS_OUTPUT_PATH': 'output_path',
            'PS_VERBOSE': 'verbose'
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Security: Validate environment variables to prevent injection
                if not self._is_safe_env_value(value, config_key):
                    self.logger.warning(f"跳过不安全的环境变量值: {env_var}={value}")
                    continue
                    
                # 处理数据类型
                if config_key in ['sample_limit', 'workers']:
                    try:
                        parsed_value = int(value)
                        # Security: Add reasonable limits to prevent resource exhaustion
                        if config_key == 'sample_limit' and parsed_value > 10000:
                            self.logger.warning(f"样本限制过大，限制为10000: {parsed_value}")
                            parsed_value = 10000
                        elif config_key == 'workers' and parsed_value > 50:
                            self.logger.warning(f"工作进程数过大，限制为50: {parsed_value}")
                            parsed_value = 50
                        env_config[config_key] = parsed_value
                    except ValueError:
                        self.logger.warning(f"环境变量 {env_var} 的值 {value} 无法转换为整数")
                elif config_key in ['verbose', 'save_results', 'parallel']:
                    env_config[config_key] = value.lower() in ['true', '1', 'yes', 'on']
                else:
                    env_config[config_key] = value
        
        return self._dict_to_config(env_config)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> ExperimentConfig:
        """将字典转换为ExperimentConfig"""
        
        # 分离已知字段和额外参数
        known_fields = {
            'model', 'attack', 'defense', 'dataset', 'judger',
            'sample_limit', 'experiment_type', 'experiment_name',
            'phase_config', 'batch_configs', 'output_path',
            'save_results', 'parallel', 'workers', 'verbose'
        }
        
        config_data = {}
        extra_params = {}
        
        for key, value in data.items():
            if key in known_fields:
                config_data[key] = value
            else:
                extra_params[key] = value
        
        if extra_params:
            config_data['extra_params'] = extra_params
        
        return ExperimentConfig(**config_data)
    
    def _merge_configs(self, base: ExperimentConfig, override: ExperimentConfig) -> ExperimentConfig:
        """合并两个配置对象"""
        
        # 将配置转换为字典进行合并
        base_dict = base.to_dict()
        override_dict = override.to_dict()
        
        # 深度合并
        merged_dict = self._deep_merge(base_dict, override_dict)
        
        return self._dict_to_config(merged_dict)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并字典"""
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config: ExperimentConfig, file_path: str):
        """保存配置到文件"""
        
        path = Path(file_path)
        config_dict = config.to_dict()
        
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix.lower() == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        else:
            raise ValueError(f"不支持的配置文件格式: {path.suffix}")
        
        self.logger.info(f"配置已保存到: {file_path}")
    
    def generate_template_configs(self) -> Dict[str, ExperimentConfig]:
        """生成模板配置"""
        
        templates = {
            "basic_evaluation": ExperimentConfig(
                model="gpt-4o",
                attack="no_attack",
                defense="no_defense",
                dataset="harmbench",
                judger="harmbench_judger",
                sample_limit=50,
                experiment_name="基础评估"
            ),
            
            "attack_evaluation": ExperimentConfig(
                model="claude-3-5-sonnet-latest",
                attack="ArtPrompt",
                defense="no_defense",
                dataset="jbb",
                judger="gpt_judger_harmful_binary",
                sample_limit=100,
                experiment_name="攻击效果评估"
            ),
            
            "defense_evaluation": ExperimentConfig(
                model="gemini-2.0-flash",
                attack="GPTFUZZER",
                defense="smooth_llm",
                dataset="airbench",
                judger="gpt_judger_contextual_harmbench",
                sample_limit=30,
                experiment_name="防御效果评估"
            ),
            
            "local_model_evaluation": ExperimentConfig(
                model="Llama-3.1-8B-Instruct",
                attack="GCGAttack",
                defense="rpo",
                dataset="harmbench",
                judger="harmbench_judger",
                sample_limit=20,
                experiment_name="本地模型白盒评估"
            ),
            
            "phase1_experiment": ExperimentConfig(
                experiment_type="phase1",
                phase_config={
                    "target_selection_count": 10,
                    "sample_limit": 30,
                    "datasets": ["harmbench", "jbb"],
                    "judgers": ["harmbench_judger", "gpt_judger_contextual_harmbench"]
                },
                experiment_name="Phase 1 模型代表性评估"
            ),
            
            "batch_experiment": ExperimentConfig(
                experiment_type="batch",
                batch_configs=[
                    {"model": "gpt-4o", "attack": "no_attack", "defense": "no_defense"},
                    {"model": "gpt-4o", "attack": "ArtPrompt", "defense": "no_defense"},
                    {"model": "gpt-4o", "attack": "ArtPrompt", "defense": "smooth_llm"}
                ],
                sample_limit=20,
                experiment_name="批量对比实验"
            )
        }
        
        return templates
    
    def validate_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """验证配置的有效性"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 验证实验类型
        valid_types = ["single", "batch", "phase1", "phase2", "phase3", "phase4"]
        if config.experiment_type not in valid_types:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"无效的实验类型: {config.experiment_type}. 有效类型: {valid_types}"
            )
        
        # 验证批量实验配置
        if config.experiment_type == "batch":
            if not config.batch_configs:
                validation_result["valid"] = False
                validation_result["errors"].append("批量实验需要提供batch_configs")
        
        # 验证阶段实验配置
        if config.experiment_type.startswith("phase"):
            if not config.phase_config:
                validation_result["warnings"].append("阶段实验建议提供phase_config")
        
        # 验证样本限制
        if config.sample_limit <= 0:
            validation_result["valid"] = False
            validation_result["errors"].append("sample_limit必须大于0")
        
        # 验证工作进程数
        if config.workers <= 0:
            validation_result["valid"] = False
            validation_result["errors"].append("workers必须大于0")
        
        return validation_result
    
    def create_cli_parser(self) -> argparse.ArgumentParser:
        """创建命令行解析器"""
        
        parser = argparse.ArgumentParser(
            description="PromptSecurity 配置管理",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # 配置文件
        parser.add_argument("--config", "-c", type=str, action="append",
                          help="配置文件路径，支持多个配置文件")
        
        # 基本组件
        parser.add_argument("--model", "-m", type=str, help="模型名称")
        parser.add_argument("--attack", "-a", type=str, help="攻击方法")
        parser.add_argument("--defense", "-d", type=str, help="防御方法")
        parser.add_argument("--dataset", type=str, help="数据集")
        parser.add_argument("--judger", "-j", type=str, help="评判器")
        
        # 实验参数
        parser.add_argument("--sample-limit", "-n", type=int, help="样本限制")
        parser.add_argument("--experiment-type", "-t", type=str,
                          choices=["single", "batch", "phase1", "phase2", "phase3", "phase4"],
                          help="实验类型")
        parser.add_argument("--experiment-name", type=str, help="实验名称")
        
        # 阶段实验
        parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], 
                          help="运行分阶段实验（等同于--experiment-type phase{N})")
        
        # 输出配置
        parser.add_argument("--output", "-o", type=str, help="输出文件路径")
        parser.add_argument("--no-save", action="store_true", help="不保存结果")
        
        # 执行配置
        parser.add_argument("--parallel", action="store_true", help="并行执行")
        parser.add_argument("--workers", "-w", type=int, default=1, help="工作进程数")
        parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
        
        # 功能操作
        parser.add_argument("--generate-template", type=str, 
                          help="生成模板配置文件")
        parser.add_argument("--validate-only", action="store_true", 
                          help="仅验证配置不执行实验")
        
        return parser
    
    def parse_cli_args(self, args: Optional[List[str]] = None) -> ExperimentConfig:
        """解析命令行参数并返回配置对象"""
        
        parser = self.create_cli_parser()
        parsed_args = parser.parse_args(args)
        
        # 处理特殊参数
        config_sources = []
        
        # 加载配置文件
        if parsed_args.config:
            config_sources.extend(parsed_args.config)
        
        # 处理phase参数
        cli_params = {}
        if parsed_args.phase:
            cli_params["experiment_type"] = f"phase{parsed_args.phase}"
        
        # 转换命令行参数
        for arg_name, arg_value in vars(parsed_args).items():
            if arg_value is not None and arg_name not in ["config", "phase", "generate_template", "validate_only"]:
                # 转换参数名（下划线转换）
                config_key = arg_name.replace("_", "_")
                if config_key == "no_save":
                    cli_params["save_results"] = not arg_value
                else:
                    cli_params[config_key] = arg_value
        
        # 加载配置
        config = self.load_config(config_sources, **cli_params)
        
        return config, parsed_args


# 全局配置管理器实例
_global_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager