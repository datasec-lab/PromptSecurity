"""
PromptSecurity 统一实验接口

提供单一用户接口，支持所有模块的所有方法作为配置选项。
支持多种输入方式：字典配置、JSON文件、命令行参数等。
"""

import json
import argparse
import os
import sys
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

# 添加项目根目录到路径
# unified_interface.py is now in experiments/core/, need to go up 3 levels
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.core.framework import ExperimentFramework

# Import new loaders instead of registry
from attacks.loader import load_attack, list_available_attacks, get_attack_info, get_attack_config_path
from defenses.loader import load_defense, list_defenses, get_defense_config, get_defense_config_path
from models.loader import load_model, list_available_models, get_model_config_path
from judgers.judger_loader import load_judger, get_available_judgers


class PromptSecurityInterface:
    """
    PromptSecurity统一接口 - 使用新的动态加载器
    
    支持所有模块的所有方法，提供灵活的配置方式
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 使用新的加载器系统
        self._initialize_loaders()
        
        # 加载所有可用方法
        self.available_methods = self._get_all_methods()
    
    def _initialize_loaders(self):
        """初始化加载器系统"""
        self.logger.info("初始化动态加载器系统...")
        
        # 缓存可用组件列表
        self._attacks = list_available_attacks()
        # 为攻击添加"all"键以保持兼容性
        self._attacks["all"] = self._attacks.get("black_box", []) + self._attacks.get("white_box", [])
        
        self._defenses = list_defenses()
        self._models = list_available_models()
        self._judgers = get_available_judgers()
        
        total_components = (
            len(self._attacks.get('black_box', [])) + 
            len(self._attacks.get('white_box', [])) +
            len(self._defenses) +
            len(self._models.get('api', [])) +
            len(self._models.get('local', [])) +
            len(self._judgers)
        )
        
        self.logger.info(f"加载器初始化完成，发现 {total_components} 个组件")
    
    def _get_all_methods(self) -> Dict[str, Any]:
        """获取所有可用方法"""
        # 为攻击添加"all"键以保持兼容性
        attacks_with_all = dict(self._attacks)
        attacks_with_all["all"] = self._attacks.get("black_box", []) + self._attacks.get("white_box", [])
        
        # 添加支持的数据集列表
        datasets = ["harmbench", "jbb", "airbench"]
        
        return {
            "attacks": attacks_with_all,
            "defenses": self._defenses,
            "models": self._models,
            "judgers": self._judgers,
            "datasets": datasets
        }
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_available_methods(self, component_type: Optional[str] = None) -> Dict[str, Any]:
        """获取可用方法"""
        
        if component_type:
            return self.available_methods.get(component_type, {})
        return self.available_methods
    
    def run_experiment(self, 
                      config: Optional[Union[str, Dict]] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        运行实验的统一接口
        
        Args:
            config: 配置文件路径或配置字典
            **kwargs: 直接指定的参数，会覆盖config中的设置
            
        Examples:
            # 方式1: 使用字典配置
            result = interface.run_experiment({
                "model": "gpt-4o",
                "attack": "ArtPrompt",
                "defense": "smooth_llm",
                "dataset": "harmbench",
                "judger": "harmbench_judger"
            })
            
            # 方式2: 使用关键字参数
            result = interface.run_experiment(
                model="claude-3-5-sonnet-latest",
                attack="no_attack",
                defense="no_defense",
                dataset="jbb",
                judger="gpt_judger_harmful_binary"
            )
            
            # 方式3: 使用配置文件
            result = interface.run_experiment("experiments/configs/my_experiment.json")
            
            # 方式4: 混合使用
            result = interface.run_experiment(
                config="base_config.json",
                model="gpt-4o",  # 覆盖配置文件中的模型
                sample_limit=50
            )
        """
        
        # 解析配置
        experiment_config = self._parse_config(config, **kwargs)
        
        # 验证配置
        validated_config = self._validate_config(experiment_config)
        
        # 决定运行方式
        if validated_config.get("experiment_type") == "phase1":
            result = self._run_phase1_experiment(validated_config)
        elif validated_config.get("experiment_type") == "batch":
            result = self._run_batch_experiment(validated_config)
        else:
            result = self._run_single_experiment(validated_config)
        
        # 占位符模式：结果已保存在占位符系统中
        # 不再自动保存到results目录
        
        return result
    
    def run_batch_experiments(self,
                            configurations: List[Dict],
                            **kwargs) -> List[Dict[str, Any]]:
        """
        运行批量实验
        
        Args:
            configurations: 实验配置列表
            **kwargs: 全局参数
        """
        
        self.logger.info(f"开始批量实验，共{len(configurations)}个配置")
        
        results = []
        for i, config in enumerate(configurations):
            self.logger.info(f"进度: {i+1}/{len(configurations)}")
            
            # 合并全局参数
            merged_config = {**config, **kwargs}
            result = self.run_experiment(merged_config)
            results.append(result)
        
        return results
    
    def run_phase_experiment(self,
                           phase: int,
                           **kwargs) -> Dict[str, Any]:
        """
        运行分阶段实验
        
        Args:
            phase: 阶段编号 (1-4)
            **kwargs: 实验参数
        """
        
        if phase == 1:
            return self._run_phase1_experiment(kwargs)
        elif phase == 2:
            return self._run_phase2_experiment(kwargs)
        elif phase == 3:
            return self._run_phase3_experiment(kwargs)
        elif phase == 4:
            return self._run_phase4_experiment(kwargs)
        else:
            raise ValueError(f"不支持的阶段: {phase}")
    
    def _parse_config(self, config: Optional[Union[str, Dict]], **kwargs) -> Dict:
        """解析配置"""
        
        final_config = {}
        
        # 处理配置文件或字典
        if isinstance(config, str):
            # 配置文件路径
            if os.path.exists(config):
                with open(config, 'r') as f:
                    final_config = json.load(f)
            else:
                raise FileNotFoundError(f"配置文件未找到: {config}")
        elif isinstance(config, dict):
            # 配置字典
            final_config = config.copy()
        
        # 关键字参数覆盖配置
        final_config.update(kwargs)
        
        return final_config
    
    def _validate_config(self, config: Dict) -> Dict:
        """验证和补全配置"""
        
        validated = config.copy()
        
        # 设置默认值
        defaults = {
            "model": "gpt-4o",
            "attack": "no_attack", 
            "defense": "no_defense",
            "dataset": "harmbench",
            "judger": "harmbench_judger",
            "sample_limit": 100
        }
        
        for key, default_value in defaults.items():
            if key not in validated:
                validated[key] = default_value
        
        # 验证组件有效性
        self._validate_component("attack", validated["attack"])
        self._validate_component("defense", validated["defense"])
        self._validate_component("model", validated["model"])
        self._validate_component("judger", validated["judger"])
        self._validate_component("dataset", validated["dataset"])
        
        # 验证兼容性
        self._validate_compatibility(validated)
        
        return validated
    
    def _validate_component(self, component_type: str, component_name):
        """验证组件有效性"""
        
        available = self.available_methods
        
        if component_type == "attack":
            all_attacks = available["attacks"]["all"]
            if component_name not in all_attacks:
                raise ValueError(f"未知攻击方法: {component_name}. 可用方法: {all_attacks}")
        
        elif component_type == "defense":
            all_defenses = available["defenses"]
            if component_name not in all_defenses:
                raise ValueError(f"未知防御方法: {component_name}. 可用方法: {all_defenses}")
        
        elif component_type == "model":
            all_models = available["models"].get("api", []) + available["models"].get("local", [])
            if component_name not in all_models:
                # 如果不在预定义列表中，尝试推断是否是有效模型名
                self.logger.warning(f"模型 {component_name} 不在预定义列表中，将尝试加载")
        
        elif component_type == "judger":
            # 处理单个judger或judger列表
            if isinstance(component_name, list):
                for jname in component_name:
                    if jname not in available["judgers"]:
                        raise ValueError(f"未知评判器: {jname}. 可用方法: {available['judgers']}")
            else:
                if component_name not in available["judgers"]:
                    raise ValueError(f"未知评判器: {component_name}. 可用方法: {available['judgers']}")
        
        elif component_type == "dataset":
            all_datasets = available["datasets"]
            if component_name not in all_datasets:
                raise ValueError(f"未知数据集: {component_name}. 可用方法: {all_datasets}")
    
    def _validate_compatibility(self, config: Dict):
        """验证组件兼容性"""
        
        attack = config["attack"]
        defense = config["defense"]
        model = config["model"]
        
        # 使用新的验证方法验证兼容性
        validation_result = self.validate_combination(attack, defense, model)
        
        if not validation_result["valid"]:
            errors = validation_result["errors"]
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")
        
        # 记录警告
        for warning in validation_result["warnings"]:
            self.logger.warning(warning)
    
    def generate_experiment_combinations(self, models, attacks, defenses, datasets, judgers, **kwargs):
        """
        生成所有有效的实验组合
        
        Args:
            models: 模型列表
            attacks: 攻击方法列表
            defenses: 防御方法列表
            datasets: 数据集列表
            judgers: 评判器列表
            **kwargs: 其他参数
            
        Returns:
            List[Dict]: 有效的实验配置列表
        """
        from itertools import product
        
        # 确保所有输入都是列表
        models = models if isinstance(models, list) else [models]
        attacks = attacks if isinstance(attacks, list) else [attacks]
        defenses = defenses if isinstance(defenses, list) else [defenses]
        datasets = datasets if isinstance(datasets, list) else [datasets]
        judgers = judgers if isinstance(judgers, list) else [judgers]
        
        valid_configs = []
        invalid_combinations = []
        
        # 生成所有组合
        for model, attack, defense, dataset, judger in product(models, attacks, defenses, datasets, judgers):
            config = {
                "model": model,
                "attack": attack,
                "defense": defense,
                "dataset": dataset,
                "judger": judger,
                **kwargs
            }
            
            try:
                # 验证组件存在性
                self._validate_component("model", model)
                self._validate_component("attack", attack)
                self._validate_component("defense", defense)
                self._validate_component("dataset", dataset)
                self._validate_component("judger", judger)
                
                # 验证兼容性
                validation_result = self.validate_combination(attack, defense, model)
                
                if validation_result["valid"]:
                    valid_configs.append(config)
                else:
                    invalid_combinations.append({
                        "config": config,
                        "reason": "; ".join(validation_result.get("errors", []))
                    })
                    
            except Exception as e:
                invalid_combinations.append({
                    "config": config,
                    "reason": str(e)
                })
        
        # 报告结果
        total_combinations = len(models) * len(attacks) * len(defenses) * len(datasets) * len(judgers)
        self.logger.info(f"生成实验组合: {len(valid_configs)}/{total_combinations} 个有效")
        
        if invalid_combinations:
            self.logger.warning(f"跳过 {len(invalid_combinations)} 个无效组合:")
            for item in invalid_combinations[:5]:  # 只显示前5个
                self.logger.warning(f"  - {item['config']['model']} + {item['config']['attack']} + {item['config']['defense']}: {item['reason']}")
            if len(invalid_combinations) > 5:
                self.logger.warning(f"  ... 还有 {len(invalid_combinations) - 5} 个无效组合")
        
        return valid_configs
    
    def _is_local_model(self, model_name: str) -> bool:
        """判断是否是本地模型"""
        
        local_prefixes = ["Llama", "Qwen", "Phi", "Yi", "gemma", "Mistral", "internlm"]
        return any(prefix in model_name for prefix in local_prefixes)
    
    def _run_single_experiment(self, config: Dict) -> Dict[str, Any]:
        """运行单次实验 - 重定向到占位符系统"""
        from experiments.core.placeholder_system import ExperimentPlaceholder
        from experiments.core.placeholder_runner import PlaceholderExperimentRunner
        
        # 创建占位符并执行
        placeholder_manager = ExperimentPlaceholder()
        placeholder_file = placeholder_manager.create_placeholder(config)
        
        runner = PlaceholderExperimentRunner()
        results = runner.run_batch_placeholders([placeholder_file], workers=1)
        
        return results[0] if results else {"status": "failed", "error": "占位符执行失败"}
    
    # 自动保存结果功能已移除 - 使用占位符系统替代
    # 所有结果保存在 experiments/placeholders/ 目录中
    
    def _run_batch_experiment(self, config: Dict) -> List[Dict[str, Any]]:
        """运行批量实验 - 重定向到占位符系统"""
        from experiments.core.placeholder_system import ExperimentPlaceholder
        from experiments.core.placeholder_runner import PlaceholderExperimentRunner
        
        configurations = config.get("configurations", [])
        placeholder_manager = ExperimentPlaceholder()
        placeholder_files = []
        
        for cfg in configurations:
            placeholder_file = placeholder_manager.create_placeholder(cfg)
            placeholder_files.append(placeholder_file)
        
        runner = PlaceholderExperimentRunner()
        return runner.run_batch_placeholders(placeholder_files, workers=1)
    
    def _run_phase1_experiment(self, config: Dict) -> Dict[str, Any]:
        """运行Phase 1实验"""
        
        framework = ExperimentFramework()
        phase1 = framework.create_phase1_experiment()
        
        # 配置实验
        phase1.configure(**config)
        
        # 执行实验
        return phase1.execute()
    
    def _run_phase2_experiment(self, config: Dict) -> Dict[str, Any]:
        """运行Phase 2实验（评判器一致性评估）"""
        # TODO: 实现Phase 2
        raise NotImplementedError("Phase 2 尚未实现")
    
    def _run_phase3_experiment(self, config: Dict) -> Dict[str, Any]:
        """运行Phase 3实验（数据集优化）"""
        # TODO: 实现Phase 3
        raise NotImplementedError("Phase 3 尚未实现")
    
    def _run_phase4_experiment(self, config: Dict) -> Dict[str, Any]:
        """运行Phase 4实验（全面攻击防御评估）"""
        from experiments.core.placeholder_system import ExperimentPlaceholder
        from experiments.core.placeholder_runner import PlaceholderExperimentRunner
        
        # 创建占位符管理器
        placeholder_manager = ExperimentPlaceholder(seed=config.get('seed', 42))
        
        # 生成Phase4占位符
        placeholder_files = placeholder_manager.generate_phase4_placeholders(
            sample_limit=config.get('sample_limit')
        )
        
        # 如果只是生成占位符模式，返回结果
        if config.get('generate_only', False):
            return {
                "status": "placeholders_generated",
                "placeholder_count": len(placeholder_files),
                "placeholders_dir": str(placeholder_manager.placeholders_dir),
                "message": "Phase4占位符生成完成，使用 --run-placeholders 执行实验"
            }
        
        # 执行占位符实验
        runner = PlaceholderExperimentRunner(
            verbose=config.get('verbose', False),
            max_length=config.get('max_length', 200),
            seed=config.get('seed', 42)
        )
        
        workers = config.get('workers', 1)
        results = runner.run_batch_placeholders(placeholder_files, workers)
        
        # 统计结果
        completed = len([r for r in results if r.get("status") == "completed"])
        failed = len([r for r in results if r.get("status") == "failed"])
        skipped = len([r for r in results if r.get("status") == "skipped"])
        
        return {
            "status": "completed",
            "phase": 4,
            "total_experiments": len(results),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "placeholder_count": len(placeholder_files),
            "execution_summary": f"✅{completed} ❌{failed} ⏭️{skipped} | 📁{len(results)}",
            "placeholders_dir": str(placeholder_manager.placeholders_dir)
        }
    
    def list_available_methods(self, component_type: Optional[str] = None) -> Dict[str, Any]:
        """列出可用方法"""
        
        return self.get_available_methods(component_type)
    
    def get_compatible_methods(self, base_method: str, base_type: str) -> Dict[str, List[str]]:
        """获取与指定方法兼容的其他方法"""
        
        # 所有组件现在都是兼容的（基于动态加载）
        return {
            "attacks": self._attacks.get("black_box", []) + self._attacks.get("white_box", []),
            "defenses": self._defenses,
            "models": self._models.get("api", []) + self._models.get("local", []),
            "judgers": self._judgers
        }
    
    def get_method_info(self, method_name: str, method_type: str) -> Optional[Dict[str, Any]]:
        """获取方法详细信息"""
        
        if method_type == "attacks":
            return get_attack_info(method_name)
        elif method_type == "defenses":
            return get_defense_config(method_name)
        elif method_type == "models":
            return {"name": method_name, "type": "api" if method_name in self._models.get("api", []) else "local"}
        elif method_type == "judgers":
            return {"name": method_name, "available": method_name in self._judgers}
        return None
    
    def refresh_methods(self):
        """刷新可用方法列表"""
        
        self._initialize_loaders()
        self.available_methods = self._get_all_methods()
        self.logger.info("方法列表已刷新")
    
    def validate_combination(self, attack: str, defense: str, model: str) -> Dict[str, Any]:
        """验证攻击、防御、模型组合的兼容性"""
        
        errors = []
        warnings = []
        
        # 检查攻击
        if attack != "no_attack":
            attack_info = get_attack_info(attack)
            if not attack_info.get("available", False):
                errors.append(f"攻击方法不存在: {attack}")
            elif attack_info.get("type") == "white_box":
                # 白盒攻击检查模型类型
                if model in self._models.get("api", []):
                    errors.append(f"白盒攻击 {attack} 不支持API模型 {model}")
        
        # 检查防御  
        if defense != "no_defense":
            if defense not in self._defenses:
                errors.append(f"防御方法不存在: {defense}")
        
        # 检查模型
        all_models = self._models.get("api", []) + self._models.get("local", [])
        if model not in all_models:
            errors.append(f"模型不存在: {model}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def generate_example_configs(self) -> Dict[str, Dict]:
        """生成示例配置"""
        
        examples = {
            "basic_evaluation": {
                "model": "gpt-4o",
                "attack": "no_attack",
                "defense": "no_defense",
                "dataset": "harmbench",
                "judger": "harmbench_judger",
                "sample_limit": 50
            },
            "attack_evaluation": {
                "model": "claude-3-5-sonnet-latest",
                "attack": "ArtPrompt",
                "defense": "no_defense",
                "dataset": "jbb",
                "judger": "gpt_judger_harmful_binary",
                "sample_limit": 100
            },
            "defense_evaluation": {
                "model": "gemini-2.0-flash",
                "attack": "GPTFUZZER",
                "defense": "smooth_llm",
                "dataset": "airbench",
                "judger": "gpt_judger_contextual_harmbench",
                "sample_limit": 30
            },
            "local_model_evaluation": {
                "model": "Llama-3.1-8B-Instruct",
                "attack": "GCGAttack",  # 白盒攻击
                "defense": "rpo",       # 白盒防御
                "dataset": "harmbench",
                "judger": "harmbench_judger",
                "sample_limit": 20
            },
            "phase1_experiment": {
                "experiment_type": "phase1",
                "target_selection_count": 10,
                "sample_limit": 30,
                "datasets": ["harmbench", "jbb"],
                "judgers": ["harmbench_judger", "gpt_judger_contextual_harmbench"]
            },
            "batch_experiment": {
                "experiment_type": "batch",
                "configurations": [
                    {"model": "gpt-4o", "attack": "no_attack", "defense": "no_defense"},
                    {"model": "gpt-4o", "attack": "ArtPrompt", "defense": "no_defense"},
                    {"model": "gpt-4o", "attack": "ArtPrompt", "defense": "smooth_llm"}
                ],
                "sample_limit": 20
            }
        }
        
        return examples


def create_cli_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    
    parser = argparse.ArgumentParser(
        description="PromptSecurity 统一实验接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础评估
  python -m experiments.unified_interface --model gpt-4o --attack no_attack --defense no_defense
  
  # 攻击评估
  python -m experiments.unified_interface --model claude-3-5-sonnet-latest --attack ArtPrompt
  
  # 防御评估  
  python -m experiments.unified_interface --model gpt-4o --attack GPTFUZZER --defense smooth_llm
  
  # 使用配置文件
  python -m experiments.unified_interface --config my_experiment.json
  
  # Phase 1实验
  python -m experiments.unified_interface --phase 1 --sample-limit 30
  
  # 列出可用方法
  python -m experiments.unified_interface --list attacks
  
  # 占位符模式已替代缓存功能
  python -m experiments.unified_interface --model gpt-4o --attack ArtPrompt
        """
    )
    
    # 基本组件选项
    parser.add_argument("--model", nargs="*", help="模型名称（可指定多个，空格分隔）")
    parser.add_argument("--attack", nargs="*", help="攻击方法（可指定多个，空格分隔）")
    parser.add_argument("--defense", nargs="*", help="防御方法（可指定多个，空格分隔）")
    parser.add_argument("--dataset", nargs="*", help="数据集（可指定多个，空格分隔）")
    parser.add_argument("--judger", nargs="*", help="评判器（可指定多个，空格分隔）")
    parser.add_argument("--multi-judger", action="store_true", 
                       help="多judger模式：在同一实验中同时使用所有指定的judger")
    
    # 实验选项
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--sample-limit", type=int, help="样本限制")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="运行分阶段实验")
    
    # 功能选项
    parser.add_argument("--list", type=str, nargs="?", const="all", 
                       choices=["all", "attacks", "defenses", "models", "judgers", "datasets"],
                       help="列出可用方法")
    parser.add_argument("--examples", action="store_true", help="显示示例配置")
    parser.add_argument("--output", type=str, help="结果输出文件")
    
    # 缓存功能已移除 - 使用占位符系统中的结果复用替代
    # 原有缓存参数不再支持: --no-cache, --clear-cache, --cache-dir, --cache-stats
    
    # 其他选项
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--stats", action="store_true",
                        help="运行完批量实验后输出统计汇总")
    parser.add_argument("--export-csv", type=str,
                        help="将完整结果导出为 CSV")
    
    return parser


def main():
    """命令行主函数"""
    
    parser = create_cli_parser()
    args = parser.parse_args()
    
    interface = PromptSecurityInterface()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 列出可用方法
    if args.list:
        methods = interface.list_available_methods(
            None if args.list == "all" else args.list
        )
        print(json.dumps(methods, indent=2, ensure_ascii=False))
        return
    
    # 显示示例配置
    if args.examples:
        examples = interface.generate_example_configs()
        print(json.dumps(examples, indent=2, ensure_ascii=False))
        return
    
    # 运行实验
    try:
        if args.phase:
            # 分阶段实验
            config = {k.replace("_", "_"): v for k, v in vars(args).items() 
                     if v is not None and k not in ["phase", "list", "examples", "output", "verbose"]}
            result = interface.run_phase_experiment(args.phase, **config)
        else:
            # 常规实验
            config = {}
            if args.config:
                config["config"] = args.config
            
            # 检查是否需要生成组合
            multi_params = False
            for key in ["model", "attack", "defense", "dataset"]:
                value = getattr(args, key)
                if value and len(value) > 1:
                    multi_params = True
                    break
            
            if multi_params:
                # 生成所有组合并运行批量实验
                models = args.model or ["gpt-4o"]  # 默认模型
                attacks = args.attack or ["no_attack"]
                defenses = args.defense or ["no_defense"]
                datasets = args.dataset or ["harmbench"]
                judgers = args.judger or ["harmbench_judger"]
                
                # 如果judger只有一个，保持字符串格式
                if len(judgers) == 1:
                    judgers = judgers[0]
                
                # 准备批量实验的全局参数
                batch_kwargs = {}
                if args.sample_limit:
                    batch_kwargs["sample_limit"] = args.sample_limit
                # 缓存参数已移除 - 使用占位符系统中的结果复用
                pass
                
                # 生成有效组合
                configs = interface.generate_experiment_combinations(
                    models=models,
                    attacks=attacks,
                    defenses=defenses,
                    datasets=datasets,
                    judgers=judgers,
                    **batch_kwargs
                )
                
                if not configs:
                    print("❌ 没有生成有效的实验组合")
                    sys.exit(1)
                
                print(f"🔄 将运行 {len(configs)} 个实验组合")
                
                # 运行批量实验
                results = interface.run_batch_experiments(configs)
                
                # 汇总结果
                print("\n📊 实验结果汇总:")
                for i, (config, result) in enumerate(zip(configs, results)):
                    if result.get("status") == "completed":
                        print(f"{i+1}. {config['model']} + {config['attack']} + {config['defense']} + {config['dataset']}:")
                        print(f"   攻击成功率: {result.get('attack_success_rate', 0):.1%}")
                        print(f"   基线安全率: {result.get('clean_safe_rate', 0):.1%}")
                        print(f"   样本数: {result.get('total_samples', 0)} "
                              f"耗时: {result.get('execution_time', 0):.1f}s")
                    else:
                        print(f"{i+1}. {config['model']} + {config['attack']} + {config['defense']} + {config['dataset']}: 失败")
                        print(f"   错误详情: {result.get('error', '未知错误')}")
                
                # 统计汇总
                if args.stats:
                    # from experiments.core.reporting import summarize, build_report, export_csv, group_by
                    
                    print("\n📈 统计汇总:")
                    stats_report = build_report(results)
                    print(stats_report)

                    # 按模型分组并展示前 5 行
                    model_stats = group_by(results, "target_llm_name")
                    if model_stats:
                        print("\n按模型平均指标 (前 5):")
                        for model, metrics in list(model_stats.items())[:5]:
                            print(f"  {model:30} 安全率:{metrics['clean_safe_rate']:.1%} "
                                  f"攻击成功率:{metrics['attack_success_rate']:.1%}")

                if args.export_csv:
                    # from experiments.core.reporting import export_csv
                    export_csv(results, args.export_csv)
                    print(f"\n✅ 已导出 CSV: {args.export_csv}")
                
                # 保存批量结果
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"\n✅ 批量结果已保存到: {args.output}")
                
                return
            
            # 单个实验的处理逻辑
            # 添加命令行参数
            for key in ["sample_limit"]:
                value = getattr(args, key.replace("_", "_"))
                if value is not None:
                    config[key] = value
            
            # 缓存参数已移除 - 使用占位符系统中的结果复用
            pass
            
            # 处理单个值的参数
            for key in ["model", "attack", "defense", "dataset"]:
                value = getattr(args, key)
                if value and len(value) > 0:
                    config[key] = value[0]
            
            # 特殊处理judger参数
            if hasattr(args, 'judger') and args.judger is not None:
                # 如果只有一个judger，保持字符串格式；如果多个，使用列表
                if len(args.judger) == 0:
                    pass  # 不设置judger
                elif len(args.judger) == 1:
                    config["judger"] = args.judger[0]
                else:
                    config["judger"] = args.judger
            
            result = interface.run_experiment(**config)
        
        # 输出结果
        output_text = json.dumps(result, indent=2, ensure_ascii=False)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"结果已保存到: {args.output}")
        else:
            print(output_text)
            
    except Exception as e:
        print(f"实验执行失败: {e}")
        import traceback
        print("\n错误详情:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()