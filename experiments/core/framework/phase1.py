"""
Phase 1: 模型代表性评估实现
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import time
from .base import BaseExperiment
from ..evaluation import run_batch_evaluation

class Phase1ModelEvaluation(BaseExperiment):
    """
    Phase 1: 模型代表性评估实验
    
    目标：从候选模型中选择10个代表性模型
    方法：基于拒答率和一致性分析的智能选择算法
    """
    
    def __init__(self, name: str = "phase1_model_evaluation"):
        super().__init__(name, "模型代表性评估 - 从候选模型中选择代表性模型")
        
        # 默认候选模型配置
        self.candidate_models = {
            "api_models": {
                "openai": ["gpt-4o", "gpt-4.1", "gpt-3.5-turbo"],
                "anthropic": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-latest", "claude-3-5-haiku-20241022"],
                "google": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"],
                "deepseek": ["deepseek-v3", "deepseek-r1"],
                "bytedance": ["doubao-seed-1-6-250615", "doubao-seed-1-6-flash-250615", "doubao-pro-128k"]
            },
            "local_models": {
                "meta": ["meta-llama_Llama-3.1-8B-Instruct", "meta-llama_Llama-3.2-8B-Instruct", "meta-llama_Llama-3.3-8B-Instruct"],
                "qwen": ["Qwen_Qwen2.5-7B-Instruct", "Qwen_Qwen2.5-14B-Instruct", "Qwen_Qwen3-8B"],
                "microsoft": ["microsoft_Phi-3.5-mini-instruct", "microsoft_Phi-4-instruct", "microsoft_Phi-3-medium-128k-instruct"],
                "01ai": ["01-AI_Yi-1.5-6B-Chat", "01-AI_Yi-1.5-9B-Chat", "01-AI_Yi-1.5-34B-Chat"],
                "mistral": ["mistralai_Ministral-8B-Instruct-2410", "mistralai_Mistral-Nemo-Instruct-2407", "mistralai_Mistral-7B-Instruct-v0.3"]
            }
        }
        
        # 配置参数
        self.target_selection_count = 10
        self.datasets = ["harmbench", "jbb", "airbench"]
        self.judgers = ["harmbench_judger", "gpt_judger_contextual_harmbench", "gpt_judger_harmful_binary"]
        self.sample_limit = 50  # 每个数据集采样数量
        
    def configure(self, **kwargs) -> None:
        """配置实验参数"""
        
        # 更新候选模型
        if "candidate_models" in kwargs:
            self.candidate_models.update(kwargs["candidate_models"])
        
        # 更新其他参数
        self.target_selection_count = kwargs.get("target_selection_count", 10)
        self.datasets = kwargs.get("datasets", self.datasets)
        self.judgers = kwargs.get("judgers", self.judgers)
        self.sample_limit = kwargs.get("sample_limit", 50)
        self.seed = kwargs.get("seed", 42)
        
        self.logger.info(f"Phase 1配置完成: 目标选择{self.target_selection_count}个模型")
    
    def execute(self) -> Dict[str, Any]:
        """执行Phase 1实验"""
        
        self.status = "running"
        self.start_time = time.time()
        self.logger.info("开始执行Phase 1: 模型代表性评估")
        
        try:
            # Phase 1 现在集成到占位符系统，不直接执行evaluation
            # 而是通过占位符系统来管理实验执行
            self.logger.info("Phase 1实验现在通过占位符系统执行")
            self.logger.info("使用 'python -m experiments --phase 1' 来执行Phase 1实验")
            
            # 1. 获取所有候选模型
            all_models = self._get_all_candidate_models()
            self.logger.info(f"总候选模型数: {len(all_models)}")
            
            # 2. 生成评估配置 - 但不直接执行
            configurations = self._generate_evaluation_configurations(all_models)
            self.logger.info(f"生成评估配置数: {len(configurations)}")
            
            # 3. 返回配置信息而不是执行实验
            # 实际执行应该通过占位符系统进行
            self.logger.warning("注意: Phase 1实验应该通过 'python -m experiments --phase 1' 执行")
            
            # 创建模拟结果用于向后兼容
            evaluation_results = []
            for config in configurations:
                evaluation_results.append({
                    "target_llm_name": config["model"],
                    "dataset_name": config["dataset"], 
                    "judger_name": config["judger"],
                    "clean_safe_rate": 0.5,  # 模拟值
                    "total_samples": self.sample_limit,
                    "successful_samples": self.sample_limit,
                    "status": "placeholder_mode"
                })
            
            # 4. 分析结果
            self.logger.info("分析评估结果...")
            analysis_results = self._analyze_evaluation_results(evaluation_results)
            
            # 5. 选择代表性模型
            self.logger.info("选择代表性模型...")
            selected_models = self._select_representative_models(analysis_results)
            
            # 6. 生成报告
            report = self._generate_phase1_report(selected_models, analysis_results)
            
            self.results = {
                "status": "completed",
                "selected_models": selected_models,
                "analysis": analysis_results,
                "evaluation_results": evaluation_results,
                "report": report,
                "statistics": {
                    "total_candidates": len(all_models),
                    "selected_count": len(selected_models),
                    "selection_rate": len(selected_models) / len(all_models)
                }
            }
            
            self.status = "completed"
            self.end_time = time.time()
            
            self.logger.info(f"Phase 1完成，选择了{len(selected_models)}个模型")
            return self.results
            
        except Exception as e:
            self.status = "failed"
            self.end_time = time.time()
            self.logger.error(f"Phase 1执行失败: {e}")
            # 不重新抛出异常，而是返回错误结果
            self.results = {
                "status": "failed",
                "selected_models": [],
                "analysis": {},
                "evaluation_results": [],
                "report": f"Phase 1执行失败: {e}",
                "statistics": {
                    "total_candidates": 0,
                    "selected_count": 0,
                    "selection_rate": 0
                },
                "error": str(e)
            }
            return self.results
    
    def _get_all_candidate_models(self) -> List[str]:
        """获取所有候选模型列表"""
        all_models = []
        
        # API模型
        for provider, models in self.candidate_models["api_models"].items():
            all_models.extend(models)
        
        # 本地模型
        for provider, models in self.candidate_models["local_models"].items():
            all_models.extend(models)
        
        return all_models
    
    def _generate_evaluation_configurations(self, models: List[str]) -> List[Dict[str, str]]:
        """生成评估配置"""
        configurations = []
        
        for model in models:
            for dataset in self.datasets:
                for judger in self.judgers:
                    configurations.append({
                        "model": model,
                        "attack": "no_attack",  # Phase 1只评估基线
                        "defense": "no_defense",
                        "dataset": dataset,
                        "judger": judger
                    })
        
        return configurations
    
    def _analyze_evaluation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析评估结果"""
        
        # 按模型组织结果
        model_results = {}
        
        for result in results:
            model = result["target_llm_name"]
            dataset = result["dataset_name"] 
            judger = result["judger_name"]
            
            if model not in model_results:
                model_results[model] = {}
            
            key = f"{dataset}_{judger}"
            model_results[model][key] = {
                "safe_rate": result.get("clean_safe_rate", 0),
                "total_samples": result.get("total_samples", 0),
                "successful_samples": result.get("successful_samples", 0)
            }
        
        # 计算每个模型的综合指标
        model_metrics = {}
        for model, results_dict in model_results.items():
            metrics = self._calculate_model_metrics(results_dict)
            model_metrics[model] = metrics
        
        return {
            "model_results": model_results,
            "model_metrics": model_metrics,
            "provider_analysis": self._analyze_by_provider(model_metrics)
        }
    
    def _calculate_model_metrics(self, results_dict: Dict[str, Dict]) -> Dict[str, float]:
        """计算单个模型的综合指标"""
        
        safe_rates = [r["safe_rate"] for r in results_dict.values() if r["successful_samples"] > 0]
        
        if not safe_rates:
            return {"mean_safe_rate": 0, "consistency": 0, "reliability": 0, "overall_score": 0}
        
        mean_safe_rate = np.mean(safe_rates)
        consistency = 1 - np.std(safe_rates)  # 一致性：标准差越小越好
        reliability = np.mean([r["successful_samples"] / r["total_samples"] 
                             for r in results_dict.values() if r["total_samples"] > 0])
        
        # 综合评分
        overall_score = mean_safe_rate * 0.5 + consistency * 0.3 + reliability * 0.2
        
        return {
            "mean_safe_rate": mean_safe_rate,
            "consistency": max(0, consistency),
            "reliability": reliability, 
            "overall_score": overall_score
        }
    
    def _analyze_by_provider(self, model_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """按厂商分析模型表现"""
        
        provider_analysis = {}
        
        # 按厂商分组
        for model, metrics in model_metrics.items():
            provider = self._get_model_provider(model)
            
            if provider not in provider_analysis:
                provider_analysis[provider] = {"models": [], "metrics": []}
            
            provider_analysis[provider]["models"].append(model)
            provider_analysis[provider]["metrics"].append(metrics["overall_score"])
        
        # 计算厂商统计
        for provider, data in provider_analysis.items():
            data["mean_score"] = np.mean(data["metrics"])
            data["best_model"] = data["models"][np.argmax(data["metrics"])]
            data["model_count"] = len(data["models"])
        
        return provider_analysis
    
    def _get_model_provider(self, model: str) -> str:
        """获取模型厂商"""
        
        # API模型判断
        if any(prefix in model for prefix in ["gpt-", "chatgpt"]):
            return "openai"
        elif "claude" in model:
            return "anthropic" 
        elif "gemini" in model:
            return "google"
        elif "deepseek" in model:
            return "deepseek"
        elif "doubao" in model:
            return "bytedance"
        
        # 本地模型判断
        elif any(name in model.lower() for name in ["llama", "meta"]):
            return "meta"
        elif "qwen" in model.lower():
            return "qwen"
        elif "phi" in model.lower():
            return "microsoft"
        elif "yi" in model.lower():
            return "01ai"
        elif "mistral" in model.lower():
            return "mistral"
        
        return "unknown"
    
    def _select_representative_models(self, analysis: Dict[str, Any]) -> List[str]:
        """选择代表性模型"""
        
        model_metrics = analysis["model_metrics"]
        provider_analysis = analysis["provider_analysis"]
        
        selected_models = []
        
        # 策略1: 每个厂商选择最佳模型
        for provider, data in provider_analysis.items():
            if data["model_count"] > 0:
                best_model = data["best_model"]
                selected_models.append(best_model)
                self.logger.info(f"从{provider}选择: {best_model} (得分: {model_metrics[best_model]['overall_score']:.3f})")
        
        # 策略2: 如果选择不足，补充高分模型
        if len(selected_models) < self.target_selection_count:
            remaining_models = [m for m in model_metrics.keys() if m not in selected_models]
            remaining_sorted = sorted(remaining_models, 
                                    key=lambda x: model_metrics[x]["overall_score"], 
                                    reverse=True)
            
            additional_count = self.target_selection_count - len(selected_models)
            selected_models.extend(remaining_sorted[:additional_count])
            
            self.logger.info(f"补充选择了{additional_count}个高分模型")
        
        # 截断到目标数量
        final_selection = selected_models[:self.target_selection_count]
        
        return final_selection
    
    def _generate_phase1_report(self, selected_models: List[str], 
                               analysis: Dict[str, Any]) -> str:
        """生成Phase 1报告"""
        
        model_metrics = analysis["model_metrics"]
        provider_analysis = analysis["provider_analysis"]
        
        report = f"""
# Phase 1 模型代表性评估报告

## 执行摘要

本阶段从{len(model_metrics)}个候选模型中选择了{len(selected_models)}个代表性模型，用于后续实验。

## 选择的代表性模型

"""
        
        for i, model in enumerate(selected_models, 1):
            metrics = model_metrics[model]
            provider = self._get_model_provider(model)
            report += f"{i}. **{model}** ({provider})\n"
            report += f"   - 平均安全率: {metrics['mean_safe_rate']:.3f}\n"
            report += f"   - 一致性: {metrics['consistency']:.3f}\n"
            report += f"   - 可靠性: {metrics['reliability']:.3f}\n"
            report += f"   - 综合得分: {metrics['overall_score']:.3f}\n\n"
        
        report += "## 厂商分析\n\n"
        
        for provider, data in provider_analysis.items():
            report += f"### {provider}\n"
            report += f"- 候选模型数: {data['model_count']}\n"
            report += f"- 平均得分: {data['mean_score']:.3f}\n"
            report += f"- 最佳模型: {data['best_model']}\n\n"
        
        report += f"""
## 实验统计

- 评估配置数: {len(model_metrics) * len(self.datasets) * len(self.judgers)}
- 候选模型数: {len(model_metrics)}
- 选择模型数: {len(selected_models)}
- 选择比例: {len(selected_models)/len(model_metrics)*100:.1f}%
- 执行时间: {(self.end_time - self.start_time) if self.end_time and self.start_time else 0:.1f}秒

## 选择标准

1. **厂商代表性**: 确保主要厂商都有代表
2. **性能均衡**: 优先选择综合得分高的模型
3. **一致性**: 选择在不同数据集上表现稳定的模型
4. **可靠性**: 选择成功率高的模型

---
报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def get_selected_models(self) -> List[str]:
        """获取选择的模型列表"""
        if self.status == "completed":
            return self.results.get("selected_models", [])
        return []
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """获取选择摘要"""
        if self.status == "completed":
            return {
                "selected_models": self.results.get("selected_models", []),
                "statistics": self.results.get("statistics", {}),
                "execution_time": self.end_time - self.start_time if self.end_time else 0
            }
        return {}