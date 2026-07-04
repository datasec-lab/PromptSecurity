#!/usr/bin/env python3
"""
占位符实验执行器 - 增强版

基于占位符文件执行实验，支持高级结果复用、智能执行计划和跨组件复用。
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.core.placeholder_system import ExperimentPlaceholder
from experiments.core.unified_interface import PromptSecurityInterface
from experiments.core.model_cache import get_global_model_cache
from experiments.core.display_components import ProgressDisplay, ContentFormatter, StatisticsDisplay
from experiments.core.value_standards import create_error_result
from attacks.loader import load_attack, get_attack_config_path
from defenses.loader import load_defense, get_defense_config_path
from models.loader import load_model, get_model_config_path

# Optional torch import for GPU memory management
try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class SmartPlaceholderRunner:
    """智能占位符实验执行器"""
    
    def __init__(self, placeholders_dir: str = "experiments/placeholders", seed: int = 42,
                 verbose: bool = False, max_length: int = 200, rerun_failed: bool = False,
                 force_rerun: bool = False):
        self.placeholder_manager = ExperimentPlaceholder(placeholders_dir, seed)
        self.interface = PromptSecurityInterface()
        self.seed = seed
        self.verbose = verbose  # 控制详细输出
        self.max_length = max_length  # 文本显示最大长度
        self.rerun_failed = rerun_failed  # 重跑除success以外的样本状态
        self.force_rerun = force_rerun  # 强制重跑全部样本（包含success/completed）
        self._judger_cache = {}  # 缓存已加载的judger实例
        
        # 初始化模型缓存和显示组件
        self.model_cache = get_global_model_cache()
        self.progress_display = ProgressDisplay(verbose=verbose)
        self.content_formatter = ContentFormatter(verbose=verbose)
        self.statistics_display = StatisticsDisplay()
        
        # 实验级defended model缓存，解决CUDA OOM问题
        self.experiment_defended_models = {}  # {cache_key: defended_model_info}
        self.current_experiment_id = None
    
    def _sanitize_error_message(self, error_msg: str) -> str:
        """清理错误消息以防止信息泄露"""
        import re
        
        # Remove sensitive file paths
        error_msg = re.sub(r'/[^/\s]*(/[^/\s]*)*', '[REDACTED_PATH]', error_msg)
        error_msg = re.sub(r'[A-Za-z]:\\[^\s]*', '[REDACTED_PATH]', error_msg)  # Windows paths
        
        # Remove API keys or tokens
        error_msg = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED_API_KEY]', error_msg)
        error_msg = re.sub(r'token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_-]{10,}', 'token=[REDACTED]', error_msg, flags=re.IGNORECASE)
        
        # Remove IP addresses
        error_msg = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[REDACTED_IP]', error_msg)
        
        # Remove user names from home directories without embedding local paths.
        home_prefix_pattern = "/" + "home" + r"/[^/\s]+"
        error_msg = re.sub(home_prefix_pattern, '[REDACTED_HOME]', error_msg)
        error_msg = re.sub(r'C:\\\\Users\\\\[^\\\\]+', r'[REDACTED_HOME]', error_msg)
        
        # Limit error message length
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "... [TRUNCATED]"
        
        return error_msg
    
    def run_single_experiment(self, config: Dict[str, Any], 
                            placeholder_data: Dict[str, Any] = None, 
                            sample_limit: int = None, 
                            progressive_saving: bool = True) -> Dict[str, Any]:
        """运行单个实验，支持实时保存和断点续传
        
        Args:
            config: 实验配置
            placeholder_data: 占位符数据
            sample_limit: 动态样本限制（覆盖占位符中的样本数量）
            progressive_saving: 是否启用实时保存和断点续传（默认True）
        """
        try:
            if self.verbose:
                print(f"🚀 开始执行实验: {config}")
                if progressive_saving:
                    print("   🔄 启用实时保存和断点续传")
            
            # 获取样本数据
            samples = placeholder_data.get("samples", []) if placeholder_data else []
            if not samples:
                if self.verbose:
                    print("❌ 错误: 没有样本数据")
                return {"status": "failed", "error": "没有样本数据"}
            
            # 应用动态样本限制
            if sample_limit is not None and sample_limit > 0:
                original_count = len(samples)
                samples = samples[:sample_limit]
                if self.verbose and len(samples) < original_count:
                    print(f"   📊 样本限制应用: {original_count} → {len(samples)} 个样本")
            
            # 选择执行方式
            if progressive_saving:
                # 使用新的实时保存方法
                if self.verbose:
                    print("   🔄 使用实时保存模式")
                
                sample_results = self._execute_with_progressive_saving(config, placeholder_data, sample_limit)
                
                # 计算统计信息
                success_count = len([r for r in sample_results if r.get("status") == "success"])
                
                # GPU显存清理：实验完成后清理所有模型显存
                if hasattr(self, 'model_cache'):
                    try:
                        self.model_cache.clear_cache()
                        if self.verbose:
                            print(f"   🧹 实验完成，GPU显存已清理")
                    except Exception as cleanup_error:
                        if self.verbose:
                            print(f"   ⚠️ 显存清理失败: {cleanup_error}")
                
                return {
                    "status": "completed",
                    "config": config,
                    "total_samples": len(samples),
                    "successful_samples": success_count,
                    "failed_samples": len(sample_results) - success_count,
                    "sample_results": sample_results,
                    "execution_time": sum(r.get("llm_response_time") or 0 for r in sample_results),
                    "progressive_saving": True
                }
            else:
                # 使用传统的批量处理方法
                if self.verbose:
                    print("   📦 使用传统批量处理模式")
                
                # 查找可复用的结果
                reusable_results = self.placeholder_manager.find_reusable_results(config, samples)
                
                if self.verbose:
                    print(f"   🔍 可复用结果: {len(reusable_results)} 个")
                    print(f"   📊 待执行样本: {len(samples)} 个")
                
                # 创建执行计划
                execution_plan = self.placeholder_manager.create_execution_plan(config, samples, reusable_results)
                
                # 显示复用统计
                reuse_stats = self.placeholder_manager.get_reuse_statistics(execution_plan)
                self._log_reuse_stats(reuse_stats)
                
                # 执行实验
                sample_results = self._execute_with_plan(config, execution_plan, reusable_results)
                
                # 计算统计信息
                success_count = len([r for r in sample_results if r.get("status") == "success"])
                
                # GPU显存清理：实验完成后清理所有模型显存
                if hasattr(self, 'model_cache'):
                    try:
                        self.model_cache.clear_cache()
                        if self.verbose:
                            print(f"   🧹 实验完成，GPU显存已清理")
                    except Exception as cleanup_error:
                        if self.verbose:
                            print(f"   ⚠️ 显存清理失败: {cleanup_error}")
                
                return {
                    "status": "completed",
                    "config": config,
                    "total_samples": len(samples),
                    "successful_samples": success_count,
                    "failed_samples": len(sample_results) - success_count,
                    "sample_results": sample_results,
                    "execution_time": sum(r.get("llm_response_time") or 0 for r in sample_results),
                    "reuse_stats": reuse_stats,
                    "progressive_saving": False
                }
            
        except Exception as e:
            # 总是显示实验执行失败信息
            experiment_name = f"{config.get('model', 'unknown')}_{config.get('attack', 'unknown')}_{config.get('defense', 'unknown')}_{config.get('judger', 'unknown')}"
            print(f"❌ 实验执行失败: {experiment_name}")
            
            # Security: Sanitize error messages to prevent information disclosure
            safe_error_msg = self._sanitize_error_message(str(e))
            print(f"   错误: {safe_error_msg}")
            
            # 详细错误信息在verbose模式下显示
            if self.verbose:
                import traceback
                print(f"错误堆栈: {traceback.format_exc()}")
            return {
                "status": "failed",
                "error": safe_error_msg,
                "config": config
            }
    
    def _execute_with_plan(self, config: Dict[str, Any], execution_plan: Dict[str, Any], 
                          reusable_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据执行计划执行实验"""
        all_results = []
        
        if self.verbose:
            print(f"   📋 执行计划详情:")
            print(f"      - 完全可复用: {execution_plan.get('fully_reusable_count', 0)} 个")
            print(f"      - 部分可复用: {execution_plan.get('partially_reusable_count', 0)} 个") 
            print(f"      - 需要执行: {execution_plan.get('needs_execution_count', 0)} 个")
        
        # 1. 处理完全可复用的样本
        for item in execution_plan.get("fully_reusable", []):
            sample = item["sample"]
            reused_result = item["reused_result"]
            
            result = reused_result["sample_data"].copy()
            result.update({
                "experiment_timestamp": time.time(),
                "reused_from": reused_result["source_file"]
            })
            
            # 验证复用的结果是否有效
            is_valid, validation_error = self._validate_sample_result(result)
            if is_valid:
                meets_success, reason = self._meets_success_criteria(result)
                if meets_success:
                    result["status"] = "success"
                else:
                    result["status"] = "completed"
                    if reason:
                        result["status_reason"] = reason
            else:
                result["status"] = "failed"
                result["error"] = f"复用结果验证失败: {validation_error}"
                if self.verbose:
                    print(f"      ⚠️ 复用结果验证失败: {validation_error}")
            
            all_results.append(result)
        
        # 2. 处理需要部分执行的样本
        # 修复：不重复计算样本，使用need_model_response作为基础样本列表
        # 因为所有组件都需要处理相同的样本，只是执行不同的步骤
        partial_execution_samples = execution_plan.get("need_model_response", [])
        
        for i, sample_plan in enumerate(partial_execution_samples):
            # Defensive programming: Validate sample_plan structure
            if not isinstance(sample_plan, dict):
                logger.error(f"Invalid sample_plan type: {type(sample_plan)} at index {i}")
                continue
            
            if "sample" not in sample_plan:
                logger.error(f"Sample_plan missing 'sample' key at index {i}. Available keys: {list(sample_plan.keys())}")
                continue
            
            sample = sample_plan["sample"]
            
            # Defensive programming: Validate sample structure with meaningful error
            if not isinstance(sample, dict):
                error_msg = f"Sample at index {i} is not a dictionary (type: {type(sample)}). This likely indicates malformed judger results being treated as sample data."
                logger.error(error_msg)
                continue
            
            if "clean_prompt" not in sample:
                error_msg = f"Sample at index {i} missing 'clean_prompt' key. Available keys: {list(sample.keys())}. This may indicate malformed judger output being processed as sample data."
                logger.error(error_msg)
                continue
            
            reusable = sample_plan["reusable"]
            sample_index = sample.get("sample_index", i)
            
            # 修复：正确计算总样本数，不重复计算
            total_samples = len(partial_execution_samples) + len(execution_plan.get('fully_reusable', []))
            
            # 使用新的进度显示
            self.progress_display.show_sample_progress(
                sample_index + 1, total_samples, sample.get('clean_prompt', '')
            )
            
            # 显示详细的原始prompt
            if self.verbose:
                formatted_prompt = self.content_formatter.format_prompt(
                    sample.get('clean_prompt', ''), prefix="原始Prompt"
                )
                print(formatted_prompt)
            
            try:
                # 执行缺失的组件
                new_results = self._execute_missing_components(config, sample, reusable, reusable_results)
                
                # 组合复用结果和新结果
                # Defensive programming: Validate sample structure before combining
                if not isinstance(sample, dict) or "clean_prompt" not in sample:
                    raise ValueError(f"Invalid sample structure for combining results. Sample type: {type(sample)}, keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
                
                combined_result = self.placeholder_manager.combine_results(
                    config, sample, reusable_results, new_results
                )
                
                # 在verbose模式下显示复用的模型响应和评判结果
                if self.verbose:
                    self._display_reused_results(combined_result, reusable, config)
                
                all_results.append(combined_result)
                
            except Exception as e:
                # 总是显示样本处理失败信息
                sample_index_str = sample.get('sample_index', 'unknown')
                print(f"❌ 样本 [{sample_index_str}] 处理失败: {str(e)}")
                # 详细错误信息在verbose模式下显示
                if self.verbose:
                    import traceback
                    print(f"错误堆栈: {traceback.format_exc()}")
                all_results.append({
                    "index": sample.get("sample_index", 0),
                    "sample_index": sample.get("sample_index", 0),
                    "status": "failed",
                    "error": str(e)
                })
            
            # 添加样本之间的分隔
            if i < len(partial_execution_samples) - 1:  # 不在最后一个样本后添加分隔
                self.progress_display.show_sample_separator()
        
        return all_results
    
    def _execute_with_progressive_saving(self, config: Dict[str, Any], placeholder_data: Dict[str, Any], 
                                        sample_limit: int = None) -> List[Dict[str, Any]]:
        """带有断点续传和实时保存的样本处理方法"""
        
        # 获取样本数据
        samples = placeholder_data.get("samples", [])
        sample_results = placeholder_data.get("sample_results", [])
        
        if not samples:
            if self.verbose:
                print("❌ 错误: 没有样本数据")
            return []
        
        # 应用样本限制
        if sample_limit is not None and sample_limit > 0:
            original_count = len(samples)
            samples = samples[:sample_limit]
            sample_results = sample_results[:sample_limit]
            if self.verbose and len(samples) < original_count:
                print(f"   📊 样本限制应用: {original_count} → {len(samples)} 个样本")
        
        # 断点续传检测
        completed_samples = []
        pending_samples = []
        
        for i, sample_result in enumerate(sample_results):
            status = sample_result.get("status")

            # 强制重跑：所有样本都重新执行（包括success/completed）
            if self.force_rerun:
                pending_samples.append(i)
                continue

            # 重跑非success：仅保留成功且有效的样本
            if self.rerun_failed:
                if status == "success":
                    is_valid, _ = self._validate_sample_result(sample_result)
                    if is_valid:
                        completed_samples.append(i)
                    else:
                        pending_samples.append(i)
                        if self.verbose:
                            print(f"      ⚠️ 样本 {i} 标记为success但数据无效，将重新执行")
                else:
                    pending_samples.append(i)
                continue

            # 默认模式：success/completed 且有效则跳过，其余重跑
            if status in {"success", "completed"}:
                is_valid, _ = self._validate_sample_result(sample_result)
                if is_valid:
                    completed_samples.append(i)
                else:
                    pending_samples.append(i)
                    if self.verbose:
                        print(f"      ⚠️ 样本 {i} 标记为{status}但数据无效，将重新执行")
            elif status in ["created", "failed", "error", "in_progress", "running", "pending", None]:
                pending_samples.append(i)
                if status == "in_progress":
                    if self._check_stale_in_progress(sample_result):
                        if self.verbose:
                            print(f"      ⚠️ 样本 {i} 处于in_progress状态超时，将重新执行")
                    else:
                        if self.verbose:
                            print(f"      ⚠️ 样本 {i} 处于in_progress状态，将重新执行")
        
        if self.verbose:
            print(f"   🔍 断点续传检测:")
            print(f"      - 已完成: {len(completed_samples)} 个样本")
            print(f"      - 需处理: {len(pending_samples)} 个样本")
            if completed_samples:
                print(f"      - 跳过样本: {min(completed_samples)}-{max(completed_samples)}")
        
        # 处理待执行的样本
        all_results = sample_results.copy()  # 保持原有结果
        
        for sample_index in pending_samples:
            if sample_index >= len(samples):
                continue
                
            sample = samples[sample_index]
            
            # 显示进度
            total_samples = len(samples)
            self.progress_display.show_sample_progress(
                sample_index + 1, total_samples, sample.get('clean_prompt', '')
            )
            
            if self.verbose:
                print(f"   🔄 处理样本 {sample_index + 1}/{total_samples}")
                formatted_prompt = self.content_formatter.format_prompt(
                    sample.get('clean_prompt', ''), prefix="原始Prompt"
                )
                print(formatted_prompt)
            
            try:
                # 标记样本为处理中
                sample_result = sample_results[sample_index].copy()
                sample_result["status"] = "in_progress"
                
                # 实时保存状态更新
                self.placeholder_manager.update_sample_result(config, sample_index, sample_result)
                
                # 执行样本处理
                processed_result = self._execute_single_sample(config, sample, sample_index)
                
                # 更新结果
                all_results[sample_index] = processed_result
                
                # 实时保存完成的结果
                success = self.placeholder_manager.update_sample_result(config, sample_index, processed_result)
                
                if self.verbose:
                    status_msg = "✅ 保存成功" if success else "❌ 保存失败"
                    print(f"   💾 {status_msg}: 样本 {sample_index}")
                    print("─" * 80)  # 添加样本间分隔线
                
            except Exception as e:
                # 处理单个样本失败
                print(f"❌ 样本 {sample_index + 1} 处理失败: {str(e)}")
                
                error_result = sample_results[sample_index].copy()
                error_result.update({
                    "status": "failed",
                    "error": str(e),
                    "experiment_timestamp": time.time()
                })
                
                all_results[sample_index] = error_result
                
                # 保存失败状态
                self.placeholder_manager.update_sample_result(config, sample_index, error_result)
                
                if self.verbose:
                    import traceback
                    print(f"错误堆栈: {traceback.format_exc()}")
        
        return all_results
    
    def _execute_single_sample(self, config: Dict[str, Any], sample: Dict[str, Any], 
                              sample_index: int) -> Dict[str, Any]:
        """执行单个样本的完整处理流程 - 遵循5要素流程"""
        
        # 从样本获取基础信息
        clean_prompt = sample.get("clean_prompt", "")
        
        # 创建结果模板
        result = {
            "status": "in_progress",
            "sample_index": sample_index,
            "clean_prompt": clean_prompt,
            "target_llm_name": config.get("model"),
            "target_llm_type": "api" if not self._is_local_model(config.get("model")) else "local",
            "experiment_timestamp": time.time()
        }
        
        # === 4要素流程（防御优先） ===
        if self.verbose:
            self.progress_display.show_four_element_flow_start()
        
        # 1. 防御处理: 创建defended_model (defense + model)
        defended_model_info = self._execute_defense_step(config, result)
        
        # 2. 攻击处理: 针对defended_model进行攻击
        attacked_prompt = self._execute_attack_step_with_defense(config, clean_prompt, defended_model_info, result)
        
        # 3. 模型响应: defended_model 对 attacked_prompt 的响应
        final_response, response_info = self._execute_model_step(config, attacked_prompt, defended_model_info, result)
        
        # 更新最终模型响应结果
        result.update(final_response)
        result.update(response_info)
        
        # 4. 评判处理: 评估(attacked_prompt, defended_response)组合
        judger_result = self._execute_judger_step(config, attacked_prompt, result)
        
        # Check if judger result is an error result
        if judger_result.get("judger_error") is not None:
            # All judgers failed - mark sample as failed
            result.update({
                "status": "failed",
                "error": judger_result.get("judger_error"),
                "judger_error_reason": judger_result.get("judger_error_reason"),
                "judger_error_detail": judger_result.get("judger_error_detail"),
                "judger_error_summary": judger_result.get("judger_error_summary"),
                "judger_name": config.get("judger"),
                "judger_config": {},
                "judger_result_on_clean": None,
                "judger_result_on_attack": None,
                "judger_result_on_clean_under_defense": None,
                "judger_result_on_attack_under_defense": None
            })
            
            if self.verbose:
                print(f"      ❌ 所有评判器失败: {judger_result.get('judger_error')}")
            
            return result
        
        # Extract judger config from the result (if available)
        judger_config = judger_result.get("config", {})
        
        # 统一评判器结果：单一字段 + 语义上下文
        # 在4要素流程中，评判器评估(attacked_prompt, defended_response)组合
        is_multi_judger = judger_result.get("multi_judger_results", False)
        
        # 根据attack和defense方法确定正确的评判结果字段
        attack_method = config.get("attack", "no_attack")
        defense_method = config.get("defense", "no_defense")
        
        if is_multi_judger:
            # 多评判器：根据attack/defense组合选择正确的结果字段
            if attack_method == "no_attack" and defense_method == "no_defense":
                multi_results = judger_result.get("judger_result_on_clean")
            elif attack_method == "no_attack" and defense_method != "no_defense":
                multi_results = judger_result.get("judger_result_on_clean_under_defense")
            elif attack_method != "no_attack" and defense_method == "no_defense":
                multi_results = judger_result.get("judger_result_on_attack")
            else:  # attack != "no_attack" and defense != "no_defense"
                multi_results = judger_result.get("judger_result_on_attack_under_defense")
            
            # 聚合多评判器结果
            if isinstance(multi_results, dict) and multi_results:
                # 计算所有评判器结果的平均值作为主要结果
                individual_scores = [score for score in multi_results.values() if isinstance(score, (int, float))]
                if individual_scores:
                    primary_result = sum(individual_scores) / len(individual_scores)
                    # 保留个体评判器结果用于详细分析
                    judger_individual_results = multi_results
                else:
                    primary_result = None
                    judger_individual_results = {}
            else:
                # 如果不是字典或为空，使用原始结果
                primary_result = multi_results
                judger_individual_results = {}
        else:
            # 单评判器：根据attack/defense组合选择正确的结果字段
            # 🔧 向后兼容：检查新格式和旧格式字段
            if attack_method == "no_attack" and defense_method == "no_defense":
                primary_result = judger_result.get("clean") or judger_result.get("judger_result_on_clean")
            elif attack_method == "no_attack" and defense_method != "no_defense":
                primary_result = judger_result.get("clean_defense") or judger_result.get("judger_result_on_clean_under_defense")
            elif attack_method != "no_attack" and defense_method == "no_defense":
                primary_result = judger_result.get("attack") or judger_result.get("judger_result_on_attack")
            else:  # attack != "no_attack" and defense != "no_defense"
                primary_result = judger_result.get("attack_defense") or judger_result.get("judger_result_on_attack_under_defense")
            
            # 单评判器没有个体结果
            judger_individual_results = {}
        
        # Debug: 显示提取的主要结果
        if self.verbose:
            print(f"       🎯 提取主要评判结果:")
            print(f"         攻击方法: {attack_method}, 防御方法: {defense_method}")
            print(f"         多评判器: {is_multi_judger}")
            if is_multi_judger and judger_individual_results:
                print(f"         个体评判器结果: {judger_individual_results}")
                print(f"         聚合结果: {primary_result} (平均值)")
            else:
                print(f"         主要结果: {primary_result}")
            print(f"         primary_result类型: {type(primary_result)}")
            print(f"         primary_result是否为None: {primary_result is None}")
        
        # 统一字段架构
        result.update({
            "judger_name": config.get("judger"),
            "judger_config": judger_config if not is_multi_judger else judger_result.get("judger_config", {}),
            "judger_result": primary_result,
            "judger_individual_results": judger_individual_results if is_multi_judger else {},
            "judger_fallback": judger_result.get("judger_fallback", False),
            "judger_fallback_detail": judger_result.get("judger_fallback_detail", {}),
            "judger_context": {
                "attack_method": config.get("attack", "no_attack"),
                "defense_method": config.get("defense", "no_defense"),
                "evaluation_target": f"({config.get('attack', 'no_attack')}_prompt, {config.get('defense', 'no_defense')}_response)",
                "is_multi_judger": is_multi_judger,
                "aggregation_method": "average" if is_multi_judger and judger_individual_results else "single"
            }
        })
        
        # 显示评判结果
        if self.verbose:
            self._display_judger_result(judger_result, "评判结果")
        
        # 验证样本结果的有效性
        is_valid, validation_error = self._validate_sample_result(result)
        
        if is_valid:
            meets_success, reason = self._meets_success_criteria(result)
            if meets_success:
                result["status"] = "success"
                # 显示4要素流程总结
                if self.verbose:
                    self.progress_display.show_four_element_summary(
                        config.get("attack"), config.get("defense"), 
                        config.get("model"), config.get("judger"), True
                    )
            else:
                result["status"] = "completed"
                if reason:
                    result["status_reason"] = reason
                if self.verbose:
                    print(f"      ⚠️ 样本完成但未满足success条件: {reason}")
                    self.progress_display.show_four_element_summary(
                        config.get("attack"), config.get("defense"), 
                        config.get("model"), config.get("judger"), True
                    )
        else:
            result["status"] = "failed"
            result["error"] = validation_error
            if self.verbose:
                print(f"      ❌ 样本验证失败: {validation_error}")
                self.progress_display.show_four_element_summary(
                    config.get("attack"), config.get("defense"), 
                    config.get("model"), config.get("judger"), False
                )
        
        return result
    
    def _execute_attack_step_with_defense(self, config: Dict[str, Any], clean_prompt: str, 
                                        defended_model_info: Dict[str, Any], result: Dict[str, Any]) -> str:
        """步骤2: 攻击处理 - 针对defended_model进行攻击"""
        attack_name = config.get("attack")
        
        if attack_name == "no_attack":
            if self.verbose:
                self.progress_display.show_attack_step(attack_name, "跳过")
            result.update({
                "attacked_prompt": clean_prompt,
                "attack_method": "no_attack",
                "attack_runtime": 0,
                "attack_query_count": 0,
                "attack_config": {}
            })
            return clean_prompt
        else:
            if self.verbose:
                self.progress_display.show_attack_step(attack_name, "执行中")
                
            # 使用defended_model作为攻击目标
            attacked_prompt, attack_info = self._execute_attack_against_defended_model(
                config, clean_prompt, defended_model_info)
            
            # 显示攻击结果
            if self.verbose:
                self.progress_display.show_attack_step(attack_name, "完成")
                # 支持通过环境变量控制完整prompt显示
                import os
                show_full_prompt = os.getenv('SHOW_FULL_PROMPT', 'false').lower() == 'true'
                attack_summary = self.content_formatter.format_attack_summary(
                    attack_name, clean_prompt, attacked_prompt, attack_info,
                    max_prompt_length=300, show_full_prompt=show_full_prompt
                )
                print(f"       {attack_summary}")
            
            result.update({
                "attacked_prompt": attacked_prompt,
                "attack_method": attack_name,
                "attack_runtime": attack_info.get("runtime", 0),
                "attack_query_count": attack_info.get("query_count", 0),
                "attack_config": attack_info.get("config", {})
            })
            return attacked_prompt or clean_prompt
    
    def _execute_defense_step(self, config: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """步骤1: 防御处理 - 创建defended_model"""
        defense_name = config.get("defense")
        model_name = config.get("model")
        
        if defense_name == "no_defense":
            if self.verbose:
                self.progress_display.show_defense_step(defense_name, model_name, "跳过")
            defended_model_info = {
                "defense_method": "no_defense",
                "defense_config": {},
                "model_name": model_name,
                "is_defended": False
            }
        else:
            if self.verbose:
                self.progress_display.show_defense_step(defense_name, model_name, "执行中")
                
            defended_model_info = self._get_or_create_defended_model(config)
            
            # 显示防御信息
            if self.verbose:
                status = "完成" if defended_model_info["is_defended"] else "执行中"
                self.progress_display.show_defense_step(defense_name, model_name, status)
                defense_summary = self.content_formatter.format_defense_summary(
                    defense_name, model_name, 
                    defended_model_info["defense_config"], 
                    defended_model_info["is_defended"]
                )
                print(f"       {defense_summary}")
        
        result.update({
            "defense_method": defended_model_info["defense_method"],
            "defense_config": defended_model_info["defense_config"]
        })
        return defended_model_info
    
    def _execute_model_step(self, config: Dict[str, Any], attacked_prompt: str, 
                           defended_model_info: Dict[str, Any], result: Dict[str, Any]) -> tuple:
        """步骤3: 模型响应 - defended_model对attacked_prompt的响应"""
        defense_name = defended_model_info.get("defense_method", "no_defense")
        model_name = defended_model_info.get("model_name", config.get("model"))
        
        if self.verbose:
            self.progress_display.show_model_response_step(model_name, defense_name, "执行中")
        
        defense_fallback_info = {"fallback": False, "fallback_events": []}
        if defended_model_info["is_defended"]:
            # 使用防御后的模型响应
            response, response_time, defense_fallback_info = self._get_defended_model_response(
                defended_model_info, attacked_prompt, config
            )
            response_type = "under_defense"
        else:
            # 使用原始模型响应
            response, response_time = self._get_model_response(
                config.get("model"), attacked_prompt, config
            )
            response_type = "direct"
        
        # 显示模型响应
        if self.verbose:
            self.progress_display.show_model_response_step(model_name, defense_name, "完成")
            response_summary = self.content_formatter.format_model_response_summary(
                response, response_time, response_type
            )
            print(f"       {response_summary}")
        
        # 统一字段架构：单一响应字段 + 语义上下文
        final_response = {
            "llm_response": response,
            "llm_response_time": response_time,
            "response_context": {
                "attack_method": config.get("attack", "no_attack"),
                "defense_method": defended_model_info.get("defense_method", "no_defense"),
                "is_defended": defended_model_info["is_defended"],
                "response_type": response_type
            }
        }
        
        response_info = {
            "response_type": response_type,
            "is_defended_response": defended_model_info["is_defended"],
            "defense_fallback": defense_fallback_info.get("fallback", False),
            "defense_fallback_events": defense_fallback_info.get("fallback_events", []),
            "defense_fallback_to_model": defense_fallback_info.get("fallback_to_model", False)
        }
        
        return final_response, response_info
    
    def _execute_judger_step(self, config: Dict[str, Any], attacked_prompt: str, 
                            result: Dict[str, Any]) -> Dict[str, Any]:
        """步骤4: 评判处理 - 评估(attacked_prompt, defended_response)组合"""
        judger_name = config.get('judger')
        
        if self.verbose:
            self.progress_display.show_judger_step(judger_name, "执行中")
        
        # 使用现有的_execute_judger方法，但确保它评估正确的组合
        judger_result = self._execute_judger(config, result)
        
        # 显示评判结果
        if self.verbose:
            self.progress_display.show_judger_step(judger_name, "完成")
            judger_summary = self.content_formatter.format_judger_evaluation_summary(
                judger_name, judger_result, attacked_prompt
            )
            print(f"       {judger_summary}")
        
        return judger_result

    def _validate_sample_result(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """验证样本结果是否有效 - 基于统一字段架构
        
        检查关键字段是否为null，判断样本是否真正执行成功
        
        Returns:
            (is_valid, error_message): 验证结果和错误信息
        """
        # 检查统一模型响应字段是否有效
        if result.get("llm_response") is None:
            return False, "模型响应为空(llm_response is null)"
        
        # 检查响应时间字段
        if result.get("llm_response_time") is None:
            return False, "模型响应时间为空(llm_response_time is null)"
        
        # 检查统一评判结果字段是否有效
        judger_result = result.get("judger_result")
        if judger_result is None:
            return False, "评判结果为空(judger_result is null)"
        
        # 检查评判器名称
        judger_name = result.get("judger_name")
        if judger_name is None:
            return False, "评判器名称为空(judger_name is null)"
        
        # 验证多judger情况的特殊处理
        judger_context = result.get("judger_context", {})
        is_multi_judger = judger_context.get("is_multi_judger", False)
        
        if is_multi_judger:
            # 多judger情况：检查个体评判器结果
            individual_results = result.get("judger_individual_results", {})
            if isinstance(individual_results, dict) and individual_results:
                # 确保至少有一个judger返回有效结果
                valid_results = [r for r in individual_results.values() if r is not None and isinstance(r, (int, float))]
                if not valid_results:
                    return False, "所有评判器都未返回有效结果"
            # 检查聚合结果是否为有效数值
            if not isinstance(judger_result, (int, float)):
                return False, f"多评判器聚合结果无效: {judger_result}"
        
        # 检查攻击相关字段（当有攻击时）
        attack_method = result.get("attack_method")
        if attack_method and attack_method != "no_attack":
            if result.get("attacked_prompt") is None:
                return False, "攻击后的prompt为空(attacked_prompt is null)"
        
        # 检查防御相关字段（当有防御时）
        defense_method = result.get("defense_method")
        if defense_method and defense_method != "no_defense":
            # 防御可能会过滤掉响应，所以这里不严格要求非空
            # 只需要确保defense_method字段存在即可
            pass
        
        # 验证响应上下文完整性（可选）
        response_context = result.get("response_context", {})
        if response_context:
            # 检查响应上下文是否与攻击/防御方法一致
            context_attack = response_context.get("attack_method")
            context_defense = response_context.get("defense_method")
            
            if context_attack != attack_method:
                return False, f"响应上下文攻击方法不一致: {context_attack} vs {attack_method}"
            if context_defense != defense_method:
                return False, f"响应上下文防御方法不一致: {context_defense} vs {defense_method}"
        
        return True, None

    def _meets_success_criteria(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """判断样本是否满足 success 标准（无 fallback 且攻击有改动）"""
        if result.get("defense_fallback") is not False:
            return False, "defense_fallback"
        if result.get("judger_fallback") is not False:
            return False, "judger_fallback"

        attacked_prompt = result.get("attacked_prompt")
        if not attacked_prompt:
            return False, "attacked_prompt_empty"

        attack_method = result.get("attack_method")
        if attack_method and attack_method != "no_attack":
            clean_prompt = result.get("clean_prompt")
            if attacked_prompt == clean_prompt:
                return False, "attacked_prompt_unchanged"

        return True, None

    def _has_incomplete_samples(self, sample_results: List[Dict[str, Any]], rerun_non_success: bool = False) -> bool:
        """检查样本中是否存在需要重跑的结果

        Args:
            sample_results: 样本结果列表
            rerun_non_success: 为True时，除status=success且结果有效外，其他状态都视为待重跑
        """
        if not sample_results:
            return True

        pending_statuses = {"created", "failed", "error", "in_progress", "running", "pending", None}
        for result in sample_results:
            if not isinstance(result, dict):
                return True

            status = result.get("status")
            if rerun_non_success:
                if status == "success":
                    is_valid, _ = self._validate_sample_result(result)
                    if not is_valid:
                        return True
                else:
                    return True
            elif status in {"success", "completed"}:
                is_valid, _ = self._validate_sample_result(result)
                if not is_valid:
                    return True
            elif status in pending_statuses:
                return True
            else:
                return True

        return False
    
    def _is_local_model(self, model_name: str) -> bool:
        """判断是否是本地模型"""
        if not model_name:
            return False
        
        local_prefixes = ["Llama", "Qwen", "Phi", "Yi", "gemma", "Mistral", "mistralai", "internlm", "meta-llama", "microsoft"]
        return any(prefix in model_name for prefix in local_prefixes)
    
    def _execute_missing_components(self, config: Dict[str, Any], sample: Dict[str, Any], 
                                  reusable: Dict[str, Any], reusable_results: Dict[str, Any]) -> Dict[str, Any]:
        """执行缺失的组件"""
        # Defensive programming: Validate sample data structure
        if not isinstance(sample, dict):
            raise ValueError(f"Sample must be a dictionary, got {type(sample)}. This may indicate malformed judger results being treated as sample data.")
        
        if "clean_prompt" not in sample:
            raise ValueError(f"Sample missing required 'clean_prompt' key. Available keys: {list(sample.keys())}. This may indicate malformed data structure.")
        
        new_results = {}
        clean_prompt = sample.get("clean_prompt", "")
        
        # 1. 如果需要模型响应
        if not reusable.get("model_response", False):
            self.progress_display.show_component_execution(
                "model", config.get('model'), "执行中", ""
            )
            model_response, response_time = self._get_model_response(
                config.get("model"), clean_prompt, config
            )
            new_results.update({
                "llm_response_on_clean": model_response,
                "llm_response_time_clean": response_time
            })
            
            # 显示模型响应详情
            self._display_model_response(model_response, response_time, "模型响应 (Clean)")
        else:
            # 如果是复用的模型响应，在verbose模式下也要显示
            if self.verbose:
                # 从复用数据中获取模型响应（需要通过其他方式获取）
                self.progress_display.show_component_execution(
                    "model", config.get('model'), "缓存命中", "(复用)"
                )
                # 注意：这里暂时不显示复用的响应，因为我们需要从结果合并后才能获取到
        
        # 2. 如果需要攻击
        if config.get("attack") != "no_attack" and not reusable.get("attack_result", False):
            self.progress_display.show_component_execution(
                "attack", config.get('attack'), "执行中", ""
            )
            attacked_prompt, attack_info = self._execute_attack(config, clean_prompt)
            
            new_results.update({
                "attacked_prompt": attacked_prompt,
                "attack_runtime": attack_info.get("runtime", 0),
                "attack_query_count": attack_info.get("query_count", 0),
                "assistant_llm_query_count": attack_info.get("assistant_query_count", 0),
                "attack_config": attack_info.get("config", {})
            })
            
            # 显示攻击对比和详情
            self._display_prompt_comparison(clean_prompt, attacked_prompt)
            self._display_attack_info(config.get('attack'), attack_info.get("config", {}), attack_info.get("runtime", 0))
            
            # 获取模型对攻击提示的响应
            if attacked_prompt != clean_prompt:
                llm_response, response_time = self._get_model_response(
                    config.get("model"), attacked_prompt, config
                )
                new_results.update({
                    "llm_response_on_attacked": llm_response,
                    "llm_response_time_attacked": response_time
                })
                
                # 显示攻击后的模型响应
                self._display_model_response(llm_response, response_time, "模型响应 (Attacked)")
        
        # 3. 如果需要防御
        if config.get("defense") != "no_defense" and not reusable.get("defense_result", False):
            self.progress_display.show_component_execution(
                "defense", config.get('defense'), "执行中", ""
            )
            defense_results = self._execute_defense(config, new_results)
            new_results.update(defense_results)
            
            # 显示防御详情
            if "defense_config" in defense_results:
                self._display_defense_info(config.get('defense'), defense_results.get("defense_config", {}))
            
            # 显示防御后的响应
            if "llm_response_on_attacked_under_defense" in defense_results:
                response = defense_results["llm_response_on_attacked_under_defense"]
                response_time = defense_results.get("llm_response_time_under_defense", 0)
                self._display_model_response(response, response_time, "模型响应 (Under Defense)")
        
        # 4. 如果需要评判
        judger_reusable = reusable.get("judger_result", False)
        if self.verbose:
            status = "缓存命中" if judger_reusable else "执行中"
            details = "(复用)" if judger_reusable else ""
            self.progress_display.show_component_execution(
                "judger", config.get('judger'), status, details
            )
        if not judger_reusable:
            
            # 准备评判数据，需要包含复用的数据和新计算的数据
            judger_data = {}
            
            # 1. 先添加基本信息
            judger_data["clean_prompt"] = clean_prompt
            
            # 2. 添加复用的数据
            # 2.1 如果模型响应被复用，需要从reusable_results中获取
            if reusable.get("model_response", False):
                model_response_data = reusable_results.get("model_responses", {}).get(clean_prompt, {})
                if model_response_data:
                    judger_data.update({
                        "llm_response_on_clean": model_response_data.get("llm_response_on_clean"),
                        "llm_response_time_clean": model_response_data.get("llm_response_time_clean"),
                        "target_llm_type": model_response_data.get("target_llm_type"),
                        "target_llm_parameters": model_response_data.get("target_llm_parameters")
                    })
            
            # 2.2 如果攻击结果被复用，需要从reusable_results中获取
            if reusable.get("attack_result", False) and config.get("attack") != "no_attack":
                attack_result_data = reusable_results.get("attack_results", {}).get(clean_prompt, {})
                if attack_result_data:
                    judger_data.update({
                        "attacked_prompt": attack_result_data.get("attacked_prompt"),
                        "llm_response_on_attacked": attack_result_data.get("llm_response_on_attacked"),
                        "llm_response_time_attacked": attack_result_data.get("llm_response_time_attacked"),
                        "attack_runtime": attack_result_data.get("attack_runtime"),
                        "attack_query_count": attack_result_data.get("attack_query_count"),
                        "assistant_llm_query_count": attack_result_data.get("assistant_llm_query_count"),
                        "attack_config": attack_result_data.get("attack_config")
                    })
            
            # 2.3 如果防御结果被复用，需要从reusable_results中获取
            if reusable.get("defense_result", False) and config.get("defense") != "no_defense":
                defense_result_data = reusable_results.get("defense_results", {}).get(clean_prompt, {})
                if defense_result_data:
                    judger_data.update({
                        "llm_response_on_clean_under_defense": defense_result_data.get("llm_response_on_clean_under_defense"),
                        "llm_response_on_attacked_under_defense": defense_result_data.get("llm_response_on_attacked_under_defense"),
                        "llm_response_time_under_defense": defense_result_data.get("llm_response_time_under_defense"),
                        "defense_config": defense_result_data.get("defense_config")
                    })
            
            # 3. 最后添加新计算的数据（这些会覆盖任何复用的数据）
            judger_data.update(new_results)
            
            # 4. 确保attacked_prompt存在
            if "attacked_prompt" not in judger_data:
                judger_data["attacked_prompt"] = clean_prompt
            
            judger_results = self._execute_judger(config, judger_data)
            new_results.update(judger_results)
            
            # 显示评判结果详情
            if judger_results and not judger_results.get("judger_error"):
                self._display_judger_results(config.get('judger'), judger_results)
        else:
            # 即使judger结果被复用，也需要在verbose模式下显示相关信息
            if self.verbose:
                self.progress_display.show_component_execution(
                    "judger", config.get('judger'), "缓存命中", "(复用)"
                )
                # 注意：具体的复用结果会在后续的_display_reused_results中显示
        
        return new_results
    
    def _normalize_attack_output(self, attack_result) -> str:
        """
        标准化攻击输出格式，确保返回字符串
        
        处理不同攻击方法的返回格式：
        - str: 直接返回
        - (count, [prompts]): 返回第一个prompt
        - [prompts]: 返回第一个prompt
        - 其他: 转换为字符串
        """
        try:
            # 如果已经是字符串，直接返回
            if isinstance(attack_result, str):
                return attack_result
            
            # 如果是tuple格式 (count, [prompts])
            if isinstance(attack_result, tuple) and len(attack_result) >= 2:
                count, prompts = attack_result[0], attack_result[1]
                if isinstance(prompts, list) and len(prompts) > 0:
                    return str(prompts[0])
                elif isinstance(prompts, str):
                    return prompts
            
            # 如果是list格式 [prompts]
            if isinstance(attack_result, list) and len(attack_result) > 0:
                return str(attack_result[0])
            
            # 其他情况，尝试转换为字符串
            return str(attack_result)
            
        except Exception as e:
            logger.warning(f"攻击输出格式转换失败: {e}, 原始结果: {attack_result}")
            # 备用方案：返回原始结果的字符串表示
            return str(attack_result)

    def _normalize_attack_name(self, attack_name: str) -> str:
        """将旧的攻击名称映射为目录/配置名"""
        if not attack_name:
            return attack_name
        name_map = {
            "AutoDANAttack": "AutoDAN",
            "COLDAttack": "COLD",
            "IFSJAttack": "IFSJ",
        }
        return name_map.get(attack_name, attack_name)
    
    def _validate_model_input(self, prompt) -> str:
        """
        验证和清理模型输入，确保为有效的字符串格式
        
        Args:
            prompt: 各种可能的输入格式
            
        Returns:
            str: 验证后的字符串输入
        """
        try:
            # 如果已经是字符串，进行基本验证
            if isinstance(prompt, str):
                if len(prompt.strip()) == 0:
                    raise ValueError("输入prompt为空字符串")
                return prompt.strip()
            
            # 如果是tuple格式，尝试提取字符串
            if isinstance(prompt, tuple):
                if len(prompt) >= 2 and isinstance(prompt[1], list) and len(prompt[1]) > 0:
                    return str(prompt[1][0]).strip()
                elif len(prompt) >= 1:
                    return str(prompt[0]).strip()
            
            # 如果是list格式，取第一个元素
            if isinstance(prompt, list) and len(prompt) > 0:
                return str(prompt[0]).strip()
            
            # 其他情况，转换为字符串
            converted = str(prompt).strip()
            if len(converted) == 0:
                raise ValueError("转换后的prompt为空")
            
            return converted
            
        except Exception as e:
            logger.error(f"模型输入验证失败: {e}, 原始输入: {prompt}")
            # 备用方案：返回错误提示作为输入
            return "输入验证失败，使用默认提示"
    
    def _execute_attack(self, config: Dict[str, Any], clean_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """执行攻击"""
        attack_name = config.get("attack")
        
        if attack_name == "no_attack":
            return clean_prompt, {"runtime": 0, "query_count": 0, "assistant_query_count": 0}
        
        try:
            # 获取攻击配置文件路径（用于记录配置），实例化使用新加载器
            normalized_attack_name = self._normalize_attack_name(attack_name)
            attack_config_path = get_attack_config_path(normalized_attack_name)
            if not attack_config_path:
                raise ValueError(f"未找到攻击配置文件: {attack_name}")
            
            # 加载实际模型实例 - 所有攻击都需要模型实例来执行query
            model_name = config.get("model")
            model_type = "local" if self._is_local_model(model_name) else "api"
            config_path = f"models/usage_examples/configs/{model_type}/{model_name}.json"
            # 使用模型缓存避免重复加载
            target_model, target_model_parameters = self.model_cache.get_model(model_name, config_path)
            
            # 使用新的攻击加载器实例化（自动合并默认配置）
            start_time = time.time()
            attack_instance = load_attack(
                normalized_attack_name,
                target_model=target_model,
                target_model_parameters=target_model_parameters,
            )
            
            # 执行攻击 - 调用攻击实例的attack方法
            attack_result = attack_instance.attack(clean_prompt)
            runtime = time.time() - start_time
            
            # 标准化攻击输出格式：确保返回字符串
            attacked_prompt = self._normalize_attack_output(attack_result)
            
            # 获取攻击配置信息（用于记录）
            with open(attack_config_path, 'r', encoding='utf-8') as f:
                attack_config = json.load(f)
            
            attack_info = {
                "runtime": runtime,
                "query_count": 1,  # 简化：假设一次查询
                "assistant_query_count": 0,  # 简化：需要根据攻击类型确定
                "config": attack_config
            }
            
            return attacked_prompt, attack_info
            
        except Exception as e:
            # 总是显示攻击失败信息
            self.progress_display.show_component_execution(
                "attack", attack_name, "错误", f"失败: {str(e)}"
            )
            # 详细错误信息在verbose模式下显示
            if self.verbose:
                print(f"❌ 攻击执行失败详情: {e}")
            # 使用统一的错误结果格式，所有数据字段都是null
            error_result = create_error_result("attack", attack_name, str(e))
            # 攻击失败时不返回attacked_prompt，返回None
            return None, error_result
    
    def _execute_attack_against_defended_model(self, config: Dict[str, Any], clean_prompt: str, 
                                              defended_model_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """执行攻击 - 针对防御后的模型"""
        attack_name = config.get("attack")
        
        if attack_name == "no_attack":
            return clean_prompt, {"runtime": 0, "query_count": 0, "assistant_query_count": 0}
        
        try:
            # 获取攻击配置文件路径（用于记录配置），实例化使用新加载器
            normalized_attack_name = self._normalize_attack_name(attack_name)
            attack_config_path = get_attack_config_path(normalized_attack_name)
            if not attack_config_path:
                raise ValueError(f"未找到攻击配置文件: {attack_name}")
            
            # 加载实际模型实例 - 所有攻击都需要模型实例来执行query
            if defended_model_info["is_defended"]:
                # 如果有防御，优先使用防御后的模型实例（如果已加载）
                defended_model = defended_model_info.get("defended_model")
                if hasattr(defended_model, '__call__'):  # 检查是否为模型实例
                    target_model = defended_model
                    target_model_parameters = defended_model_info.get("model_parameters", config.get("model_parameters", {}))
                else:
                    # 如果防御信息中没有模型实例，需要加载原始模型
                    model_name = config.get("model")
                    model_type = "local" if self._is_local_model(model_name) else "api"
                    config_path = f"models/usage_examples/configs/{model_type}/{model_name}.json"
                    # 使用模型缓存避免重复加载
                    target_model, target_model_parameters = self.model_cache.get_model(model_name, config_path)
            else:
                # 如果没有防御，加载原始模型实例
                model_name = config.get("model")
                model_type = "local" if self._is_local_model(model_name) else "api"
                config_path = f"models/usage_examples/configs/{model_type}/{model_name}.json"
                # 使用模型缓存避免重复加载
                target_model, target_model_parameters = self.model_cache.get_model(model_name, config_path)
            
            # 使用新的攻击加载器实例化（自动合并默认配置）
            start_time = time.time()
            attack_instance = load_attack(
                normalized_attack_name,
                target_model=target_model,
                target_model_parameters=target_model_parameters,
            )
            
            # 执行攻击 - 调用攻击实例的attack方法
            attack_result = attack_instance.attack(clean_prompt)
            runtime = time.time() - start_time
            
            # 标准化攻击输出格式：确保返回字符串
            attacked_prompt = self._normalize_attack_output(attack_result)
            
            # 获取攻击配置信息
            with open(attack_config_path, 'r', encoding='utf-8') as f:
                attack_config = json.load(f)
            
            attack_info = {
                "runtime": runtime,
                "query_count": 1,  # 简化：假设一次查询
                "assistant_query_count": 0,  # 简化：需要根据攻击类型确定
                "config": attack_config
            }
            
            return attacked_prompt, attack_info
            
        except Exception as e:
            # 总是显示攻击失败信息
            self.progress_display.show_component_execution(
                "attack", attack_name, "错误", f"失败: {str(e)}"
            )
            # 详细错误信息在verbose模式下显示
            if self.verbose:
                print(f"❌ 攻击执行失败详情: {e}")
            # 使用统一的错误结果格式，所有数据字段都是null
            error_result = create_error_result("attack", attack_name, str(e))
            # 攻击失败时不返回attacked_prompt，返回None
            return None, error_result
    
    def _check_stale_in_progress(self, sample_result: Dict[str, Any], timeout_minutes: int = 30) -> bool:
        """检测卡住的in_progress样本
        
        Args:
            sample_result: 样本结果
            timeout_minutes: 超时时间（分钟）
            
        Returns:
            是否为卡住的样本
        """
        if sample_result.get("status") != "in_progress":
            return False
        
        # 检查最后更新时间
        last_update = sample_result.get("experiment_timestamp", 0)
        if last_update == 0:
            return True  # 没有时间戳，视为卡住
        
        # 计算时间差
        time_diff = time.time() - last_update
        return time_diff > timeout_minutes * 60
    
    def _get_model_response(self, model_name: str, prompt: str, config: Dict[str, Any]) -> Tuple[str, float]:
        """获取模型响应"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # 使用新的加载器系统加载模型
                from models.loader import load_model
                
                # 直接使用新的加载器
                start_time = time.time()
                model_instance, model_parameters = load_model(model_name)
                
                # 显示模型加载状态
                if self.model_cache.is_model_cached(model_name):
                    self.progress_display.show_component_execution(
                        "model", model_name, "缓存命中", ""
                    )
                else:
                    self.progress_display.show_component_execution(
                        "model", model_name, "执行中", "(首次加载)"
                    )
                
                # 监控实际传递给模型的参数
                self._monitor_model_parameters(model_name, model_parameters)
                
                # 验证和清理输入prompt
                validated_prompt = self._validate_model_input(prompt)
                
                # 执行推理，传递模型参数
                response = model_instance.generate(validated_prompt, **model_parameters)
                response_time = time.time() - start_time
                
                # 验证响应长度合规性
                self._monitor_token_compliance(response, model_parameters, model_name)
                
                # 显示完成状态
                self.progress_display.show_component_execution(
                    "model", model_name, "完成", f"({response_time:.1f}s)"
                )
                
                # 允许空响应作为有效响应传递给judger评估
                # 只有None才表示技术性失败，空字符串是合法的模型拒绝响应
                return response, response_time
                
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"       ⚠️ 模型推理失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}, {retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    if self.verbose:
                        print(f"       ❌ 模型推理最终失败: {error_msg}")
                    self.progress_display.show_component_execution(
                        "model", model_name, "错误", error_msg
                    )
                    # 使用null值而不是空字符串和0.0来避免歧义
                    return None, None
        
        # 所有重试都失败时返回null值
        return None, None
    
    def _execute_defense(self, config: Dict[str, Any], base_results: Dict[str, Any]) -> Dict[str, Any]:
        """执行防御"""
        defense_name = config.get("defense")
        
        if defense_name == "no_defense":
            return {}
        
        try:
            # 获取防御配置文件路径（用于记录配置），实例化使用新加载器
            defense_config_path = get_defense_config_path(defense_name)
            if not defense_config_path:
                raise ValueError(f"未找到防御配置文件: {defense_name}")
            
            # 从当前模型获取target_model
            target_model = config.get("model")
            tokenizer = config.get("tokenizer")  # 某些防御需要tokenizer
            
            # 使用新的防御加载器实例化（自动合并默认配置）
            defense_instance = load_defense(
                defense_name,
                target_model=target_model,
                tokenizer=tokenizer,
            )
            
            # 获取要防御的prompt
            prompt_to_defend = base_results.get("attacked_prompt", base_results.get("clean_prompt"))
            
            # 执行防御 - 调用防御实例的generate方法
            defended_response = defense_instance.generate(prompt_to_defend)
            
            # 获取防御配置信息
            with open(defense_config_path, 'r', encoding='utf-8') as f:
                defense_config = json.load(f)
            
            return {
                "defense_config": defense_config,
                "llm_response_on_attacked_under_defense": defended_response,
                "llm_response_on_clean_under_defense": defended_response if base_results.get("attacked_prompt") == base_results.get("clean_prompt") else "",
                "llm_response_time_under_defense": 0.0  # 简化：需要实际测量
            }
            
        except Exception as e:
            # 总是显示防御失败信息
            self.progress_display.show_component_execution(
                "defense", defense_name, "错误", f"失败: {str(e)}"
            )
            # 详细错误信息在verbose模式下显示
            if self.verbose:
                print(f"❌ 防御执行失败详情: {e}")
            # 使用统一的错误结果格式，所有数据字段都是null
            return create_error_result("defense", defense_name, str(e))
    
    def _execute_judger(self, config: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """执行评判"""
        judger_name = config.get("judger")
        
        # Defensive programming: Validate input parameters
        if not isinstance(config, dict):
            error_msg = f"Config must be a dictionary, got {type(config)}"
            logger.error(error_msg)
            return create_error_result("judger", str(judger_name), error_msg)
        
        if not isinstance(results, dict):
            error_msg = f"Results must be a dictionary, got {type(results)}"
            logger.error(error_msg)
            return create_error_result("judger", str(judger_name), error_msg)
        
        # Validate that results contain expected sample data structure
        required_keys = ["clean_prompt"]
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            error_msg = f"Results missing required keys: {missing_keys}"
            logger.error(error_msg)
            return create_error_result("judger", str(judger_name), error_msg)
        
        try:
            # 处理多judger情况
            if isinstance(judger_name, list):
                # 对于多个judger，我们需要聚合结果而不是返回原始结果字典
                all_results = {}
                judger_configs = {}
                successful_judges = 0
                failure_reasons = []
                failure_details = {}
                
                for judge in judger_name:
                    try:
                        if self.verbose:
                            print(f"       🔍 执行评判器: {judge}")
                        
                        result = self._single_judge_evaluate(judge, results, config)
                        
                        # Defensive programming: Validate single judge result structure
                        if not isinstance(result, dict):
                            error_msg = f"Single judge {judge} returned non-dict result: {type(result)}"
                            logger.error(error_msg)
                            if self.verbose:
                                print(f"       ❌ 评判器 {judge} 返回无效结果类型: {type(result)}")
                            failure_reasons.append("non_dict_result")
                            failure_details[str(judge)] = {"reason": "non_dict_result", "detail": str(type(result))}
                            continue
                        
                        # Check for judger-level errors
                        if result.get("error"):
                            error_msg = f"Single judge {judge} failed: {result.get('error')}"
                            logger.warning(error_msg)
                            if self.verbose:
                                print(f"       ❌ 评判器 {judge} 执行失败: {result.get('error')}")
                            failure_reasons.append("judge_error")
                            failure_details[str(judge)] = {"reason": "judge_error", "detail": result.get("error")}
                            continue
                        
                        judger_configs[judge] = result.get("config", {})
                        
                        # 聚合每个评判器的结果到标准格式
                        judge_has_results = False
                        for eval_type in ["clean", "attack", "clean_defense", "attack_defense"]:
                            if eval_type in result and result[eval_type] is not None:
                                # 为多judger创建组合键名，正确映射字段名
                                if eval_type == "clean":
                                    key = "judger_result_on_clean"
                                elif eval_type == "attack":
                                    key = "judger_result_on_attack" 
                                elif eval_type == "clean_defense":
                                    key = "judger_result_on_clean_under_defense"
                                elif eval_type == "attack_defense":
                                    key = "judger_result_on_attack_under_defense"
                                if key not in all_results:
                                    all_results[key] = {}
                                all_results[key][judge] = result[eval_type]
                                judge_has_results = True
                                
                                if self.verbose:
                                    # 根据评判结果使用SAFE/UNSAFE显示
                                    score = result[eval_type]
                                    if score == 0:
                                        status = "✅ SAFE"  # 安全/拒绝攻击
                                    elif score == 1:
                                        status = "❌ UNSAFE"  # 危险/攻击成功
                                    else:
                                        status = f"⚠️ UNKNOWN({score})"  # 其他结果
                                    print(f"       {status} 评判器 {judge} [{eval_type}]")
                        
                        if judge_has_results:
                            successful_judges += 1
                        elif self.verbose:
                            print(f"       ⚠️ 评判器 {judge} 未返回有效结果")
                        if not judge_has_results:
                            failure_reasons.append("no_valid_results")
                            failure_details[str(judge)] = {"reason": "no_valid_results", "detail": "no eval results"}
                        
                    except Exception as judge_error:
                        error_msg = f"Error evaluating with judge {judge}: {judge_error}"
                        logger.error(error_msg)
                        if self.verbose:
                            print(f"       ❌ 评判器 {judge} 异常: {judge_error}")
                            import traceback
                            print(f"       错误详情: {traceback.format_exc()}")
                        failure_reasons.append("exception")
                        failure_details[str(judge)] = {"reason": "exception", "detail": str(judge_error)}
                        # Continue with other judges even if one fails
                        continue
                
                # 添加聚合的配置信息
                all_results["judger_config"] = judger_configs
                all_results["multi_judger_results"] = True
                if failure_reasons:
                    all_results["judger_fallback"] = True
                    all_results["judger_fallback_detail"] = {
                        "failure_reasons": failure_reasons,
                        "failure_details": failure_details
                    }
                else:
                    all_results["judger_fallback"] = False
                
                # Defensive programming: Validate multi-judger result structure
                if successful_judges == 0:
                    error_msg = "All judges failed to produce valid results"
                    logger.warning(error_msg)
                    if self.verbose:
                        print(f"       ❌ 所有评判器失败，成功数量: {successful_judges}")
                    error_result = create_error_result("judger", str(judger_name), error_msg)
                    error_result["judger_error_reason"] = "all_judges_failed"
                    error_result["judger_error_detail"] = {
                        "failure_reasons": failure_reasons,
                        "failure_details": failure_details
                    }
                    if failure_reasons:
                        summary = {}
                        for reason in failure_reasons:
                            summary[reason] = summary.get(reason, 0) + 1
                        error_result["judger_error_summary"] = summary
                    return error_result
                
                if self.verbose:
                    print(f"       📊 多评判器聚合结果: {successful_judges} 个成功")
                    for key, value in all_results.items():
                        if key not in ["judger_config", "multi_judger_results"]:
                            print(f"         {key}: {value}")
                
                return all_results
            else:
                result = self._single_judge_evaluate(judger_name, results, config)
                
                # Defensive programming: Validate single judger result structure
                if not isinstance(result, dict):
                    error_msg = f"Single judge {judger_name} returned non-dict result: {type(result)}"
                    logger.error(error_msg)
                    return create_error_result("judger", str(judger_name), error_msg)
                error_detail = {k: v for k, v in result.items() if k.endswith("_error") and v}
                
                return {
                    "judger_result_on_clean": result.get("clean", None),
                    "judger_result_on_attack": result.get("attack", None),
                    "judger_result_on_clean_under_defense": result.get("clean_defense", None),
                    "judger_result_on_attack_under_defense": result.get("attack_defense", None),
                    "judger_config": result.get("config", {}),
                    "multi_judger_results": False,
                    "judger_fallback": bool(error_detail),
                    "judger_fallback_detail": error_detail
                }
                
        except Exception as e:
            import traceback
            # 总是显示评判失败信息
            self.progress_display.show_component_execution(
                "judger", config.get('judger'), "错误", f"失败: {str(e)}"
            )
            # 详细错误信息在verbose模式下显示
            if self.verbose:
                print(f"❌ 评判失败详情: {e}")
                print(f"错误堆栈: {traceback.format_exc()}")
            # 使用统一的错误结果格式，所有数据字段都是null
            return create_error_result("judger", config.get('judger', 'unknown'), str(e))
    
    def _single_judge_evaluate(self, judger_name: str, results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """单个评判器评估"""
        try:
            # 使用新的加载器系统检查评判器
            from judgers.judger_loader import get_available_judgers
            available_judgers = get_available_judgers()
            if judger_name not in available_judgers:
                raise ValueError(f"未找到评判器: {judger_name}")
            
            # 检查缓存中是否已有judger实例
            cache_key = f"{judger_name}_default"
            if cache_key in self._judger_cache:
                judger_instance = self._judger_cache[cache_key]
            else:
                # 使用judger_loader直接加载（支持完整名称）
                from judgers.judger_loader import load_judger
                
                try:
                    # 直接使用完整的judger名称（包含_judger后缀）
                    judger_instance = load_judger(judger_name)
                    if self.verbose:
                        print(f"      ✅ 加载Judger: {judger_name}")
                except Exception as e:
                    raise ValueError(f"无法加载Judger '{judger_name}': {e}")
                
                # 缓存judger实例
                self._judger_cache[cache_key] = judger_instance
            
            # 初始化judger配置
            judger_config = {}
            
            judger_results = {}
            
            # 🔧 向后兼容性映射：从统一字段恢复旧字段名
            # 根据response_context确定字段映射
            response_context = results.get('response_context', {})
            attack_method = response_context.get('attack_method', results.get('attack_method', 'unknown'))
            defense_method = response_context.get('defense_method', results.get('defense_method', 'unknown'))
            
            
            # 创建具有旧字段名的映射副本
            results_with_legacy_fields = results.copy()
            
            if attack_method == "no_attack" and defense_method == "no_defense":
                # no_attack + no_defense: llm_response -> llm_response_on_clean
                results_with_legacy_fields["llm_response_on_clean"] = results.get("llm_response")
            elif attack_method == "no_attack" and defense_method != "no_defense":
                # no_attack + defense: llm_response -> llm_response_on_clean_under_defense
                results_with_legacy_fields["llm_response_on_clean_under_defense"] = results.get("llm_response")
            elif attack_method != "no_attack" and defense_method == "no_defense":
                # attack + no_defense: llm_response -> llm_response_on_attacked
                results_with_legacy_fields["llm_response_on_attacked"] = results.get("llm_response")
                results_with_legacy_fields["attacked_prompt"] = results.get("attacked_prompt", results.get("clean_prompt"))
            elif attack_method != "no_attack" and defense_method != "no_defense":
                # attack + defense: llm_response -> llm_response_on_attacked_under_defense
                results_with_legacy_fields["llm_response_on_attacked_under_defense"] = results.get("llm_response")
                results_with_legacy_fields["attacked_prompt"] = results.get("attacked_prompt", results.get("clean_prompt"))

            # Fallback: if no legacy response fields are set but llm_response exists, infer target slot
            if (
                results.get("llm_response") is not None
                and results_with_legacy_fields.get("llm_response_on_clean") is None
                and results_with_legacy_fields.get("llm_response_on_attacked") is None
                and results_with_legacy_fields.get("llm_response_on_clean_under_defense") is None
                and results_with_legacy_fields.get("llm_response_on_attacked_under_defense") is None
            ):
                inferred_attack = attack_method or results.get("attack_method", "unknown")
                inferred_defense = defense_method or results.get("defense_method", "unknown")
                if inferred_attack not in ("no_attack", "unknown") and inferred_defense not in ("no_defense", "unknown"):
                    results_with_legacy_fields["llm_response_on_attacked_under_defense"] = results.get("llm_response")
                    results_with_legacy_fields["attacked_prompt"] = results.get("attacked_prompt", results.get("clean_prompt"))
                elif inferred_attack not in ("no_attack", "unknown"):
                    results_with_legacy_fields["llm_response_on_attacked"] = results.get("llm_response")
                    results_with_legacy_fields["attacked_prompt"] = results.get("attacked_prompt", results.get("clean_prompt"))
                elif inferred_defense not in ("no_defense", "unknown"):
                    results_with_legacy_fields["llm_response_on_clean_under_defense"] = results.get("llm_response")
                else:
                    if results.get("clean_prompt") is not None:
                        results_with_legacy_fields["llm_response_on_clean"] = results.get("llm_response")
            
            # 使用映射后的results进行评判
            results = results_with_legacy_fields
            
            # 评判clean响应（包含空字符串）
            if results.get("llm_response_on_clean") is not None:
                if self.verbose:
                    print(f"       🔍 评判clean响应...")
                try:
                    # 添加详细的输入输出调试信息
                    input_data = {
                        "behavior": results.get("clean_prompt", ""),
                        "generation": results.get("llm_response_on_clean")
                    }
                    
                    clean_result = judger_instance.judge(
                        behavior=results.get("clean_prompt", ""),
                        generation=results.get("llm_response_on_clean")
                    )
                    
                    
                    if self.verbose:
                        formatted_result = self.content_formatter.format_judger_result(judger_name, clean_result)
                        print(f"       ✅ Clean: {formatted_result}")
                    judger_results["clean"] = clean_result
                except Exception as judge_error:
                    # 处理评判器错误，但不中断整个实验
                    error_msg = self._format_judger_error(judge_error)
                    if self.verbose:
                        print(f"       ❌ Clean evaluation failed: {error_msg}")
                    judger_results["clean"] = None
                    judger_results["clean_error"] = str(judge_error)
            
            # 评判attack响应（包含空字符串）
            if results.get("llm_response_on_attacked") is not None:
                if self.verbose:
                    print(f"       🔍 评判attack响应...")
                try:
                    # 添加详细的输入输出调试信息
                    input_data = {
                        "behavior": results.get("attacked_prompt", ""),
                        "generation": results.get("llm_response_on_attacked")
                    }
                    
                    attack_result = judger_instance.judge(
                        behavior=results.get("attacked_prompt", ""),
                        generation=results.get("llm_response_on_attacked")
                    )
                    
                    
                    if self.verbose:
                        formatted_result = self.content_formatter.format_judger_result(judger_name, attack_result)
                        print(f"       ✅ Attack: {formatted_result}")
                    judger_results["attack"] = attack_result
                except Exception as judge_error:
                    # 处理评判器错误，但不中断整个实验
                    error_msg = self._format_judger_error(judge_error)
                    if self.verbose:
                        print(f"       ❌ Attack evaluation failed: {error_msg}")
                    judger_results["attack"] = None
                    judger_results["attack_error"] = str(judge_error)
            
            # 评判defense响应（包含空字符串）
            if results.get("llm_response_on_clean_under_defense") is not None:
                try:
                    # 添加详细的输入输出调试信息
                    input_data = {
                        "behavior": results.get("clean_prompt", ""),
                        "generation": results.get("llm_response_on_clean_under_defense")
                    }
                    
                    clean_defense_result = judger_instance.judge(
                        behavior=results.get("clean_prompt", ""),
                        generation=results.get("llm_response_on_clean_under_defense")
                    )
                    
                    
                    judger_results["clean_defense"] = clean_defense_result
                except Exception as judge_error:
                    error_msg = self._format_judger_error(judge_error)
                    if self.verbose:
                        print(f"       ❌ Clean defense evaluation failed: {error_msg}")
                    judger_results["clean_defense"] = None
                    judger_results["clean_defense_error"] = str(judge_error)
            
            if results.get("llm_response_on_attacked_under_defense") is not None:
                try:
                    # 添加详细的输入输出调试信息
                    input_data = {
                        "behavior": results.get("attacked_prompt", ""),
                        "generation": results.get("llm_response_on_attacked_under_defense")
                    }
                    
                    attack_defense_result = judger_instance.judge(
                        behavior=results.get("attacked_prompt", ""),
                        generation=results.get("llm_response_on_attacked_under_defense")
                    )
                    
                    
                    judger_results["attack_defense"] = attack_defense_result
                except Exception as judge_error:
                    error_msg = self._format_judger_error(judge_error)
                    if self.verbose:
                        print(f"       ❌ Attack defense evaluation failed: {error_msg}")
                    judger_results["attack_defense"] = None
                    judger_results["attack_defense_error"] = str(judge_error)
            
            judger_results["config"] = judger_config
            return judger_results
            
        except Exception as e:
            import traceback
            # 总是显示单个评判器失败信息
            self.progress_display.show_component_execution(
                "judger", judger_name, "错误", f"失败: {str(e)}"
            )
            # 详细错误信息在verbose模式下显示
            if self.verbose:
                print(f"❌ 单个评判器失败详情 ({judger_name}): {e}")
                print(f"错误堆栈: {traceback.format_exc()}")
            # 返回null值结果而不是-1
            return {
                "error": str(e),
                "clean": None,  # 使用null表示执行失败
                "attack": None,
                "clean_defense": None,
                "attack_defense": None,
                "config": None
            }
    
    def _format_judger_error(self, error: Exception) -> str:
        """格式化评判器错误信息"""
        try:
            # 尝试导入自定义异常类
            from judgers.gpt_judger import (
                JudgerQuotaExceededError, 
                JudgerRateLimitError, 
                JudgerNetworkError,
                JudgerParsingError,
                JudgerAPIError
            )
            
            if isinstance(error, JudgerQuotaExceededError):
                return f"API配额不足 - 请检查您的API计费详情"
            elif isinstance(error, JudgerRateLimitError):
                retry_msg = ""
                if hasattr(error, 'retry_after') and error.retry_after:
                    retry_msg = f"，{error.retry_after}秒后重试"
                return f"API请求频率过高{retry_msg}"
            elif isinstance(error, JudgerNetworkError):
                return f"网络连接错误 - 请检查网络连接"
            elif isinstance(error, JudgerParsingError):
                return f"响应解析失败 - 模型响应格式不正确"
            elif isinstance(error, JudgerAPIError):
                return f"API调用错误 - {error.error_code or '未知错误'}"
            else:
                return f"未知错误 - {str(error)}"
        except ImportError:
            # 如果无法导入自定义异常类，回退到通用错误处理
            error_msg = str(error).lower()
            if "quota" in error_msg or "insufficient_quota" in error_msg:
                return "API配额不足 - 请检查您的API计费详情"
            elif "rate" in error_msg or "429" in error_msg:
                return "API请求频率过高"
            elif "network" in error_msg or "connection" in error_msg:
                return "网络连接错误"
            else:
                return f"未知错误 - {str(error)}"

    def _log_reuse_stats(self, reuse_stats: Dict[str, Any]):
        """记录复用统计信息"""
        if not reuse_stats:
            return
        
        total_samples = reuse_stats.get("total_samples", 0)
        if total_samples == 0:
            return
        
        formatted_stats = self.content_formatter.format_reuse_stats(reuse_stats)
        print(f"   🔄 {formatted_stats}")
    
    def _monitor_model_parameters(self, model_name: str, model_parameters: dict):
        """监控实际传递给模型的参数"""
        if not self.verbose:
            return
            
        print(f"       📋 实际传递给模型 {model_name} 的参数:")
        
        # 重点关注token限制参数
        token_params = []
        if 'max_tokens' in model_parameters:
            token_params.append(f"max_tokens: {model_parameters['max_tokens']}")
        if 'max_new_tokens' in model_parameters:
            token_params.append(f"max_new_tokens: {model_parameters['max_new_tokens']}")
        
        if token_params:
            print(f"         🎯 Token限制: {', '.join(token_params)}")
        else:
            print(f"         ⚠️ 未找到token限制参数!")
            
        # 显示其他重要参数
        other_params = []
        for key in ['temperature', 'top_p', 'top_k', 'do_sample']:
            if key in model_parameters:
                other_params.append(f"{key}: {model_parameters[key]}")
        
        if other_params:
            print(f"         📊 其他参数: {', '.join(other_params)}")
        
        # 如果存在未知参数，也显示出来
        known_params = {'max_tokens', 'max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample'}
        unknown_params = {k: v for k, v in model_parameters.items() if k not in known_params}
        if unknown_params:
            print(f"         🔍 其他参数: {unknown_params}")
    
    def _monitor_token_compliance(self, response: str, model_parameters: dict, model_name: str):
        """监控token合规性"""
        if not response:
            return
            
        # 计算响应的token数量（粗略估算）
        estimated_tokens = len(response) // 4
        
        # 获取预期的token限制
        expected_limit = model_parameters.get('max_tokens') or model_parameters.get('max_new_tokens')
        
        if expected_limit:
            compliance_status = "✅" if estimated_tokens <= expected_limit else "❌"
            if self.verbose:
                print(f"       📏 响应长度检查: {len(response)}字符 ≈ {estimated_tokens} tokens")
                print(f"         {compliance_status} 预期限制: {expected_limit} tokens")
                
                if estimated_tokens > expected_limit:
                    excess = estimated_tokens - expected_limit
                    print(f"         ⚠️ 超出限制: +{excess} tokens ({excess/expected_limit*100:.1f}%)")
        else:
            if self.verbose:
                print(f"       📏 响应长度: {len(response)}字符 ≈ {estimated_tokens} tokens (无限制参数)")
                print(f"         ⚠️ 无法验证合规性：未找到token限制参数")
    
    
    def _display_prompt_comparison(self, clean_prompt: str, attacked_prompt: str):
        """显示原始prompt vs 攻击后prompt的对比"""
        if not self.verbose:
            return
            
        # 显示原始prompt
        formatted_clean = self.content_formatter.format_prompt(clean_prompt, prefix="原始Prompt")
        print(formatted_clean)
        
        # 如果有攻击，显示攻击后的prompt
        if attacked_prompt and attacked_prompt != clean_prompt:
            formatted_attacked = self.content_formatter.format_prompt(attacked_prompt, prefix="攻击后Prompt")
            print(formatted_attacked)
        elif attacked_prompt and attacked_prompt == clean_prompt:
            print("       📋 攻击后Prompt与原始Prompt相同")
        # 如果attacked_prompt为空，说明只是显示原始prompt
    
    def _display_model_response(self, response: str, response_time: float, label: str = "模型响应"):
        """显示模型响应详情"""
        if not self.verbose:
            return
            
        if response:
            formatted_response = self.content_formatter.format_response(response, prefix=label)
            print(formatted_response)
            print(f"       ⏱️ 响应时间: {response_time:.2f}s, 响应长度: {len(response)}字符")
        else:
            print(f"       ⚠️ {label}: [空响应]")
    
    def _display_judger_results(self, judger_name: str, judger_results: Dict[str, Any]):
        """显示评判结果详情"""
        if not self.verbose:
            return
            
        result_lines = []
        
        # 检查是否是多judger结果
        is_multi_judger = judger_results.get("multi_judger_results", False)
        
        if is_multi_judger:
            # 处理多judger结果格式
            for eval_type in ["clean", "attack", "clean_defense", "attack_defense"]:
                key = f"judger_result_on_{eval_type}" if eval_type in ["clean", "attack"] else f"judger_result_on_{eval_type}"
                if key in judger_results and judger_results[key]:
                    eval_type_display = eval_type.replace("_", " ").title()
                    for judge, result in judger_results[key].items():
                        if result is not None:
                            result_lines.append(f"{eval_type_display} ({judge}): {result}")
            
            # 显示配置信息
            if "judger_config" in judger_results and judger_results["judger_config"]:
                for judge, config in judger_results["judger_config"].items():
                    if config:
                        config_str = str(config)
                        if len(config_str) > 50:
                            config_str = config_str[:47] + "..."
                        result_lines.append(f"配置 ({judge}): {config_str}")
        else:
            # 处理单judger结果格式 (保持向后兼容)
            if "clean" in judger_results and judger_results["clean"] is not None:
                clean_result = judger_results["clean"]
                result_lines.append(f"Clean Response: {clean_result}")
            
            if "attack" in judger_results and judger_results["attack"] is not None:
                attack_result = judger_results["attack"]
                result_lines.append(f"Attack Response: {attack_result}")
            
            if "clean_defense" in judger_results and judger_results["clean_defense"] is not None:
                clean_defense_result = judger_results["clean_defense"]
                result_lines.append(f"Clean Defense: {clean_defense_result}")
            
            if "attack_defense" in judger_results and judger_results["attack_defense"] is not None:
                attack_defense_result = judger_results["attack_defense"]
                result_lines.append(f"Attack Defense: {attack_defense_result}")
            
            # 显示配置信息
            if "config" in judger_results and judger_results["config"]:
                config_str = str(judger_results["config"])
                if len(config_str) > 50:
                    config_str = config_str[:47] + "..."
                result_lines.append(f"配置: {config_str}")
            
            # 显示新格式的结果 (用于向后兼容)
            for result_key in ["judger_result_on_clean", "judger_result_on_attack", "judger_result_on_clean_under_defense", "judger_result_on_attack_under_defense"]:
                if result_key in judger_results and judger_results[result_key] is not None:
                    result_type = result_key.replace("judger_result_on_", "").replace("_", " ").title()
                    result_lines.append(f"{result_type}: {judger_results[result_key]}")
                    
            if "judger_config" in judger_results and judger_results["judger_config"]:
                config_str = str(judger_results["judger_config"])
                if len(config_str) > 50:
                    config_str = config_str[:47] + "..."
                result_lines.append(f"配置: {config_str}")
        
        # 显示错误信息
        error_messages = []
        for error_key in ["error", "clean_error", "attack_error", "clean_defense_error", "attack_defense_error"]:
            if error_key in judger_results and judger_results[error_key]:
                error_type = error_key.replace("_error", "").replace("_", " ").title()
                if error_type == "Error":
                    error_type = "General"
                error_messages.append(f"{error_type}: {self._format_judger_error(Exception(judger_results[error_key]))}")
        
        if error_messages:
            for error_msg in error_messages:
                print(f"       ❌ {error_msg}")
        
        if result_lines:
            for line in result_lines:
                result_formatted = self.content_formatter.format_judger_result(judger_name, line)
                print(f"       {result_formatted}")
        
        # 检查是否所有评判都失败了
        has_any_result = any(
            judger_results.get(key) is not None 
            for key in ["clean", "attack", "clean_defense", "attack_defense"]
        )
        
        if not has_any_result and not error_messages:
            print(f"       ⚠️ 评判器 {judger_name}: 没有可显示的结果")
    
    def _display_judger_result(self, judger_result: Dict[str, Any], label: str = "评判结果"):
        """显示单个评判结果"""
        if not self.verbose:
            return
        
        print(f"      📊 {label}:")
        
        # 显示clean结果
        clean_result = judger_result.get("clean_result")
        if clean_result is not None:
            if isinstance(clean_result, dict):
                print(f"        Clean: {clean_result}")
            else:
                print(f"        Clean: {clean_result}")
        
        # 显示attack结果
        attack_result = judger_result.get("attack_result")
        if attack_result is not None:
            if isinstance(attack_result, dict):
                print(f"        Attack: {attack_result}")
            else:
                print(f"        Attack: {attack_result}")
        
        # 显示defense结果
        clean_defense_result = judger_result.get("clean_defense_result")
        if clean_defense_result is not None:
            print(f"        Clean+Defense: {clean_defense_result}")
        
        attack_defense_result = judger_result.get("attack_defense_result")
        if attack_defense_result is not None:
            print(f"        Attack+Defense: {attack_defense_result}")
    
    def _display_attack_info(self, attack_name: str, attack_config: Dict[str, Any], runtime: float):
        """显示攻击信息"""
        if not self.verbose:
            return
            
        config_str = str(attack_config) if attack_config else "{}"
        if len(config_str) > 100:
            config_str = config_str[:97] + "..."
        
        print(f"       ⚔️ 攻击配置: {config_str}")
        print(f"       ⏱️ 攻击时间: {runtime:.2f}s")
    
    def _display_defense_info(self, defense_name: str, defense_config: Dict[str, Any]):
        """显示防御信息"""
        if not self.verbose:
            return
            
        config_str = str(defense_config) if defense_config else "{}"
        if len(config_str) > 100:
            config_str = config_str[:97] + "..."
        
        print(f"       🛡️ 防御配置: {config_str}")
    
    def _get_or_create_defended_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """获取或创建防御后的模型配置（实验级缓存）"""
        defense_name = config.get("defense")
        model_name = config.get("model")
        
        # 如果无防御，直接返回
        if defense_name == "no_defense":
            return {
                "defense_method": "no_defense",
                "defense_config": {},
                "defense_instance": None,
                "model_name": model_name,
                "is_defended": False
            }
        
        # 实验级缓存key
        cache_key = f"{model_name}+{defense_name}"
        
        # 检查缓存
        if cache_key in self.experiment_defended_models:
            if self.verbose:
                logger.info(f"💼 复用defended model: {cache_key}")
            return self.experiment_defended_models[cache_key]
        
        # 首次创建，记录到缓存
        if self.verbose:
            logger.info(f"🆕 创建defended model: {cache_key}")
        
        defended_model_info = self._create_defended_model_internal(config)
        self.experiment_defended_models[cache_key] = defended_model_info
        
        return defended_model_info
    
    def _create_defended_model_internal(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建防御后的模型配置（内部方法）"""
        defense_name = config.get("defense")
        
        try:
            # 获取防御配置文件路径
            defense_config_path = get_defense_config_path(defense_name)
            if not defense_config_path:
                raise ValueError(f"未找到防御配置文件: {defense_name}")
            
            # 加载防御配置
            with open(defense_config_path, 'r', encoding='utf-8') as f:
                defense_config = json.load(f)
            
            # 从当前模型获取target_model - 需要加载实际的模型实例而非字符串
            model_name = config.get("model")
            model_config_path = get_model_config_path(model_name)
            if not model_config_path:
                raise ValueError(f"未找到模型配置文件: {model_name}")
            
            # 使用模型缓存加载模型实例，避免重复加载
            target_model, target_model_parameters = self.model_cache.get_model(model_name, model_config_path)
            tokenizer = getattr(target_model, 'get_tokenizer', lambda: None)() if hasattr(target_model, 'get_tokenizer') else None
            
            # 使用新的防御加载器创建防御实例（自动合并默认配置）
            defense_instance = load_defense(
                defense_name,
                target_model=target_model,
                tokenizer=tokenizer,
            )
            
            return {
                "defense_method": defense_name,
                "defense_config": defense_config,
                "defense_instance": defense_instance,  # 返回实例而非函数
                "model_name": config.get("model"),
                "model_parameters": target_model_parameters,
                "is_defended": True
            }
            
        except Exception as e:
            logger.error(f"创建防御模型失败: {e}")
            return {
                "defense_method": "no_defense",
                "defense_config": {},
                "defense_instance": None,
                "model_name": config.get("model"),
                "is_defended": False,
                "error": str(e)
            }
    
    def _get_defended_model_response(self, defended_model_info: Dict[str, Any], 
                                   prompt: str, config: Dict[str, Any]) -> tuple:
        """获取防御后模型的响应"""
        if not defended_model_info["is_defended"]:
            # 如果没有防御，直接调用原始模型
            response, response_time = self._get_model_response(defended_model_info["model_name"], prompt, config)
            return response, response_time, {"fallback": False, "fallback_events": []}
        
        try:
            # 使用防御实例处理prompt并获取响应
            defense_instance = defended_model_info["defense_instance"]
            if defense_instance is None:
                raise ValueError("防御实例为空")
            
            # 调用防御实例的generate方法获取响应
            start_time = time.time()
            if hasattr(defense_instance, "clear_fallback_events"):
                defense_instance.clear_fallback_events()
            defended_response = defense_instance.generate(prompt)
            response_time = time.time() - start_time
            fallback_events = []
            if hasattr(defense_instance, "get_fallback_events"):
                fallback_events = defense_instance.get_fallback_events()
            fallback_info = {
                "fallback": bool(fallback_events),
                "fallback_events": fallback_events
            }
            return defended_response, response_time, fallback_info
            
        except Exception as e:
            logger.error(f"防御模型响应失败: {e}")
            # 降级到原始模型
            response, response_time = self._get_model_response(defended_model_info["model_name"], prompt, config)
            fallback_info = {
                "fallback": True,
                "fallback_events": [f"defense_generate_error: {e}"],
                "fallback_to_model": True
            }
            return response, response_time, fallback_info
    
    def _display_attack_results(self, attack_name: str, clean_prompt: str, 
                               attacked_prompt: str, attack_info: Dict[str, Any]):
        """增强的攻击结果显示"""
        if not self.verbose:
            return
        
        print(f"       ⚔️ 攻击方法: {attack_name}")
        
        # 显示攻击配置
        config_str = str(attack_info.get("config", {}))
        if len(config_str) > 100:
            config_str = config_str[:97] + "..."
        print(f"       📋 攻击配置: {config_str}")
        
        # 显示执行时间和查询次数
        runtime = attack_info.get("runtime", 0)
        query_count = attack_info.get("query_count", 0)
        print(f"       ⏱️ 攻击耗时: {runtime:.2f}s ({query_count} 次查询)")
        
        # 显示prompt对比
        self._display_prompt_comparison(clean_prompt, attacked_prompt)
    
    def _display_defense_results(self, defense_name: str, defended_model_info: Dict[str, Any]):
        """增强的防御结果显示"""
        if not self.verbose:
            return
        
        print(f"       🛡️ 防御方法: {defense_name}")
        print(f"       🤖 防御模型: {defended_model_info['model_name']} + {defense_name}")
        
        # 显示防御配置
        config_str = str(defended_model_info.get("defense_config", {}))
        if len(config_str) > 100:
            config_str = config_str[:97] + "..."
        print(f"       📋 防御配置: {config_str}")
        
        # 显示防御状态
        if defended_model_info.get("error"):
            print(f"       ⚠️ 防御加载失败: {defended_model_info['error']}")
        elif defended_model_info["is_defended"]:
            print(f"       ✅ 防御模型已就绪")
        else:
            print(f"       ⚪ 使用原始模型")
    
    def _display_judger_results_enhanced(self, judger_name: str, judger_results: Dict[str, Any], 
                                       attacked_prompt: str):
        """增强的评判结果显示"""
        if not self.verbose:
            return
        
        print(f"       ⚖️ 评判器: {judger_name}")
        print(f"       📝 评判目标: (attacked_prompt, defended_response) 组合")
        
        # 显示攻击prompt的摘要
        prompt_summary = attacked_prompt[:60] + "..." if len(attacked_prompt) > 60 else attacked_prompt
        print(f"       🎯 攻击Prompt: {prompt_summary}")
        
        # 显示评判结果
        self._display_judger_results(judger_name, judger_results)

    def _display_reused_results(self, combined_result: Dict[str, Any], reusable: Dict[str, Any], config: Dict[str, Any]):
        """显示复用的结果信息"""
        if not self.verbose:
            return
            
        # 显示复用的模型响应
        if reusable.get("model_response", False) and combined_result.get("llm_response_on_clean"):
            response = combined_result["llm_response_on_clean"]
            response_time = combined_result.get("llm_response_time_clean", 0)
            self._display_model_response(response, response_time, "模型响应 (Clean - 复用)")
        
        # 显示复用的评判结果
        if reusable.get("judger_result", False):
            # 转换judger结果键名格式以匹配_display_judger_results的期望
            judger_results = {}
            key_mappings = {
                "judger_result_on_clean": "clean",
                "judger_result_on_attack": "attack", 
                "judger_result_on_clean_under_defense": "clean_defense",
                "judger_result_on_attack_under_defense": "attack_defense",
                "judger_config": "config"
            }
            
            for full_key, simple_key in key_mappings.items():
                if full_key in combined_result:
                    judger_results[simple_key] = combined_result[full_key]
            
            if judger_results:
                # 获取judger名称，优先从当前config获取
                judger_name = (config.get("judger") or 
                              combined_result.get("judger_name") or 
                              combined_result.get("judger") or 
                              "unknown")
                
                print(f"       🔍 评判结果 (复用):")
                self._display_judger_results(judger_name, judger_results)
            else:
                # 调试信息：如果没有judger结果，显示可用的键
                available_keys = [k for k in combined_result.keys() if 'judger' in k.lower()]
                if available_keys:
                    print(f"       ⚠️ 复用judger结果为空，可用键: {available_keys}")
                else:
                    print(f"       ⚠️ 复用judger结果为空，无相关键")
    
    def run_placeholder_experiment(self, placeholder_file: str, sample_limit: int = None) -> Dict[str, Any]:
        """运行单个占位符实验
        
        Args:
            placeholder_file: 占位符文件路径
            sample_limit: 动态样本限制（可选，覆盖占位符中的样本数量）
        """
        try:
            # 读取占位符文件
            with open(placeholder_file, 'r', encoding='utf-8') as f:
                placeholder_data = json.load(f)
            
            config = placeholder_data.get("config", {})
            status = placeholder_data.get("status", "pending")
            
            # 检查sample-limit特定的完成状态
            if sample_limit is not None:
                # 为不同sample-limit维护独立状态
                sample_limit_statuses = placeholder_data.get("sample_limit_statuses", {})
                sample_limit_key = f"limit_{sample_limit}"
                sample_limit_entry = sample_limit_statuses.get(sample_limit_key, "pending")
                if isinstance(sample_limit_entry, dict):
                    sample_limit_status = sample_limit_entry.get("status", "pending")
                else:
                    sample_limit_status = sample_limit_entry

                if self.force_rerun:
                    if self.verbose:
                        print(f"🔁 强制重跑样本限制 {sample_limit}: {placeholder_file}")
                elif sample_limit_status == "success":
                    if not self.rerun_failed:
                        if self.verbose:
                            print(f"⏭️ 样本限制 {sample_limit} 的实验已完成，跳过: {placeholder_file}")
                        return {"status": "skipped", "reason": f"already_completed_for_limit_{sample_limit}"}

                    results = None
                    if isinstance(sample_limit_entry, dict):
                        results = sample_limit_entry.get("results")
                    if results is None:
                        results = placeholder_data.get("sample_results", [])
                    if results:
                        results = results[:sample_limit]

                    if not self._has_incomplete_samples(results, rerun_non_success=self.rerun_failed):
                        if self.verbose:
                            print(f"⏭️ 样本限制 {sample_limit} 的实验无需重跑样本，跳过: {placeholder_file}")
                        return {"status": "skipped", "reason": f"already_completed_for_limit_{sample_limit}"}
                    if self.verbose:
                        print(f"🔁 样本限制 {sample_limit} 的实验存在可重跑样本，继续运行: {placeholder_file}")
            else:
                # 传统完整实验检查
                if self.force_rerun:
                    if self.verbose:
                        print(f"🔁 强制重跑实验: {placeholder_file}")
                elif status == "success":
                    if not self.rerun_failed:
                        if self.verbose:
                            print(f"⏭️ 实验已完成，跳过: {placeholder_file}")
                        return {"status": "skipped", "reason": "already_completed"}

                    sample_results = placeholder_data.get("sample_results", [])
                    if not self._has_incomplete_samples(sample_results, rerun_non_success=self.rerun_failed):
                        if self.verbose:
                            print(f"⏭️ 实验已完成且无需重跑样本，跳过: {placeholder_file}")
                        return {"status": "skipped", "reason": "already_completed"}
                    if self.verbose:
                        print(f"🔁 实验存在可重跑样本，继续运行: {placeholder_file}")
            
            # 更新状态为运行中
            self.placeholder_manager.update_placeholder_status(config, "running")
            
            # 执行实验，传递sample_limit
            result = self.run_single_experiment(config, placeholder_data, sample_limit)
            
            # 更新占位符状态
            if result.get("status") == "completed":
                sample_results = result.get("sample_results", [])
                
                if self.verbose:
                    print(f"   💾 保存实验结果: {len(sample_results)} 个样本结果")
                
                if sample_limit is not None:
                    # 更新sample-limit特定状态
                    success = self.placeholder_manager.update_placeholder_sample_limit_status(
                        config, sample_limit, "success", sample_results)
                    if self.verbose:
                        status_msg = "成功" if success else "失败"
                        print(f"   📝 更新样本限制状态 (limit={sample_limit}): {status_msg}")
                else:
                    # 更新传统完整实验状态
                    success = self.placeholder_manager.update_placeholder_status(config, "success", sample_results)
                    if self.verbose:
                        status_msg = "成功" if success else "失败"
                        print(f"   📝 更新占位符状态: {status_msg}")
                
                if not success and self.verbose:
                    print(f"   ⚠️ 警告: 占位符状态更新失败")
                    
            elif result.get("status") == "failed":
                error_info = result.get("error", "Unknown error")
                if sample_limit is not None:
                    self.placeholder_manager.update_placeholder_sample_limit_status(
                        config, sample_limit, "failed", None, error_info)
                else:
                    self.placeholder_manager.update_placeholder_status(config, "failed", None, error_info)
                
                if self.verbose:
                    print(f"   ❌ 实验失败，已更新占位符状态: {error_info[:100]}...")
            else:
                if self.verbose:
                    print(f"   ❓ 未知实验状态: {result.get('status', 'None')}")
            
            # 刷新复用索引以便后续实验可以复用结果
            self.placeholder_manager.refresh_reuse_index()
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 运行占位符实验失败 {placeholder_file}: {e}")
            
            # 更新占位符状态为失败
            try:
                if 'config' in locals():
                    self.placeholder_manager.update_placeholder_status(config, "failed", None, str(e))
                else:
                    # 如果配置还未加载，尝试从占位符文件获取
                    with open(placeholder_file, 'r', encoding='utf-8') as f:
                        placeholder_data = json.load(f)
                    config = placeholder_data.get("config", {})
                    self.placeholder_manager.update_placeholder_status(config, "failed", None, str(e))
            except Exception as update_error:
                if self.verbose:
                    print(f"❌ 更新占位符状态失败: {update_error}")
            
            return {"status": "failed", "error": str(e)}
    
    def run_batch_placeholders(self, placeholder_files: List[str], workers: int = 1, sample_limit: int = None) -> List[Dict[str, Any]]:
        """批量运行占位符实验
        
        Args:
            placeholder_files: 占位符文件列表
            workers: 工作进程数
            sample_limit: 动态样本限制（可选，覆盖每个占位符中的样本数量）
        """
        try:
            if workers == 1:
                # 串行执行
                results = []
                for i, placeholder_file in enumerate(placeholder_files):
                    experiment_name = Path(placeholder_file).name
                    
                    # 显示实验开始
                    # 读取占位符文件
                    with open(placeholder_file, 'r', encoding='utf-8') as f:
                        placeholder_data = json.load(f)
                    config = placeholder_data.get("config", {})
                    self.progress_display.show_experiment_start(
                        i + 1, len(placeholder_files), experiment_name, config
                    )
                    
                    result = self.run_placeholder_experiment(placeholder_file, sample_limit)
                    
                    # 显示实验完成
                    if result.get("status") == "completed":
                        self.progress_display.show_experiment_summary(experiment_name, result)
                        self.statistics_display.add_experiment_result(experiment_name, result)
                    
                    results.append(result)
                return results
            else:
                # 并行执行
                results = [None] * len(placeholder_files)
                
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    # 提交任务
                    future_to_index = {}
                    for i, placeholder_file in enumerate(placeholder_files):
                        future = executor.submit(self.run_placeholder_experiment, placeholder_file)
                        future_to_index[future] = i
                    
                    # 收集结果
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            result = future.result()
                            results[index] = result
                            print(f"✅ 完成实验 {index+1}/{len(placeholder_files)}")
                            print("=" * 80)  # 样本分割线
                        except Exception as e:
                            print(f"❌ 实验 {index+1} 失败: {e}")
                            results[index] = {"status": "failed", "error": str(e)}
                
                return results
        finally:
            # 清理实验级defended model缓存
            self._cleanup_experiment_cache()
    
    def _cleanup_experiment_cache(self):
        """清理实验级defended model缓存"""
        if self.experiment_defended_models:
            cleanup_count = len(self.experiment_defended_models)
            if self.verbose:
                logger.info(f"🧹 清理实验级defended model缓存: {cleanup_count}个")
            
            # 清理defense instances
            for cache_key, defended_model_info in self.experiment_defended_models.items():
                defense_instance = defended_model_info.get("defense_instance")
                if defense_instance and hasattr(defense_instance, '__del__'):
                    try:
                        del defense_instance
                    except Exception as e:
                        logger.warning(f"清理defense instance失败 {cache_key}: {e}")
            
            self.experiment_defended_models.clear()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if self.verbose:
                logger.info("✅ 实验级缓存清理完成")


# 向后兼容的类名
PlaceholderExperimentRunner = SmartPlaceholderRunner


def main():
    """测试占位符执行器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="智能占位符实验执行器")
    parser.add_argument("--run", metavar="PLACEHOLDER_FILE", help="运行单个占位符实验")
    parser.add_argument("--run-all", action="store_true", help="批量运行占位符实验")
    parser.add_argument("--workers", type=int, default=1, help="并行工作数")
    parser.add_argument("--show-reuse-stats", action="store_true", help="显示复用统计")
    parser.add_argument("--verbose", action="store_true", help="显示详细的实验过程和结果")
    parser.add_argument("--max-length", type=int, default=200, help="文本显示的最大长度 (默认: 200)")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="重跑除success外的所有样本状态（如 completed/failed/pending）")
    parser.add_argument("--force-rerun", action="store_true",
                        help="强制重跑全部样本（包含success和completed）")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    runner = SmartPlaceholderRunner(
        verbose=args.verbose,
        max_length=args.max_length,
        rerun_failed=args.rerun_failed,
        force_rerun=args.force_rerun
    )
    
    if args.run:
        # 运行单个实验
        result = runner.run_placeholder_experiment(args.run)
        print(f"实验结果: {result.get('status')}")
        
        if result.get("status") == "completed":
            sample_results = result.get("sample_results") or []
            if isinstance(sample_results, list) and sample_results:
                total_samples = result.get("total_samples", len(sample_results))
                successful_samples = sum(1 for s in sample_results if s.get("status") == "success")
                completed_samples = sum(1 for s in sample_results if s.get("status") == "completed")
            else:
                total_samples = result.get("total_samples", 0)
                successful_samples = result.get("successful_samples", 0)
                completed_samples = 0
            print(f"成功样本: {successful_samples}/{total_samples}")
            print(f"完成样本: {completed_samples}/{total_samples}")
            
            if args.show_reuse_stats:
                reuse_stats = result.get("reuse_stats", {})
                if reuse_stats:
                    print("\n复用统计:")
                    for key, value in reuse_stats.items():
                        print(f"  {key}: {value}")
    
    elif args.run_all:
        # 运行所有pending实验
        placeholders = runner.placeholder_manager.list_placeholders("pending")
        
        if not placeholders:
            print("没有待执行的实验")
            return
        
        placeholder_files = [
            str(runner.placeholder_manager.placeholders_dir / p["filename"])
            for p in placeholders
        ]
        
        print(f"开始执行 {len(placeholder_files)} 个待执行实验...")
        results = runner.run_batch_placeholders(placeholder_files, args.workers)
        
        # 统计结果
        completed = len([r for r in results if r.get("status") == "completed"])
        success = len([r for r in results if r.get("status") == "success"])
        failed = len([r for r in results if r.get("status") == "failed"])
        skipped = len([r for r in results if r.get("status") == "skipped"])
        
        print(f"\n执行完成:")
        print(f"  完成: {completed}")
        print(f"  成功: {success}")
        print(f"  失败: {failed}")
        print(f"  跳过: {skipped}")
        print(f"  总计: {len(results)}")
        
        # 显示复用统计
        if args.show_reuse_stats:
            total_reuse_stats = {}
            for result in results:
                if result.get("status") == "completed":
                    reuse_stats = result.get("reuse_stats", {})
                    for key, value in reuse_stats.items():
                        if key not in total_reuse_stats:
                            total_reuse_stats[key] = []
                        total_reuse_stats[key].append(value)
            
            if total_reuse_stats:
                print("\n整体复用统计:")
                for key, values in total_reuse_stats.items():
                    if values and isinstance(values[0], (int, float)):
                        avg_value = sum(values) / len(values)
                        print(f"  {key}: {avg_value:.2f}")


if __name__ == "__main__":
    main()
