"""
Display Components for PromptSecurity Experiments

Provides structured, hierarchical display components for experiment progress,
content formatting, and statistics display.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import textwrap


logger = logging.getLogger(__name__)


class ProgressDisplay:
    """
    Handles progress display for experiments and samples.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()
        self.experiment_start_times = {}
        
    def show_system_init(self, stats: Dict[str, Any]):
        """Show system initialization information."""
        print("\n" + "="*60)
        print("🚀 PromptSecurity实验执行器")
        print("="*60)
        print(f"📊 系统初始化:")
        print(f"   ├── 攻击方法: {stats.get('attacks', 0)} 个")
        print(f"   ├── 防御方法: {stats.get('defenses', 0)} 个") 
        print(f"   ├── 模型: {stats.get('models', 0)} 个")
        print(f"   ├── 评判器: {stats.get('judgers', 0)} 个")
        print(f"   ├── 数据集: {stats.get('datasets', 0)} 个")
        print(f"   └── 复用实验: {stats.get('cached_experiments', 0)} 个")
        print()
        
    def show_experiment_start(self, current: int, total: int, experiment_name: str, config: Dict[str, Any]):
        """Show the start of a new experiment."""
        print("─" * 80)
        print(f"🔄 实验 [{current:03d}/{total:03d}] {experiment_name}")
        print(f"   📋 配置: {config.get('model')} + {config.get('attack')} + {config.get('defense')} + {config.get('judger')}")
        
        # Calculate and show overall progress
        progress_percent = (current - 1) / total * 100
        progress_bar = self._create_progress_bar(progress_percent, 30)
        
        # Estimate remaining time
        elapsed = time.time() - self.start_time
        if current > 1:
            avg_time_per_experiment = elapsed / (current - 1)
            remaining_experiments = total - current + 1
            estimated_remaining = avg_time_per_experiment * remaining_experiments
            remaining_str = self._format_duration(estimated_remaining)
        else:
            remaining_str = "计算中..."
            
        print(f"   📈 总体进度: {progress_bar} {progress_percent:.1f}% ({remaining_str}剩余)")
        
        self.experiment_start_times[experiment_name] = time.time()
        
    def show_sample_progress(self, current: int, total: int, sample_title: str):
        """Show progress for individual samples."""
        progress_percent = current / total * 100
        progress_bar = self._create_progress_bar(progress_percent, 20, small=True)
        
        # Truncate sample title for display
        truncated_title = self._truncate_text(sample_title, 50)
        
        print(f"   📝 样本 [{current:03d}/{total:03d}] {progress_bar} {truncated_title}")
        
    def show_sample_separator(self):
        """Show separator between samples."""
        print()
        
    def show_component_execution(self, component_type: str, component_name: str, 
                                status: str = "执行中", details: str = ""):
        """Show execution of individual components."""
        icons = {
            "model": "🤖",
            "attack": "⚔️", 
            "defense": "🛡️",
            "judger": "⚖️"
        }
        icon = icons.get(component_type, "🔧")
        
        if status == "执行中":
            print(f"       {icon} {component_type.title()}: {component_name} {details}")
        elif status == "完成":
            print(f"       ✅ {component_type.title()}: {component_name} {details}")
        elif status == "缓存命中":
            print(f"       💾 {component_type.title()}: {component_name} (缓存命中) {details}")
        elif status == "错误":
            print(f"       ❌ {component_type.title()}: {component_name} - {details}")
    
    def show_four_element_flow_start(self):
        """Show the start of 4-element experimental flow."""
        print(f"   🔄 开始4要素实验流程:")
        print(f"      1️⃣ 防御模型 → 2️⃣ 攻击处理 → 3️⃣ 模型响应 → 4️⃣ 评判分析")
    
    def show_attack_step(self, attack_name: str, status: str = "执行中"):
        """Show attack step in defense-first flow."""
        if status == "执行中":
            print(f"      2️⃣ ⚔️ 攻击处理: {attack_name}")
        elif status == "完成":
            print(f"      2️⃣ ✅ 攻击完成: {attack_name}")
        elif status == "跳过":
            print(f"      2️⃣ ⚪ 无攻击: 使用原始prompt")
    
    def show_defense_step(self, defense_name: str, model_name: str, status: str = "执行中"):
        """Show defense step in defense-first flow."""
        if status == "执行中":
            print(f"      1️⃣ 🛡️ 防御模型: {model_name} + {defense_name}")
        elif status == "完成":
            print(f"      1️⃣ ✅ 防御就绪: {model_name} + {defense_name}")
        elif status == "跳过":
            print(f"      1️⃣ ⚪ 无防御: 使用原始模型 {model_name}")
    
    def show_model_response_step(self, model_name: str, defense_name: str, status: str = "执行中"):
        """Show model response step in 4-element flow."""
        model_desc = f"{model_name}+{defense_name}" if defense_name != "no_defense" else model_name
        if status == "执行中":
            print(f"      3️⃣ 🤖 模型响应: {model_desc} 处理attacked_prompt")
        elif status == "完成":
            print(f"      3️⃣ ✅ 响应完成: {model_desc}")
    
    def show_judger_step(self, judger_name: str, status: str = "执行中"):
        """Show judger evaluation step in 4-element flow."""
        if status == "执行中":
            print(f"      4️⃣ ⚖️ 评判分析: {judger_name} 评估(attacked_prompt, defended_response)")
        elif status == "完成":
            print(f"      4️⃣ ✅ 评判完成: {judger_name}")
    
    def show_four_element_summary(self, attack_name: str, defense_name: str, model_name: str, 
                                 judger_name: str, success: bool = True):
        """Show summary of 4-element flow completion."""
        status_icon = "✅" if success else "❌"
        status_text = "成功" if success else "失败"
        
        print(f"   {status_icon} 4要素实验{status_text}:")
        print(f"      攻击: {attack_name} → 防御: {defense_name} → 模型: {model_name} → 评判: {judger_name}")
            
    def show_experiment_summary(self, experiment_name: str, results: Dict[str, Any]):
        """Show summary for a completed experiment."""
        if experiment_name in self.experiment_start_times:
            duration = time.time() - self.experiment_start_times[experiment_name]
            duration_str = self._format_duration(duration)
        else:
            duration_str = "未知"
            
        sample_results = results.get("sample_results") or []
        if isinstance(sample_results, list) and sample_results:
            total_samples = results.get("total_samples", len(sample_results))
            successful_samples = sum(1 for s in sample_results if s.get("status") == "success")
            completed_samples = sum(1 for s in sample_results if s.get("status") == "completed")
        else:
            total_samples = results.get("total_samples", 0)
            successful_samples = results.get("successful_samples", 0)
            completed_samples = 0
        success_rate = (successful_samples / total_samples * 100) if total_samples > 0 else 0
        completed_rate = (completed_samples / total_samples * 100) if total_samples > 0 else 0
        
        # Calculate safety metrics if available
        clean_safe_rate = results.get('clean_safe_rate')
        attack_success_rate = results.get('attack_success_rate')
        
        print(f"   📊 实验汇总:")
        print(f"       ├── 样本: {successful_samples}/{total_samples} 成功 ({success_rate:.1f}%)")
        print(f"       ├── 完成: {completed_samples}/{total_samples} ({completed_rate:.1f}%)")
        if clean_safe_rate is not None:
            print(f"       ├── 安全率: {clean_safe_rate:.1f}%")
        if attack_success_rate is not None:
            print(f"       ├── 攻击成功率: {attack_success_rate:.1f}%")
        print(f"       └── 耗时: {duration_str}")
        
    def show_final_summary(self, total_experiments: int, success_count: int, total_duration: float):
        """Show final summary of all experiments."""
        print("\n" + "="*60)
        print("✅ 实验执行完成")
        print("="*60)
        
        success_rate = (success_count / total_experiments * 100) if total_experiments > 0 else 0
        duration_str = self._format_duration(total_duration)
        
        print(f"📊 总体统计:")
        print(f"   ├── 实验总数: {total_experiments}")
        print(f"   ├── 成功实验: {success_count}")
        print(f"   ├── 成功率: {success_rate:.1f}%")
        print(f"   └── 总耗时: {duration_str}")
        print()
        
    def _create_progress_bar(self, percent: float, width: int = 30, small: bool = False) -> str:
        """Create a visual progress bar."""
        if small:
            filled = int(percent / 100 * width)
            bar = "█" * filled + "░" * (width - filled)
        else:
            filled = int(percent / 100 * width)
            bar = "█" * filled + "░" * (width - filled)
        return bar
        
    def _format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable way."""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"
            
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."


class ContentFormatter:
    """
    Handles formatting of content for display.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def format_prompt(self, prompt: str, max_length: int = 200, prefix: str = "原始Prompt") -> str:
        """Format a prompt for display."""
        if not self.verbose:
            # In non-verbose mode, just show truncated version
            truncated = self._truncate_text(prompt, max_length)
            return truncated
            
        # In verbose mode, show with simple formatting (no box)
        lines = textwrap.wrap(prompt, width=max_length)
        if not lines:
            return ""
            
        result = f"📋 {prefix}:\n"
        for line in lines:
            result += f"   {line}\n"
        
        return result.rstrip()
        
    def format_response(self, response: str, max_length: int = 200, prefix: str = "模型响应") -> str:
        """Format a model response for display."""
        if not self.verbose:
            # In non-verbose mode, show truncated version
            truncated = self._truncate_text(response, max_length)
            return truncated
            
        # In verbose mode, show with simple formatting (no box)
        lines = textwrap.wrap(response, width=max_length)
        if not lines:
            return ""
            
        result = f"💬 {prefix}:\n"
        for line in lines:
            result += f"   {line}\n"
        
        return result.rstrip()
        
    def format_timing_info(self, timing_info: Dict[str, float]) -> str:
        """Format timing information."""
        parts = []
        
        if 'model_time' in timing_info:
            parts.append(f"模型{timing_info['model_time']:.1f}s")
        if 'attack_time' in timing_info:
            parts.append(f"攻击{timing_info['attack_time']:.1f}s")
        if 'defense_time' in timing_info:
            parts.append(f"防御{timing_info['defense_time']:.1f}s")
        if 'judger_time' in timing_info:
            parts.append(f"评判{timing_info['judger_time']:.1f}s")
            
        total_time = sum(timing_info.values())
        parts.append(f"总计{total_time:.1f}s")
        
        return " + ".join(parts)
        
    def format_judger_result(self, judger_name: str, result: Any) -> str:
        """Format judger result for display."""
        if isinstance(result, (int, float)):
            if result == 0:
                return f"{judger_name} → 🟢 SAFE"
            elif result == 1:
                return f"{judger_name} → 🔴 UNSAFE"
            else:
                return f"{judger_name} → {result}"
        else:
            return f"{judger_name} → {result}"
    
    def format_attack_summary(self, attack_name: str, clean_prompt: str, attacked_prompt: str, 
                             attack_info: Dict[str, Any], max_prompt_length: int = 200, 
                             show_full_prompt: bool = False) -> str:
        """Format attack summary for display."""
        if attack_name == "no_attack":
            return "⚪ 无攻击: 使用原始prompt"
        
        # 格式化prompt对比 - 支持完整显示模式
        if show_full_prompt:
            clean_summary = clean_prompt
            attacked_summary = attacked_prompt or ""
        else:
            clean_summary = self._truncate_text(clean_prompt, max_prompt_length)
            attacked_summary = self._truncate_text(attacked_prompt or "", max_prompt_length)
        
        # 攻击统计信息
        runtime = attack_info.get("runtime", 0)
        query_count = attack_info.get("query_count", 0)
        
        result = f"⚔️ {attack_name}\n"
        result += f"   ├─ 原始: {clean_summary}\n"
        result += f"   ├─ 攻击后: {attacked_summary}\n"
        
        # 添加变化状态指示
        if attacked_prompt and attacked_prompt != clean_prompt:
            result += f"   ├─ 状态: ✅ 攻击成功修改了prompt\n"
        else:
            result += f"   ├─ 状态: ⚠️ 攻击未修改prompt（可能失败或无需修改）\n"
        
        result += f"   └─ 统计: {runtime:.2f}s, {query_count}次查询"
        
        return result
    
    def format_defense_summary(self, defense_name: str, model_name: str, 
                              defense_config: Dict[str, Any], is_defended: bool) -> str:
        """Format defense summary for display."""
        if defense_name == "no_defense":
            return f"⚪ 无防御: 使用原始模型 {model_name}"
        
        status_icon = "✅" if is_defended else "⚠️"
        status_text = "已就绪" if is_defended else "加载失败"
        
        result = f"🛡️ {defense_name}\n"
        result += f"   ├─ 模型: {model_name}\n"
        result += f"   ├─ 状态: {status_icon} 防御{status_text}\n"
        
        # 显示配置摘要
        if defense_config:
            config_summary = str(defense_config)
            if len(config_summary) > 80:
                config_summary = config_summary[:77] + "..."
            result += f"   └─ 配置: {config_summary}"
        else:
            result += f"   └─ 配置: 默认配置"
        
        return result
    
    def format_model_response_summary(self, response: str, response_time: float, 
                                    response_type: str, max_length: int = 150) -> str:
        """Format model response summary for display."""
        type_icons = {
            "clean": "🤖",
            "direct": "🤖", 
            "under_defense": "🛡️🤖",
            "attacked": "⚔️🤖"
        }
        
        icon = type_icons.get(response_type, "🤖")
        response_summary = self._truncate_text(response or "无响应", max_length)
        
        result = f"{icon} 模型响应 ({response_type})\n"
        result += f"   ├─ 内容: {response_summary}\n"
        result += f"   └─ 耗时: {response_time:.2f}s" if response_time is not None else f"   └─ 耗时: 未知"
        
        return result
    
    def format_judger_evaluation_summary(self, judger_name: str, judger_results: Dict[str, Any], 
                                       attacked_prompt: str) -> str:
        """Format judger evaluation summary for display."""
        prompt_summary = self._truncate_text(attacked_prompt, 80)
        
        result = f"⚖️ {judger_name} 评判结果\n"
        result += f"   ├─ 目标: (attacked_prompt, defended_response) 组合\n"
        result += f"   ├─ Prompt: {prompt_summary}\n"
        
        # 显示各种评判结果
        evaluation_results = []
        for key, value in judger_results.items():
            if key.startswith("judger_result_on_") and value is not None:
                result_type = key.replace("judger_result_on_", "").replace("_", " ")
                if isinstance(value, (int, float)):
                    if value == 0:
                        evaluation_results.append(f"{result_type}: 🟢 SAFE")
                    elif value == 1:
                        evaluation_results.append(f"{result_type}: 🔴 UNSAFE")
                    else:
                        evaluation_results.append(f"{result_type}: {value}")
                else:
                    evaluation_results.append(f"{result_type}: {value}")
        
        if evaluation_results:
            result += f"   └─ 结果: " + " | ".join(evaluation_results)
        else:
            result += f"   └─ 结果: 无有效评判结果"
        
        return result
            
    def format_reuse_stats(self, stats: Dict[str, Any]) -> str:
        """Format reuse statistics."""
        if not stats:
            return "复用统计: 无数据"
            
        try:
            if not self.verbose:
                total_reuse = stats.get('total_reuse_rate', 0)
                if isinstance(total_reuse, (int, float)):
                    return f"复用率: {total_reuse:.1f}%"
                else:
                    return f"复用率: {total_reuse}"
                
            parts = []
            for component, rate in stats.items():
                if component != 'total_reuse_rate':
                    if isinstance(rate, (int, float)):
                        parts.append(f"{component}: {rate:.1f}%")
                    else:
                        parts.append(f"{component}: {rate}")
                    
            return " | ".join(parts) if parts else "复用统计: 无详细数据"
        except Exception as e:
            return f"复用统计: 格式化错误 - {e}"
        
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."


class StatisticsDisplay:
    """
    Handles display of statistics and analytics.
    """
    
    def __init__(self):
        self.experiment_stats = []
        
    def add_experiment_result(self, experiment_name: str, results: Dict[str, Any]):
        """Add an experiment result for statistics tracking."""
        self.experiment_stats.append({
            'name': experiment_name,
            'timestamp': time.time(),
            'results': results
        })
        
    def show_model_performance(self, model_name: str, performance_data: Dict[str, Any]):
        """Show performance metrics for a specific model."""
        print(f"   🚀 {model_name} 性能指标:")
        
        if 'avg_response_time' in performance_data:
            print(f"       ├── 平均响应时间: {performance_data['avg_response_time']:.2f}s")
        if 'cache_hit_rate' in performance_data:
            print(f"       ├── 缓存命中率: {performance_data['cache_hit_rate']:.1f}%")
        if 'total_samples' in performance_data:
            print(f"       └── 处理样本数: {performance_data['total_samples']}")
            
    def show_safety_analysis(self, safety_stats: Dict[str, Any]):
        """Show safety analysis results."""
        print(f"   🛡️ 安全性分析:")
        
        if 'baseline_safety_rate' in safety_stats:
            print(f"       ├── 基线安全率: {safety_stats['baseline_safety_rate']:.1f}%")
        if 'attack_success_rate' in safety_stats:
            print(f"       ├── 攻击成功率: {safety_stats['attack_success_rate']:.1f}%")
        if 'defense_effectiveness' in safety_stats:
            print(f"       └── 防御有效性: {safety_stats['defense_effectiveness']:.1f}%")
            
    def show_cache_statistics(self, cache_stats: Dict[str, Any]):
        """Show model cache statistics."""
        if not cache_stats.get('cached_models'):
            return
            
        print(f"   💾 模型缓存状态:")
        print(f"       ├── 缓存使用率: {cache_stats.get('cache_utilization', 0):.1f}%")
        print(f"       ├── 当前缓存模型: {len(cache_stats.get('cached_models', []))}")
        
        for model in cache_stats.get('cached_models', []):
            last_used = cache_stats.get('last_used', {}).get(model, 0)
            if last_used > 0:
                print(f"       │   └── {model} (最后使用: {last_used:.1f}s前)")
            else:
                print(f"       │   └── {model}")
                
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall statistics across all experiments."""
        if not self.experiment_stats:
            return {}
            
        total_experiments = len(self.experiment_stats)
        successful_experiments = sum(1 for exp in self.experiment_stats 
                                   if exp['results'].get('successful_samples', 0) > 0)
        
        return {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': (successful_experiments / total_experiments * 100) if total_experiments > 0 else 0,
            'avg_samples_per_experiment': sum(exp['results'].get('total_samples', 0) 
                                            for exp in self.experiment_stats) / total_experiments if total_experiments > 0 else 0
        }
