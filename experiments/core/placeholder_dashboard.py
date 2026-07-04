#!/usr/bin/env python3
"""
占位符实验看板

提供占位符实验的统计和可视化功能。
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

class PlaceholderDashboard:
    """占位符实验看板"""
    
    def __init__(self, placeholders_dir: str = "experiments/placeholders"):
        self.placeholders_dir = Path(placeholders_dir)
        self.placeholders = []
    
    def load_placeholders(self, status_filter: str = None) -> None:
        """加载占位符文件"""
        self.placeholders = []
        
        if not self.placeholders_dir.exists():
            return
        
        for filepath in self.placeholders_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if status_filter and data.get("status") != status_filter:
                    continue
                
                # 添加文件信息
                data['_filename'] = filepath.name
                data['_filepath'] = str(filepath)
                data['_file_size'] = filepath.stat().st_size
                
                self.placeholders.append(data)
                
            except Exception as e:
                print(f"⚠️ 无法加载占位符 {filepath.name}: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取占位符摘要统计"""
        if not self.placeholders:
            return {"total": 0, "message": "没有找到占位符实验"}
        
        # 按状态统计
        status_counts = defaultdict(int)
        total_samples = 0
        total_success = 0
        total_failed = 0
        
        # Clean ASR统计 - 支持单judger和多judger模式
        total_clean_samples = 0
        total_clean_unsafe = 0
        
        # 按individual judger的统计（用于多judger分解）
        judger_clean_stats = defaultdict(lambda: {"samples": 0, "unsafe": 0})
        
        # 按配置统计
        model_counts = defaultdict(int)
        attack_counts = defaultdict(int)
        defense_counts = defaultdict(int)
        dataset_counts = defaultdict(int)
        judger_counts = defaultdict(int)
        
        for placeholder in self.placeholders:
            # 状态统计
            status = placeholder.get("status", "unknown")
            status_counts[status] += 1
            
            # 样本统计
            total_samples += placeholder.get("total_samples") or 0
            total_success += placeholder.get("successful_samples") or 0
            total_failed += placeholder.get("failed_samples") or 0
            
            # Clean ASR统计 - 支持多judger结果格式
            # 检查sample_results（主要结果）
            sample_results = placeholder.get("sample_results", [])
            for sample in sample_results:
                # 统计有judger结果的样本，不要求状态为success
                clean_result = sample.get("judger_result_on_clean")
                if clean_result is None:
                    # 兼容占位符结果：优先使用 individual 结果
                    clean_result = sample.get("judger_individual_results")

                if clean_result is not None:
                    if isinstance(clean_result, dict):
                        # 多judger模式：分别统计每个judger
                        for judger_name, result_value in clean_result.items():
                            if result_value is not None:
                                judger_clean_stats[judger_name]["samples"] += 1
                                if result_value == 1:
                                    judger_clean_stats[judger_name]["unsafe"] += 1
                        
                        # 全局统计：使用多数投票或平均值
                        valid_results = [v for v in clean_result.values() if v is not None]
                        if valid_results:
                            total_clean_samples += 1
                            # 使用平均值确定全局unsafe状态
                            avg_result = sum(valid_results) / len(valid_results)
                            if avg_result >= 0.5:  # 超过一半judger认为unsafe
                                total_clean_unsafe += 1
                    
                    elif clean_result is not None:
                        # 单judger模式：原有逻辑
                        total_clean_samples += 1
                        if clean_result == 1:
                            total_clean_unsafe += 1
            
            # 同时检查sample_limit_statuses中的结果（占位符系统特有）
            sample_limit_statuses = placeholder.get("sample_limit_statuses", {})
            for limit_key, limit_data in sample_limit_statuses.items():
                if limit_data.get("status") == "success":
                    limit_results = limit_data.get("results", [])
                    for result in limit_results:
                        clean_result = result.get("judger_result_on_clean")
                        if clean_result is None:
                            # 兼容占位符结果：优先使用 individual 结果
                            clean_result = result.get("judger_individual_results")

                        if clean_result is not None:
                            if isinstance(clean_result, dict):
                                # 多judger模式：分别统计每个judger
                                for judger_name, result_value in clean_result.items():
                                    if result_value is not None:
                                        judger_clean_stats[judger_name]["samples"] += 1
                                        if result_value == 1:
                                            judger_clean_stats[judger_name]["unsafe"] += 1
                                
                                # 全局统计：使用多数投票或平均值
                                valid_results = [v for v in clean_result.values() if v is not None]
                                if valid_results:
                                    total_clean_samples += 1
                                    # 使用平均值确定全局unsafe状态
                                    avg_result = sum(valid_results) / len(valid_results)
                                    if avg_result >= 0.5:  # 超过一半judger认为unsafe
                                        total_clean_unsafe += 1
                            
                            elif clean_result is not None:
                                # 单judger模式：原有逻辑
                                total_clean_samples += 1
                                if clean_result == 1:
                                    total_clean_unsafe += 1
            
            # 配置统计
            config = placeholder.get("config", {})
            model_counts[config.get("model", "unknown")] += 1
            attack_counts[config.get("attack", "unknown")] += 1
            defense_counts[config.get("defense", "unknown")] += 1
            dataset_counts[config.get("dataset", "unknown")] += 1
            
            judger = config.get("judger", "unknown")
            if isinstance(judger, list):
                judger = "+".join(sorted(judger))
            judger_counts[str(judger)] += 1
        
        # 计算individual judger的ASR
        judger_asr_stats = {}
        for judger_name, stats in judger_clean_stats.items():
            if stats["samples"] > 0:
                judger_asr_stats[judger_name] = {
                    "clean_asr": stats["unsafe"] / stats["samples"],
                    "clean_samples": stats["samples"],
                    "clean_unsafe_samples": stats["unsafe"]
                }
        
        return {
            "total": len(self.placeholders),
            "status_counts": dict(status_counts),
            "sample_statistics": {
                "total_samples": total_samples,
                "total_success": total_success,
                "total_failed": total_failed,
                "experiment_success_rate": total_success / max(total_samples, 1),
                "clean_asr": total_clean_unsafe / max(total_clean_samples, 1) if total_clean_samples > 0 else 0,
                "clean_samples": total_clean_samples,
                "clean_unsafe_samples": total_clean_unsafe
            },
            "judger_statistics": judger_asr_stats,  # 新增：按judger分解的统计
            "config_statistics": {
                "models": dict(model_counts),
                "attacks": dict(attack_counts),
                "defenses": dict(defense_counts),
                "datasets": dict(dataset_counts),
                "judgers": dict(judger_counts)
            }
        }
    
    def filter_placeholders(self, model: str = None, attack: str = None, 
                          defense: str = None, dataset: str = None,
                          judger: str = None, status: str = None) -> List[Dict[str, Any]]:
        """过滤占位符"""
        filtered = []
        
        for placeholder in self.placeholders:
            config = placeholder.get("config", {})
            
            # 应用过滤条件
            if model and config.get("model") != model:
                continue
            if attack and config.get("attack") != attack:
                continue
            if defense and config.get("defense") != defense:
                continue
            if dataset and config.get("dataset") != dataset:
                continue
            if status and placeholder.get("status") != status:
                continue
            
            if judger:
                placeholder_judger = config.get("judger", "")
                if isinstance(placeholder_judger, list):
                    placeholder_judger = "+".join(sorted(placeholder_judger))
                if str(placeholder_judger) != judger:
                    continue
            
            filtered.append(placeholder)
        
        return filtered
    
    def _calculate_realtime_progress(self, placeholder: Dict[str, Any]) -> Dict[str, Any]:
        """计算实时进度信息，包括部分ASR"""
        
        sample_results = placeholder.get("sample_results", [])
        total_samples = len(sample_results)
        
        # 统计各种状态的样本数
        completed_samples = 0
        samples_with_judger = 0
        
        for sample in sample_results:
            status = sample.get("status", "created")
            if status == "success":
                completed_samples += 1
            if sample.get("judger_result_on_clean") is not None:
                samples_with_judger += 1
        
        # 计算进度百分比
        completion_rate = completed_samples / total_samples if total_samples > 0 else 0.0
        judger_progress = samples_with_judger / total_samples if total_samples > 0 else 0.0
        
        return {
            "total_samples": total_samples,
            "completed_samples": completed_samples,
            "samples_with_judger": samples_with_judger,
            "completion_rate": completion_rate,
            "judger_progress": judger_progress
        }
    
    def _calculate_individual_judger_asrs(self, placeholder: Dict[str, Any]) -> tuple[List[float], float]:
        """计算单个占位符的各个judger ASR
        
        Returns:
            tuple[List[float], float]: (按Phase1顺序排列的judger ASR列表, 平均ASR)
            [harmbench_judger, gpt_judger_contextual_harmbench, gpt_judger_harmful_binary,
             gpt_judger_harmbench_style, gpt_judger_openai_policy, gpt_judger_tap_style, rejection_prefix_judger]
        """
        # Phase1 judger固定顺序
        phase1_judgers = [
            "harmbench_judger",
            "gpt_judger_contextual_harmbench", 
            "gpt_judger_harmful_binary",
            "gpt_judger_harmbench_style",
            "gpt_judger_openai_policy",
            "gpt_judger_tap_style",
            "rejection_prefix_judger"
        ]
        
        # 初始化每个judger的统计
        judger_stats = {judger: {"samples": 0, "unsafe": 0} for judger in phase1_judgers}
        
        # 检查sample_results中的结果
        sample_results = placeholder.get("sample_results", [])
        for sample in sample_results:
            # 检查是否有judger结果，不要求样本状态为success
            clean_result = sample.get("judger_result_on_clean")
            if clean_result is None:
                # 兼容占位符结果：优先使用 individual 结果
                clean_result = sample.get("judger_individual_results")
            if isinstance(clean_result, dict):
                # 多judger模式：分别统计每个judger
                for judger_name, result_value in clean_result.items():
                    if judger_name in judger_stats and result_value is not None:
                        judger_stats[judger_name]["samples"] += 1
                        if result_value == 1:
                            judger_stats[judger_name]["unsafe"] += 1
        
        # 检查sample_limit_statuses中的结果 - 这是关键修复
        sample_limit_statuses = placeholder.get("sample_limit_statuses", {})
        for limit_key, limit_data in sample_limit_statuses.items():
            if limit_data.get("status") == "success":
                limit_results = limit_data.get("results", [])
                for result in limit_results:
                    clean_result = result.get("judger_result_on_clean")
                    if clean_result is None:
                        # 兼容占位符结果：优先使用 individual 结果
                        clean_result = result.get("judger_individual_results")
                    if clean_result is not None and isinstance(clean_result, dict):
                        # 多judger模式：分别统计每个judger
                        for judger_name, result_value in clean_result.items():
                            if judger_name in judger_stats and result_value is not None:
                                judger_stats[judger_name]["samples"] += 1
                                if result_value == 1:
                                    judger_stats[judger_name]["unsafe"] += 1
        
        # 计算每个judger的ASR，按固定顺序返回
        judger_asrs = []
        valid_asrs = []
        for judger in phase1_judgers:
            stats = judger_stats[judger]
            if stats["samples"] > 0:
                asr = stats["unsafe"] / stats["samples"]
                judger_asrs.append(asr)
                valid_asrs.append(asr)
            else:
                judger_asrs.append(0.0)  # 无样本时默认为0.0
        
        # 计算平均ASR（仅对有结果的judger取平均）
        avg_asr = sum(valid_asrs) / len(valid_asrs) if valid_asrs else 0.0
        
        return judger_asrs, avg_asr
    
    def _has_judger_data(self, placeholder: Dict[str, Any]) -> bool:
        """检查占位符是否有实际的judger数据"""
        
        # 检查sample_results中的judger数据
        sample_results = placeholder.get("sample_results", [])
        for sample in sample_results:
            # 检查是否有judger结果，不要求样本状态为success
            clean_result = sample.get("judger_result_on_clean")
            if clean_result is None:
                # 兼容占位符结果：优先使用 individual 结果
                clean_result = sample.get("judger_individual_results")
            if isinstance(clean_result, dict) and clean_result:
                return True
            elif clean_result is not None:
                return True
        
        # 检查sample_limit_statuses中的judger数据
        sample_limit_statuses = placeholder.get("sample_limit_statuses", {})
        for limit_key, limit_data in sample_limit_statuses.items():
            if limit_data.get("status") == "success":
                limit_results = limit_data.get("results", [])
                for result in limit_results:
                    clean_result = result.get("judger_result_on_clean")
                    if clean_result is None:
                        # 兼容占位符结果：优先使用 individual 结果
                        clean_result = result.get("judger_individual_results")
                    if isinstance(clean_result, dict) and clean_result:
                        return True
                    elif clean_result is not None:
                        return True
        
        return False

    def display_table(self, sort_by: str = "created_time", limit: int = None,
                     show_details: bool = False, only_asr: bool = False) -> str:
        """显示占位符表格"""
        if not self.placeholders:
            return "📊 没有找到占位符实验"

        if only_asr:
            self.placeholders = [
                p for p in self.placeholders
                if p.get("status") == "success" and self._has_judger_data(p)
            ]
            if not self.placeholders:
                return "📊 没有包含ASR结果的占位符实验"
        
        # 排序 - 支持多种排序方式
        # 预计算ASR缓存，避免排序时重复计算
        asr_cache = {}
        def get_avg_asr(placeholder):
            """获取平均ASR用于排序 - 基于7个judger的平均值"""
            exp_id = placeholder.get("experiment_id", "")
            if exp_id in asr_cache:
                return asr_cache[exp_id]
            
            try:
                _, avg_asr = self._calculate_individual_judger_asrs(placeholder)
                asr_cache[exp_id] = avg_asr
                return avg_asr
            except Exception:
                asr_cache[exp_id] = 0.0
                return 0.0
        
        sort_key_map = {
            "created_time": lambda x: x.get("created_time", 0),
            "time": lambda x: x.get("created_time", 0),  # 别名
            "status": lambda x: x.get("status", ""),
            "samples": lambda x: x.get("total_samples") or 0,
            "success_rate": lambda x: (x.get("successful_samples") or 0) / max((x.get("total_samples") or 1), 1),
            "model": lambda x: x.get("config", {}).get("model", ""),
            "dataset": lambda x: x.get("config", {}).get("dataset", ""),
            "attack": lambda x: x.get("config", {}).get("attack", ""),
            "defense": lambda x: x.get("config", {}).get("defense", ""),
            "asr": get_avg_asr,  # 按平均ASR排名（7个judger的平均值）
            "clean_asr": get_avg_asr,  # 别名，按平均ASR排名
            "experiment_id": lambda x: x.get("experiment_id", "")
        }
        
        # 判断是否为数值排序（需要倒序）
        numeric_sorts = ["created_time", "time", "samples", "success_rate", "asr", "clean_asr"]
        reverse_sort = sort_by in numeric_sorts
        
        if sort_by in sort_key_map:
            sorted_placeholders = sorted(self.placeholders, 
                                       key=sort_key_map[sort_by], 
                                       reverse=reverse_sort)
        else:
            sorted_placeholders = self.placeholders
        
        # 限制数量
        if limit:
            sorted_placeholders = sorted_placeholders[:limit]
        
        # 构建表格
        output = ["📊 占位符实验看板"]
        output.append("=" * 230)  # 增加宽度以容纳新列
        
        # 添加Judger顺序说明
        output.append("📋 Phase1 Judger顺序 (Individual ASRs): [HB, GPT-C, GPT-H, GPT-HS, GPT-O, GPT-T, REJ]")
        output.append("   HB=harmbench_judger, GPT-C=gpt_judger_contextual_harmbench, GPT-H=gpt_judger_harmful_binary")
        output.append("   GPT-HS=gpt_judger_harmbench_style, GPT-O=gpt_judger_openai_policy, GPT-T=gpt_judger_tap_style, REJ=rejection_prefix_judger")
        output.append("=" * 230)
        
        # 表头
        if show_details:
            header = f"{'状态':<8} {'模型':<30} {'攻击':<15} {'防御':<15} {'数据集':<12} {'评判器':<20} {'样本':<8} {'成功率':<8} {'Individual Judger ASRs':<55} {'平均ASR':<10} {'ID':<12}"
        else:
            header = f"{'状态':<8} {'模型':<30} {'攻击':<15} {'防御':<15} {'样本':<8} {'成功率':<8} {'Individual Judger ASRs':<55} {'平均ASR':<10} {'创建时间':<19}"
        
        output.append(header)
        output.append("-" * 230)
        
        # 数据行
        for placeholder in sorted_placeholders:
            config = placeholder.get("config", {})
            status = placeholder.get("status", "unknown")
            
            # 状态图标
            status_icon = {
                "pending": "⏳",
                "running": "🔄", 
                "success": "✅",
                "failed": "❌"
            }.get(status, "❓")
            
            # 配置信息
            model = config.get("model", "N/A")[:28]
            attack = config.get("attack", "N/A")[:13]
            defense = config.get("defense", "N/A")[:13]
            dataset = config.get("dataset", "N/A")[:10]
            judger = config.get("judger", "N/A")
            if isinstance(judger, list):
                judger = "+".join(judger)[:18]
            else:
                judger = str(judger)[:18]
            
            # 统计信息
            total_samples = placeholder.get("total_samples") or 0
            success_count = placeholder.get("successful_samples") or 0
            success_rate = success_count / max(total_samples, 1) if total_samples > 0 else 0
            
            # 计算Individual Judger ASRs和实时进度
            judger_asrs, avg_asr = self._calculate_individual_judger_asrs(placeholder)
            progress_info = self._calculate_realtime_progress(placeholder)
            
            # 格式化Judger ASRs为显示字符串
            # 检查是否有实际的judger数据（不管ASR是否为0）
            has_judger_data = self._has_judger_data(placeholder)
            
            if has_judger_data:
                # 有实际judger数据时显示百分比，保留1位小数
                asr_display = "[" + ", ".join([f"{asr:.1%}" for asr in judger_asrs]) + "]"
                avg_asr_display = f"{avg_asr:.1%}"
                # 添加实时进度信息
                samples_with_judger = progress_info["samples_with_judger"]
                if samples_with_judger < total_samples:
                    asr_display += f" ({samples_with_judger}/{total_samples})"
            else:
                # 无judger数据时显示N/A，但显示进度
                progress_samples = progress_info["samples_with_judger"]
                if progress_samples > 0:
                    asr_display = f"[Processing... {progress_samples}/{total_samples}]"
                    avg_asr_display = "N/A"
                else:
                    asr_display = "[N/A, N/A, N/A, N/A, N/A, N/A, N/A]"
                    avg_asr_display = "N/A"
            
            # 截断过长的ASR显示字符串
            if len(asr_display) > 50:
                asr_display = asr_display[:47] + "...]" 
            
            # 时间信息
            created_time = placeholder.get("created_time", 0)
            if created_time:
                time_str = datetime.fromtimestamp(created_time).strftime("%Y-%m-%d %H:%M")
            else:
                time_str = "N/A"
            
            if show_details:
                exp_id = placeholder.get("experiment_id", "N/A")[:10]
                row = f"{status_icon} {status:<6} {model:<30} {attack:<18} {defense:<18} {dataset:<12} {judger:<28} {total_samples:<8} {success_rate:<9.1%} {asr_display:<55} {avg_asr_display:<10} {exp_id:<12}"
            else:
                row = f"{status_icon} {status:<6} {model:<30} {attack:<18} {defense:<18} {total_samples:<8} {success_rate:<9.1%} {asr_display:<55} {avg_asr_display:<10} {time_str:<19}"
            
            output.append(row)
        
        return "\n".join(output)
    
    def display_summary(self) -> str:
        """显示摘要统计"""
        summary = self.get_summary()
        
        if summary.get("total", 0) == 0:
            return summary.get("message", "没有数据")
        
        output = ["📈 占位符实验统计摘要"]
        output.append("=" * 60)
        
        # 总体统计
        output.append(f"总实验数: {summary['total']}")
        
        # 状态统计
        status_counts = summary["status_counts"]
        output.append("\n📊 实验状态分布:")
        for status, count in status_counts.items():
            icon = {"pending": "⏳", "running": "🔄", "success": "✅", "failed": "❌"}.get(status, "❓")
            percentage = count / summary['total'] * 100
            output.append(f"  {icon} {status:<8}: {count:<6} ({percentage:5.1f}%)")
        
        # 样本统计
        sample_stats = summary["sample_statistics"]
        output.append(f"\n🔢 样本统计:")
        output.append(f"  总样本数: {sample_stats['total_samples']}")
        output.append(f"  成功样本: {sample_stats['total_success']}")
        output.append(f"  失败样本: {sample_stats['total_failed']}")
        output.append(f"  实验成功率: {sample_stats['experiment_success_rate']:.1%}")
        output.append(f"  Clean ASR: {sample_stats['clean_asr']:.1%} ({sample_stats['clean_unsafe_samples']}/{sample_stats['clean_samples']})")
        
        # Individual Judger统计 (用于多judger分解)
        judger_stats = summary.get("judger_statistics", {})
        if judger_stats:
            output.append(f"\n⚖️ 各Judger Clean ASR分解:")
            sorted_judgers = sorted(judger_stats.items(), key=lambda x: x[1]['clean_asr'], reverse=True)
            for judger_name, stats in sorted_judgers:
                asr = stats['clean_asr']
                samples = stats['clean_samples'] 
                unsafe = stats['clean_unsafe_samples']
                output.append(f"  {judger_name:<35}: {asr:6.1%} ({unsafe}/{samples})")
        
        # 配置统计
        config_stats = summary["config_statistics"]
        output.append(f"\n⚙️ 配置分布:")
        
        for category, stats in config_stats.items():
            if stats:
                output.append(f"  {category.capitalize()}:")
                sorted_items = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]  # 显示前5个
                for name, count in sorted_items:
                    output.append(f"    {name}: {count}")
        
        return "\n".join(output)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """获取进度信息"""
        summary = self.get_summary()
        
        if summary.get("total", 0) == 0:
            return {"message": "没有实验数据"}
        
        status_counts = summary["status_counts"]
        total = summary["total"]
        
        completed = status_counts.get("success", 0)
        failed = status_counts.get("failed", 0)
        running = status_counts.get("running", 0)
        pending = status_counts.get("pending", 0)
        
        progress = {
            "total_experiments": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "completion_rate": completed / total if total > 0 else 0,
            "failure_rate": failed / total if total > 0 else 0
        }
        
        return progress
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误信息摘要"""
        if not self.placeholders:
            return {"message": "没有找到占位符实验"}
        
        error_summary = {
            "total_failed_samples": 0,
            "experiments_with_errors": 0,
            "error_types": defaultdict(int),
            "error_details": [],
            "most_common_errors": []
        }
        
        for placeholder in self.placeholders:
            sample_results = placeholder.get("sample_results", [])
            experiment_name = placeholder.get("experiment_name", "unknown")
            
            failed_samples = [s for s in sample_results if s.get("status") == "failed"]
            
            if failed_samples:
                error_summary["experiments_with_errors"] += 1
                error_summary["total_failed_samples"] += len(failed_samples)
                
                # 统计错误类型
                for sample in failed_samples:
                    error_msg = sample.get("error", "Unknown error")
                    if error_msg:
                        # 提取错误类型（前50个字符）
                        error_type = error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                        error_summary["error_types"][error_type] += 1
                        
                        # 收集详细错误信息
                        if len(error_summary["error_details"]) < 10:  # 只显示前10个
                            error_summary["error_details"].append({
                                "experiment": experiment_name,
                                "sample_index": sample.get("sample_index", "unknown"),
                                "error": error_msg,
                                "clean_prompt": sample.get("clean_prompt", "")[:100] + "..." if len(sample.get("clean_prompt", "")) > 100 else sample.get("clean_prompt", "")
                            })
        
        # 排序最常见错误
        error_summary["most_common_errors"] = sorted(
            error_summary["error_types"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return error_summary
    
    def display_error_summary(self) -> str:
        """显示错误信息摘要"""
        error_info = self.get_error_summary()
        
        if "message" in error_info:
            return error_info["message"]
        
        output = ["🚫 实验错误信息摘要"]
        output.append("=" * 60)
        
        # 基本统计
        output.append(f"存在错误的实验数: {error_info['experiments_with_errors']}")
        output.append(f"失败样本总数: {error_info['total_failed_samples']}")
        
        # 最常见错误
        if error_info["most_common_errors"]:
            output.append("\n🔽 最常见错误类型:")
            for error_type, count in error_info["most_common_errors"]:
                output.append(f"  • {error_type} ({count}次)")
        
        # 错误详情示例
        if error_info["error_details"]:
            output.append("\n🔍 错误详情示例:")
            for i, detail in enumerate(error_info["error_details"][:3], 1):  # 只显示前3个
                output.append(f"\n  {i}. 实验: {detail['experiment']}")
                output.append(f"     样本: {detail['sample_index']}")
                output.append(f"     提示: {detail['clean_prompt']}")
                output.append(f"     错误: {detail['error'][:200]}..." if len(detail['error']) > 200 else f"     错误: {detail['error']}")
        
        return "\n".join(output)


class UnifiedDashboard:
    """统一看板：同时显示占位符和结果"""
    
    def __init__(self, placeholders_dir: str = "experiments/placeholders",
                 results_dir: str = "experiments/results"):
        self.placeholder_dashboard = PlaceholderDashboard(placeholders_dir)
        # 如果需要也可以集成原有的ResultsDashboard
        self.placeholders_dir = Path(placeholders_dir)
        self.results_dir = Path(results_dir)
    
    def display_unified_view(self, sort_by: str = "created_time", only_asr: bool = False) -> str:
        """显示统一视图"""
        self.placeholder_dashboard.load_placeholders()
        
        output = ["🎯 PromptSecurity 实验总览"]
        output.append("=" * 80)
        
        # 占位符统计
        output.append(self.placeholder_dashboard.display_summary())
        
        # 进度信息
        progress = self.placeholder_dashboard.get_progress_info()
        if "message" not in progress:
            output.append(f"\n🚀 执行进度:")
            output.append(f"  总实验: {progress['total_experiments']}")
            output.append(f"  已完成: {progress['completed']} ({progress['completion_rate']:.1%})")
            output.append(f"  执行中: {progress['running']}")
            output.append(f"  待执行: {progress['pending']}")
            output.append(f"  失败数: {progress['failed']} ({progress['failure_rate']:.1%})")
        
        # 全部实验
        output.append(f"\n📋 全部实验:")
        all_table = self.placeholder_dashboard.display_table(
            sort_by=sort_by, 
            limit=None,  # 显示全部，不限制数量
            show_details=True,  # 显示详细信息
            only_asr=only_asr
        )
        output.append(all_table)
        
        return "\n".join(output)


def main():
    """测试看板功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description="占位符实验看板")
    parser.add_argument("--summary", action="store_true", help="显示摘要统计")
    parser.add_argument("--table", action="store_true", help="显示实验表格")
    parser.add_argument("--unified", action="store_true", help="显示统一视图")
    parser.add_argument("--status", choices=["pending", "running", "success", "failed"], help="过滤状态")
    parser.add_argument("--limit", type=int, help="限制显示数量")
    parser.add_argument("--sort", choices=["created_time", "status", "samples", "success_rate"], 
                       default="created_time", help="排序方式")
    
    args = parser.parse_args()
    
    if args.unified:
        dashboard = UnifiedDashboard()
        print(dashboard.display_unified_view())
    else:
        dashboard = PlaceholderDashboard()
        dashboard.load_placeholders(args.status)
        
        if args.summary or (not args.table and not args.summary):
            print(dashboard.display_summary())
        
        if args.table:
            print("\n" + dashboard.display_table(
                sort_by=args.sort,
                limit=args.limit,
                show_details=True
            ))


if __name__ == "__main__":
    main()
