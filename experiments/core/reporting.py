# experiments/reporting.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import re

class ResultsDashboard:
    """基于results文件夹的看板"""
    
    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results = []
        
    def load_results(self, pattern: Optional[str] = None) -> None:
        """加载results文件夹中的所有结果文件"""
        self.results = []
        
        # 获取所有JSON文件
        json_files = list(self.results_dir.glob("*.json"))
        
        # 如果指定了pattern，过滤文件
        if pattern:
            json_files = [f for f in json_files if pattern in f.name]
        
        # 加载每个文件
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 检查是否是批量实验文件（数组格式）
                    if isinstance(data, list):
                        # 跳过批量实验文件
                        continue
                    
                    # 只处理单个实验结果（字典格式）
                    if isinstance(data, dict):
                        # 添加文件名信息
                        data['_filename'] = file_path.name
                        data['_filepath'] = str(file_path)
                        self.results.append(data)
            except Exception as e:
                print(f"⚠️ 无法加载 {file_path.name}: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        if not self.results:
            return {"total": 0, "message": "没有加载任何结果"}
        
        summary = {
            "total": len(self.results),
            "models": defaultdict(int),
            "attacks": defaultdict(int),
            "defenses": defaultdict(int),
            "datasets": defaultdict(int),
            "judgers": defaultdict(int),
            "date_range": {"earliest": None, "latest": None}
        }
        
        # 统计各种组合
        for result in self.results:
            summary["models"][result.get("target_llm_name", "unknown")] += 1
            summary["attacks"][result.get("attack_method", "unknown")] += 1
            summary["defenses"][result.get("defense_method", "unknown")] += 1
            summary["datasets"][result.get("dataset_name", "unknown")] += 1
            
            # 处理judger_name可能是列表的情况
            judger = result.get("judger_name", "unknown")
            if isinstance(judger, list):
                judger = ", ".join(judger)
            summary["judgers"][judger] += 1
            
            # 从文件名提取时间戳
            match = re.search(r'(\d{8}_\d{6})', result.get("_filename", ""))
            if match:
                timestamp_str = match.group(1)
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    if summary["date_range"]["earliest"] is None or timestamp < summary["date_range"]["earliest"]:
                        summary["date_range"]["earliest"] = timestamp
                    if summary["date_range"]["latest"] is None or timestamp > summary["date_range"]["latest"]:
                        summary["date_range"]["latest"] = timestamp
                except:
                    pass
        
        # 转换defaultdict为普通dict
        summary["models"] = dict(summary["models"])
        summary["attacks"] = dict(summary["attacks"])
        summary["defenses"] = dict(summary["defenses"])
        summary["datasets"] = dict(summary["datasets"])
        summary["judgers"] = dict(summary["judgers"])
        
        return summary
    
    def filter_results(self, 
                      model: Optional[str] = None,
                      attack: Optional[str] = None,
                      defense: Optional[str] = None,
                      dataset: Optional[str] = None,
                      judger: Optional[str] = None) -> List[Dict[str, Any]]:
        """根据条件过滤结果"""
        filtered = self.results
        
        if model:
            filtered = [r for r in filtered if r.get("target_llm_name") == model]
        if attack:
            filtered = [r for r in filtered if r.get("attack_method") == attack]
        if defense:
            filtered = [r for r in filtered if r.get("defense_method") == defense]
        if dataset:
            filtered = [r for r in filtered if r.get("dataset_name") == dataset]
        if judger:
            filtered = [r for r in filtered if r.get("judger_name") == judger]
        
        return filtered
    
    def display_dashboard(self, sort_by: str = "time", limit: Optional[int] = None) -> str:
        """生成表格化看板视图"""
        lines = [
            "# 📊 实验结果看板",
            f"\n总计: {len(self.results)} 个实验结果\n"
        ]
        
        if not self.results:
            return "\n".join(lines + ["没有加载任何实验结果"])
        
        # 计算列宽
        col_widths = {
            'dataset': 12,
            'attack': 15,
            'defense': 15,
            'model': 25,
            'judger': 20,
            'samples': 8,
            'clean_asr': 10,
            'asr': 8,
            'queries': 8,
            'def_rate': 10,
            'errors': 8
        }
        
        # 表头
        header = (
            f"{'数据集':<{col_widths['dataset']}} "
            f"{'攻击':<{col_widths['attack']}} "
            f"{'防御':<{col_widths['defense']}} "
            f"{'模型':<{col_widths['model']}} "
            f"{'评判器':<{col_widths['judger']}} "
            f"{'样本数':<{col_widths['samples']}} "
            f"{'Clean ASR':<{col_widths['clean_asr']}} "
            f"{'ASR':<{col_widths['asr']}} "
            f"{'查询数':<{col_widths['queries']}} "
            f"{'防御率':<{col_widths['def_rate']}} "
            f"{'错误':<{col_widths['errors']}}"
        )
        
        separator = "=" * len(header)
        
        lines.extend([
            header,
            separator
        ])
        
        # 根据指定方式排序结果
        if sort_by == "time":
            sorted_results = sorted(self.results, 
                                  key=lambda x: x.get("_filename", ""), 
                                  reverse=True)
        elif sort_by == "asr":
            sorted_results = sorted(self.results, 
                                  key=lambda x: x.get("attack_success_rate", 0), 
                                  reverse=True)
        elif sort_by == "clean_asr":
            sorted_results = sorted(self.results, 
                                  key=lambda x: 1 - x.get("clean_safe_rate", 1), 
                                  reverse=True)
        elif sort_by == "samples":
            sorted_results = sorted(self.results, 
                                  key=lambda x: x.get("total_samples", 0), 
                                  reverse=True)
        else:
            sorted_results = self.results
        
        # 应用限制
        if limit and limit > 0:
            sorted_results = sorted_results[:limit]
            lines[1] = f"\n显示: {len(sorted_results)} / {len(self.results)} 个实验结果 (按 {sort_by} 排序)\n"
        
        # 生成每行数据
        for result in sorted_results:
            # 提取基本信息
            dataset = self._truncate(result.get("dataset_name", "unknown"), col_widths['dataset'])
            attack = self._truncate(result.get("attack_method", "unknown"), col_widths['attack'])
            defense = self._truncate(result.get("defense_method", "unknown"), col_widths['defense'])
            model = self._truncate(result.get("target_llm_name", "unknown"), col_widths['model'])
            
            # 处理judger (可能是列表)
            judger = result.get("judger_name", "unknown")
            if isinstance(judger, list):
                judger = "+".join(judger)
            judger = self._truncate(judger, col_widths['judger'])
            
            # 样本统计
            total_samples = result.get("total_samples", 0)
            failed_samples = result.get("failed_samples", 0)
            successful_samples = result.get("successful_samples", 0)
            
            # Clean ASR (clean sample attack success rate)
            clean_asr = 1 - result.get("clean_safe_rate", 0)
            clean_asr_str = f"{clean_asr:.1%}" if clean_asr >= 0 else "N/A"
            
            # ASR (attack success rate)
            asr = result.get("attack_success_rate", 0)
            asr_str = f"{asr:.1%}" if asr >= 0 else "N/A"
            
            # 查询数统计
            total_queries = self._calculate_total_queries(result)
            queries_str = str(total_queries) if total_queries >= 0 else "N/A"
            
            # 防御率 (defense rate = 1 - attack_success_rate when defense is used)
            if defense == "no_defense":
                def_rate_str = "N/A"
            else:
                def_rate = 1 - asr if asr >= 0 else 0
                def_rate_str = f"{def_rate:.1%}"
            
            # 错误统计
            errors_str = str(failed_samples) if failed_samples >= 0 else "N/A"
            
            # 格式化行
            row = (
                f"{dataset:<{col_widths['dataset']}} "
                f"{attack:<{col_widths['attack']}} "
                f"{defense:<{col_widths['defense']}} "
                f"{model:<{col_widths['model']}} "
                f"{judger:<{col_widths['judger']}} "
                f"{total_samples:<{col_widths['samples']}} "
                f"{clean_asr_str:<{col_widths['clean_asr']}} "
                f"{asr_str:<{col_widths['asr']}} "
                f"{queries_str:<{col_widths['queries']}} "
                f"{def_rate_str:<{col_widths['def_rate']}} "
                f"{errors_str:<{col_widths['errors']}}"
            )
            
            lines.append(row)
        
        # 添加说明
        lines.extend([
            "",
            separator,
            "说明:",
            "- Clean ASR: 无攻击时的不安全响应率",
            "- ASR: 攻击成功率",
            "- 查询数: 攻击过程中的总查询次数",
            "- 防御率: 1 - ASR (仅当使用防御时)",
            "- 错误: 失败的样本数量"
        ])
        
        return "\n".join(lines)
    
    def _truncate(self, text: str, max_length: int) -> str:
        """截断文本到指定长度"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _calculate_total_queries(self, result: Dict[str, Any]) -> int:
        """计算总查询次数"""
        total_queries = 0
        
        # 从sample_results中统计
        if "sample_results" in result:
            for sample in result.get("sample_results", []):
                attack_queries = sample.get("attack_query_count", 0)
                assistant_queries = sample.get("assistant_llm_query_count", 0)
                total_queries += attack_queries + assistant_queries
        
        return total_queries
    
    def get_experiment_details(self, filename: str) -> Optional[Dict[str, Any]]:
        """获取特定实验的详细信息"""
        for result in self.results:
            if result.get("_filename") == filename:
                return result
        return None

# 使用示例
if __name__ == "__main__":
    dashboard = ResultsDashboard()
    dashboard.load_results()
    print(dashboard.display_dashboard())