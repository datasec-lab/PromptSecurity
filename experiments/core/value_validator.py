#!/usr/bin/env python3
"""
值验证和歧义检测工具

用于验证实验结果中是否存在歧义值，帮助识别和修复数据质量问题。
支持批量验证占位符文件、实验结果和任意JSON数据。
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from value_standards import ValueStandards, validate_values

logger = logging.getLogger(__name__)


class ValueValidator:
    """值验证器"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.validation_stats = {
            "total_files": 0,
            "files_with_issues": 0,
            "total_issues": 0,
            "issue_categories": defaultdict(int)
        }
    
    def validate_placeholder_file(self, filepath: Path) -> Dict[str, Any]:
        """验证单个占位符文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self._validate_experiment_data(data, str(filepath))
            
        except Exception as e:
            logger.error(f"读取文件失败 {filepath}: {e}")
            return {
                "filepath": str(filepath),
                "valid": False,
                "error": f"文件读取失败: {e}",
                "issues": [],
                "critical_issues": []
            }
    
    def validate_placeholder_directory(self, directory: Path) -> Dict[str, Any]:
        """验证整个占位符目录"""
        if not directory.exists():
            return {
                "directory": str(directory),
                "valid": False,
                "error": "目录不存在",
                "file_results": []
            }
        
        file_results = []
        total_issues = 0
        files_with_issues = 0
        
        # 扫描所有JSON文件
        json_files = list(directory.glob("*.json"))
        
        if not json_files:
            return {
                "directory": str(directory),
                "valid": True,
                "message": "目录中没有JSON文件",
                "file_results": []
            }
        
        print(f"🔍 开始验证目录: {directory}")
        print(f"📁 发现 {len(json_files)} 个JSON文件")
        
        for filepath in json_files:
            file_result = self.validate_placeholder_file(filepath)
            file_results.append(file_result)
            
            if not file_result["valid"] or file_result["issues"]:
                files_with_issues += 1
                total_issues += len(file_result.get("issues", []))
            
            # 更新统计
            self.validation_stats["total_files"] += 1
            if file_result.get("issues"):
                self.validation_stats["files_with_issues"] += 1
                self.validation_stats["total_issues"] += len(file_result["issues"])
                
                # 分类统计问题
                for issue in file_result.get("issues", []):
                    category = self._categorize_issue(issue)
                    self.validation_stats["issue_categories"][category] += 1
        
        return {
            "directory": str(directory),
            "valid": files_with_issues == 0,
            "total_files": len(json_files),
            "files_with_issues": files_with_issues,
            "total_issues": total_issues,
            "file_results": file_results
        }
    
    def _validate_experiment_data(self, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """验证实验数据"""
        result = {
            "filepath": source,
            "valid": True,
            "issues": [],
            "critical_issues": [],
            "warnings": []
        }
        
        # 1. 验证顶级字段的歧义值
        top_level_issues = validate_values(data)
        result["issues"].extend(top_level_issues)
        
        # 2. 验证sample_results中的歧义值
        sample_results = data.get("sample_results", [])
        if isinstance(sample_results, list):
            for i, sample in enumerate(sample_results):
                if isinstance(sample, dict):
                    sample_issues = validate_values(sample)
                    # 为每个issue添加sample索引信息
                    for issue in sample_issues:
                        result["issues"].append(f"样本[{i}] {issue}")
        
        # 3. 验证状态一致性
        status_issues = self._validate_status_consistency(data)
        result["issues"].extend(status_issues)
        
        # 4. 验证关键字段缺失
        missing_field_issues = self._validate_required_fields(data)
        result["critical_issues"].extend(missing_field_issues)
        
        # 5. 验证数据类型一致性
        type_issues = self._validate_data_types(data)
        result["warnings"].extend(type_issues)
        
        # 总体验证状态
        if result["issues"] or result["critical_issues"]:
            result["valid"] = False
        
        return result
    
    def _validate_status_consistency(self, data: Dict[str, Any]) -> List[str]:
        """验证状态一致性"""
        issues = []
        
        experiment_status = data.get("status", "unknown")
        sample_results = data.get("sample_results", [])
        
        if not isinstance(sample_results, list):
            return issues
        
        # 检查实验状态与样本状态的一致性
        sample_statuses = [s.get("status", "unknown") for s in sample_results if isinstance(s, dict)]
        
        if experiment_status == "success":
            # 如果实验状态是success，所有样本应该也是success
            failed_samples = [s for s in sample_statuses if s != "success"]
            if failed_samples:
                issues.append(f"实验状态为success但有 {len(failed_samples)} 个样本状态不是success")
        
        elif experiment_status == "pending":
            # 如果实验状态是pending，应该没有样本是success
            success_samples = [s for s in sample_statuses if s == "success"]
            if success_samples:
                issues.append(f"实验状态为pending但有 {len(success_samples)} 个样本状态是success")
        
        return issues
    
    def _validate_required_fields(self, data: Dict[str, Any]) -> List[str]:
        """验证必需字段"""
        issues = []
        
        # 关键的必需字段
        required_fields = [
            "experiment_id", "config", "status", "sample_results"
        ]
        
        for field in required_fields:
            if field not in data:
                issues.append(f"缺少关键字段: {field}")
            elif data[field] is None:
                issues.append(f"关键字段为null: {field}")
        
        # 验证config结构
        config = data.get("config", {})
        if isinstance(config, dict):
            config_required = ["model", "attack", "defense", "dataset", "judger"]
            for field in config_required:
                if field not in config:
                    issues.append(f"配置中缺少字段: {field}")
        
        return issues
    
    def _validate_data_types(self, data: Dict[str, Any]) -> List[str]:
        """验证数据类型一致性"""
        warnings = []
        
        # 检查时间字段应该是数字类型
        time_fields = ["created_time", "last_updated", "experiment_timestamp"]
        for field in time_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], (int, float)):
                    warnings.append(f"时间字段 {field} 应该是数字类型，当前是 {type(data[field])}")
        
        # 检查计数字段应该是整数
        count_fields = ["total_samples", "successful_samples", "failed_samples", "success_count", "failed_count"]
        for field in count_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], int):
                    warnings.append(f"计数字段 {field} 应该是整数类型，当前是 {type(data[field])}")
        
        return warnings
    
    def _categorize_issue(self, issue: str) -> str:
        """将问题分类"""
        issue_lower = issue.lower()
        
        if "时间" in issue or "time" in issue_lower:
            return "时间字段歧义"
        elif "count" in issue_lower or "计数" in issue:
            return "计数字段歧义"
        elif "response" in issue_lower or "响应" in issue:
            return "响应字段歧义"
        elif "config" in issue_lower or "配置" in issue:
            return "配置字段歧义"
        elif "状态" in issue or "status" in issue_lower:
            return "状态不一致"
        elif "缺少" in issue or "missing" in issue_lower:
            return "字段缺失"
        else:
            return "其他问题"
    
    def generate_validation_report(self, validation_result: Dict[str, Any]) -> str:
        """生成验证报告"""
        if validation_result.get("error"):
            return f"❌ 验证失败: {validation_result['error']}"
        
        directory = validation_result.get("directory", "单文件")
        file_results = validation_result.get("file_results", [])
        
        if not file_results:
            return f"✅ {directory}: 没有文件需要验证"
        
        total_files = len(file_results)
        files_with_issues = sum(1 for f in file_results if not f["valid"] or f.get("issues"))
        total_issues = sum(len(f.get("issues", [])) for f in file_results)
        
        report_lines = [
            f"📊 验证报告: {directory}",
            f"├── 总文件数: {total_files}",
            f"├── 有问题文件: {files_with_issues}",
            f"├── 总问题数: {total_issues}",
            f"└── 验证状态: {'❌ 发现问题' if files_with_issues > 0 else '✅ 全部通过'}"
        ]
        
        if files_with_issues > 0 and self.verbose:
            report_lines.append("\\n🔍 详细问题:")
            
            for file_result in file_results:
                if not file_result["valid"] or file_result.get("issues"):
                    filepath = Path(file_result["filepath"]).name
                    report_lines.append(f"\\n📁 {filepath}:")
                    
                    if file_result.get("error"):
                        report_lines.append(f"   ❌ 错误: {file_result['error']}")
                    
                    for issue in file_result.get("issues", []):
                        report_lines.append(f"   ⚠️  {issue}")
                    
                    for critical in file_result.get("critical_issues", []):
                        report_lines.append(f"   🚨 严重: {critical}")
                    
                    if self.verbose and file_result.get("warnings"):
                        for warning in file_result.get("warnings", []):
                            report_lines.append(f"   💡 警告: {warning}")
        
        return "\\n".join(report_lines)
    
    def generate_summary_report(self) -> str:
        """生成汇总统计报告"""
        stats = self.validation_stats
        
        if stats["total_files"] == 0:
            return "📊 没有文件被验证"
        
        report_lines = [
            "📈 验证统计汇总:",
            f"├── 验证文件总数: {stats['total_files']}",
            f"├── 有问题文件数: {stats['files_with_issues']}",
            f"├── 问题总数: {stats['total_issues']}",
            f"└── 通过率: {((stats['total_files'] - stats['files_with_issues']) / stats['total_files'] * 100):.1f}%"
        ]
        
        if stats["issue_categories"]:
            report_lines.append("\\n🏷️  问题分类:")
            for category, count in sorted(stats["issue_categories"].items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"   ├── {category}: {count} 个")
        
        return "\\n".join(report_lines)
    
    def fix_common_issues(self, filepath: Path, backup: bool = True) -> Dict[str, Any]:
        """尝试自动修复常见问题"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            original_data = json.dumps(data, ensure_ascii=False)
            fixes_applied = []
            
            # 创建备份
            if backup:
                backup_path = filepath.with_suffix('.json.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_data)
                fixes_applied.append(f"创建备份: {backup_path}")
            
            # 修复常见的歧义值
            fixes_applied.extend(self._fix_ambiguous_values(data))
            
            # 修复状态不一致
            fixes_applied.extend(self._fix_status_inconsistency(data))
            
            # 如果有修复，保存文件
            if len(fixes_applied) > (1 if backup else 0):
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                fixes_applied.append(f"保存修复后的文件: {filepath}")
            
            return {
                "filepath": str(filepath),
                "success": True,
                "fixes_applied": fixes_applied
            }
            
        except Exception as e:
            return {
                "filepath": str(filepath),
                "success": False,
                "error": str(e),
                "fixes_applied": []
            }
    
    def _fix_ambiguous_values(self, data: Dict[str, Any]) -> List[str]:
        """修复歧义值"""
        fixes = []
        
        # 修复顶级字段
        for key, value in data.items():
            if self._is_ambiguous_value(key, value):
                data[key] = None
                fixes.append(f"修复字段 {key}: {value} → null")
        
        # 修复sample_results
        sample_results = data.get("sample_results", [])
        if isinstance(sample_results, list):
            for i, sample in enumerate(sample_results):
                if isinstance(sample, dict):
                    for key, value in sample.items():
                        if self._is_ambiguous_value(key, value):
                            sample[key] = None
                            fixes.append(f"修复样本[{i}].{key}: {value} → null")
        
        return fixes
    
    def _fix_status_inconsistency(self, data: Dict[str, Any]) -> List[str]:
        """修复状态不一致"""
        fixes = []
        
        sample_results = data.get("sample_results", [])
        if not isinstance(sample_results, list):
            return fixes
        
        # 检查样本状态，决定实验状态
        sample_statuses = [s.get("status", "unknown") for s in sample_results if isinstance(s, dict)]
        
        if not sample_statuses:
            return fixes
        
        success_count = sample_statuses.count("success")
        total_count = len(sample_statuses)
        
        # 根据样本状态确定实验状态
        if success_count == total_count:
            expected_status = "success"
        elif success_count == 0:
            expected_status = "pending"
        else:
            expected_status = "partial"
        
        current_status = data.get("status", "unknown")
        if current_status != expected_status:
            data["status"] = expected_status
            fixes.append(f"修复实验状态: {current_status} → {expected_status}")
        
        return fixes
    
    def _is_ambiguous_value(self, key: str, value: Any) -> bool:
        """判断是否是歧义值"""
        if value is None:
            return False
        
        # 数值类型的歧义值
        if isinstance(value, (int, float)):
            if key.endswith(('_time', '_runtime', '_count', '_rate')) and value in [-1, 0, 0.0]:
                return True
        
        # 字符串类型的歧义值
        elif isinstance(value, str):
            if key.endswith(('_response', '_prompt', '_result', '_config')) and value.lower() in ["", "unknown", "none", "null"]:
                return True
        
        # 列表和字典类型的歧义值
        elif isinstance(value, list) and value == []:
            if key.endswith(('_results', '_list', '_data')):
                return True
        elif isinstance(value, dict) and value == {}:
            if key.endswith(('_config', '_params', '_metadata')):
                return True
        
        return False


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="值验证和歧义检测工具")
    parser.add_argument("path", help="要验证的文件或目录路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    parser.add_argument("--fix", action="store_true", help="尝试自动修复常见问题")
    parser.add_argument("--no-backup", action="store_true", help="修复时不创建备份")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建验证器
    validator = ValueValidator(verbose=args.verbose)
    
    # 验证路径
    path = Path(args.path)
    if not path.exists():
        print(f"❌ 路径不存在: {path}")
        return 1
    
    # 执行验证
    if path.is_file():
        print(f"🔍 验证文件: {path}")
        result = validator.validate_placeholder_file(path)
        
        if args.fix and not result["valid"]:
            print("🔧 尝试自动修复...")
            fix_result = validator.fix_common_issues(path, backup=not args.no_backup)
            if fix_result["success"]:
                print("✅ 修复完成:")
                for fix in fix_result["fixes_applied"]:
                    print(f"   - {fix}")
                
                # 重新验证
                result = validator.validate_placeholder_file(path)
            else:
                print(f"❌ 修复失败: {fix_result['error']}")
        
        print(validator.generate_validation_report({"file_results": [result]}))
        
    elif path.is_dir():
        print(f"🔍 验证目录: {path}")
        result = validator.validate_placeholder_directory(path)
        
        if args.fix and result.get("files_with_issues", 0) > 0:
            print("🔧 批量修复...")
            fixed_count = 0
            for file_result in result.get("file_results", []):
                if not file_result["valid"] or file_result.get("issues"):
                    file_path = Path(file_result["filepath"])
                    fix_result = validator.fix_common_issues(file_path, backup=not args.no_backup)
                    if fix_result["success"]:
                        fixed_count += 1
            print(f"✅ 修复了 {fixed_count} 个文件")
            
            # 重新验证
            result = validator.validate_placeholder_directory(path)
        
        print(validator.generate_validation_report(result))
        
        if args.stats:
            print("\\n" + validator.generate_summary_report())
    
    else:
        print(f"❌ 不支持的路径类型: {path}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())