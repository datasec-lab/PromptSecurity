#!/usr/bin/env python3
"""
值标准管理器

统一管理实验中的默认值、错误值和null值，确保所有组件遵循一致的值标准，
避免与正常执行结果产生混淆。

核心原则：
- null: 表示未执行、执行失败或创建时的默认状态
- 有效值: 正常执行结果的实际值  
- 错误标识: 只在error字段中使用，其他字段必须为null
"""

import time
from typing import Dict, List, Any, Optional, Union


class ValueStandards:
    """值标准管理器"""
    
    # 状态定义
    STATUS_CREATED = "created"      # 刚创建，未开始执行
    STATUS_RUNNING = "running"      # 正在执行中
    STATUS_IN_PROGRESS = "in_progress"  # 样本正在处理中
    STATUS_SUCCESS = "success"      # 执行成功
    STATUS_ERROR = "error"          # 执行错误，可重新执行
    STATUS_PENDING = "pending"      # 等待重新执行（从error恢复）
    STATUS_FAILED = "failed"        # 配置/环境问题，需要人工干预
    
    @staticmethod
    def create_null_sample_template(sample_index: int, clean_prompt: str, attack_method: str = "no_attack") -> Dict[str, Any]:
        """创建包含所有null字段的sample模板
        
        用于placeholder创建时的初始状态，确保所有数据字段都是null
        
        Args:
            sample_index: 样本索引
            clean_prompt: 原始prompt
            attack_method: 攻击方法名，用于决定attacked_prompt的初始值
            
        Returns:
            包含所有null字段的sample字典
        """
        # 决定attacked_prompt的初始值
        if attack_method == "no_attack":
            attacked_prompt = None  # 没有攻击时为null
        else:
            attacked_prompt = clean_prompt  # 有攻击时初始值为clean_prompt
            
        return {
            # 基础标识字段（不能为null）
            "status": ValueStandards.STATUS_CREATED,
            "sample_index": sample_index,
            "clean_prompt": clean_prompt,
            "attacked_prompt": attacked_prompt,
            
            # 执行相关字段（创建时都是null）
            "llm_response_on_clean": None,
            "llm_response_on_attacked": None,
            "llm_response_on_clean_under_defense": None,
            "llm_response_on_attacked_under_defense": None,
            
            # 时间相关字段（创建时都是null）
            "llm_response_time_clean": None,
            "llm_response_time_attacked": None,
            "llm_response_time_under_defense": None,
            
            # 攻击相关字段（创建时都是null）
            "attack_runtime": None,
            "attack_query_count": None,
            "assistant_llm_query_count": None,
            "attack_config": None,
            
            # 防御相关字段（创建时都是null）
            "defense_config": None,
            
            # 评判相关字段（创建时都是null）
            "judger_result_on_clean": None,
            "judger_result_on_attack": None,
            "judger_result_on_clean_under_defense": None,
            "judger_result_on_attack_under_defense": None,
            "judger_config": None,
            
            # 模型相关字段（创建时都是null）
            "target_llm_type": None,
            "target_llm_parameters": None,
            
            # 元数据字段
            "sample_metadata": {},  # 可以包含原始数据，但执行相关信息为null
            "error": None  # 错误信息字段
        }
    
    @staticmethod
    def create_null_experiment_template(config: Dict[str, Any], experiment_id: str, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建包含所有null字段的实验模板
        
        用于placeholder创建时的初始状态
        
        Args:
            config: 实验配置
            experiment_id: 实验ID
            samples: 样本列表
            
        Returns:
            包含所有null字段的实验字典
        """
        judger = config.get('judger', 'unknown')
        if isinstance(judger, list):
            judger_str = "+".join(sorted(str(j) for j in judger))
        else:
            judger_str = str(judger)
            
        current_time = time.time()
        
        return {
            # 占位符系统字段
            "experiment_id": experiment_id,
            "config": config,
            "status": ValueStandards.STATUS_CREATED,
            "created_time": current_time,
            "last_updated": None,  # 创建时未更新过
            "dependencies": [],  # 稍后由其他逻辑填充
            "metadata": {
                "placeholder_version": "3.0",
                "value_standards_version": "1.0"
            },
            
            # 实验结果字段（创建时都是null）
            "experiment_name": f"evaluation_{config.get('model', 'unknown')}_{config.get('attack', 'no_attack')}_{config.get('defense', 'no_defense')}_{config.get('dataset', 'unknown')}_{judger_str}",
            "experiment_timestamp": None,  # 执行完成时才设置
            "execution_time": None,  # 执行完成时才设置
            
            # 配置信息（不能为null）
            "target_llm_name": config.get("model", "unknown"),
            "attack_method": config.get("attack", "no_attack"),
            "defense_method": config.get("defense", "no_defense"),
            "dataset_name": config.get("dataset", "unknown"),
            "judger_name": judger,
            
            # 统计字段（创建时都是null）
            "total_samples": len(samples),
            "successful_samples": None,  # 执行完成时才设置
            "failed_samples": None,  # 执行完成时才设置
            "clean_safe_rate": None,  # 执行完成时才设置
            "attack_success_rate": None,  # 执行完成时才设置
            
            # 样本结果（创建时都是空的模板）
            "sample_results": samples,
            
            # 向后兼容字段
            "sample_limit": len(samples),
            "samples": samples,
            "success_count": None,  # 执行完成时才设置
            "failed_count": None,  # 执行完成时才设置
            
            # 错误信息字段
            "error": None
        }
    
    @staticmethod 
    def create_error_result(component_type: str, component_name: str, error_message: str) -> Dict[str, Any]:
        """创建统一的错误结果
        
        所有组件执行失败时都应该返回这种格式的结果，
        确保所有数据字段都是null，只有error字段包含错误信息
        
        Args:
            component_type: 组件类型 ("judger", "attack", "defense", "model")
            component_name: 组件名称
            error_message: 错误消息
            
        Returns:
            统一格式的错误结果字典
        """
        base_error = {
            f"{component_type}_error": error_message,
            "error_component": component_type,
            "error_component_name": component_name,
            "error_timestamp": time.time()
        }
        
        if component_type == "judger":
            base_error.update({
                "judger_result_on_clean": None,
                "judger_result_on_attack": None,
                "judger_result_on_clean_under_defense": None,
                "judger_result_on_attack_under_defense": None,
                "judger_config": None
            })
        elif component_type == "attack":
            base_error.update({
                "attacked_prompt": None,  # 失败时不返回clean_prompt
                "attack_runtime": None,
                "attack_query_count": None,
                "assistant_llm_query_count": None,
                "attack_config": None,
                "llm_response_on_attacked": None,
                "llm_response_time_attacked": None
            })
        elif component_type == "defense":
            base_error.update({
                "defense_config": None,
                "llm_response_on_attacked_under_defense": None,
                "llm_response_on_clean_under_defense": None,
                "llm_response_time_under_defense": None
            })
        elif component_type == "model":
            base_error.update({
                "llm_response": None,
                "llm_response_time": None,
                "target_llm_type": None,
                "target_llm_parameters": None
            })
            
        return base_error
    
    @staticmethod
    def validate_no_ambiguous_values(result: Dict[str, Any]) -> List[str]:
        """验证结果中没有歧义值
        
        检测可能造成混淆的值，如-1, 0, "", []等
        
        Args:
            result: 要验证的结果字典
            
        Returns:
            发现的歧义值问题列表
        """
        issues = []
        
        # 定义歧义值模式
        ambiguous_numeric = [-1, 0, 0.0]
        ambiguous_string = ["", "unknown", "none", "null"]
        ambiguous_list = []
        ambiguous_dict = {}
        
        def check_field(key: str, value: Any, path: str = ""):
            field_path = f"{path}.{key}" if path else key
            
            # 跳过error字段和基础标识字段的检查
            if key in ["error", "sample_index", "status", "experiment_id", "config", "created_time", "total_samples", "sample_limit"]:
                return
            
            # 检查数值字段
            if isinstance(value, (int, float)) and value in ambiguous_numeric:
                if any(pattern in key for pattern in ['_time', '_runtime', '_count', '_rate']):
                    issues.append(f"字段 '{field_path}' 有歧义值 {value}，应该使用 null")
            
            # 检查字符串字段  
            elif isinstance(value, str) and value.lower() in ambiguous_string:
                if any(pattern in key for pattern in ['_response', '_prompt', '_result', '_config']):
                    issues.append(f"字段 '{field_path}' 有歧义值 '{value}'，应该使用 null")
            
            # 检查列表字段
            elif isinstance(value, list) and value == ambiguous_list:
                if key.endswith(('_results', '_list', '_data')):
                    issues.append(f"字段 '{field_path}' 有歧义值 []，应该使用 null")
            
            # 检查字典字段
            elif isinstance(value, dict) and value == ambiguous_dict:
                if key.endswith(('_config', '_params', '_metadata')):
                    issues.append(f"字段 '{field_path}' 有歧义值 {{}}，应该使用 null")
            
            # 递归检查嵌套字典
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    check_field(sub_key, sub_value, field_path)
        
        # 递归检查所有字段
        for key, value in result.items():
            check_field(key, value)
        
        return issues
    
    @staticmethod  
    def is_valid_status_transition(from_status: str, to_status: str) -> bool:
        """验证状态转换是否合法
        
        合法的状态转换：
        created → running → success
        created → running → error → pending → running → ...
        created → running → failed
        
        Args:
            from_status: 当前状态
            to_status: 目标状态
            
        Returns:
            状态转换是否合法
        """
        valid_transitions = {
            ValueStandards.STATUS_CREATED: [ValueStandards.STATUS_RUNNING, ValueStandards.STATUS_IN_PROGRESS],
            ValueStandards.STATUS_RUNNING: [ValueStandards.STATUS_SUCCESS, ValueStandards.STATUS_ERROR, ValueStandards.STATUS_FAILED, ValueStandards.STATUS_IN_PROGRESS],
            ValueStandards.STATUS_IN_PROGRESS: [ValueStandards.STATUS_SUCCESS, ValueStandards.STATUS_ERROR, ValueStandards.STATUS_FAILED, ValueStandards.STATUS_RUNNING],
            ValueStandards.STATUS_ERROR: [ValueStandards.STATUS_PENDING, ValueStandards.STATUS_FAILED],
            ValueStandards.STATUS_PENDING: [ValueStandards.STATUS_RUNNING, ValueStandards.STATUS_IN_PROGRESS],
            ValueStandards.STATUS_SUCCESS: [],  # 成功状态是终态
            ValueStandards.STATUS_FAILED: []    # 失败状态是终态
        }
        
        return to_status in valid_transitions.get(from_status, [])
    
    @staticmethod
    def get_status_description(status: str) -> str:
        """获取状态描述"""
        descriptions = {
            ValueStandards.STATUS_CREATED: "已创建，尚未开始执行",
            ValueStandards.STATUS_RUNNING: "正在执行中",
            ValueStandards.STATUS_IN_PROGRESS: "样本正在处理中",
            ValueStandards.STATUS_SUCCESS: "执行成功完成",
            ValueStandards.STATUS_ERROR: "执行出错，可重新执行", 
            ValueStandards.STATUS_PENDING: "等待重新执行",
            ValueStandards.STATUS_FAILED: "执行失败，需要人工干预"
        }
        return descriptions.get(status, f"未知状态: {status}")


# 便捷函数
def create_null_sample(sample_index: int, clean_prompt: str, attack_method: str = "no_attack") -> Dict[str, Any]:
    """创建null值sample的便捷函数"""
    return ValueStandards.create_null_sample_template(sample_index, clean_prompt, attack_method)

def create_null_experiment(config: Dict[str, Any], experiment_id: str, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """创建null值experiment的便捷函数"""
    return ValueStandards.create_null_experiment_template(config, experiment_id, samples)

def create_error_result(component_type: str, component_name: str, error_message: str) -> Dict[str, Any]:
    """创建错误结果的便捷函数"""
    return ValueStandards.create_error_result(component_type, component_name, error_message)

def validate_values(result: Dict[str, Any]) -> List[str]:
    """验证值的便捷函数"""
    return ValueStandards.validate_no_ambiguous_values(result)

def is_valid_transition(from_status: str, to_status: str) -> bool:
    """验证状态转换的便捷函数"""
    return ValueStandards.is_valid_status_transition(from_status, to_status)


if __name__ == "__main__":
    # 测试代码
    import json
    
    print("=== 值标准管理器测试 ===")
    
    # 测试创建null sample模板
    print("\n1. 创建null sample模板:")
    sample = create_null_sample(0, "测试prompt", "ArtPrompt")
    print(json.dumps(sample, indent=2, ensure_ascii=False))
    
    # 测试验证功能
    print("\n2. 测试值验证:")
    
    # 创建一个有歧义值的结果
    bad_result = {
        "llm_response_time_clean": 0.0,  # 歧义值
        "attack_query_count": -1,        # 歧义值
        "judger_result_on_clean": "",    # 歧义值
        "attack_config": {},             # 歧义值
        "sample_index": 0,               # 这个是合法的
        "clean_prompt": "测试"           # 这个是合法的
    }
    
    issues = validate_values(bad_result)
    print(f"发现 {len(issues)} 个歧义值问题:")
    for issue in issues:
        print(f"  - {issue}")
    
    # 测试状态转换
    print("\n3. 测试状态转换:")
    transitions = [
        ("created", "running"),
        ("running", "success"),
        ("running", "error"),
        ("error", "pending"),
        ("pending", "running"),
        ("success", "running"),  # 非法
    ]
    
    for from_status, to_status in transitions:
        valid = is_valid_transition(from_status, to_status)
        print(f"  {from_status} → {to_status}: {'✅ 合法' if valid else '❌ 非法'}")