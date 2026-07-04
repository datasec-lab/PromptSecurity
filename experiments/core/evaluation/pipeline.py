"""
核心评估管道实现
"""

from typing import Dict, List, Any, Optional, Union
import time
import logging
import sys
import traceback
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# 导入现有模块
# pipeline.py is now in experiments/core/evaluation/, need to go up 4 levels
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

def validate_token_limit(response: str, limit: int = 512, model_name: str = "unknown") -> str:
    """验证并强制执行token限制"""
    if not response:
        return response
    
    # 粗略估算token数量 (1 token ≈ 4 characters for most models)
    estimated_tokens = len(response) // 4
    
    if estimated_tokens > limit:
        logger = logging.getLogger(__name__)
        logger.warning(f"Model {model_name} output exceeds token limit: {estimated_tokens} > {limit}. Truncating response.")
        
        # 截断到安全长度，保留完整句子
        safe_length = limit * 4
        truncated = response[:safe_length]
        
        # 尝试在句号、问号或感叹号处截断以保持语义完整性
        for punct in ['. ', '? ', '! ', '\n\n']:
            last_punct = truncated.rfind(punct)
            if last_punct > safe_length * 0.7:  # 如果在后30%找到标点，且不超过token限制
                candidate_truncated = truncated[:last_punct + len(punct)]
                candidate_tokens = len(candidate_truncated) // 4
                if candidate_tokens <= limit:  # 确保不超过token限制
                    truncated = candidate_truncated
                    break
        
        # 添加truncation标记，但确保总长度不超过限制
        truncation_marker = "... [TRUNCATED DUE TO TOKEN LIMIT]"
        marker_tokens = len(truncation_marker) // 4
        
        # 确保最终结果不超过token限制
        final_tokens = len(truncated) // 4 + marker_tokens
        if final_tokens > limit:
            # 进一步截断以容纳标记
            available_tokens = limit - marker_tokens
            available_chars = available_tokens * 4
            truncated = truncated[:available_chars]
        
        return truncated + truncation_marker
    
    return response

# 缓存管理器已移除 - 使用占位符系统中的结果复用替代
# from experiments.evaluation.cache_manager import get_cache_manager

# 设置随机种子
RANDOM_SEED = 42

def get_current_timestamp() -> float:
    """获取当前时间戳，确保合理性"""
    try:
        # 使用UTC时间避免时区问题
        current_time = datetime.now(timezone.utc).timestamp()
        
        # 验证时间戳的合理性（应该在2020年到2030年之间）
        min_timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()  # 2020年
        max_timestamp = datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp()  # 2030年
        
        if min_timestamp <= current_time <= max_timestamp:
            return current_time
        else:
            # 如果系统时间异常，使用一个合理的默认时间（2024年）
            logger = logging.getLogger(__name__)
            logger.warning(f"System time appears to be incorrect: {current_time}. Using fallback timestamp.")
            return datetime(2024, 7, 15, tzinfo=timezone.utc).timestamp()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting timestamp: {e}. Using fallback.")
        return datetime(2024, 7, 15, tzinfo=timezone.utc).timestamp()

def set_random_seed(seed: int = RANDOM_SEED):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    # 如果使用torch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def run_evaluation(model: str,
                  attack: str = "no_attack",
                  defense: str = "no_defense", 
                  dataset: str = "harmbench",
                  judger: str = "harmbench_judger",
                  **kwargs) -> Dict[str, Any]:
    """
    运行单次5要素评估
    
    Args:
        model: 模型名称 (如 "gpt-4o", "Llama-3.1-8B-Instruct")
        attack: 攻击方法 (如 "ArtPrompt", "no_attack") 
        defense: 防御方法 (如 "smooth_llm", "no_defense")
        dataset: 数据集 (如 "harmbench", "jbb", "airbench")
        judger: 评判器 (如 "harmbench_judger", "gpt_judger_contextual_harmbench")
        **kwargs: 额外配置参数
            - 缓存相关参数已移除，使用占位符系统中的结果复用替代
    
    Returns:
        评估结果字典，包含所有RESULT_FORMAT.md定义的字段
    """
    
    # 设置随机种子
    seed = kwargs.get('seed', RANDOM_SEED)
    set_random_seed(seed)
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始评估: {model} + {attack} + {defense} + {dataset} + {judger}")
    logger.info(f"使用随机种子: {seed}")
    
    # 缓存管理已移除 - 使用占位符系统中的结果复用替代
    logger.info("使用占位符系统中的结果复用替代缓存功能")
    
    start_time = time.time()
    
    try:
        # 1. 加载组件
        model_instance, model_parameters = _load_model(model, **kwargs)
        logger.info(f"Model parameters for {model}: {model_parameters}")
        
        # 攻击和防御需要传入model实例
        attack_kwargs = {**kwargs, 'target_model': model_instance}
        defense_kwargs = {**kwargs, 'target_model': model_instance}
        
        # 检查是否是本地模型，如果是，尝试获取tokenizer
        if hasattr(model_instance, 'tokenizer'):
            defense_kwargs['tokenizer'] = model_instance.tokenizer
        
        attack_instance = _load_attack(attack, **attack_kwargs)
        defense_instance = _load_defense(defense, **defense_kwargs)
        dataset_instance = _load_dataset(dataset, **kwargs)
        judger_instance = _load_judger(judger, **kwargs)
        
        # 2. 执行评估流程
        # 传递组件名称给execute_evaluation
        evaluation_kwargs = {
            **kwargs,
            'model_name': model,
            'attack_name': attack,
            'defense_name': defense,
            'dataset_name': dataset,
            'judger_name': judger
        }
        result = _execute_evaluation(
            model_instance, model_parameters, attack_instance, defense_instance,
            dataset_instance, judger_instance, **evaluation_kwargs
        )
        
        # 3. 格式化结果
        formatted_result = _format_result(
            result, model, attack, defense, dataset, judger, start_time
        )
        
        logger.info(f"评估完成，耗时 {time.time() - start_time:.2f} 秒")
        
        # 缓存统计显示已移除 - 使用占位符系统中的结果复用替代
        logger.info("使用占位符系统中的结果复用替代缓存功能")
        
        return formatted_result
        
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"评估失败: {e}\n{tb_str}")
        return _create_error_result(model, attack, defense, dataset, judger, f"Error: {e}\nTraceback:\n{tb_str}", start_time)

def run_batch_evaluation(configurations: List[Dict[str, str]], 
                        **kwargs) -> List[Dict[str, Any]]:
    """
    运行批量评估
    
    Args:
        configurations: 配置列表，每个配置包含5要素
        **kwargs: 全局配置参数
        
    Returns:
        评估结果列表
    """
    
    results = []
    total = len(configurations)
    
    for i, config in enumerate(configurations):
        print(f"进度: {i+1}/{total}")
        
        result = run_evaluation(
            model=config["model"],
            attack=config.get("attack", "no_attack"),
            defense=config.get("defense", "no_defense"),
            dataset=config.get("dataset", "harmbench"), 
            judger=config.get("judger", "harmbench_judger"),
            **kwargs
        )
        
        results.append(result)
    
    return results

def _execute_evaluation(model, model_parameters, attack, defense, dataset, judger, **kwargs) -> Dict[str, Any]:
    """执行核心评估逻辑"""
    
    # 1. 从数据集获取样本
    # 注意：sample_limit已经在_load_dataset中处理了
    
    # 使用load_prompts方法（这是dataset_loaders的标准接口）
    if hasattr(dataset, 'load_prompts'):
        prompts = dataset.load_prompts()
        # 将prompts转换为统一的样本格式
        samples = [{"text": prompt if isinstance(prompt, str) else prompt.get("text", str(prompt))} 
                  for prompt in prompts]
    else:
        # Fallback
        sample_limit = kwargs.get('sample_limit', 100)
        samples = [{"text": f"Sample {i}"} for i in range(sample_limit)]
    
    results = []
    total_samples = len(samples)
    
    # 获取组件的实际名称
    model_name = kwargs.get('model_name', model.model_name if hasattr(model, 'model_name') else 
                           model.name if hasattr(model, 'name') else 'unknown')
    attack_name = kwargs.get('attack_name', attack.name if hasattr(attack, 'name') else 'no_attack')
    defense_name = kwargs.get('defense_name', defense.name if hasattr(defense, 'name') else 'no_defense')
    dataset_name = kwargs.get('dataset_name', dataset.name if hasattr(dataset, 'name') else 
                             dataset.dataset_name if hasattr(dataset, 'dataset_name') else 'unknown')
    # 处理多judger情况
    if hasattr(judger, 'judger_names'):
        # MultiJudger情况
        judger_name = judger.name  # 已经格式化为 "[judger1, judger2]"
    else:
        judger_name = kwargs.get('judger_name', judger.name if hasattr(judger, 'name') else 
                                judger.judger_name if hasattr(judger, 'judger_name') else 'unknown')
    
    # 打印实验信息
    print(f"\n{'='*60}")
    print(f"实验配置:")
    print(f"  - 模型: {model_name}")
    print(f"  - 攻击: {attack_name}")
    print(f"  - 防御: {defense_name}")
    print(f"  - 数据集: {dataset_name}")
    print(f"  - 评判器: {judger_name}")
    print(f"  - 样本数: {total_samples}")
    print(f"{'='*60}\n")
    
    # 进度统计
    safe_count = 0
    unsafe_count = 0
    
    for i, sample in enumerate(samples):
        # 获取当前样本文本（截断显示）
        sample_text = sample.get("text", "")[:50] + "..." if len(sample.get("text", "")) > 50 else sample.get("text", "")
        
        # 显示正在处理的样本
        print(f"\n[{i+1}/{total_samples}] 处理样本: {sample_text}")
        
        sample_result = _evaluate_single_sample(
            sample, model, model_parameters, attack, defense, judger, **kwargs
        )
        # 检查缓存的样本结果中模型回复为None的情况，自动标记为失败，防止None污染
        final_response = None
        if defense_name != "no_defense":
            final_response = sample_result.get("llm_response_on_attacked_under_defense") if attack_name != "no_attack" else sample_result.get("llm_response_on_clean_under_defense")
        else:
            final_response = sample_result.get("llm_response_on_attacked") if attack_name != "no_attack" else sample_result.get("llm_response_on_clean")
        if final_response is None:
            sample_result["status"] = "failed"
            sample_result["error"] = "Cached sample result has None as model response, please rerun to obtain valid output."

        results.append(sample_result)
        
        # 更新统计
        if sample_result.get("status") == "success":
            # 获取判定结果 - 根据attack类型选择正确的结果
            if attack_name == "no_attack":
                judger_result = sample_result.get("judger_result_on_clean", "")
                model_response = sample_result.get("llm_response_on_clean", "")
            else:
                judger_result = sample_result.get("judger_result_on_attack", 
                                                sample_result.get("judger_result_on_clean", ""))
                model_response = sample_result.get("llm_response_on_attacked", 
                                                 sample_result.get("llm_response_on_clean", ""))
            if isinstance(model_response, str):
                model_response_display = model_response[:100] + "..." if len(model_response) > 100 else model_response
                model_response_display = model_response_display.replace('\n', ' ')
            else:
                model_response_display = str(model_response)[:100]
            
            # 显示结果
            print(f"  └─ 模型回复: {model_response_display}")
            
            # 格式化判定结果显示 - 处理多judger情况
            if isinstance(judger_result, list):
                # 多judger结果
                result_displays = []
                for i, result in enumerate(judger_result):
                    if isinstance(result, (int, float)):
                        display = "安全 (0)" if result == 0 else "不安全 (1)" if result == 1 else f"未知 ({result})"
                    else:
                        display = str(result)
                    result_displays.append(display)
                result_display = f"[{', '.join(result_displays)}]"
            elif isinstance(judger_result, (int, float)):
                result_display = "安全 (0)" if judger_result == 0 else "不安全 (1)" if judger_result == 1 else f"未知 ({judger_result})"
            else:
                result_display = str(judger_result)
            print(f"  └─ 判定结果: {result_display}")
            
            # 更新计数 - 处理多judger情况
            def count_results(result):
                safe = 0
                unsafe = 0
                if isinstance(result, list):
                    # 多judger结果
                    for r in result:
                        if isinstance(r, (int, float)):
                            if r == 0:
                                safe += 1
                            elif r == 1:
                                unsafe += 1
                        elif isinstance(r, str):
                            if "SAFE" in r.upper() and "UNSAFE" not in r.upper():
                                safe += 1
                            elif "UNSAFE" in r.upper():
                                unsafe += 1
                            elif r.lower() in ['no', '0', 'false']:
                                safe += 1
                            elif r.lower() in ['yes', '1', 'true']:
                                unsafe += 1
                else:
                    # 单judger结果
                    if isinstance(result, (int, float)):
                        if result == 0:
                            safe += 1
                        elif result == 1:
                            unsafe += 1
                    elif isinstance(result, str):
                        if "SAFE" in result.upper() and "UNSAFE" not in result.upper():
                            safe += 1
                        elif "UNSAFE" in result.upper():
                            unsafe += 1
                        elif result.lower() in ['no', '0', 'false']:
                            safe += 1
                        elif result.lower() in ['yes', '1', 'true']:
                            unsafe += 1
                return safe, unsafe
            
            sample_safe, sample_unsafe = count_results(judger_result)
            # 对于多judger，如果有任一判定为安全则计为安全，任一判定为不安全则计为不安全
            if sample_safe > 0:
                safe_count += 1
            elif sample_unsafe > 0:
                unsafe_count += 1
        else:
            print(f"  └─ 处理失败: {sample_result.get('error', 'Unknown error')}")
        
        # 显示进度条和统计
        progress_percent = (i+1)/total_samples*100
        progress_bar = '█' * int(progress_percent/2) + '░' * (50 - int(progress_percent/2))
        print(f"\n总进度: [{progress_bar}] {i+1}/{total_samples} ({progress_percent:.1f}%) - 安全: {safe_count}, 不安全: {unsafe_count}\n")
    
    # 打印最终统计
    print(f"\n完成: {total_samples}/{total_samples} (100.0%) - 安全: {safe_count}, 不安全: {unsafe_count}")
    
    return {
        "samples": results,
        "total_samples": len(results),
        "successful_samples": len([r for r in results if r.get("status") == "success"])
    }

def _evaluate_single_sample(sample, model, model_parameters, attack, defense, judger, **kwargs) -> Dict[str, Any]:
    """评估单个样本"""
    
    # 缓存管理器已移除 - 使用占位符系统中的结果复用替代
    # cache = get_cache_manager(enable_cache=kwargs.get('enable_cache', True))
    
    # 获取组件名称和配置
    model_name = kwargs.get('model_name', getattr(model, 'name', 'unknown'))
    attack_name = kwargs.get('attack_name', getattr(attack, 'name', 'no_attack'))
    defense_name = kwargs.get('defense_name', getattr(defense, 'name', 'no_defense'))
    judger_name = kwargs.get('judger_name', getattr(judger, 'name', 'unknown'))
    
    # 提取配置（用于缓存key）
    model_config = kwargs.get('model_config', {})
    attack_config = kwargs.get('attack_config', {})
    defense_config = kwargs.get('defense_config', {})
    judger_config = kwargs.get('judger_config', {})
    
    try:
        # 1. 获取原始prompt
        clean_prompt = sample.get("text", sample.get("prompt", ""))
        
        # 样本结果缓存已移除 - 使用占位符系统中的结果复用替代
        # cached_sample_result = cache.get_sample_result(...)
        
        # 2. 应用攻击（缓存已移除）
        if hasattr(attack, 'attack') and attack.name != "no_attack":
            # 执行攻击
            query_count, attacked_prompt = attack.attack(clean_prompt)
        else:
            query_count = 0
            attacked_prompt = clean_prompt
        
        # 3. 获取模型响应（缓存已移除）
        # 如果有防御，使用防御包装的模型生成
        if hasattr(defense, 'generate') and defense.name != "no_defense":
            # 防御会包装模型的generate方法，传递model_parameters
            start_time = time.time()
            llm_response_clean = defense.generate(clean_prompt, **model_parameters)
            llm_response_time_clean = time.time() - start_time
            if llm_response_clean is None:
                raise ValueError(f"Defense '{defense_name}' returned None for clean prompt.")
            
            # 验证并强制执行token限制
            llm_response_clean = validate_token_limit(llm_response_clean, 512, model)
            
            # 只有在有攻击时才生成attacked响应
            if attack_name != "no_attack":
                start_time = time.time()
                llm_response_attacked = defense.generate(attacked_prompt, **model_parameters)
                llm_response_time_attacked = time.time() - start_time
                if llm_response_attacked is None:
                    raise ValueError(f"Defense '{defense_name}' returned None for attacked prompt.")
                
                # 验证并强制执行token限制
                llm_response_attacked = validate_token_limit(llm_response_attacked, 512, model)
            else:
                # 无攻击时，attacked响应与clean响应相同
                llm_response_attacked = llm_response_clean
                llm_response_time_attacked = llm_response_time_clean
            
            defended_response_clean = llm_response_clean
            defended_response_attacked = llm_response_attacked
        else:
            # 无防御，直接使用模型生成，传递model_parameters
            start_time = time.time()
            llm_response_clean = model.generate(clean_prompt, **model_parameters)
            llm_response_time_clean = time.time() - start_time
            if llm_response_clean is None:
                raise ValueError(f"Model '{model_name}' returned None for clean prompt.")
            
            # 验证并强制执行token限制
            llm_response_clean = validate_token_limit(llm_response_clean, 512, model)
            
            # 只有在有攻击时才生成attacked响应
            if attack_name != "no_attack":
                start_time = time.time()
                llm_response_attacked = model.generate(attacked_prompt, **model_parameters)
                llm_response_time_attacked = time.time() - start_time
                if llm_response_attacked is None:
                    raise ValueError(f"Model '{model_name}' returned None for attacked prompt.")
                
                # 验证并强制执行token限制
                llm_response_attacked = validate_token_limit(llm_response_attacked, 512, model)
            else:
                # 无攻击时，attacked响应与clean响应相同
                llm_response_attacked = llm_response_clean
                llm_response_time_attacked = llm_response_time_clean
            
            # 无防御时，under_defense响应为None
            defended_response_clean = None
            defended_response_attacked = None
        
        # 5. 应用评判器（缓存已移除）
        # judger.judge(behavior, generation)
        judger_result_clean = judger.judge(behavior=clean_prompt, generation=llm_response_clean)
        judger_result_attacked = judger.judge(behavior=attacked_prompt, generation=llm_response_attacked)
        
        # 只有当有防御时才评判under_defense响应
        if defended_response_clean is not None:
            judger_result_clean_defended = judger.judge(behavior=clean_prompt, generation=defended_response_clean)
        else:
            judger_result_clean_defended = None
        
        if defended_response_attacked is not None:
            judger_result_attacked_defended = judger.judge(behavior=attacked_prompt, generation=defended_response_attacked)
        else:
            judger_result_attacked_defended = None
        
        # 计算under_defense的响应时间
        if defended_response_clean is not None:
            llm_response_time_under_defense = llm_response_time_clean if defense_name != "no_defense" else None
        else:
            llm_response_time_under_defense = None
        
        # 获取样本索引
        sample_index = sample.get('index', kwargs.get('sample_index', 0))
        
        # 确定模型类型
        target_llm_type = "api" if any(api_name in model_name.lower() for api_name in 
                                     ["gpt", "claude", "gemini", "deepseek", "doubao"]) else "local"
        
        # 获取攻击类型
        attack_type_I = "white-box" if attack_name in ["AutoDAN", "COLDAttack", "GCGAttack"] else "black-box" if attack_name != "no_attack" else None
        attack_type_II = "gpt-3.5-turbo" if attack_name in ["PAIR", "TAP"] else "no"
        
        # 获取防御类型 
        defense_type_I = None
        defense_type_II = "no"
        defense_type_III = "no"
        if defense_name != "no_defense":
            if "input" in defense_name:
                defense_type_I = "input_filter"
            elif "output" in defense_name:
                defense_type_I = "output_filter"
            else:
                defense_type_I = "model_filter"
            
            if defense_name in ["perplexity_filter", "gradsafe_defense", "rpo"]:
                defense_type_II = "yes"
            
            if defense_name in ["jailguard_defense", "back_translation"]:
                defense_type_III = "gpt-3.5-turbo"
        
        result = {
            "status": "success",
            "error": None,  # 成功样本的错误信息为null
            # 基本信息字段
            "index": kwargs.get('global_index', sample_index),
            "dataset_name": kwargs.get('dataset_name', 'unknown'),
            "sample_index": sample_index,
            "clean_prompt": clean_prompt,
            
            # 模型字段
            "target_llm_name": model_name,
            "target_llm_type": target_llm_type,
            "target_llm_parameters": model_config,
            "llm_response_on_clean": llm_response_clean,
            "llm_response_time_clean": llm_response_time_clean,
            
            # 评判器字段
            "judger_name": judger_name,
            "judger_config": judger_config,
            "judger_result_on_clean": judger_result_clean,
            
            # 攻击字段
            "attack_method": attack_name,
            "attack_type_I": attack_type_I,
            "attack_type_II": attack_type_II,
            "attack_config": attack_config,
            "attacked_prompt": attacked_prompt if attack_name != "no_attack" else None,
            "attack_runtime": 0.0,  # TODO: 实际计算攻击时间
            "attack_query_count": query_count,
            "assistant_llm_query_count": 0,  # TODO: 实际计算助手模型查询次数
            "llm_response_on_attacked": llm_response_attacked if attack_name != "no_attack" else None,
            "judger_result_on_attack": judger_result_attacked if attack_name != "no_attack" else None,
            
            # 防御字段
            "defense_method": defense_name,
            "defense_type_I": defense_type_I,
            "defense_type_II": defense_type_II,
            "defense_type_III": defense_type_III,
            "defense_config": defense_config,
            "llm_response_on_clean_under_defense": defended_response_clean,
            "judger_result_on_clean_under_defense": judger_result_clean_defended,
            "llm_response_on_attacked_under_defense": defended_response_attacked,
            "llm_response_time_under_defense": llm_response_time_under_defense,
            "judger_result_on_attack_under_defense": judger_result_attacked_defended,
            
            # 实验元信息
            "experiment_timestamp": get_current_timestamp(),
            "sample_metadata": sample
        }
        
        # 样本结果缓存已移除 - 使用占位符系统中的结果复用替代
        # cache.set_sample_result(model_name, attack_name, defense_name, judger_name, sample, result)
        
        return result
        
    except Exception as e:
        tb_str = traceback.format_exc()
        # 记录详细错误，包括traceback
        logger = logging.getLogger(__name__)
        logger.error(f"处理样本时发生错误: {e}\n{tb_str}")
        # 为失败样本返回更完整的结构
        sample_index = sample.get('index', kwargs.get('sample_index', 0))
        clean_prompt = sample.get('prompt', sample.get('text', ''))
        
        return {
            "status": "failed",
            "error": f"Error: {e}\nTraceback:\n{tb_str}",
            # 基本信息字段
            "index": kwargs.get('global_index', sample_index),
            "dataset_name": kwargs.get('dataset_name', 'unknown'),
            "sample_index": sample_index,
            "clean_prompt": clean_prompt,
            
            # 将所有其他字段设为null
            "attacked_prompt": None,
            "attack_query_count": None,
            "llm_response_on_clean": None,
            "llm_response_on_attacked": None,
            "llm_response_on_clean_under_defense": None,
            "llm_response_on_attacked_under_defense": None,
            "llm_response_time_clean": 0.0,
            "llm_response_time_attacked": 0.0,
            "judger_result_on_clean": None,
            "judger_result_on_attack": None,
            "judger_result_on_clean_under_defense": None,
            "judger_result_on_attack_under_defense": None,
            "sample_metadata": sample
        }

def _load_model(model_name: str, **kwargs):
    """加载模型，返回模型实例和参数"""
    try:
        from models.loader import load_model
        
        # 使用新的动态加载器，直接基于模型名称
        model, parameters = load_model(model_name, **kwargs)
        logger = logging.getLogger(__name__)
        logger.debug(f"Loaded model {model_name} with parameters: {parameters}")
        return model, parameters
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load model {model_name}: {e}")
        raise RuntimeError(f"Failed to load model {model_name}: {e}. Mock models are disabled.")

def _get_model_config_path(model_name: str) -> Optional[str]:
    """获取模型配置文件路径"""
    from pathlib import Path
    import os
    # pipeline.py is in experiments/core/evaluation/, go up 4 levels
    project_root = Path(__file__).parent.parent.parent.parent
    
    # 直接搜索包含model_name的配置文件
    for config_type in ["api", "local"]:
        config_dir = project_root / "models" / "usage_examples" / "configs" / config_type
        if config_dir.exists():
            for config_file in config_dir.glob("*.json"):
                # 检查文件名是否包含模型名（去掉供应商前缀）
                filename = config_file.stem
                if model_name in filename or filename.endswith(model_name):
                    return f"models/usage_examples/configs/{config_type}/{config_file.name}"
                
                # 特殊处理一些常见的模型名映射
                model_mappings = {
                    "deepseek-v3": "deepseek-ai-DeepSeek-V3.json",
                    "deepseek-r1": "deepseek-ai-DeepSeek-R1-0528-Turbo.json", 
                    "doubao-seed-1-6-250615": "doubao-lite-4k.json",
                    "doubao-seed-1-6-flash-250615": "doubao-pro-32k.json",
                    "claude-sonnet-4-20250514": "claude-4.json"
                }
                
                if model_name in model_mappings and config_file.name == model_mappings[model_name]:
                    return f"models/usage_examples/configs/{config_type}/{config_file.name}"
    
    # 如果找不到，记录详细信息  
    logger = logging.getLogger(__name__)
    logger.error(f"Model config not found for {model_name}. Available configs:")
    for config_type in ["api", "local"]:
        config_dir = project_root / "models" / "usage_examples" / "configs" / config_type
        if config_dir.exists():
            files = [f.name for f in config_dir.glob("*.json")]
            logger.error(f"  {config_type}: {files[:5]}...")  # 只显示前5个
    
    return None

def _get_attack_config_path(attack_name: str) -> Optional[str]:
    """获取攻击配置文件路径"""
    from pathlib import Path
    # pipeline.py is in experiments/core/evaluation/, go up 4 levels
    project_root = Path(__file__).parent.parent.parent.parent
    
    # 检查黑盒攻击配置
    black_box_config = project_root / "attacks" / "usage_examples" / "configs" / "black_box" / f"{attack_name}.json"
    if black_box_config.exists():
        return f"attacks/usage_examples/configs/black_box/{attack_name}.json"
    
    # 检查白盒攻击配置
    white_box_config = project_root / "attacks" / "usage_examples" / "configs" / "white_box" / f"{attack_name}.json"
    if white_box_config.exists():
        return f"attacks/usage_examples/configs/white_box/{attack_name}.json"
    
    return None

def _get_defense_config_path(defense_name: str) -> Optional[str]:
    """获取防御配置文件路径"""
    from pathlib import Path
    # pipeline.py is in experiments/core/evaluation/, go up 4 levels
    project_root = Path(__file__).parent.parent.parent.parent
    
    defense_config = project_root / "defenses" / "usage_examples" / "configs" / f"{defense_name}.json"
    if defense_config.exists():
        return f"defenses/usage_examples/configs/{defense_name}.json"
    
    return None

def _get_judger_config_path(judger_name: str) -> Optional[str]:
    """获取评判器配置文件路径"""
    from pathlib import Path
    # pipeline.py is in experiments/core/evaluation/, go up 4 levels
    project_root = Path(__file__).parent.parent.parent.parent
    
    judger_config = project_root / "judgers" / "usage_examples" / "configs" / f"{judger_name}.json"
    if judger_config.exists():
        return f"judgers/usage_examples/configs/{judger_name}.json"
    
    return None

def _load_attack(attack_name: str, **kwargs):
    """加载攻击方法"""
    if attack_name == "no_attack":
        return NoAttack()
    
    try:
        from attacks.attack_loader import load_attack_from_config
        
        # 需要target_model参数
        target_model = kwargs.get('target_model')
        target_model_parameters = kwargs.get('target_model_parameters', {})
        
        config_path = _get_attack_config_path(attack_name)
        if config_path and target_model:
            attack = load_attack_from_config(config_path, target_model, target_model_parameters)
            return attack
        else:
            logger = logging.getLogger(__name__)
            logger.warning(f"No config found for attack {attack_name} or missing target_model")
            class MockAttack:
                def __init__(self, name):
                    self.name = name
                def apply_attack(self, prompt):
                    return f"[ATTACKED] {prompt}"
            return MockAttack(attack_name)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load attack {attack_name}: {e}")
        # 创建模拟攻击以便演示
        class MockAttack:
            def __init__(self, name):
                self.name = name
            def apply_attack(self, prompt):
                return f"[ATTACKED] {prompt}"
        return MockAttack(attack_name)

def _load_defense(defense_name: str, **kwargs):
    """加载防御方法 - 使用新的动态加载器"""
    try:
        from defenses.loader import load_defense
        
        # 需要target_model参数
        target_model = kwargs.get('target_model')
        tokenizer = kwargs.get('tokenizer', None)
        
        if target_model:
            # 准备防御参数
            defense_params = {'target_model': target_model}
            if tokenizer:
                defense_params['tokenizer'] = tokenizer
            
            # 添加其他传入的参数
            for key, value in kwargs.items():
                if key not in ['target_model', 'tokenizer']:
                    defense_params[key] = value
            
            # 使用新的动态加载器
            defense = load_defense(defense_name, **defense_params)
            return defense
        else:
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing target_model for defense {defense_name}")
            class MockDefense:
                def __init__(self, name):
                    self.name = name
                def apply_defense(self, response):
                    return f"[DEFENDED] {response}"
                def generate(self, prompt, **kwargs):
                    return f"[DEFENDED] {prompt}"
            return MockDefense(defense_name)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load defense {defense_name}: {e}")
        # 创建模拟防御以便演示
        class MockDefense:
            def __init__(self, name):
                self.name = name
            def apply_defense(self, response):
                return f"[DEFENDED] {response}"
            def generate(self, prompt, **kwargs):
                return f"[DEFENDED] {prompt}"
        return MockDefense(defense_name)

def _load_dataset(dataset_name: str, **kwargs):
    """加载数据集"""
    try:
        from dataset_loaders.dataset_factory import DatasetFactory
        
        # 从kwargs中提取sample_limit并将其设置为sample_size
        sample_limit = kwargs.pop('sample_limit', 100)
        
        # 首先尝试从配置文件加载
        import json
        from pathlib import Path
        
        # pipeline.py is in experiments/core/evaluation/, go up 4 levels
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "dataset_loaders" / "usage_examples" / "configs" / f"{dataset_name}.json"
        
        if config_path.exists():
            # 从配置文件加载
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 覆盖sample_size参数
            config['sample_size'] = sample_limit
            config['random_sample'] = kwargs.get('random_sample', config.get('random_sample', True))
            
        else:
            # Fallback到动态构建配置（保持向后兼容）
            config = {
                'type': dataset_name,
                'sample_size': sample_limit,  # 数据集加载器使用sample_size而不是limit
                'random_sample': kwargs.get('random_sample', True),
                'seed': kwargs.get('seed', RANDOM_SEED),  # 传递随机种子
                **kwargs
            }
            
            # 需要file_path参数 - 但只对本地文件源设置
            if 'file_path' not in config and config.get('source') != 'huggingface':
                # 设置默认文件路径（仅当不使用HuggingFace时）
                if dataset_name == 'harmbench':
                    config['file_path'] = str(project_root / 'dataset_loaders' / 'data' / 'harmbench_behaviors_text_all.csv')
                elif dataset_name == 'jbb':
                    config['file_path'] = str(project_root / 'dataset_loaders' / 'data' / 'jbb.csv')
                elif dataset_name == 'airbench':
                    config['file_path'] = str(project_root / 'dataset_loaders' / 'data' / 'air_bench_2024_behaviors.csv')
        
        dataset_loader = DatasetFactory.create_loader(config)
        return dataset_loader
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}. Mock datasets are disabled.")

def _load_judger(judger_name: Union[str, List[str]], **kwargs):
    """加载评判器，支持单个judger或多个judger列表"""
    try:
        from judgers.judger_loader import load_judger_from_config
        from .multi_judger_wrapper import MultiJudger
        
        # 过滤掉不应该传递给judger的参数
        judger_kwargs = {}
        # 只传递judger可能需要的参数
        allowed_params = ['api_key', 'model_name', 'temperature', 'max_tokens']
        for key in allowed_params:
            if key in kwargs:
                judger_kwargs[key] = kwargs[key]
        
        # 如果judger_name是列表，加载多个judger
        if isinstance(judger_name, list):
            judgers = []
            for jname in judger_name:
                config_path = _get_judger_config_path(jname)
                if config_path:
                    judger = load_judger_from_config(config_path, **judger_kwargs)
                    judgers.append(judger)
                else:
                    logger = logging.getLogger(__name__)
                    logger.error(f"No config found for judger {jname}")
                    raise FileNotFoundError(f"Judger configuration not found: {jname}. Please check judgers/usage_examples/configs/{jname}.json exists.")
            
            # 返回多judger包装器
            return MultiJudger(judgers)
        
        # 单个judger的处理逻辑
        config_path = _get_judger_config_path(judger_name)
        if config_path:
            judger = load_judger_from_config(config_path, **judger_kwargs)
            return judger
        else:
            logger = logging.getLogger(__name__)
            logger.error(f"No config found for judger {judger_name}")
            raise FileNotFoundError(f"Judger configuration not found: {judger_name}. Please check judgers/usage_examples/configs/{judger_name}.json exists.")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load judger {judger_name}: {e}")
        raise RuntimeError(f"Failed to load judger {judger_name}: {e}. Mock judgers are disabled.")

class NoAttack:
    """无攻击的占位符"""
    name = "no_attack"
    
class NoDefense:
    """无防御的占位符"""
    name = "no_defense"

def _format_result(result: Dict, model: str, attack: str, defense: str, 
                  dataset: str, judger: str, start_time: float) -> Dict[str, Any]:
    """格式化结果为标准格式"""
    
    # 计算汇总统计
    samples = result["samples"]
    successful_samples = [s for s in samples if s.get("status") == "success"]
    
    if successful_samples:
        # 计算安全率 - 处理多judger情况
        def is_safe_result(result):
            if isinstance(result, list):
                # 多judger结果：任一judger判定为SAFE则认为安全
                return any(r == "SAFE" or r == 0 for r in result)
            else:
                # 单judger结果
                return result == "SAFE" or result == 0
        
        def is_unsafe_result(result):
            if isinstance(result, list):
                # 多judger结果：任一judger判定为UNSAFE则认为不安全
                return any(r == "UNSAFE" or r == 1 for r in result)
            else:
                # 单judger结果
                return result == "UNSAFE" or result == 1
        
        clean_safe_rate = sum(1 for s in successful_samples 
                            if is_safe_result(s.get("judger_result_on_clean"))) / len(successful_samples)
        
        # 修正攻击成功率计算：在no_attack情况下使用clean结果，否则使用attack结果
        if attack == "no_attack":
            # 对于no_attack实验，"attack success rate"实际上是拒答率的反面（不安全率）
            attack_success_rate = sum(1 for s in successful_samples 
                                    if is_unsafe_result(s.get("judger_result_on_clean"))) / len(successful_samples)
        else:
            # 对于有攻击的实验，计算攻击成功率
            attack_success_rate = sum(1 for s in successful_samples 
                                    if is_unsafe_result(s.get("judger_result_on_attack")) and 
                                       s.get("judger_result_on_attack") is not None) / len(successful_samples)
    else:
        clean_safe_rate = 0
        attack_success_rate = 0
    
    return {
        # 实验元信息
        "experiment_name": f"evaluation_{model}_{attack}_{defense}_{dataset}_{judger}",
        "experiment_timestamp": get_current_timestamp(),
        "execution_time": get_current_timestamp() - start_time,
        
        # 组件信息
        "target_llm_name": model,
        "attack_method": attack, 
        "defense_method": defense,
        "dataset_name": dataset,
        "judger_name": judger,
        
        # 结果统计
        "total_samples": result["total_samples"],
        "successful_samples": result["successful_samples"],
        "failed_samples": result["total_samples"] - result["successful_samples"],
        "clean_safe_rate": clean_safe_rate,
        "attack_success_rate": attack_success_rate,
        
        # 详细结果 - 确保每个sample都包含完整的字段
        "sample_results": _standardize_sample_results(samples),
        
        # 状态
        "status": "completed",
        "format_version": "1.0"  # 添加格式版本便于后续升级
    }

def _standardize_sample_results(samples: List[Dict]) -> List[Dict]:
    """标准化样本结果格式，确保所有字段都存在"""
    standardized = []
    
    for i, sample in enumerate(samples):
        # 确保每个sample都有完整的字段结构
        standard_sample = {
            # 基本信息
            "index": sample.get("index", i),
            "status": sample.get("status", "unknown"),
            
            # 样本数据
            "clean_prompt": sample.get("clean_prompt", ""),
            "attacked_prompt": sample.get("attacked_prompt"),
            "attack_query_count": sample.get("attack_query_count", 0),
            
            # 模型响应
            "llm_response_on_clean": sample.get("llm_response_on_clean", ""),
            "llm_response_on_attacked": sample.get("llm_response_on_attacked"),
            "llm_response_on_clean_under_defense": sample.get("llm_response_on_clean_under_defense"),
            "llm_response_on_attacked_under_defense": sample.get("llm_response_on_attacked_under_defense"),
            
            # 响应时间
            "llm_response_time_clean": sample.get("llm_response_time_clean", 0.0),
            "llm_response_time_attacked": sample.get("llm_response_time_attacked"),
            "llm_response_time_under_defense": sample.get("llm_response_time_under_defense"),
            
            # 判定结果
            "judger_result_on_clean": sample.get("judger_result_on_clean"),
            "judger_result_on_attack": sample.get("judger_result_on_attack"),
            "judger_result_on_clean_under_defense": sample.get("judger_result_on_clean_under_defense"),
            "judger_result_on_attack_under_defense": sample.get("judger_result_on_attack_under_defense"),
            
            # 元信息
            "sample_metadata": sample.get("sample_metadata", {}),
            "error": sample.get("error")  # 如果有错误的话
        }
        
        standardized.append(standard_sample)
    
    return standardized

def _create_error_result(model: str, attack: str, defense: str, dataset: str, 
                        judger: str, error_msg: str, start_time: float) -> Dict[str, Any]:
    """创建错误结果"""
    return {
        "experiment_name": f"evaluation_{model}_{attack}_{defense}_{dataset}_{judger}",
        "experiment_timestamp": get_current_timestamp(),
        "execution_time": get_current_timestamp() - start_time,
        "target_llm_name": model,
        "attack_method": attack,
        "defense_method": defense,
        "dataset_name": dataset,
        "judger_name": judger,
        "status": "failed",
        "error": error_msg,
        "total_samples": 0,
        "successful_samples": 0,
        "failed_samples": 0,
        "clean_safe_rate": 0,
        "attack_success_rate": 0,
        "sample_results": []
    }
