# models/loader.py

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from .api_models.api_gpt_model import APIGPTModel
from .api_models.api_claude_model import APIClaudeModel
from .api_models.api_gemini_model import APIGeminiModel
from .api_models.api_deepinfra_model import APIDeepInfraModel
from .api_models.api_doubao_model import APIDoubaoModel
from .api_keys import OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPINFRA_API_KEY, DOUBAO_API_KEY

try:
    # Local stack (torch/transformers) is optional for API-only runs.
    from .local_models.local_huggingface_model import LocalHuggingFaceModel
    _LOCAL_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on environment packages
    LocalHuggingFaceModel = None
    _LOCAL_IMPORT_ERROR = e

logger = logging.getLogger(__name__)

# 模型类型检测规则
MODEL_TYPE_RULES = {
    'api_claude': {
        'patterns': ['claude'],
        'class': APIClaudeModel,
        'api_key': ANTHROPIC_API_KEY,
        'api_key_name': 'ANTHROPIC_API_KEY'
    },
    'api_gpt': {
        'patterns': ['gpt', 'o1-'],
        'class': APIGPTModel,
        'api_key': OPENAI_API_KEY,
        'api_key_name': 'OPENAI_API_KEY'
    },
    'api_gemini': {
        'patterns': ['gemini'],
        'class': APIGeminiModel,
        'api_key': GEMINI_API_KEY,
        'api_key_name': 'GEMINI_API_KEY'
    },
    'api_doubao': {
        'patterns': ['doubao', 'deepseek'],
        'class': APIDoubaoModel,
        'api_key': DOUBAO_API_KEY,
        'api_key_name': 'DOUBAO_API_KEY'
    },
    'api_deepinfra': {
        'patterns': [],  # Default for other API models
        'class': APIDeepInfraModel,
        'api_key': DEEPINFRA_API_KEY,
        'api_key_name': 'DEEPINFRA_API_KEY'
    }
}

def _detect_model_type(model_name: str) -> str:
    """根据模型名称检测模型类型"""
    model_name_lower = model_name.lower()
    
    # 检查API模型模式（优先）
    for model_type, rules in MODEL_TYPE_RULES.items():
        if any(pattern in model_name_lower for pattern in rules['patterns']):
            return model_type
    
    # 检查是否为本地模型（常见的本地模型标识）
    local_patterns = ['llama', 'qwen', 'phi', 'gemma', 'yi', 'mistral', 'internlm', 'meta-llama', 'microsoft']
    if any(pattern in model_name_lower for pattern in local_patterns):
        return 'local'
    
    # 默认为API模型（DeepInfra）
    return 'api_deepinfra'

def _get_config_path(model_name: str) -> Optional[Path]:
    """根据模型名称获取配置文件路径"""
    project_root = Path(__file__).parent.parent
    
    # 首先检测模型类型
    model_type = _detect_model_type(model_name)
    
    if model_type == 'local':
        config_dir = project_root / "models" / "usage_examples" / "configs" / "local"
    else:
        config_dir = project_root / "models" / "usage_examples" / "configs" / "api"
    
    # 尝试多种文件名模式
    possible_names = [
        f"{model_name}.json",
        f"{model_name.replace('/', '_')}.json",
        f"{model_name.replace('_', '-')}.json",  # 下划线转连字符
        f"{model_name.replace('-', '_')}.json"   # 连字符转下划线
    ]
    
    for name in possible_names:
        config_path = config_dir / name
        if config_path.exists():
            return config_path
    
    logger.warning(f"No config file found for model {model_name}. Tried: {possible_names}")
    return None

def _validate_api_key(api_key: Optional[str], model_type: str, model_name: str) -> str:
    """验证API密钥"""
    if not api_key or api_key.startswith("your-"):
        api_key_name = MODEL_TYPE_RULES.get(model_type, {}).get('api_key_name', 'API_KEY')
        raise ValueError(f"API key is required for {model_name}. Please set {api_key_name} in environment variables or config file.")
    return api_key

def load_model(model_name: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    动态加载模型实例
    
    Args:
        model_name: 模型名称，配置文件名即模型名
        **kwargs: 额外参数，会覆盖配置文件中的参数
    
    Returns:
        Tuple[model_instance, parameters]: 模型实例和生成参数
    
    Examples:
        # 加载API模型
        model, params = load_model("gpt-4o")
        model, params = load_model("claude-3-5-sonnet-latest")
        
        # 加载本地模型
        model, params = load_model("meta-llama_Llama-3.1-8B-Instruct")
        
        # 覆盖参数
        model, params = load_model("gpt-4o", temperature=0.5, max_tokens=1024)
    """
    
    # 获取配置文件路径
    config_path = _get_config_path(model_name)
    
    if config_path and config_path.exists():
        # 从配置文件加载
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_type = config.get('model_type')
        config_model_name = config.get('model_name', model_name)
        init_parameters = config.get('init_parameters', {})
        parameters = config.get('parameters', {})
        
        # 用户参数覆盖配置文件参数
        parameters.update(kwargs)
        
        logger.info(f"Loading model {model_name} from config: {config_path}")
    else:
        # 基于约定的默认配置
        model_type = _detect_model_type(model_name)
        config_model_name = model_name
        init_parameters = {}
        parameters = {
            'temperature': 0.0,
            'max_tokens' if model_type != 'local' else 'max_new_tokens': 512,
            **kwargs
        }
        
        logger.info(f"Loading model {model_name} with default config (type: {model_type})")
    
    # 根据模型类型创建模型实例
    if model_type == 'local':
        # 本地模型
        if LocalHuggingFaceModel is None:
            raise ImportError(
                "Local model dependencies are missing. Install local stack (e.g., torch/transformers) "
                f"before loading local models. Original error: {_LOCAL_IMPORT_ERROR}"
            )
        model = LocalHuggingFaceModel(model_name_or_path=config_model_name, **init_parameters)
    else:
        # API模型
        model_rules = MODEL_TYPE_RULES.get(model_type, MODEL_TYPE_RULES['api_deepinfra'])
        model_class = model_rules['class']
        
        # 获取API密钥
        api_key = config.get('api_key') if config_path else None
        api_key = api_key or model_rules['api_key']
        api_key = _validate_api_key(api_key, model_type, model_name)
        
        # 创建模型实例
        if model_type == 'api_doubao':
            endpoint_url = init_parameters.get('endpoint_url')
            model = model_class(api_key=api_key, model_name=config_model_name, endpoint_url=endpoint_url)
        else:
            model = model_class(api_key=api_key, model_name=config_model_name)
    
    # 验证并记录token限制参数
    _validate_and_log_token_parameters(model_type, model_name, parameters)
    
    return model, parameters

def _validate_and_log_token_parameters(model_type: str, model_name: str, parameters: Dict[str, Any]):
    """验证并记录token限制参数"""
    token_limit_param = None
    token_limit_value = None
    
    if model_type.startswith('api') or model_type != 'local':
        # API模型应该使用max_tokens
        if 'max_tokens' in parameters:
            token_limit_param = 'max_tokens'
            token_limit_value = parameters['max_tokens']
            logger.info(f"✅ API模型 {model_name} 配置了 max_tokens: {token_limit_value}")
        elif 'max_new_tokens' in parameters:
            logger.warning(f"⚠️ API模型 {model_name} 使用了 max_new_tokens({parameters['max_new_tokens']})，建议使用 max_tokens")
            token_limit_param = 'max_new_tokens'
            token_limit_value = parameters['max_new_tokens']
        else:
            logger.warning(f"⚠️ API模型 {model_name} 未配置token限制参数 (max_tokens)")
    
    elif model_type == 'local':
        # 本地模型应该使用max_new_tokens
        if 'max_new_tokens' in parameters:
            token_limit_param = 'max_new_tokens'
            token_limit_value = parameters['max_new_tokens']
            logger.info(f"✅ 本地模型 {model_name} 配置了 max_new_tokens: {token_limit_value}")
        elif 'max_tokens' in parameters:
            logger.warning(f"⚠️ 本地模型 {model_name} 使用了 max_tokens({parameters['max_tokens']})，建议使用 max_new_tokens")
            token_limit_param = 'max_tokens'
            token_limit_value = parameters['max_tokens']
        else:
            logger.warning(f"⚠️ 本地模型 {model_name} 未配置token限制参数 (max_new_tokens)")
    
    # 验证token限制值的合理性
    if token_limit_value is not None:
        if not isinstance(token_limit_value, int) or token_limit_value <= 0:
            logger.error(f"❌ 模型 {model_name} 的 {token_limit_param} 值无效: {token_limit_value}")
        elif token_limit_value > 4096:
            logger.warning(f"⚠️ 模型 {model_name} 的 {token_limit_param} 值较大: {token_limit_value}")
        elif token_limit_value < 10:
            logger.warning(f"⚠️ 模型 {model_name} 的 {token_limit_param} 值较小: {token_limit_value}")
    
    # 记录所有generation参数以便调试
    generation_params = {k: v for k, v in parameters.items() 
                        if k in ['temperature', 'max_tokens', 'max_new_tokens', 'top_p', 'top_k', 'do_sample']}
    if generation_params:
        logger.debug(f"📋 模型 {model_name} 的生成参数: {generation_params}")
    else:
        logger.debug(f"📋 模型 {model_name} 未配置生成参数")

def list_available_models() -> Dict[str, list]:
    """列出所有可用的模型"""
    project_root = Path(__file__).parent.parent
    
    models = {'api': [], 'local': []}
    
    # 扫描API模型配置
    api_config_dir = project_root / "models" / "usage_examples" / "configs" / "api"
    if api_config_dir.exists():
        for config_file in api_config_dir.glob("*.json"):
            model_name = config_file.stem
            models['api'].append(model_name)
    
    # 扫描本地模型配置
    local_config_dir = project_root / "models" / "usage_examples" / "configs" / "local"
    if local_config_dir.exists():
        for config_file in local_config_dir.glob("*.json"):
            model_name = config_file.stem
            models['local'].append(model_name)
    
    return models

def get_model_config_path(model_name: str) -> Optional[str]:
    """获取模型配置文件路径
    
    Args:
        model_name: 模型名称
        
    Returns:
        配置文件路径或None（如果不存在）
    """
    config_path = _get_config_path(model_name)
    return str(config_path) if config_path else None

# 向后兼容
load_model_from_config = load_model
