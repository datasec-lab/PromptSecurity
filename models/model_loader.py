# models/model_loader.py

import json
import os
import logging
try:
    from .api_models.api_gpt_model import APIGPTModel
except Exception:  # pragma: no cover
    APIGPTModel = None
try:
    from .api_models.api_claude_model import APIClaudeModel
except Exception:  # pragma: no cover
    APIClaudeModel = None
try:
    from .api_models.api_gemini_model import APIGeminiModel
except Exception:  # pragma: no cover
    APIGeminiModel = None
try:
    from .api_models.api_deepinfra_model import APIDeepInfraModel  # Import Deep Infra model
except Exception:  # pragma: no cover
    APIDeepInfraModel = None
try:
    from .api_models.api_doubao_model import APIDoubaoModel  # Import Doubao model
except Exception:  # pragma: no cover
    APIDoubaoModel = None
try:
    from .local_models.local_huggingface_model import LocalHuggingFaceModel
except Exception:  # pragma: no cover
    LocalHuggingFaceModel = None
from .api_keys import OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPINFRA_API_KEY, DOUBAO_API_KEY

logger = logging.getLogger(__name__)

def load_model_from_config(config_path: str):
    # Get the project's root directory (assuming this script is located within the project structure)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Construct the full path to the configuration file
    full_config_path = os.path.join(project_root, config_path)

    # Load the configuration file
    with open(full_config_path, 'r') as file:
        config = json.load(file)

    model_type = config.get('model_type')
    model_name = config.get('model_name')
    init_parameters = config.get('init_parameters', {})
    parameters = config.get('parameters', {})

    if model_type == 'api' or model_type.startswith('api_'):
        # Handle specific API types and generic 'api' type
        if model_type == 'api_claude' or (model_type == 'api' and 'claude' in model_name.lower()):
            api_key = config.get('api_key') or ANTHROPIC_API_KEY
            if not api_key or api_key == "your-anthropic-api-key":
                raise ValueError("API key is required for Claude models. Please set it in the config file or as an environment variable.")
            if APIClaudeModel is None:
                raise ImportError("APICLaudeModel backend is unavailable. Install required dependencies (e.g., anthropic).")
            model = APIClaudeModel(api_key=api_key, model_name=model_name)
        elif model_type == 'api_gemini' or (model_type == 'api' and 'gemini' in model_name.lower()):
            api_key = config.get('api_key') or GEMINI_API_KEY
            if not api_key or api_key == "your-gemini-api-key":
                raise ValueError("API key is required for Google Gemini models. Please set it in the config file or as an environment variable.")
            if APIGeminiModel is None:
                raise ImportError("APIGeminiModel backend is unavailable. Install required dependencies (e.g., google-generativeai).")
            model = APIGeminiModel(api_key=api_key, model_name=model_name)
        elif model_type == 'api_gpt' or (model_type == 'api' and ('gpt' in model_name.lower() or 'o1-' in model_name.lower())):
            api_key = config.get('api_key') or OPENAI_API_KEY
            if not api_key or api_key == "your-openai-api-key":
                raise ValueError("API key is required for OpenAI models. Please set it in the config file or as an environment variable.")
            if APIGPTModel is None:
                raise ImportError("APIGPTModel backend is unavailable. Install required dependencies (e.g., openai).")
            model = APIGPTModel(api_key=api_key, model_name=model_name)
        elif model_type == 'api_doubao' or (model_type == 'api' and ('doubao' in model_name.lower() or 'deepseek' in model_name.lower())):
            api_key = config.get('api_key') or DOUBAO_API_KEY
            if not api_key or api_key == "your-doubao-api-key":
                raise ValueError("API key is required for Doubao/DeepSeek models. Please set it in the config file or as an environment variable.")
            if APIDoubaoModel is None:
                raise ImportError("APIDoubaoModel backend is unavailable. Install required dependencies (e.g., volcengine-python-sdk[ark]).")
            endpoint_url = init_parameters.get('endpoint_url')
            model = APIDoubaoModel(api_key=api_key, model_name=model_name, endpoint_url=endpoint_url)
        else:
            # Default to DeepInfra for api_deepinfra or unknown api models
            api_key = config.get('api_key') or DEEPINFRA_API_KEY
            if not api_key or api_key == "your-deepinfra-api-key":
                raise ValueError(
                    "API key is required for Deep Infra models. Please set it in the config file or as an environment variable.")
            if APIDeepInfraModel is None:
                raise ImportError("APIDeepInfraModel backend is unavailable. Install required dependencies (e.g., openai).")
            model = APIDeepInfraModel(api_key=api_key, model_name=model_name)
    elif model_type == 'local':
        if LocalHuggingFaceModel is None:
            raise ImportError("LocalHuggingFaceModel backend is unavailable. Install required dependencies (e.g., transformers, torch).")
        model = LocalHuggingFaceModel(model_name_or_path=model_name, **init_parameters)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Token limit parameter validation and logging
    _validate_and_log_token_parameters(model_type, model_name, parameters)

    return model, parameters


def _validate_and_log_token_parameters(model_type: str, model_name: str, parameters: dict):
    """验证并记录token限制参数"""
    token_limit_param = None
    token_limit_value = None
    
    if model_type.startswith('api'):
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
