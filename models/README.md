# Models Module

The models module provides a unified interface for loading and interacting with various language models, including both local (Hugging Face) models and API-based models (OpenAI, Claude, Gemini, etc.). It follows a modular registry-based architecture for easy extensibility and consistent usage patterns.

## Overview

This module abstracts away the differences between various model providers and deployment methods, providing:
- **Unified interface**: Consistent API across all model types
- **Flexible configuration**: JSON-based model configuration
- **Provider support**: Local models, OpenAI, Claude, Gemini, DeepInfra, and more
- **Parameter management**: Consistent parameter handling across providers
- **Error handling**: Robust error handling and retry mechanisms

## Architecture

```
models/
├── __init__.py                 # Module exports
├── base_model.py              # Base class for all models
├── model_registry.py          # Registry of available models
├── model_loader.py            # Configuration-based loader
├── config.py                  # Global model configuration
├── api_keys.py               # API key management
├── api_models/               # API-based model implementations
│   ├── __init__.py
│   ├── api_base.py           # Base class for API models
│   ├── api_gpt_model.py      # OpenAI GPT models
│   ├── api_claude_model.py   # Anthropic Claude models
│   ├── api_gemini_model.py   # Google Gemini models
│   └── api_deepinfra_model.py # DeepInfra models
├── local_models/             # Local model implementations
│   ├── __init__.py
│   ├── local_base.py         # Base class for local models
│   └── local_huggingface_model.py # Hugging Face models
└── usage_examples/
    ├── configs/              # Model configuration files
    │   ├── api/             # API model configs
    │   └── local/           # Local model configs
    ├── api_generation_example.py
    ├── api_gpt_example.py
    └── local_huggingface_example.py
```

## Supported Model Providers

### API-based Models

#### 1. **OpenAI Models**
- **Supported Models**: GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo, o1-preview, o1-mini
- **Features**: Chat completions, function calling, streaming
- **Authentication**: API key required
- **Rate Limits**: Handled automatically with exponential backoff

```json
{
  "model_type": "api",
  "model_name": "gpt-4o-mini",
  "api_provider": "openai",
  "parameters": {
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }
}
```

#### 2. **Anthropic Claude Models**
- **Supported Models**: Claude-3 Opus, Claude-3.5 Sonnet, Claude-3 Haiku
- **Features**: Long context, vision capabilities, tool use
- **Authentication**: API key required
- **Strengths**: Long context understanding, safety alignment

```json
{
  "model_type": "api",
  "model_name": "claude-3-5-sonnet-20241022",
  "api_provider": "anthropic",
  "parameters": {
    "max_tokens": 4096,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 5
  }
}
```

#### 3. **Google Gemini Models**
- **Supported Models**: Gemini-1.5 Pro, Gemini-1.5 Flash, Gemini-1.5 Flash-8B
- **Features**: Multimodal capabilities, long context, function calling
- **Authentication**: API key required
- **Strengths**: Multimodal understanding, efficiency

```json
{
  "model_type": "api",
  "model_name": "gemini-1.5-pro",
  "api_provider": "google",
  "parameters": {
    "max_tokens": 2048,
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 40
  }
}
```

#### 4. **DeepInfra Models**
- **Supported Models**: Meta-Llama, Qwen, Mistral, and many others
- **Features**: Cost-effective API access to open-source models
- **Authentication**: API key required
- **Strengths**: Wide model selection, competitive pricing

```json
{
  "model_type": "api",
  "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "api_provider": "deepinfra",
  "parameters": {
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

### Local Models

#### 1. **Hugging Face Models**
- **Supported Models**: Any model compatible with transformers library
- **Features**: Local inference, custom models, fine-tuned models
- **Requirements**: GPU memory, model weights
- **Advantages**: No API costs, full control, privacy

```json
{
  "model_type": "local",
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "parameters": {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": true,
    "device_map": "auto",
    "torch_dtype": "float16"
  }
}
```

## Configuration Format

All models use a standardized JSON configuration format:

```json
{
  "model_type": "api|local",
  "model_name": "provider/model-name",
  "api_provider": "openai|anthropic|google|deepinfra",
  "parameters": {
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 1.0,
    "additional_params": "value"
  }
}
```

### Parameter Mapping

The module automatically handles parameter differences between providers:

| Unified Parameter | OpenAI | Anthropic | Google | Local (HF) |
|------------------|---------|-----------|---------|------------|
| `max_tokens` | `max_tokens` | `max_tokens` | `max_output_tokens` | `max_new_tokens` |
| `temperature` | `temperature` | `temperature` | `temperature` | `temperature` |
| `top_p` | `top_p` | `top_p` | `top_p` | `top_p` |
| `stop_sequences` | `stop` | `stop_sequences` | `stop_sequences` | `stop_sequences` |

## Usage Examples

### Basic Usage

```python
from models.model_loader import load_model_from_config

# Load any model using configuration
model, params = load_model_from_config(
    "models/usage_examples/configs/api/gpt-4o-mini.json"
)

# Generate text
response = model.generate(
    prompt="What is artificial intelligence?",
    max_tokens=100,
    temperature=0.7
)
print(response)
```

### Chat-based Generation

```python
# For chat models, use messages format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."}
]

response = model.generate(
    messages=messages,
    max_tokens=500,
    temperature=0.8
)
print(response)
```

### Batch Generation

```python
# Generate multiple responses
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is deep learning?"
]

responses = []
for prompt in prompts:
    response = model.generate(prompt=prompt, max_tokens=200)
    responses.append(response)

print(responses)
```

### Model Comparison

```python
# Compare different models on the same task
model_configs = [
    "models/usage_examples/configs/api/gpt-4o-mini.json",
    "models/usage_examples/configs/api/claude-3-haiku-20240307.json",
    "models/usage_examples/configs/local/meta-llama-Llama-3.2-3B-Instruct.json"
]

test_prompt = "Explain the concept of attention in transformers."

for config_path in model_configs:
    model, params = load_model_from_config(config_path)
    response = model.generate(prompt=test_prompt, max_tokens=300)
    print(f"Model: {config_path}")
    print(f"Response: {response}\n")
```

## API Key Management

### Setting API Keys

API keys can be provided in several ways:

1. **Environment Variables** (Recommended):
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-claude-key"
export GOOGLE_API_KEY="your-gemini-key"
export DEEPINFRA_API_KEY="your-deepinfra-key"
```

2. **Configuration File**:
```json
{
  "openai_api_key": "your-openai-key",
  "anthropic_api_key": "your-claude-key",
  "google_api_key": "your-gemini-key",
  "deepinfra_api_key": "your-deepinfra-key"
}
```

3. **Direct Parameter**:
```python
model, params = load_model_from_config(
    config_path,
    api_key="your-api-key"
)
```

### Key Security Best Practices

- **Never commit API keys** to version control
- **Use environment variables** in production
- **Rotate keys regularly**
- **Monitor usage** and set billing alerts
- **Use separate keys** for development and production

## Advanced Features

### Custom Model Implementation

```python
from models.base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self.custom_param = kwargs.get('custom_param', 'default')
    
    def generate(self, prompt: str = None, messages: list = None, **kwargs) -> str:
        """Implement your custom generation logic."""
        if messages:
            # Handle chat format
            prompt = self._format_messages(messages)
        
        # Your custom inference logic here
        response = self._custom_inference(prompt, **kwargs)
        return response
    
    def _custom_inference(self, prompt: str, **kwargs) -> str:
        """Custom inference implementation."""
        # Implement your model's inference logic
        return f"Custom response to: {prompt}"
    
    def _format_messages(self, messages: list) -> str:
        """Convert messages to prompt format."""
        formatted = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted += f"{role}: {content}\n"
        return formatted

# Register your custom model
from models.model_registry import MODELS
MODELS['custom'] = CustomModel
```

### Model Caching and Optimization

```python
class CachedModel:
    def __init__(self, base_model, cache_size=100):
        self.base_model = base_model
        self.cache = {}
        self.cache_size = cache_size
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Create cache key
        cache_key = self._create_cache_key(prompt, **kwargs)
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate response
        response = self.base_model.generate(prompt=prompt, **kwargs)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = response
        return response
    
    def _create_cache_key(self, prompt: str, **kwargs) -> str:
        """Create a cache key from prompt and parameters."""
        import hashlib
        key_data = f"{prompt}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
```

### Model Ensemble

```python
class ModelEnsemble:
    def __init__(self, model_configs, voting_strategy='majority'):
        self.models = []
        self.voting_strategy = voting_strategy
        
        for config in model_configs:
            model, params = load_model_from_config(config)
            self.models.append((model, params))
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using ensemble of models."""
        responses = []
        
        for model, params in self.models:
            try:
                response = model.generate(prompt=prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                print(f"Model failed: {e}")
                responses.append("")
        
        if self.voting_strategy == 'majority':
            return self._majority_vote(responses)
        elif self.voting_strategy == 'longest':
            return max(responses, key=len)
        elif self.voting_strategy == 'first':
            return next((r for r in responses if r), "")
        else:
            return responses[0] if responses else ""
    
    def _majority_vote(self, responses):
        """Simple majority voting based on response similarity."""
        from collections import Counter
        # Simplified: return most common response
        return Counter(responses).most_common(1)[0][0]
```

## Performance Optimization

### Local Model Optimization

```python
# GPU optimization for local models
{
  "model_type": "local",
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "parameters": {
    "device_map": "auto",
    "torch_dtype": "float16",
    "attn_implementation": "flash_attention_2",
    "load_in_4bit": true,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": true,
    "max_memory": {0: "20GB", 1: "20GB"}
  }
}
```

### API Rate Limiting

```python
import time
from functools import wraps

def rate_limit(calls_per_minute=60):
    """Rate limiting decorator for API calls."""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Apply to model generation
@rate_limit(calls_per_minute=30)
def safe_generate(model, prompt, **kwargs):
    return model.generate(prompt=prompt, **kwargs)
```

### Batch Processing

```python
class BatchProcessor:
    def __init__(self, model, batch_size=10, delay=1.0):
        self.model = model
        self.batch_size = batch_size
        self.delay = delay
    
    def process_batch(self, prompts: list, **kwargs) -> list:
        """Process prompts in batches with rate limiting."""
        results = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            batch_results = []
            
            for prompt in batch:
                try:
                    result = self.model.generate(prompt=prompt, **kwargs)
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error processing prompt: {e}")
                    batch_results.append("")
            
            results.extend(batch_results)
            
            # Rate limiting delay
            if i + self.batch_size < len(prompts):
                time.sleep(self.delay)
        
        return results
```

## Error Handling and Reliability

### Retry Mechanisms

```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    """Exponential backoff retry decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    
                    # Calculate delay with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, 0.1) * delay
                    time.sleep(delay + jitter)
                    
                    print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
            
        return wrapper
    return decorator

# Apply to model generation
@retry_with_backoff(max_retries=3)
def reliable_generate(model, prompt, **kwargs):
    return model.generate(prompt=prompt, **kwargs)
```

### Fallback Models

```python
class FallbackModel:
    def __init__(self, primary_config, fallback_configs):
        self.primary_model, self.primary_params = load_model_from_config(primary_config)
        
        self.fallback_models = []
        for config in fallback_configs:
            model, params = load_model_from_config(config)
            self.fallback_models.append((model, params))
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Try primary model, fall back to alternatives on failure."""
        # Try primary model
        try:
            return self.primary_model.generate(prompt=prompt, **kwargs)
        except Exception as e:
            print(f"Primary model failed: {e}")
        
        # Try fallback models
        for i, (model, params) in enumerate(self.fallback_models):
            try:
                print(f"Trying fallback model {i + 1}")
                return model.generate(prompt=prompt, **kwargs)
            except Exception as e:
                print(f"Fallback model {i + 1} failed: {e}")
                continue
        
        raise Exception("All models failed")
```

## Monitoring and Logging

### Usage Tracking

```python
import logging
from datetime import datetime

class ModelMonitor:
    def __init__(self, model):
        self.model = model
        self.usage_stats = {
            'total_calls': 0,
            'total_tokens': 0,
            'errors': 0,
            'avg_latency': 0.0
        }
        self.logger = logging.getLogger(__name__)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Monitored generation with usage tracking."""
        start_time = time.time()
        
        try:
            response = self.model.generate(prompt=prompt, **kwargs)
            
            # Update stats
            self.usage_stats['total_calls'] += 1
            self.usage_stats['total_tokens'] += len(response.split())
            
            latency = time.time() - start_time
            self.usage_stats['avg_latency'] = (
                (self.usage_stats['avg_latency'] * (self.usage_stats['total_calls'] - 1) + latency) /
                self.usage_stats['total_calls']
            )
            
            # Log successful request
            self.logger.info(f"Generated response in {latency:.2f}s")
            
            return response
            
        except Exception as e:
            self.usage_stats['errors'] += 1
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return self.usage_stats.copy()
```

### Cost Tracking

```python
class CostTracker:
    PRICING = {
        'gpt-4o': {'input': 0.0025, 'output': 0.01},  # per 1K tokens
        'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
        'claude-3-5-sonnet': {'input': 0.003, 'output': 0.015},
        'gemini-1.5-pro': {'input': 0.00125, 'output': 0.005}
    }
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
    
    def track_usage(self, input_text: str, output_text: str):
        """Track token usage and calculate cost."""
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4
        
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        
        if self.model_name in self.PRICING:
            pricing = self.PRICING[self.model_name]
            cost = (
                (input_tokens / 1000) * pricing['input'] +
                (output_tokens / 1000) * pricing['output']
            )
            self.total_cost += cost
    
    def get_cost_summary(self) -> dict:
        """Get cost summary."""
        return {
            'total_cost': self.total_cost,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.input_tokens + self.output_tokens
        }
```

## Configuration Management

### Environment-based Configuration

```python
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self):
        self.env_mapping = {
            'development': 'dev_configs',
            'staging': 'staging_configs',
            'production': 'prod_configs'
        }
    
    def get_model_config(self, model_name: str, environment: str = None) -> Dict[str, Any]:
        """Get model configuration based on environment."""
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'development')
        
        config_dir = self.env_mapping.get(environment, 'dev_configs')
        config_path = f"models/usage_examples/configs/{config_dir}/{model_name}.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Override with environment variables
        if 'OPENAI_API_KEY' in os.environ:
            config['openai_api_key'] = os.environ['OPENAI_API_KEY']
        
        return config
    
    def load_model_for_environment(self, model_name: str, environment: str = None):
        """Load model with environment-specific configuration."""
        config = self.get_model_config(model_name, environment)
        return load_model_from_config(config)
```

### Dynamic Configuration Updates

```python
class DynamicModel:
    def __init__(self, initial_config):
        self.config = initial_config
        self.model, self.params = load_model_from_config(initial_config)
        self.generation_count = 0
    
    def update_config(self, new_params: dict):
        """Update model configuration dynamically."""
        self.config['parameters'].update(new_params)
        print(f"Updated config: {new_params}")
    
    def adaptive_generation(self, prompt: str, **kwargs) -> str:
        """Generate with adaptive parameters based on usage."""
        # Adapt temperature based on generation count
        if self.generation_count > 100:
            kwargs['temperature'] = min(kwargs.get('temperature', 0.7) + 0.1, 1.0)
        
        response = self.model.generate(prompt=prompt, **kwargs)
        self.generation_count += 1
        
        return response
```

## Testing and Validation

### Model Testing Framework

```python
import unittest
from typing import List

class ModelTestSuite:
    def __init__(self, model_configs: List[str]):
        self.model_configs = model_configs
        self.test_cases = [
            ("Hello, how are you?", "greeting"),
            ("What is 2+2?", "math"),
            ("Explain quantum computing", "technical"),
            ("Write a poem about cats", "creative")
        ]
    
    def run_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run test suite on all models."""
        results = {}
        
        for config_path in self.model_configs:
            model, params = load_model_from_config(config_path)
            model_results = self.test_model(model, config_path)
            results[config_path] = model_results
        
        return results
    
    def test_model(self, model, model_name: str) -> Dict[str, Any]:
        """Test individual model."""
        results = {
            'passed': 0,
            'failed': 0,
            'responses': {},
            'errors': []
        }
        
        for prompt, category in self.test_cases:
            try:
                response = model.generate(prompt=prompt, max_tokens=100)
                results['responses'][category] = response
                results['passed'] += 1
            except Exception as e:
                results['errors'].append(f"{category}: {str(e)}")
                results['failed'] += 1
        
        return results
```

### Model Validation

```python
def validate_model_output(response: str, expected_type: str = "text") -> bool:
    """Validate model output format and content."""
    if not response or not isinstance(response, str):
        return False
    
    if expected_type == "text":
        return len(response.strip()) > 0
    elif expected_type == "json":
        try:
            json.loads(response)
            return True
        except:
            return False
    elif expected_type == "code":
        # Basic code validation (contains code-like patterns)
        code_indicators = ['def ', 'function', 'import ', 'class ', '```']
        return any(indicator in response for indicator in code_indicators)
    
    return True

# Usage in testing
def test_model_responses():
    model, params = load_model_from_config("path/to/config.json")
    
    test_cases = [
        ("Write a Python function", "code"),
        ("Return JSON data", "json"),
        ("Explain AI", "text")
    ]
    
    for prompt, expected_type in test_cases:
        response = model.generate(prompt=prompt, max_tokens=200)
        is_valid = validate_model_output(response, expected_type)
        print(f"Test '{prompt}': {'PASS' if is_valid else 'FAIL'}")
```

## Best Practices

### 1. **Model Selection**
- **Match model to task**: Use appropriate models for specific use cases
- **Consider cost vs. performance**: Balance quality needs with budget constraints
- **Test multiple providers**: Compare outputs and reliability across providers

### 2. **Configuration Management**
- **Version control configs**: Track configuration changes
- **Environment separation**: Use different configs for dev/staging/prod
- **Parameter documentation**: Document the purpose of each parameter

### 3. **Error Handling**
- **Implement retries**: Handle transient failures gracefully
- **Fallback strategies**: Have backup models for critical applications
- **Monitoring and alerting**: Track model availability and performance

### 4. **Security and Privacy**
- **Secure API keys**: Never expose keys in code or logs
- **Data handling**: Be careful with sensitive data sent to APIs
- **Compliance**: Follow relevant data protection regulations

### 5. **Performance Optimization**
- **Caching**: Cache responses for identical requests
- **Batch processing**: Process multiple requests efficiently
- **Resource management**: Monitor memory and GPU usage for local models

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   ```python
   # Check if API key is set
   import os
   print(f"OpenAI key set: {'OPENAI_API_KEY' in os.environ}")
   ```

2. **Memory Issues (Local Models)**:
   ```python
   # Use smaller models or quantization
   {
     "load_in_4bit": true,
     "device_map": "auto",
     "max_memory": {0: "10GB"}
   }
   ```

3. **Rate Limiting**:
   ```python
   # Implement proper delays
   import time
   time.sleep(1)  # Wait between requests
   ```

4. **Model Loading Failures**:
   ```python
   # Check model name and availability
   from transformers import AutoTokenizer
   try:
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       print("Model available")
   except:
       print("Model not found")
   ```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test model loading step by step
config = load_config("path/to/config.json")
print(f"Config loaded: {config}")

model, params = load_model_from_config(config)
print(f"Model loaded: {type(model)}")

response = model.generate(prompt="test", max_tokens=10)
print(f"Response: {response}")
```

## Contributing

### Adding New Model Providers

1. **Create provider class**:
   ```python
   from models.api_models.api_base import APIBase
   
   class NewProviderModel(APIBase):
       def __init__(self, model_name: str, **kwargs):
           super().__init__(model_name, **kwargs)
           self.api_endpoint = "https://api.newprovider.com/v1"
       
       def _make_request(self, prompt: str, **kwargs) -> str:
           # Implement provider-specific API call
           pass
   ```

2. **Register the provider**:
   ```python
   # In model_registry.py
   from .new_provider import NewProviderModel
   MODELS['new_provider'] = NewProviderModel
   ```

3. **Create configuration template and tests**

### Model Integration Guidelines
- Follow the base model interface
- Handle provider-specific parameters correctly
- Implement proper error handling
- Add comprehensive tests
- Document new features and limitations

## References

### Provider Documentation
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude API](https://docs.anthropic.com)
- [Google Gemini API](https://ai.google.dev)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

### Model Information
- Model cards and documentation for specific models
- Performance benchmarks and comparisons
- Usage guidelines and best practices

For detailed implementation information, see the individual provider documentation and example configurations in the `usage_examples/` directory.