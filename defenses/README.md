# Defenses Module

The defenses module provides a comprehensive collection of defensive mechanisms to protect Large Language Models (LLMs) against various attacks and harmful inputs. It follows a modular registry-based architecture that mirrors the attacks and models modules.

## Overview

This module contains implementations of various defense methods categorized into:
- **Input filters**: Preprocess and filter inputs before they reach the model
- **Output filters**: Post-process and filter model outputs
- **Model-based defenses**: Integrate defensive mechanisms into the model itself
- **Detection systems**: Identify potentially harmful inputs or outputs

All defenses inherit from `BaseDefendedModel` and can be loaded via configuration files for reproducible experiments.

## Architecture

```
defenses/
├── __init__.py                 # Module exports
├── base_defended_model.py     # Base class for all defenses
├── defense_registry.py        # Registry of available defenses
├── defense_loader.py          # Configuration-based loader
├── defenses/                  # Defense implementations
│   ├── BackTranslation/       # Back-translation defense
│   ├── GradSafe/             # Gradient-based safety mechanism
│   ├── JailGuard/            # Jailbreak detection and prevention
│   ├── PerplexityFilter/     # Perplexity-based filtering
│   ├── PrimeGuard/           # Multi-layer defense system
│   ├── RPO/                  # Robust Prompt Optimization
│   ├── SmoothLLM/            # Input smoothing defense
│   ├── input_filter_defense.py   # Generic input filtering
│   └── output_filter_defense.py  # Generic output filtering
└── usage_examples/
    ├── configs/              # Defense configuration files
    ├── defense_example.py    # Usage examples
    └── defense_example_rpo.py # RPO-specific examples
```

## Available Defenses

### Input Filtering Defenses

#### 1. **JailGuard**
- **Description**: Advanced jailbreak detection using similarity-based analysis
- **Features**: Text and image input support, augmentation techniques, similarity scoring
- **Effectiveness**: High accuracy against known jailbreak patterns
- **Use Cases**: Real-time jailbreak detection, content filtering

```python
# JailGuard configuration
{
  "defense_name": "JailGuardDefense",
  "parameters": {
    "threshold": 0.7,
    "similarity_metric": "cosine",
    "augmentation_enabled": true
  }
}
```

#### 2. **PerplexityFilter**
- **Description**: Filters inputs based on perplexity scores
- **Features**: Language model-based perplexity calculation, adaptive thresholds
- **Effectiveness**: Good at detecting out-of-distribution attacks
- **Use Cases**: Preprocessing unusual or crafted inputs

```python
# PerplexityFilter configuration
{
  "defense_name": "PerplexityFilterDefense",
  "parameters": {
    "threshold": 50.0,
    "model_name": "gpt2",
    "window_size": 100
  }
}
```

#### 3. **BackTranslation**
- **Description**: Uses translation round-trips to normalize inputs
- **Features**: Multi-language support, semantic preservation
- **Effectiveness**: Effective against character-level and encoding attacks
- **Use Cases**: Input preprocessing, attack mitigation

```python
# BackTranslation configuration
{
  "defense_name": "BackTranslationDefense",
  "parameters": {
    "intermediate_language": "french",
    "translation_model": "opus-mt",
    "confidence_threshold": 0.8
  }
}
```

#### 4. **SmoothLLM**
- **Description**: Smooths inputs through multiple perturbations
- **Features**: Character-level perturbations, ensemble voting
- **Effectiveness**: Robust against character-substitution attacks
- **Use Cases**: Input preprocessing, adversarial robustness

```python
# SmoothLLM configuration
{
  "defense_name": "SmoothLLMDefense",
  "parameters": {
    "num_copies": 10,
    "perturbation_rate": 0.1,
    "voting_strategy": "majority"
  }
}
```

### Model-Based Defenses

#### 1. **RPO** (Robust Prompt Optimization)
- **Description**: Optimizes prompts to be robust against adversarial suffixes
- **Features**: Gradient-based optimization, suffix detection
- **Effectiveness**: High robustness against GCG-style attacks
- **Use Cases**: Prompt hardening, adversarial training

```python
# RPO configuration
{
  "defense_name": "RPODefense",
  "parameters": {
    "optimization_steps": 100,
    "learning_rate": 0.01,
    "suffix_length": 20
  }
}
```

#### 2. **GradSafe**
- **Description**: Gradient-based safety mechanism for model outputs
- **Features**: Real-time gradient analysis, safety scoring
- **Effectiveness**: Good at detecting model manipulation
- **Use Cases**: Model monitoring, safety assurance

```python
# GradSafe configuration
{
  "defense_name": "GradSafeDefense",
  "parameters": {
    "safety_threshold": 0.5,
    "gradient_norm_limit": 1.0,
    "monitoring_layers": ["attention", "output"]
  }
}
```

### Output Filtering Defenses

#### 1. **PrimeGuard**
- **Description**: Multi-layer output analysis and filtering system
- **Features**: Content analysis, safety routing, template-based filtering
- **Effectiveness**: Comprehensive output safety checking
- **Use Cases**: Production deployment, content moderation

```python
# PrimeGuard configuration
{
  "defense_name": "PrimeGuardDefense",
  "parameters": {
    "safety_model": "safety-classifier-v2",
    "content_filters": ["violence", "toxicity", "hate"],
    "routing_strategy": "conservative"
  }
}
```

#### 2. **OutputFilterDefense**
- **Description**: Generic output filtering with customizable rules
- **Features**: Keyword filtering, pattern matching, ML-based classification
- **Effectiveness**: Flexible and configurable
- **Use Cases**: Custom filtering needs, rapid prototyping

```python
# OutputFilter configuration
{
  "defense_name": "OutputFilterDefense",
  "parameters": {
    "filter_type": "keyword",
    "blocked_keywords": ["harmful", "dangerous"],
    "use_ml_classifier": true
  }
}
```

### Generic Defenses

#### 1. **InputFilterDefense**
- **Description**: Configurable input filtering framework
- **Features**: Multiple filter types, chaining support, custom rules
- **Effectiveness**: Depends on configuration
- **Use Cases**: Custom input filtering, research experiments

```python
# InputFilter configuration
{
  "defense_name": "InputFilterDefense",
  "parameters": {
    "filters": [
      {"type": "length", "max_length": 1000},
      {"type": "profanity", "severity": "high"},
      {"type": "regex", "pattern": ".*harmful.*"}
    ]
  }
}
```

## Configuration Format

All defenses use JSON configuration files following this structure:

```json
{
  "defense_name": "DefenseClassName",
  "parameters": {
    "param1": "value1",
    "param2": "value2",
    "model_config": "path/to/model/config.json"
  }
}
```

## Usage Examples

### Basic Usage

```python
from defenses import load_defense_from_config
from models.model_loader import load_model_from_config

# Load base model
base_model, base_model_params = load_model_from_config(
    "models/usage_examples/configs/local/meta-llama-Llama-3.2-3B-Instruct.json"
)

# Load defense
defended_model = load_defense_from_config(
    "defenses/usage_examples/configs/jailguard_defense.json",
    base_model,
    base_model_params
)

# Use defended model
safe_response = defended_model.generate(
    prompt="Potentially harmful input",
    max_tokens=100
)
print(f"Safe response: {safe_response}")
```

### Defense Chaining

```python
# Chain multiple defenses
defense_configs = [
    "defenses/usage_examples/configs/input_filter_defense.json",
    "defenses/usage_examples/configs/smooth_llm.json",
    "defenses/usage_examples/configs/prime_guard.json"
]

# Apply defenses in sequence
defended_model = base_model
for config_path in defense_configs:
    defended_model = load_defense_from_config(
        config_path,
        defended_model,
        base_model_params
    )

# Final defended model has all protections
response = defended_model.generate(prompt="test input")
```

### Custom Defense Implementation

```python
from defenses.base_defended_model import BaseDefendedModel

class MyCustomDefense(BaseDefendedModel):
    def __init__(self, base_model, base_model_parameters, custom_param=None, **kwargs):
        super().__init__(base_model, base_model_parameters)
        self.custom_param = custom_param
    
    def defend_input(self, prompt: str, **kwargs) -> str:
        """Filter/modify input before passing to model."""
        # Custom input defense logic
        if self.is_harmful_input(prompt):
            return "I cannot process this request."
        return prompt
    
    def defend_output(self, output: str, **kwargs) -> str:
        """Filter/modify output before returning."""
        # Custom output defense logic
        if self.is_harmful_output(output):
            return "I cannot provide that information."
        return output
    
    def is_harmful_input(self, prompt: str) -> bool:
        """Custom harmfulness detection."""
        return any(word in prompt.lower() for word in ['harmful', 'dangerous'])
    
    def is_harmful_output(self, output: str) -> bool:
        """Custom output safety check."""
        return any(word in output.lower() for word in ['illegal', 'unsafe'])

# Register your defense
from defenses.defense_registry import DEFENSES
DEFENSES['MyCustomDefense'] = MyCustomDefense
```

## Defense Evaluation

### Effectiveness Metrics

#### 1. **Attack Success Rate (ASR)**
Percentage of attacks that succeed against the defense:

```python
def evaluate_defense_asr(defended_model, attack_dataset):
    successful_attacks = 0
    total_attacks = len(attack_dataset)
    
    for attack_prompt in attack_dataset:
        response = defended_model.generate(prompt=attack_prompt)
        if is_attack_successful(response):
            successful_attacks += 1
    
    asr = successful_attacks / total_attacks
    return asr
```

#### 2. **False Positive Rate (FPR)**
Percentage of benign inputs incorrectly flagged as harmful:

```python
def evaluate_defense_fpr(defended_model, benign_dataset):
    false_positives = 0
    total_benign = len(benign_dataset)
    
    for benign_prompt in benign_dataset:
        response = defended_model.generate(prompt=benign_prompt)
        if is_blocked_response(response):
            false_positives += 1
    
    fpr = false_positives / total_benign
    return fpr
```

#### 3. **Utility Preservation**
How well the defense maintains model utility:

```python
def evaluate_defense_utility(defended_model, base_model, test_dataset):
    defended_scores = []
    base_scores = []
    
    for prompt in test_dataset:
        defended_response = defended_model.generate(prompt=prompt)
        base_response = base_model.generate(prompt=prompt)
        
        defended_scores.append(quality_score(defended_response))
        base_scores.append(quality_score(base_response))
    
    utility_retention = np.mean(defended_scores) / np.mean(base_scores)
    return utility_retention
```

### Benchmarking Framework

```python
from defenses.evaluation import DefenseBenchmark

# Create benchmark
benchmark = DefenseBenchmark(
    attack_datasets=["harmful_prompts.json", "jailbreak_prompts.json"],
    benign_datasets=["normal_prompts.json", "helpful_requests.json"],
    metrics=["asr", "fpr", "utility", "latency"]
)

# Evaluate defense
results = benchmark.evaluate(
    defense_config="defenses/usage_examples/configs/jailguard_defense.json",
    base_model_config="models/usage_examples/configs/local/llama3-8b.json"
)

print(f"Defense Performance: {results}")
```

## Performance Considerations

### Latency Impact

```python
import time

def measure_defense_latency(defended_model, base_model, test_prompts):
    # Measure base model latency
    base_times = []
    for prompt in test_prompts:
        start_time = time.time()
        base_model.generate(prompt=prompt)
        base_times.append(time.time() - start_time)
    
    # Measure defended model latency
    defended_times = []
    for prompt in test_prompts:
        start_time = time.time()
        defended_model.generate(prompt=prompt)
        defended_times.append(time.time() - start_time)
    
    latency_overhead = np.mean(defended_times) / np.mean(base_times) - 1
    return latency_overhead
```

### Memory Usage

```python
import psutil
import torch

def measure_memory_usage(defended_model, base_model):
    # Measure base memory
    torch.cuda.empty_cache()
    base_memory = torch.cuda.memory_allocated()
    
    # Measure defended memory
    defended_model.load()  # Ensure defense is loaded
    defended_memory = torch.cuda.memory_allocated()
    
    memory_overhead = (defended_memory - base_memory) / base_memory
    return memory_overhead
```

### Optimization Strategies

1. **Caching**: Cache defense computations for repeated inputs
2. **Batching**: Process multiple inputs together when possible
3. **Early stopping**: Stop processing when threat is detected
4. **Model selection**: Use appropriate model sizes for defense components

## Adaptive Defenses

### Dynamic Threshold Adjustment

```python
class AdaptiveDefense(BaseDefendedModel):
    def __init__(self, base_model, base_model_parameters, initial_threshold=0.5):
        super().__init__(base_model, base_model_parameters)
        self.threshold = initial_threshold
        self.attack_history = []
    
    def adapt_threshold(self, recent_attacks):
        """Adjust threshold based on recent attack patterns."""
        if len(recent_attacks) > 10:  # High attack frequency
            self.threshold *= 0.9  # Lower threshold (more strict)
        elif len(recent_attacks) < 2:  # Low attack frequency
            self.threshold *= 1.1  # Higher threshold (less strict)
    
    def defend_input(self, prompt: str, **kwargs) -> str:
        risk_score = self.calculate_risk(prompt)
        
        if risk_score > self.threshold:
            self.attack_history.append(prompt)
            self.adapt_threshold(self.attack_history[-20:])  # Last 20 attempts
            return "Request blocked for safety."
        
        return prompt
```

### Ensemble Defenses

```python
class EnsembleDefense(BaseDefendedModel):
    def __init__(self, base_model, base_model_parameters, defense_configs):
        super().__init__(base_model, base_model_parameters)
        self.defenses = []
        
        for config in defense_configs:
            defense = load_defense_from_config(config, base_model, base_model_parameters)
            self.defenses.append(defense)
    
    def defend_input(self, prompt: str, **kwargs) -> str:
        votes = []
        for defense in self.defenses:
            try:
                result = defense.defend_input(prompt, **kwargs)
                votes.append(result != prompt)  # True if blocked
            except Exception:
                votes.append(True)  # Conservative: block on error
        
        # Majority voting
        if sum(votes) > len(votes) // 2:
            return "Request blocked by ensemble defense."
        return prompt
```

## Integration with Judgers

Defenses can leverage the judgers module for evaluation:

```python
from judgers import load_judger_from_config

class JudgerBasedDefense(BaseDefendedModel):
    def __init__(self, base_model, base_model_parameters, judger_config, threshold=0.5):
        super().__init__(base_model, base_model_parameters)
        self.judger = load_judger_from_config(judger_config)
        self.threshold = threshold
    
    def defend_input(self, prompt: str, **kwargs) -> str:
        # Use judger to evaluate input harmfulness
        harm_score = self.judger.judge(generation=prompt)
        
        if harm_score > self.threshold:
            return "Input blocked by safety judger."
        return prompt
    
    def defend_output(self, output: str, **kwargs) -> str:
        # Use judger to evaluate output safety
        harm_score = self.judger.judge(generation=output)
        
        if harm_score > self.threshold:
            return "I cannot provide that information."
        return output
```

## Deployment Considerations

### Production Deployment

```python
class ProductionDefense(BaseDefendedModel):
    def __init__(self, base_model, base_model_parameters, config):
        super().__init__(base_model, base_model_parameters)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = MetricsCollector()
    
    def generate(self, prompt: str, **kwargs) -> str:
        start_time = time.time()
        
        try:
            # Input defense
            filtered_prompt = self.defend_input(prompt, **kwargs)
            
            # Model generation
            response = self.base_model.generate(
                prompt=filtered_prompt,
                **kwargs
            )
            
            # Output defense
            safe_response = self.defend_output(response, **kwargs)
            
            # Log metrics
            self.metrics.record_success(time.time() - start_time)
            return safe_response
            
        except Exception as e:
            self.logger.error(f"Defense error: {e}")
            self.metrics.record_error()
            return "I apologize, but I cannot process your request at this time."
```

### Monitoring and Alerting

```python
class DefenseMonitor:
    def __init__(self, defense, alert_threshold=10):
        self.defense = defense
        self.alert_threshold = alert_threshold
        self.blocked_count = 0
        self.total_count = 0
    
    def monitor_request(self, prompt: str, response: str):
        self.total_count += 1
        
        if "blocked" in response.lower():
            self.blocked_count += 1
            
            # Alert on high block rate
            block_rate = self.blocked_count / self.total_count
            if block_rate > self.alert_threshold:
                self.send_alert(f"High block rate: {block_rate:.2%}")
    
    def send_alert(self, message: str):
        # Implement alerting logic (email, Slack, etc.)
        print(f"ALERT: {message}")
```

## Best Practices

### 1. **Defense Selection**
- **Layer multiple defenses**: Use complementary defense mechanisms
- **Consider threat model**: Choose defenses appropriate for your use case
- **Balance security vs. utility**: Avoid over-filtering legitimate requests

### 2. **Configuration Management**
- **Version control**: Track defense configurations and changes
- **A/B testing**: Compare defense effectiveness systematically
- **Gradual rollout**: Deploy new defenses incrementally

### 3. **Monitoring and Maintenance**
- **Continuous evaluation**: Regularly assess defense effectiveness
- **Attack pattern analysis**: Monitor emerging attack vectors
- **Performance tracking**: Watch for latency and utility impacts

### 4. **Security Operations**
- **Incident response**: Have procedures for defense failures
- **Regular updates**: Keep defense models and rules current
- **Threat intelligence**: Incorporate new attack information

## Troubleshooting

### Common Issues

1. **High False Positive Rate**
   ```python
   # Adjust thresholds
   defense.threshold *= 1.2  # Less aggressive filtering
   
   # Add whitelisting
   defense.add_whitelist_patterns(["legitimate", "helpful"])
   ```

2. **Performance Degradation**
   ```python
   # Enable caching
   defense.enable_cache(max_size=1000)
   
   # Use faster defense components
   defense.use_lightweight_models()
   ```

3. **Memory Issues**
   ```python
   # Clear caches periodically
   defense.clear_cache()
   
   # Use model offloading
   defense.enable_model_offloading()
   ```

### Debug Mode

```python
# Enable detailed logging
defense = load_defense_from_config(
    config_path,
    base_model,
    base_model_params,
    debug=True
)

# Check defense decisions
result = defense.generate(prompt="test", explain_decision=True)
print(f"Defense reasoning: {result.explanation}")
```

## Contributing

### Adding New Defenses

1. **Implement BaseDefendedModel**:
   ```python
   from defenses.base_defended_model import BaseDefendedModel
   
   class NewDefense(BaseDefendedModel):
       def defend_input(self, prompt: str, **kwargs) -> str:
           # Input defense logic
           pass
       
       def defend_output(self, output: str, **kwargs) -> str:
           # Output defense logic
           pass
   ```

2. **Register the defense**:
   ```python
   # In defense_registry.py
   from .new_defense import NewDefense
   DEFENSES['NewDefense'] = NewDefense
   ```

3. **Create configuration and tests**

### Research Contributions
- Implement new defense mechanisms from recent papers
- Contribute evaluation metrics and benchmarks
- Add integration with new model architectures

## Security Considerations

### Defense Evasion
- **Adaptive attacks**: Expect attackers to adapt to your defenses
- **Defense evaluation**: Regularly test against new attack methods
- **Ensemble robustness**: Use multiple defense layers

### Privacy Protection
- **Data handling**: Ensure defense mechanisms protect user data
- **Logging**: Be careful about what defense logs contain
- **Compliance**: Follow relevant privacy regulations

## References

### Defense Papers
- **SmoothLLM**: Defending Large Language Models Against Jailbreaking Attacks
- **JailGuard**: Jailbreak Defense for LLMs
- **RPO**: Robust Prompt Optimization for Large Language Models

### Security Frameworks
- OWASP AI Security Guidelines
- NIST AI Risk Management Framework
- Common attack and defense taxonomies

For detailed implementation information, see the individual defense documentation in their respective directories.
