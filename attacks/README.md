# Attacks Module

The attacks module provides a comprehensive collection of jailbreaking and prompt injection attacks for evaluating LLM security. It follows a modular registry-based architecture for easy extensibility and configuration.

## Overview

This module contains implementations of various attack methods categorized into:
- **Black-box attacks**: No access to model weights or gradients
- **White-box attacks**: Full access to model internals

All attacks inherit from `BaseAttack` and can be loaded via configuration files for reproducible experiments.

## Architecture

```
attacks/
├── __init__.py                 # Module exports
├── base_attack.py             # Base class for all attacks
├── attack_registry.py         # Registry of available attacks
├── attack_loader.py           # Configuration-based loader
├── black_box/                 # Black-box attack implementations
│   ├── ReNeLLM/              # Rewrite/Nest attack
│   ├── PAIR/                 # Prompt Automatic Iterative Refinement
│   ├── TapAttack/            # Tree of Attacks with Pruning
│   ├── ABJAttack/            # Anti-Basic-Jailbreak attack
│   ├── ArtPrompt/            # ASCII art-based attacks
│   ├── GPTFUZZER/            # GPT fuzzing framework
│   ├── DRA/                  # Direct Resistance Attack
│   ├── CodeAttack/           # Code-based prompt injection
│   ├── CodeChameleon/        # Encryption-based attacks
│   ├── FlipAttack/           # Character flipping attacks
│   ├── InceptionAttack/      # Nested prompt attacks
│   ├── MultilingualJailbreak/ # Cross-language attacks
│   ├── PastTense/            # Tense modification attacks
│   ├── PersuasiveInContext/  # Persuasion-based attacks
│   ├── DrAttack/             # Direct resistance attacks
│   └── IFSJ/                 # Instruction following attacks
├── white_box/                 # White-box attack implementations
│   ├── AutoDAN/              # Automatic gradient-based attacks
│   ├── COLD/                 # Contrastive learning attacks
│   └── GCGAttack/            # Greedy coordinate gradient attacks
└── usage_examples/
    ├── configs/              # Attack configuration files
    └── attack_example.py     # Usage examples
```

## Available Attacks

### Black-box Attacks

#### 1. **ReNeLLM** (Rewrite and Nest)
- **Description**: Iteratively rewrites harmful prompts and nests them in scenarios
- **Features**: Multi-operation prompt transformation, scenario-based nesting
- **Usage**: Effective against content filters and safety training
- **Judge Required**: Yes (harmful classification)

#### 2. **PAIR** (Prompt Automatic Iterative Refinement)
- **Description**: Adversarial conversation framework with iterative refinement
- **Features**: Multi-stream conversations, automatic prompt optimization
- **Usage**: Sophisticated multi-turn jailbreaking
- **Judge Required**: Yes (HarmBench classifier)

#### 3. **TapAttack** (Tree of Attacks with Pruning)
- **Description**: Tree-structured attack generation with evaluator-guided pruning
- **Features**: Branching factor control, evaluator-based pruning
- **Usage**: Systematic exploration of attack space
- **Judge Required**: Yes (custom evaluator)

#### 4. **ABJAttack** (Anti-Basic-Jailbreak)
- **Description**: Character/feature/job-based prompt crafting
- **Features**: Structured prompt generation, assist model integration
- **Usage**: Role-playing and persona-based attacks
- **Judge Required**: Optional (assist model)

#### 5. **ArtPrompt**
- **Description**: ASCII art-based semantic attacks
- **Features**: Visual encoding, multiple safety evaluators
- **Usage**: Bypass visual content filters
- **Judge Required**: Yes (GPT-4 + substring matching)

#### 6. **GPTFUZZER**
- **Description**: Template-based fuzzing with mutation strategies
- **Features**: Seed mutation, jailbreak prediction
- **Usage**: Automated attack generation
- **Judge Required**: Yes (RoBERTa classifier)

#### 7. **DRA** (Direct Resistance Attack)
- **Description**: Direct prompt generation with content moderation
- **Features**: HarmBench evaluation, toxicity filtering
- **Usage**: Straightforward jailbreaking attempts
- **Judge Required**: Yes (HarmBench + Detoxify)

#### 8. **Simple Transformation Attacks**
- **CodeAttack**: Code-based prompt encoding
- **CodeChameleon**: Encryption/decryption-based attacks
- **FlipAttack**: Character replacement attacks
- **InceptionAttack**: Nested instruction attacks
- **MultilingualJailbreak**: Cross-language translation attacks
- **PastTense**: Tense modification attacks
- **PersuasiveInContext**: Persuasion taxonomy-based attacks

### White-box Attacks

#### 1. **AutoDAN**
- **Description**: Automatic gradient-based adversarial prompt generation
- **Features**: Hierarchical genetic algorithm, gradient-based optimization
- **Usage**: Direct optimization against model weights
- **Requirements**: Model access, gradient computation

#### 2. **COLD** (Contrastive Learning)
- **Description**: Contrastive learning-based adversarial examples
- **Features**: Paraphrase generation, BLEU loss optimization
- **Usage**: Semantic-preserving adversarial examples
- **Requirements**: Model access, embedding computation

#### 3. **GCGAttack** (Greedy Coordinate Gradient)
- **Description**: Suffix-based gradient optimization
- **Features**: Token-level optimization, greedy coordinate ascent
- **Usage**: Append adversarial suffixes to prompts
- **Requirements**: Model access, gradient computation

## Configuration Format

All attacks use JSON configuration files following this structure:

```json
{
  "attack_name": "AttackClassName",
  "parameters": {
    "param1": "value1",
    "param2": "value2",
    "judge_model_config": "path/to/judger/config.json"
  }
}
```

### Example Configurations

**ReNeLLM Attack:**
```json
{
  "attack_name": "ReNeLLM",
  "parameters": {
    "rewrite_model_config": "models/usage_examples/configs/api/gpt-4o-mini.json",
    "judge_model_config": "judgers/usage_examples/configs/gpt_judger_harmful_binary.json",
    "iter_max": 10,
    "temperature": 0.0,
    "max_tokens": 512
  }
}
```

**PAIR Attack:**
```json
{
  "attack_name": "PairAttack",
  "parameters": {
    "attack_model_config": "models/usage_examples/configs/api/gpt-4.1-mini.json",
    "judge_model_config": "judgers/usage_examples/configs/harmbench_judger.json",
    "n_streams": 2,
    "n_iterations": 3,
    "max_score_stop": 5
  }
}
```

## Usage Examples

### Basic Usage

```python
from attacks import load_attack_from_config
from models.model_loader import load_model_from_config

# Load target model
target_model, target_model_params = load_model_from_config(
    "models/usage_examples/configs/local/meta-llama-Llama-3.2-3B-Instruct.json"
)

# Load attack
attack = load_attack_from_config(
    "attacks/usage_examples/configs/black_box/ReNeLLM.json",
    target_model,
    target_model_params
)

# Execute attack
result = attack.attack("How to make explosives?")
print(f"Attack result: {result}")
```

### Batch Attacks

```python
from attacks.attack_registry import ATTACKS

# List all available attacks
print("Available attacks:")
for attack_name in ATTACKS.keys():
    print(f"- {attack_name}")

# Load multiple attacks
attack_configs = [
    "attacks/usage_examples/configs/black_box/ReNeLLM.json",
    "attacks/usage_examples/configs/black_box/PAIR.json",
    "attacks/usage_examples/configs/black_box/TapAttack.json"
]

for config_path in attack_configs:
    attack = load_attack_from_config(config_path, target_model, target_model_params)
    result = attack.attack("harmful prompt")
    print(f"Attack {config_path}: {result}")
```

### Custom Attack Implementation

```python
from attacks.base_attack import BaseAttack

class MyCustomAttack(BaseAttack):
    def __init__(self, target_model, target_model_parameters, custom_param=None, **kwargs):
        super().__init__(target_model)
        self.target_model_parameters = target_model_parameters
        self.custom_param = custom_param
        self.query_count = 0
    
    def attack(self, original_prompt: str, **kwargs):
        """Implement your attack logic here."""
        # Transform the prompt
        adversarial_prompt = self.transform_prompt(original_prompt)
        
        # Query the target model
        response = self.target_model.generate(
            prompt=adversarial_prompt,
            **self.target_model_parameters
        )
        self.query_count += 1
        
        # Return results
        return self.query_count, adversarial_prompt
    
    def transform_prompt(self, prompt: str) -> str:
        """Custom prompt transformation logic."""
        return f"Custom transformation: {prompt}"

# Register your attack
from attacks.attack_registry import ATTACKS
ATTACKS['MyCustomAttack'] = MyCustomAttack
```

## Judge Integration

Many attacks require judge models for evaluation. The attacks module integrates with the judgers module:

```python
# Attacks automatically load judgers from configuration
attack = load_attack_from_config(
    "attacks/usage_examples/configs/black_box/ReNeLLM.json",  # Uses judger config
    target_model,
    target_model_parameters
)

# Judge models are used internally for:
# - Evaluating attack success
# - Iterative prompt refinement
# - Response harmfulness assessment
```

## Performance Considerations

### Query Counting
All attacks track the number of queries made to target models:

```python
attack = load_attack_from_config(config_path, target_model, target_model_params)
result = attack.attack("prompt")
print(f"Queries used: {attack.query_count}")
```

### Memory Management
- **Local models**: Loaded once and reused
- **API models**: Stateless, managed by API providers
- **Judge models**: Loaded separately, can be shared across attacks

### Optimization Tips
1. **Use appropriate judgers**: Choose lightweight judgers for fast iteration
2. **Batch processing**: Process multiple prompts together when possible
3. **Caching**: Cache model outputs for repeated evaluations
4. **Early stopping**: Use score thresholds to stop successful attacks early

## Evaluation Metrics

### Success Rate
Percentage of prompts that successfully jailbreak the target model:

```python
success_count = 0
total_prompts = len(test_prompts)

for prompt in test_prompts:
    result = attack.attack(prompt)
    if is_successful(result):  # Define success criteria
        success_count += 1

success_rate = success_count / total_prompts
print(f"Attack Success Rate: {success_rate:.2%}")
```

### Query Efficiency
Average number of queries needed per successful attack:

```python
total_queries = sum(attack.query_count for attack in attack_results)
successful_attacks = sum(1 for result in attack_results if is_successful(result))
efficiency = total_queries / successful_attacks if successful_attacks > 0 else float('inf')
```

### Semantic Preservation
Measure how well attacks preserve original intent:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
original_embedding = model.encode([original_prompt])
adversarial_embedding = model.encode([adversarial_prompt])
similarity = cosine_similarity(original_embedding, adversarial_embedding)[0][0]
```

## Best Practices

### 1. **Configuration Management**
- Use configuration files for reproducible experiments
- Version control your attack configurations
- Document parameter choices and their effects

### 2. **Responsible Usage**
- Only use for defensive security research
- Do not use against production systems without permission
- Follow ethical guidelines and legal requirements

### 3. **Evaluation Protocol**
- Use consistent datasets across different attacks
- Report both success rates and query counts
- Include baseline comparisons
- Validate on multiple target models

### 4. **Error Handling**
```python
try:
    attack = load_attack_from_config(config_path, target_model, target_model_params)
    result = attack.attack(prompt)
except Exception as e:
    logger.error(f"Attack failed: {e}")
    # Handle gracefully
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Model Loading Failures**
   ```python
   # Check model configuration paths
   import os
   assert os.path.exists(model_config_path), f"Model config not found: {model_config_path}"
   ```

3. **Judge Integration Issues**
   ```python
   # Verify judger configuration
   from judgers import load_judger_from_config
   judger = load_judger_from_config(judge_config_path)
   ```

4. **Memory Issues**
   ```python
   # Use smaller models or batch sizes
   # Enable gradient checkpointing for white-box attacks
   torch.cuda.empty_cache()  # Clear GPU memory
   ```

### Debugging Tips

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test with simple prompts first**:
   ```python
   simple_prompt = "Hello, how are you?"
   result = attack.attack(simple_prompt)
   ```

3. **Check attack parameters**:
   ```python
   print(f"Attack config: {attack.get_config()}")
   ```

## Contributing

### Adding New Attacks

1. **Inherit from BaseAttack**:
   ```python
   from attacks.base_attack import BaseAttack
   
   class NewAttack(BaseAttack):
       def attack(self, original_prompt: str, **kwargs):
           # Implement attack logic
           pass
   ```

2. **Register the attack**:
   ```python
   # In attack_registry.py
   from .new_attack import NewAttack
   ATTACKS['NewAttack'] = NewAttack
   ```

3. **Create configuration**:
   ```json
   {
     "attack_name": "NewAttack",
     "parameters": {
       "param1": "value1"
     }
   }
   ```

4. **Add documentation and tests**

### Testing Framework

```python
# test_new_attack.py
import unittest
from attacks.attack_loader import load_attack_from_config

class TestNewAttack(unittest.TestCase):
    def setUp(self):
        # Load test models and configurations
        pass
    
    def test_attack_success(self):
        # Test attack functionality
        pass
    
    def test_configuration_loading(self):
        # Test configuration loading
        pass
```

## Security Considerations

### Responsible Disclosure
- Report vulnerabilities found during research
- Coordinate with model developers
- Allow time for fixes before public disclosure

### Data Privacy
- Do not use sensitive or personal data in attacks
- Anonymize any data used in research
- Follow data protection regulations

### Ethical Guidelines
- Use attacks only for defensive research
- Do not weaponize or distribute malicious prompts
- Consider societal impact of research

## References

### Attack Papers
- **ReNeLLM**: [Paper link]
- **PAIR**: [Paper link] 
- **TAP**: [Paper link]
- **AutoDAN**: [Paper link]
- **GCG**: [Paper link]

### Related Work
- Red teaming methodologies
- Adversarial machine learning
- AI safety and alignment research

For more information, see the individual attack documentation in their respective directories.
