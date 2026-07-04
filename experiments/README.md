# PromptSecurity 统一实验系统

一个简洁、强大的AI安全评估实验系统，提供统一接口访问所有模块的所有方法。

## 🚀 核心特性

- **统一接口**：单一函数访问200+种攻击、防御、模型、评判器和数据集组合
- **模块自动发现**：自动扫描并注册所有可用组件，无需手动维护
- **智能兼容性检查**：自动验证白盒/黑盒方法与模型的兼容性
- **灵活配置管理**：支持字典、JSON文件、YAML文件、环境变量、命令行等多种输入方式
- **分阶段实验支持**：内置Phase 1-4实验框架
- **向后兼容**：保持原有接口可用，平滑升级

## 📦 可用方法统计

| 组件类型 | 数量 | 说明 |
|---------|------|------|
| 攻击方法 | 20个 | 17个黑盒攻击 + 3个白盒攻击 |
| 防御方法 | 10个 | 输入/输出过滤 + 模型基础 + 白盒防御 |
| 模型配置 | 161个 | 86个API模型 + 75个本地模型 |
| 评判器 | 7个 | 多种评估标准 |
| 数据集 | 3个 | harmbench, jbb, airbench |
| **总计** | **200+** | **所有组合自动可用** |

## 🔧 快速开始

### 1. 最简单的使用方式

```python
import experiments

# 基础安全评估
result = experiments.run_experiment(
    model="gpt-4o",
    attack="ArtPrompt",
    defense="smooth_llm"
)

print(f"攻击成功率: {result['attack_success_rate']:.1%}")
print(f"基线安全率: {result['clean_safe_rate']:.1%}")
```

### 2. 查看可用方法

```python
# 列出所有可用方法
methods = experiments.list_methods()
print(f"可用攻击: {methods['attacks']['black_box']}")
print(f"可用防御: {methods['defenses']['all']}")
print(f"可用模型: {len(methods['models']['all'])} 个")

# 列出特定类型
attacks = experiments.list_methods("attacks")
print(f"黑盒攻击: {attacks['black_box']}")
print(f"白盒攻击: {attacks['white_box']}")
```

### 3. 兼容性检查

```python
# 检查白盒攻击的兼容模型
compatible = experiments.get_compatible("GCGAttack", "attack")
print(f"兼容的本地模型: {len(compatible['models'])} 个")

# 检查API模型的兼容攻击
compatible = experiments.get_compatible("gpt-4o", "model")
print(f"兼容的攻击方法: {len(compatible['attacks'])} 个")
```

## 📋 详细使用方法

### Python API使用

#### 方式1: 直接参数

```python
import experiments

# 基础评估
result = experiments.run_experiment(
    model="gpt-4o",
    attack="no_attack",
    defense="no_defense",
    dataset="harmbench",
    judger="harmbench_judger",
    sample_limit=50
)
```

#### 方式2: 字典配置

```python
config = {
    "model": "claude-3-5-sonnet-latest",
    "attack": "GPTFUZZER", 
    "defense": "jailguard_defense",
    "dataset": "jbb",
    "judger": "gpt_judger_harmful_binary",
    "sample_limit": 100,
    "experiment_name": "我的安全评估实验"
}

result = experiments.run_experiment(config)
```

#### 方式3: 配置文件

```python
# 支持JSON和YAML格式
result = experiments.run_experiment("my_experiment.json")
result = experiments.run_experiment("my_experiment.yaml")
```

**配置文件示例 (JSON)**:
```json
{
  "model": "gpt-4o",
  "attack": "ArtPrompt",
  "defense": "smooth_llm", 
  "dataset": "harmbench",
  "judger": "harmbench_judger",
  "sample_limit": 100,
  "experiment_name": "ArtPrompt攻击测试"
}
```

**配置文件示例 (YAML)**:
```yaml
model: gpt-4o
attack: ArtPrompt
defense: smooth_llm
dataset: harmbench
judger: harmbench_judger
sample_limit: 100
experiment_name: ArtPrompt攻击测试
```

#### 方式4: 环境变量

```bash
export PS_MODEL=gpt-4o
export PS_ATTACK=ArtPrompt
export PS_DEFENSE=smooth_llm
export PS_SAMPLE_LIMIT=50
```

```python
import experiments
result = experiments.run_experiment()  # 自动使用环境变量
```

### 命令行使用

#### 基础实验

```bash
# 最简单的使用
python -m experiments --model gpt-4o --attack ArtPrompt

# 完整参数
python -m experiments \
  --model claude-3-5-sonnet-latest \
  --attack GPTFUZZER \
  --defense smooth_llm \
  --dataset harmbench \
  --judger gpt_judger_harmful_binary \
  --sample-limit 100
```

#### 使用配置文件

```bash
python -m experiments --config my_experiment.json
python -m experiments --config my_experiment.yaml
```

#### 查看可用方法

```bash
# 列出所有攻击方法
python -m experiments --list attacks

# 列出所有防御方法  
python -m experiments --list defenses

# 列出所有模型
python -m experiments --list models

# 列出所有方法
python -m experiments --list all
```

#### Phase实验

```bash
# 运行Phase 1模型评估
python -m experiments --phase 1 --sample-limit 30

# 运行Phase 2评判器一致性评估  
python -m experiments --phase 2

# 自定义Phase配置
python -m experiments --phase 1 \
  --experiment-name "自定义Phase1" \
  --sample-limit 50
```

#### 其他功能

```bash
# 显示使用示例
python -m experiments --show-examples

# 查看方法详细信息
python -m experiments --info ArtPrompt attack

# 保存结果到文件
python -m experiments --model gpt-4o --attack ArtPrompt --output result.json
```

## 🔄 批量实验

### Python API批量实验

```python
interface = experiments.get_interface()

# 定义多个配置进行对比
configs = [
    {"model": "gpt-4o", "attack": "no_attack", "defense": "no_defense"},
    {"model": "gpt-4o", "attack": "ArtPrompt", "defense": "no_defense"},  
    {"model": "gpt-4o", "attack": "ArtPrompt", "defense": "smooth_llm"}
]

results = interface.run_batch_experiments(configs, sample_limit=20)

# 对比结果
for i, result in enumerate(results):
    attack = result['attack_method']
    defense = result['defense_method'] 
    safe_rate = result['clean_safe_rate']
    print(f"实验{i+1}: {attack}+{defense} -> 安全率:{safe_rate:.1%}")
```

### 配置文件批量实验

**批量配置文件示例**:
```json
{
  "experiment_type": "batch",
  "batch_configs": [
    {
      "model": "gpt-4o",
      "attack": "no_attack", 
      "defense": "no_defense"
    },
    {
      "model": "gpt-4o",
      "attack": "ArtPrompt",
      "defense": "no_defense"
    },
    {
      "model": "gpt-4o", 
      "attack": "ArtPrompt",
      "defense": "smooth_llm"
    }
  ],
  "sample_limit": 20,
  "experiment_name": "攻击防御对比实验"
}
```

```bash
python -m experiments --config batch_experiment.json
```

## 🏗️ Phase实验系统

### Phase 1: 模型代表性评估

```python
# Python API
interface = experiments.get_interface()
result = interface.run_phase_experiment(
    phase=1,
    target_selection_count=10,
    sample_limit=30,
    datasets=["harmbench", "jbb"],
    judgers=["harmbench_judger", "gpt_judger_contextual_harmbench"]
)

print(f"选择的代表性模型: {result['selected_models']}")
print(f"选择比例: {result['statistics']['selection_rate']:.1%}")
```

```bash
# 命令行
python -m experiments --phase 1 --sample-limit 30
```

### Phase 2-4: 即将支持

- **Phase 2**: 评判器一致性评估
- **Phase 3**: 数据集优化  
- **Phase 4**: 全面攻击防御评估

## ⚡ 高级功能

### 方法信息查询

```python
interface = experiments.get_interface()

# 查询攻击方法详细信息
info = interface.get_method_info("ArtPrompt", "attacks") 
print(f"配置文件: {info['config_path']}")
print(f"配置参数: {info['config']}")

# 查询防御方法信息
info = interface.get_method_info("smooth_llm", "defenses")
```

### 配置验证

```python
from experiments.config_manager import get_config_manager

config_manager = get_config_manager()
config = config_manager.load_config(["my_config.json"])

# 验证配置有效性
validation = config_manager.validate_config(config)
if not validation["valid"]:
    print(f"配置错误: {validation['errors']}")
```

### 刷新方法列表

```python
interface = experiments.get_interface()

# 添加新的攻击/防御方法后，刷新注册表
interface.refresh_methods()
```

## 🔧 扩展和自定义

### 添加新的攻击/防御方法

1. 在相应目录添加实现文件
2. 在`usage_examples/configs/`添加配置文件
3. 系统自动发现新方法，无需修改代码

### 自定义Phase实验

```python
from experiments.framework import ExperimentFramework

framework = ExperimentFramework()

# 创建自定义实验
class MyCustomExperiment(BaseExperiment):
    def execute(self):
        # 自定义实验逻辑
        pass

framework.register_experiment("my_experiment", MyCustomExperiment)
```

## 🐛 错误处理

### 常见错误和解决方案

**1. 配置验证失败**
```bash
❌ 配置验证失败: 白盒攻击 GCGAttack 只能用于本地模型，当前模型 gpt-4o 是API模型
```
解决方案：使用本地模型或改用黑盒攻击

**2. 未知方法**
```bash
❌ 未知攻击方法: MyAttack. 可用方法: ['ArtPrompt', 'FlipAttack', ...]
```
解决方案：检查方法名拼写，或使用`--list`查看可用方法

**3. 配置文件不存在**
```bash
❌ 配置文件未找到: my_config.json
```
解决方案：确认文件路径正确

## 📊 性能优化

### 并行执行

```bash
# 并行执行批量实验
python -m experiments --config batch.json --parallel --workers 4
```

### 样本限制

```python
# 快速测试：限制样本数
result = experiments.run_experiment(
    model="gpt-4o",
    attack="ArtPrompt", 
    sample_limit=10  # 仅处理10个样本
)
```

## 🔄 向后兼容

原有接口继续可用：

```python
# 原有方式仍然支持
from experiments.evaluation import run_evaluation
result = run_evaluation(model="gpt-4o", attack="ArtPrompt")

# 原有框架仍然支持  
from experiments.framework import ExperimentFramework
framework = ExperimentFramework()
phase1 = framework.create_phase1_experiment()
```

## 📚 更多示例

查看完整示例：

```bash
# 运行完整功能演示
python experiments/examples/unified_interface_demo.py

# 运行快速测试
python experiments/examples/quick_test.py

# 查看基础使用示例
python experiments/examples/basic_evaluation.py

# 查看Phase实验示例
python experiments/examples/phase1_experiment.py
```

## 🏗 系统架构

### 新架构 (v2.0+)

```
experiments/
├── __init__.py                 # 统一导出接口
├── __main__.py                 # 命令行入口
├── unified_interface.py        # 核心统一接口
├── module_registry.py          # 模块注册和自动发现
├── config_manager.py           # 配置管理系统
├── evaluation/                 # 🎯 通用5要素评估管道
│   ├── __init__.py            # 导出核心函数
│   ├── pipeline.py            # 核心评估逻辑
│   └── config.py              # 配置管理
├── framework/                  # 🔧 可扩展实验框架
│   ├── __init__.py            # 导出框架类
│   ├── base.py                # 基础实验类
│   └── phase1.py              # Phase 1模型评估
├── examples/                   # 📚 使用示例
│   ├── basic_evaluation.py    # 基础评估示例
│   ├── phase1_experiment.py   # Phase 1实验示例
│   ├── unified_interface_demo.py # 统一接口演示
│   └── quick_test.py          # 快速测试
├── results/                    # 📊 实验结果存储目录
├── logs/                      # 📝 日志文件目录
└── README.md                   # 本文档
```

### 5要素说明

PromptSecurity使用5要素评估框架：

| 要素 | 说明 | 示例值 |
|------|------|--------|
| **Model** | 目标评估模型 | `"gpt-4o"`, `"Llama-3.1-8B-Instruct"` |
| **Attack** | 攻击方法 | `"ArtPrompt"`, `"FlipAttack"`, `"no_attack"` |
| **Defense** | 防御机制 | `"smooth_llm"`, `"input_filter"`, `"no_defense"` |
| **Dataset** | 评估数据集 | `"harmbench"`, `"jbb"`, `"airbench"` |
| **Judger** | 安全评判器 | `"harmbench_judger"`, `"gpt_judger_contextual_harmbench"` |

## 📊 结果格式与保存

### 自动结果保存

**新功能**: 实验结果现在会自动保存，无需手动指定输出文件！

- **自动保存路径**: `experiments/results/`
- **文件命名格式**: `{实验类型}_{模型名}_{攻击方法}_{时间戳}.json`
- **示例文件名**: `evaluation_gpt-4o_ArtPrompt_20250714_143025.json`

```bash
# 运行实验后自动保存
python -m experiments --model gpt-4o --attack ArtPrompt --sample-limit 100
# ✅ 结果已保存到: experiments/results/evaluation_gpt-4o_ArtPrompt_20250714_143025.json

# 仍可手动指定输出路径
python -m experiments --model gpt-4o --attack ArtPrompt --output my_result.json
```

```python
# Python API也支持自动保存
result = experiments.run_experiment(
    model="gpt-4o",
    attack="ArtPrompt",
    sample_limit=100
)
# 结果已自动保存到 experiments/results/

# 禁用自动保存
result = experiments.run_experiment(
    model="gpt-4o", 
    attack="ArtPrompt",
    auto_save=False  # 禁用自动保存
)
```

### 结果格式

所有评估结果都遵循统一格式，包含以下关键字段：

```python
{
    "target_llm_name": "gpt-4o",
    "attack_method": "ArtPrompt",
    "defense_method": "no_defense",
    "dataset_name": "harmbench",
    "judger_name": "harmbench_judger",
    
    "clean_safe_rate": 0.85,      # 基线安全率
    "attack_success_rate": 0.23,   # 攻击成功率
    "total_samples": 100,          # 总样本数
    "successful_samples": 98,      # 成功处理样本数
    
    "execution_time": 45.2,        # 执行时间（秒）
    "status": "completed"          # 执行状态
}
```

## 🚨 注意事项

### 1. 模型兼容性

- **白盒攻击** (`AutoDANAttack`, `GCGAttack`, `COLDAttack`) 只能用于本地模型
- **白盒防御** (`perplexity_filter`, `gradsafe_defense`, `rpo`) 只能用于本地模型
- 系统会自动检查兼容性并给出警告

### 2. 资源管理

- **API模型**: 注意速率限制和成本控制
- **本地模型**: 注意GPU内存使用
- 建议先用小样本测试，再进行大规模实验

### 3. 结果可靠性

- 使用固定随机种子保证可重现性
- 建议使用足够的样本数量（≥50）
- 多次运行验证结果稳定性

## 🔄 扩展计划

### Phase 2: 评判器一致性分析
- 评估不同评判器的一致性
- 选择最可靠的评判器

### Phase 3: 数据集优化
- 分析样本质量
- 构建优化的评估数据集

### Phase 4: 全面攻击防御评估
- 基于前期结果的全面评估
- 生成综合分析报告

## 🤝 贡献指南

1. 添加新方法时，请在相应的`usage_examples/configs/`目录添加配置文件
2. 确保配置文件遵循现有的JSON格式规范
3. 系统会自动发现新方法，无需修改注册代码
4. 运行测试确保兼容性：`python experiments/examples/quick_test.py`

## 📞 技术支持

- 查看使用示例：`python -m experiments --show-examples`
- 列出可用方法：`python -m experiments --list all`
- 查看方法信息：`python -m experiments --info <METHOD_NAME> <TYPE>`
- 遇到问题请查看详细日志：`python -m experiments --verbose`

---

**版本**: 2.0.0  
**更新日期**: 2025-07-14  
**维护者**: PromptSecurity Team