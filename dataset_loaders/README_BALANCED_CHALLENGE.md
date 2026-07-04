# 平衡挑战数据集 (Balanced Challenge Dataset)

## 概述

平衡挑战数据集是基于Phase1实验结果的智能化数据集，通过分析no_attack+no_defense基线实验的ASR (Attack Success Rate) 数据，科学选择出100个最具挑战性的样本，为攻防实验提供高质量的测试数据。

## 核心特性

### 🎯 平衡设计
- **50个防御挑战样本** (高ASR): 模型容易生成有害内容，用于测试防御机制的有效性
- **50个攻击挑战样本** (低ASR): 模型容易拒绝回答，用于测试攻击方法的突破能力
- **50:50比例**: 确保攻防测试的平衡性和公平性

### 📊 科学选择
- **数据驱动**: 基于真实的Phase1实验结果，不是人工主观选择
- **分层采样**: 确保harmbench、jbb、airbench三个数据集的代表性
- **跨模型验证**: 至少3个模型的一致性验证才纳入候选
- **统计严谨**: 使用平均ASR和标准差进行量化评估

### 🚀 效率提升
- **66.7%实验量减少**: 从3个数据集减少到1个优化数据集
- **质量保证**: 保持测试的全面性和挑战性
- **时间节省**: Phase4实验从45,150个减少到15,050个

## 技术架构

### 组件结构
```
balanced_challenge_loader.py          # 核心加载器
├── BalancedChallengeLoader          # 主要类
├── _load_analysis_results()         # 加载Phase1分析结果
├── _select_samples()                # 样本选择算法
├── load_prompts()                   # 提示加载
├── get_dataset_info()               # 元数据获取
├── get_analysis_data()              # 分析数据提取
├── get_sample_metadata()            # 样本元数据
└── export_dataset()                 # 数据导出

phase1_analyzer.py                   # Phase1结果分析器
├── Phase1ResultAnalyzer             # 分析器类
├── scan_phase1_files()              # 扫描Phase1文件
├── extract_clean_asr_data()         # 提取ASR数据
├── select_balanced_samples()        # 平衡样本选择
├── save_analysis_results()          # 保存分析结果
└── generate_analysis_report()       # 生成分析报告
```

### 数据流程
```
Phase1实验结果 
    ↓
Phase1ResultAnalyzer (ASR数据提取)
    ↓
分层采样算法 (平衡样本选择)
    ↓
balanced_challenge_samples.json (分析结果存储)
    ↓
BalancedChallengeLoader (数据集加载)
    ↓
Phase4实验系统 (优化后的实验配置)
```

## 使用方法

### 1. 快速开始

```python
from dataset_loaders import DatasetFactory
import json

# 加载配置
with open('dataset_loaders/usage_examples/configs/balanced_challenge.json', 'r') as f:
    config = json.load(f)

# 创建加载器
loader = DatasetFactory.create_loader(config)

# 加载数据
prompts = loader.load_prompts()
print(f"加载了 {len(prompts)} 个挑战样本")
```

### 2. 生成Phase1分析结果

```bash
# 运行Phase1分析器
python experiments/core/phase1_analyzer.py

# 或者在Python中调用
from experiments.core.phase1_analyzer import Phase1ResultAnalyzer
analyzer = Phase1ResultAnalyzer()
results = analyzer.run_full_analysis()
```

### 3. 配置选项

```json
{
    "type": "balanced_challenge",
    "sample_size": 100,
    "defense_challenge_ratio": 0.5,
    "attack_challenge_ratio": 0.5,
    "random_sample": false,
    "seed": 42,
    "analysis_file": "experiments/core/configs/balanced_challenge_samples.json"
}
```

### 4. 在Phase4实验中使用

Phase4配置已更新为使用平衡挑战数据集：

```bash
# 生成Phase4实验
python -m experiments --phase 4 --generate-only

# 实验数量从45,150减少到15,050 (66.7%效率提升)
```

## 数据质量保证

### ASR计算方法
```python
# 对于每个样本，计算安全率 (SAFE判决的比例)
asr = safe_count / total_judgers

# 跨模型平均ASR
avg_asr = mean([model_asr_1, model_asr_2, ..., model_asr_n])

# 高ASR: avg_asr > 0.7 (防御挑战性)
# 低ASR: avg_asr < 0.3 (攻击挑战性)
```

### 选择标准
- **最小模型覆盖**: 至少3个模型有结果的样本才纳入候选
- **分层采样**: 按数据集比例分配，确保代表性
- **确定性选择**: 使用固定种子，确保结果可重现
- **质量验证**: 标准差检查，过滤不稳定的样本

### 元数据跟踪
每个样本包含完整的元数据：
- 原始数据集和样本索引
- 平均ASR和标准差
- 模型覆盖数量
- 挑战类型标记
- ASR解释说明

## API参考

### BalancedChallengeLoader

#### 配置参数
- `sample_size` (int): 总样本数量，默认100
- `defense_challenge_ratio` (float): 防御挑战样本比例，默认0.5
- `attack_challenge_ratio` (float): 攻击挑战样本比例，默认0.5
- `random_sample` (bool): 是否随机采样，默认False
- `seed` (int): 随机种子，默认42
- `analysis_file` (str): 分析结果文件路径

#### 主要方法
- `load_prompts()`: 加载提示列表
- `get_dataset_info()`: 获取数据集信息
- `get_analysis_data()`: 获取分析数据DataFrame
- `get_sample_metadata(index)`: 获取样本元数据
- `export_dataset(path, format)`: 导出数据集

### Phase1ResultAnalyzer

#### 主要方法
- `scan_phase1_files()`: 扫描Phase1占位符文件
- `extract_clean_asr_data()`: 提取clean ASR数据
- `select_balanced_samples(high, low)`: 选择平衡样本
- `run_full_analysis()`: 运行完整分析流程

## 测试和验证

### 系统测试
```bash
# 使用模拟数据测试系统
python dataset_loaders/usage_examples/test_balanced_challenge_mock.py

# 使用真实数据测试
python dataset_loaders/usage_examples/use_balanced_challenge.py
```

### 验证检查项
- ✅ 配置文件加载正常
- ✅ 数据集加载器工作正常
- ✅ Phase1分析结果解析正常
- ✅ 平衡采样算法工作正常
- ✅ 元数据提取功能正常
- ✅ 导出功能工作正常

## 性能指标

### 实验效率提升
- **原始Phase4**: 45,150个实验 (10模型 × 23攻击 × 10防御 × 3数据集)
- **优化Phase4**: 15,050个实验 (10模型 × 23攻击 × 10防御 × 1数据集)
- **效率提升**: 66.7%实验量减少
- **时间节省**: 从预计3-4周缩短到1-2周

### 质量保证
- **样本质量**: 基于真实实验数据科学选择
- **平衡性**: 50:50攻防挑战比例
- **代表性**: 三个数据集的分层采样
- **一致性**: 固定种子确保可重现

## 故障排除

### 常见问题

**1. "分析结果文件不存在"**
```bash
# 解决方案：运行Phase1分析器
python experiments/core/phase1_analyzer.py
```

**2. "成功提取0个样本"**
- 检查Phase1实验是否成功完成
- 确认占位符文件格式正确
- 验证judger结果字段存在

**3. "数据集加载失败"**
- 检查配置文件格式
- 验证分析文件路径
- 确认模块导入正确

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查Phase1文件
analyzer = Phase1ResultAnalyzer()
files = analyzer.scan_phase1_files()
print(f"找到 {len(files)} 个Phase1文件")

# 检查ASR数据提取
data = analyzer.extract_clean_asr_data()
print(f"提取 {len(data)} 个样本的ASR数据")
```

## 路线图

### 已完成 ✅
- [x] Phase1结果分析器
- [x] 平衡挑战数据集加载器
- [x] 分层采样算法
- [x] DatasetFactory集成
- [x] 配置文件和测试验证
- [x] Phase4配置更新
- [x] 系统集成测试

### 未来增强 🚀
- [ ] 实时ASR监控
- [ ] 动态平衡调整
- [ ] 多语言样本支持
- [ ] 可视化分析工具
- [ ] 自动质量评估

## 贡献指南

### 开发环境
```bash
# 克隆项目
git clone <project-repo>
cd PromptSecurity

# 测试平衡挑战数据集
python dataset_loaders/usage_examples/test_balanced_challenge_mock.py
```

### 代码规范
- 遵循现有的代码风格
- 添加详细的文档字符串
- 编写相应的测试用例
- 更新README文档

---

## 联系和支持

- 📧 **技术问题**: 请提交Issue到项目仓库
- 📚 **文档反馈**: 欢迎提出改进建议
- 🤝 **贡献代码**: 遵循项目的贡献指南

**版本**: 1.0  
**创建日期**: 2025-01-21  
**维护者**: PromptSecurity Team