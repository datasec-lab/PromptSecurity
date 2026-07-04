"""
Balanced Challenge Dataset Loader
================================

基于Phase1实验结果创建的平衡挑战数据集加载器。
包含50个高ASR样本(防御挑战性)和50个低ASR样本(攻击挑战性)。
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from rich.console import Console

from .base_dataset_loader import BaseDatasetLoader

logger = logging.getLogger(__name__)


class BalancedChallengeLoader(BaseDatasetLoader):
    """平衡挑战数据集加载器"""
    
    def __init__(self, config: Dict[str, Any], console: Optional[Console] = None):
        """
        初始化平衡挑战数据集加载器
        
        Args:
            config: 数据集配置
            console: Rich控制台
        """
        super().__init__(config, console)
        
        # 配置参数
        self.sample_size = config.get('sample_size', 100)
        self.defense_challenge_ratio = config.get('defense_challenge_ratio', 0.5)
        self.attack_challenge_ratio = config.get('attack_challenge_ratio', 0.5)
        self.random_sample = config.get('random_sample', False)
        self.seed = config.get('seed', 42)
        
        # 分析结果文件路径
        self.analysis_file = config.get(
            'analysis_file', 
            'experiments/core/configs/balanced_challenge_samples.json'
        )
        
        # 内部数据
        self.balanced_samples = []
        self.high_asr_samples = []  # 防御挑战性样本
        self.low_asr_samples = []   # 攻击挑战性样本
        
        # 设置随机种子
        random.seed(self.seed)
        
    def _load_analysis_results(self) -> Dict[str, Any]:
        """
        加载Phase1分析结果
        
        Returns:
            分析结果字典
        """
        analysis_path = Path(self.analysis_file)
        
        if not analysis_path.exists():
            # 如果分析结果不存在，尝试运行分析
            logger.warning(f"分析结果文件不存在: {analysis_path}")
            logger.info("尝试运行Phase1分析...")
            
            try:
                from experiments.core.phase1_analyzer import Phase1ResultAnalyzer
                analyzer = Phase1ResultAnalyzer()
                results = analyzer.run_full_analysis(save_results=True)
                
                if not results:
                    raise ValueError("Phase1分析失败")
                    
                return results
                
            except Exception as e:
                logger.error(f"无法运行Phase1分析: {e}")
                raise ValueError(f"分析结果文件不存在且无法生成: {analysis_path}")
        
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logger.info(f"成功加载分析结果: {analysis_path}")
            return results
            
        except Exception as e:
            logger.error(f"无法加载分析结果文件 {analysis_path}: {e}")
            raise
    
    def _select_samples(self, all_samples: List[Dict], target_count: int) -> List[Dict]:
        """
        从样本列表中选择指定数量的样本
        
        Args:
            all_samples: 所有候选样本
            target_count: 目标样本数量
            
        Returns:
            选择的样本列表
        """
        if len(all_samples) <= target_count:
            return all_samples
        
        if self.random_sample:
            return random.sample(all_samples, target_count)
        else:
            # 使用确定性选择，按ASR排序确保一致性
            return all_samples[:target_count]
    
    def load_prompts(self) -> List[str]:
        """
        加载平衡挑战数据集的提示
        
        Returns:
            提示字符串列表
        """
        if self.prompts:
            return self.prompts
        
        try:
            # 加载分析结果
            analysis_results = self._load_analysis_results()
            
            high_asr_samples = analysis_results.get('high_asr_samples', [])
            low_asr_samples = analysis_results.get('low_asr_samples', [])
            
            # 计算实际的样本数量分配
            total_available = len(high_asr_samples) + len(low_asr_samples)
            actual_sample_size = min(self.sample_size, total_available)
            
            defense_target = int(actual_sample_size * self.defense_challenge_ratio)
            attack_target = actual_sample_size - defense_target
            
            # 选择样本
            selected_high_asr = self._select_samples(high_asr_samples, defense_target)
            selected_low_asr = self._select_samples(low_asr_samples, attack_target)
            
            # 存储选择的样本用于元数据
            self.high_asr_samples = selected_high_asr
            self.low_asr_samples = selected_low_asr
            self.balanced_samples = selected_high_asr + selected_low_asr
            
            # 提取提示文本
            prompts = []
            for sample in self.balanced_samples:
                prompt = sample.get('prompt', '')
                if prompt:
                    prompts.append(prompt)
            
            # 混洗顺序确保随机性
            if self.random_sample:
                random.shuffle(prompts)
            
            self.prompts = prompts
            
            self.console.print(f"📊 成功加载平衡挑战数据集:")
            self.console.print(f"   防御挑战样本: {len(selected_high_asr)} 个 (高ASR)")
            self.console.print(f"   攻击挑战样本: {len(selected_low_asr)} 个 (低ASR)")
            self.console.print(f"   总样本数: {len(prompts)} 个")
            
            return prompts
            
        except Exception as e:
            logger.error(f"加载平衡挑战数据集失败: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Returns:
            数据集元数据字典
        """
        if not self.balanced_samples:
            self.load_prompts()
        
        # 统计信息
        high_asr_count = len(self.high_asr_samples)
        low_asr_count = len(self.low_asr_samples)
        
        # 按数据集统计
        dataset_distribution = {}
        for sample in self.balanced_samples:
            dataset = sample.get('dataset', 'unknown')
            challenge_type = sample.get('challenge_type', 'unknown')
            
            if dataset not in dataset_distribution:
                dataset_distribution[dataset] = {'defense_challenge': 0, 'attack_challenge': 0}
            
            if challenge_type == 'high_asr':
                dataset_distribution[dataset]['defense_challenge'] += 1
            elif challenge_type == 'low_asr':
                dataset_distribution[dataset]['attack_challenge'] += 1
        
        # ASR统计
        high_asr_values = [s.get('avg_asr', 0) for s in self.high_asr_samples]
        low_asr_values = [s.get('avg_asr', 0) for s in self.low_asr_samples]
        
        return {
            'name': 'balanced_challenge',
            'description': 'Phase1结果驱动的平衡挑战数据集 - 攻防双重测试',
            'version': '1.0',
            'total_samples': len(self.balanced_samples),
            'composition': {
                'defense_challenge_samples': high_asr_count,
                'attack_challenge_samples': low_asr_count,
                'defense_challenge_ratio': high_asr_count / len(self.balanced_samples) if self.balanced_samples else 0,
                'attack_challenge_ratio': low_asr_count / len(self.balanced_samples) if self.balanced_samples else 0
            },
            'asr_statistics': {
                'high_asr_mean': sum(high_asr_values) / len(high_asr_values) if high_asr_values else 0,
                'low_asr_mean': sum(low_asr_values) / len(low_asr_values) if low_asr_values else 0,
                'high_asr_range': [min(high_asr_values), max(high_asr_values)] if high_asr_values else [0, 0],
                'low_asr_range': [min(low_asr_values), max(low_asr_values)] if low_asr_values else [0, 0]
            },
            'dataset_distribution': dataset_distribution,
            'source_analysis': {
                'analysis_file': self.analysis_file,
                'selection_method': 'random' if self.random_sample else 'deterministic',
                'seed': self.seed
            },
            'challenge_types': {
                'defense_challenge': '高ASR样本 - 模型容易生成有害内容，测试防御能力',
                'attack_challenge': '低ASR样本 - 模型容易拒绝回答，测试攻击突破能力'
            }
        }
    
    def get_analysis_data(self) -> Optional[pd.DataFrame]:
        """
        获取原始数据集用于分析
        
        Returns:
            包含样本详细信息的DataFrame
        """
        if not self.balanced_samples:
            self.load_prompts()
        
        if not self.balanced_samples:
            return None
        
        # 构建DataFrame
        data_rows = []
        for i, sample in enumerate(self.balanced_samples):
            row = {
                'index': i,
                'sample_key': sample.get('sample_key', f'unknown_{i}'),
                'dataset': sample.get('dataset', 'unknown'),
                'prompt': sample.get('prompt', ''),
                'challenge_type': sample.get('challenge_type', 'unknown'),
                'avg_asr': sample.get('avg_asr', 0.0),
                'std_asr': sample.get('std_asr', 0.0),
                'model_count': sample.get('model_count', 0),
                'sample_index': sample.get('sample_index', i)
            }
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        return df
    
    def get_sample_metadata(self, sample_index: int) -> Dict[str, Any]:
        """
        获取特定样本的元数据
        
        Args:
            sample_index: 样本索引
            
        Returns:
            样本元数据
        """
        if not self.balanced_samples or sample_index >= len(self.balanced_samples):
            return {}
        
        sample = self.balanced_samples[sample_index]
        return {
            'sample_key': sample.get('sample_key'),
            'original_dataset': sample.get('dataset'),
            'challenge_type': sample.get('challenge_type'),
            'avg_asr': sample.get('avg_asr'),
            'asr_interpretation': (
                '防御挑战性 - 模型容易生成有害内容' if sample.get('challenge_type') == 'high_asr'
                else '攻击挑战性 - 模型容易拒绝回答'
            ),
            'model_count': sample.get('model_count'),
            'phase1_sample_index': sample.get('sample_index')
        }
    
    def export_dataset(self, output_path: str, format: str = 'json') -> None:
        """
        导出数据集到文件
        
        Args:
            output_path: 输出文件路径
            format: 导出格式 ('json', 'csv', 'jsonl')
        """
        if not self.balanced_samples:
            self.load_prompts()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            export_data = {
                'dataset_info': self.get_dataset_info(),
                'samples': [
                    {
                        'index': i,
                        'prompt': prompt,
                        'metadata': self.get_sample_metadata(i)
                    }
                    for i, prompt in enumerate(self.prompts)
                ]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
        elif format.lower() == 'csv':
            df = self.get_analysis_data()
            if df is not None:
                df.to_csv(output_file, index=False, encoding='utf-8')
                
        elif format.lower() == 'jsonl':
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, prompt in enumerate(self.prompts):
                    line = {
                        'index': i,
                        'prompt': prompt,
                        'metadata': self.get_sample_metadata(i)
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        logger.info(f"数据集已导出到: {output_file}")


def main():
    """测试函数"""
    config = {
        'type': 'balanced_challenge',
        'sample_size': 100,
        'defense_challenge_ratio': 0.5,
        'attack_challenge_ratio': 0.5,
        'random_sample': False,
        'seed': 42
    }
    
    console = Console()
    loader = BalancedChallengeLoader(config, console)
    
    try:
        prompts = loader.load_prompts()
        info = loader.get_dataset_info()
        
        console.print(f"\n📋 数据集测试结果:")
        console.print(f"加载的提示数量: {len(prompts)}")
        console.print(f"数据集信息: {info['name']}")
        console.print(f"防御挑战样本: {info['composition']['defense_challenge_samples']}")
        console.print(f"攻击挑战样本: {info['composition']['attack_challenge_samples']}")
        
        # 显示几个示例
        console.print(f"\n前3个样本示例:")
        for i in range(min(3, len(prompts))):
            metadata = loader.get_sample_metadata(i)
            console.print(f"{i+1}. [{metadata['challenge_type']}] {prompts[i][:100]}...")
        
        console.print("✅ 平衡挑战数据集加载器测试成功!")
        
    except Exception as e:
        console.print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main()