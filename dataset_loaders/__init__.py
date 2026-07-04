"""
Dataset Loaders Module
======================

This module provides dataset loaders for various evaluation datasets including:
- HarmBench
- JBB (Jailbreak Bench)
- AirBench
- Balanced Challenge (Phase1-driven balanced dataset)
- Custom datasets

Each loader implements the BaseDatasetLoader interface for consistency.

Usage:
    from dataset_loaders import DatasetFactory
    
    # Create a dataset loader
    config = {
        'type': 'harmbench',
        'file_path': 'data/harmbench.csv',
        'sample_size': 100
    }
    loader = DatasetFactory.create_loader(config)
    prompts = loader.load_prompts()
"""

from .base_dataset_loader import BaseDatasetLoader
from .harmbench_loader import HarmBenchLoader
from .jbb_loader import JBBLoader
from .airbench_loader import AirBenchLoader
from .combined_dataset_loader import CombinedDatasetLoader
from .balanced_challenge_loader import BalancedChallengeLoader
from .dataset_factory import DatasetFactory

__all__ = [
    'BaseDatasetLoader',
    'HarmBenchLoader', 
    'JBBLoader',
    'AirBenchLoader',
    'CombinedDatasetLoader',
    'BalancedChallengeLoader',
    'DatasetFactory'
]