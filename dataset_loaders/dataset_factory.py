"""
Dataset Factory
===============

Factory class for creating appropriate dataset loaders.
"""

from typing import Dict, Any
from rich.console import Console
from .base_dataset_loader import BaseDatasetLoader
from .harmbench_loader import HarmBenchLoader
from .jbb_loader import JBBLoader
from .airbench_loader import AirBenchLoader
from .combined_dataset_loader import CombinedDatasetLoader
from .balanced_challenge_loader import BalancedChallengeLoader


class DatasetFactory:
    """Factory for creating dataset loaders"""
    
    LOADERS = {
        'harmbench': HarmBenchLoader,
        'jbb': JBBLoader,
        'airbench': AirBenchLoader,
        'combined': CombinedDatasetLoader,
        'balanced_challenge': BalancedChallengeLoader
    }
    
    @classmethod
    def create_loader(cls, config: Dict[str, Any], console: Console = None) -> BaseDatasetLoader:
        """
        Create a dataset loader based on configuration.
        
        Args:
            config: Dataset configuration with 'type' field
            console: Rich console for logging
            
        Returns:
            Appropriate dataset loader instance
            
        Raises:
            ValueError: If dataset type is not supported
        """
        dataset_type = config.get('type', '').lower()
        
        if dataset_type not in cls.LOADERS:
            available_types = ', '.join(cls.LOADERS.keys())
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Available types: {available_types}")
        
        loader_class = cls.LOADERS[dataset_type]
        return loader_class(config, console)
    
    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported dataset types"""
        return list(cls.LOADERS.keys())
    
    @classmethod
    def register_loader(cls, dataset_type: str, loader_class: type):
        """
        Register a new dataset loader.
        
        Args:
            dataset_type: Type identifier for the dataset
            loader_class: Loader class that inherits from BaseDatasetLoader
        """
        if not issubclass(loader_class, BaseDatasetLoader):
            raise ValueError("Loader class must inherit from BaseDatasetLoader")
        
        cls.LOADERS[dataset_type] = loader_class