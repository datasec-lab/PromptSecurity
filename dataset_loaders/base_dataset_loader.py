"""
Base Dataset Loader Interface
=============================

Defines the common interface for all dataset loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
from rich.console import Console


class BaseDatasetLoader(ABC):
    """Base class for dataset loaders"""
    
    def __init__(self, config: Dict[str, Any], console: Optional[Console] = None):
        """
        Initialize the dataset loader.
        
        Args:
            config: Dataset configuration
            console: Rich console for logging
        """
        self.config = config
        self.console = console or Console()
        self.dataset_data = None
        self.prompts = []
        self.metadata = {}
    
    @abstractmethod
    def load_prompts(self) -> List[str]:
        """
        Load prompts from the dataset.
        
        Returns:
            List of prompt strings
        """
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset metadata
        """
        pass
    
    @abstractmethod
    def get_analysis_data(self) -> Optional[pd.DataFrame]:
        """
        Get the raw dataset for analysis.
        
        Returns:
            DataFrame with raw dataset or None if not available
        """
        pass
    
    def get_sample_size(self) -> int:
        """Get the configured sample size"""
        return self.config.get('sample_size', len(self.prompts))
    
    def get_random_sample(self) -> bool:
        """Check if random sampling is enabled"""
        return self.config.get('random_sample', False)
    
    def filter_prompts(self, prompts: List[str]) -> List[str]:
        """
        Filter out invalid prompts.
        
        Args:
            prompts: List of prompts to filter
            
        Returns:
            List of valid prompts
        """
        valid_prompts = []
        for prompt in prompts:
            if prompt and isinstance(prompt, str) and len(prompt.strip()) > 0:
                valid_prompts.append(prompt.strip())
        
        if len(valid_prompts) != len(prompts):
            self.console.print(f"⚠️  Filtered out {len(prompts) - len(valid_prompts)} invalid prompts")
        
        return valid_prompts
    
    def apply_sampling(self, prompts: List[str]) -> List[str]:
        """
        Apply sampling to prompts if configured.
        
        Args:
            prompts: List of prompts to sample from
            
        Returns:
            Sampled prompts
        """
        sample_size = self.get_sample_size()
        if sample_size and sample_size < len(prompts):
            if self.get_random_sample():
                # Random sampling with seed
                import random
                # 使用配置的种子，默认为42
                seed = self.config.get('seed', 42)
                random.seed(seed)
                sampled_prompts = random.sample(prompts, sample_size)
                self.console.print(f"🎲 Randomly sampled {sample_size} prompts (seed={seed})")
            else:
                # Take first N prompts
                sampled_prompts = prompts[:sample_size]
                self.console.print(f"📋 Using first {sample_size} prompts")
            return sampled_prompts
        
        return prompts
    
    def load(self) -> List[str]:
        """
        Load dataset prompts. This is the main entry point for loading data.
        
        Returns:
            List of prompt strings
        """
        return self.load_prompts()