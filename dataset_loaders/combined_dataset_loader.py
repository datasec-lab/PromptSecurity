"""
Combined Dataset Loader
=======================

Loads data from multiple datasets and combines them into a single dataset.
Supports sampling from each dataset individually.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from .base_dataset_loader import BaseDatasetLoader
from .harmbench_loader import HarmBenchLoader
from .jbb_loader import JBBLoader
from .airbench_loader import AirBenchLoader


class CombinedDatasetLoader(BaseDatasetLoader):
    """Loader for combining multiple datasets"""
    
    def __init__(self, config: Dict[str, Any], console=None):
        """
        Initialize the combined dataset loader.
        
        Args:
            config: Dataset configuration with 'datasets' list
            console: Rich console for logging
        """
        super().__init__(config, console)
        self.sub_loaders = []
        self.sub_datasets_info = []
        
    def load_prompts(self) -> List[str]:
        """Load prompts from multiple datasets and combine them"""
        
        datasets_config = self.config.get('datasets', [])
        if not datasets_config:
            raise ValueError("No datasets configured for combined loader")
        
        all_prompts = []
        combined_metadata = {
            'total_datasets': len(datasets_config),
            'datasets_info': [],
            'total_prompts': 0
        }
        
        self.console.print(f"🔄 Loading {len(datasets_config)} datasets for combination...")
        
        for i, dataset_config in enumerate(datasets_config):
            dataset_type = dataset_config.get('type', '').lower()
            dataset_name = dataset_config.get('name', f'Dataset_{i+1}')
            
            self.console.print(f"📚 Loading {dataset_name} ({dataset_type})...")
            
            # Create appropriate loader
            loader = self._create_sub_loader(dataset_type, dataset_config)
            if loader:
                try:
                    # Load prompts from this dataset
                    prompts = loader.load_prompts()
                    all_prompts.extend(prompts)
                    
                    # Collect metadata
                    dataset_info = loader.get_dataset_info()
                    dataset_info['dataset_name'] = dataset_name
                    dataset_info['dataset_type'] = dataset_type
                    dataset_info['loaded_prompts'] = len(prompts)
                    combined_metadata['datasets_info'].append(dataset_info)
                    
                    self.console.print(f"✅ Loaded {len(prompts)} prompts from {dataset_name}")
                    
                    # Store sub-loader for later use
                    self.sub_loaders.append(loader)
                    
                except Exception as e:
                    self.console.print(f"❌ Failed to load {dataset_name}: {str(e)}")
                    continue
        
        # Apply global sampling if configured
        if self.config.get('global_sample_size'):
            global_sample_size = self.config['global_sample_size']
            if global_sample_size < len(all_prompts):
                if self.get_random_sample():
                    import random
                    all_prompts = random.sample(all_prompts, global_sample_size)
                    self.console.print(f"🎲 Randomly sampled {global_sample_size} prompts globally")
                else:
                    all_prompts = all_prompts[:global_sample_size]
                    self.console.print(f"📋 Using first {global_sample_size} prompts globally")
        
        # Store results
        self.prompts = all_prompts
        combined_metadata['total_prompts'] = len(all_prompts)
        self.metadata = combined_metadata
        
        self.console.print(f"🎉 Successfully combined {len(all_prompts)} prompts from {len(datasets_config)} datasets")
        return all_prompts
    
    def _create_sub_loader(self, dataset_type: str, dataset_config: Dict[str, Any]) -> Optional[BaseDatasetLoader]:
        """Create a sub-loader for a specific dataset type"""
        
        loader_classes = {
            'harmbench': HarmBenchLoader,
            'jbb': JBBLoader,
            'airbench': AirBenchLoader
        }
        
        if dataset_type not in loader_classes:
            self.console.print(f"⚠️  Unsupported dataset type: {dataset_type}")
            return None
        
        loader_class = loader_classes[dataset_type]
        return loader_class(dataset_config, self.console)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the combined dataset"""
        return {
            'type': 'combined',
            'total_datasets': self.metadata.get('total_datasets', 0),
            'datasets_info': self.metadata.get('datasets_info', []),
            'total_prompts': len(self.prompts),
            'sample_size': self.get_sample_size(),
            'random_sample': self.get_random_sample()
        }
    
    def get_analysis_data(self) -> Optional[pd.DataFrame]:
        """Get combined analysis data from all sub-datasets"""
        if not self.sub_loaders:
            return None
        
        combined_data = []
        for i, loader in enumerate(self.sub_loaders):
            sub_data = loader.get_analysis_data()
            if sub_data is not None:
                # Add source dataset information
                sub_data = sub_data.copy()
                dataset_info = self.metadata['datasets_info'][i]
                sub_data['source_dataset'] = dataset_info['dataset_name']
                sub_data['source_type'] = dataset_info['dataset_type']
                combined_data.append(sub_data)
        
        if combined_data:
            return pd.concat(combined_data, ignore_index=True)
        return None
    
    def get_datasets_breakdown(self) -> Dict[str, int]:
        """Get breakdown of prompts by dataset"""
        breakdown = {}
        for dataset_info in self.metadata.get('datasets_info', []):
            dataset_name = dataset_info.get('dataset_name', 'Unknown')
            prompt_count = dataset_info.get('loaded_prompts', 0)
            breakdown[dataset_name] = prompt_count
        return breakdown