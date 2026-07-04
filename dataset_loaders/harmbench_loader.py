"""
HarmBench Dataset Loader
========================

Loads data from HarmBench CSV files.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from .base_dataset_loader import BaseDatasetLoader


class HarmBenchLoader(BaseDatasetLoader):
    """Loader for HarmBench dataset"""
    
    def load_prompts(self) -> List[str]:
        """Load prompts from HarmBench CSV dataset"""
        
        file_path = self.config['file_path']
        behavior_column = self.config.get('behavior_column', 'Behavior')
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            self.dataset_data = df
            self.console.print(f"📊 Loaded HarmBench dataset with {len(df)} behaviors")
            
            # Extract behavior column
            if behavior_column not in df.columns:
                available_cols = ', '.join(df.columns)
                raise ValueError(f"Column '{behavior_column}' not found. Available columns: {available_cols}")
            
            behaviors = df[behavior_column].tolist()
            
            # Filter and sample
            valid_behaviors = self.filter_prompts(behaviors)
            sampled_behaviors = self.apply_sampling(valid_behaviors)
            
            # Store prompts for reuse
            self.prompts = sampled_behaviors
            self.metadata = {
                'total_behaviors': len(df),
                'valid_behaviors': len(valid_behaviors),
                'sampled_behaviors': len(sampled_behaviors),
                'behavior_column': behavior_column
            }
            
            self.console.print(f"✅ Loaded {len(sampled_behaviors)} behaviors from HarmBench")
            return sampled_behaviors
            
        except Exception as e:
            self.console.print(f"❌ Error loading HarmBench dataset: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get HarmBench dataset information"""
        return {
            'type': 'harmbench',
            'name': 'HarmBench',
            'description': 'Harmful behavior benchmark dataset',
            'total_prompts': self.metadata.get('total_behaviors', 0),
            'loaded_prompts': len(self.prompts),
            'behavior_column': self.metadata.get('behavior_column', 'Behavior')
        }
    
    def get_analysis_data(self) -> Optional[pd.DataFrame]:
        """Get HarmBench dataset for analysis"""
        return self.dataset_data
    
    def get_categories(self) -> Dict[str, Any]:
        """Get category information from HarmBench dataset"""
        if self.dataset_data is None:
            return {}
        
        categories = {}
        
        # Functional categories
        if 'FunctionalCategory' in self.dataset_data.columns:
            categories['functional'] = self.dataset_data['FunctionalCategory'].value_counts().to_dict()
        
        # Semantic categories
        if 'SemanticCategory' in self.dataset_data.columns:
            categories['semantic'] = self.dataset_data['SemanticCategory'].value_counts().to_dict()
        
        return categories