"""
AirBench Dataset Loader
=======================

Loads data from AirBench dataset files and HuggingFace AIR-Bench-2024 dataset.
Supports both local files and direct HuggingFace dataset loading with L4 name sampling.
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from .base_dataset_loader import BaseDatasetLoader

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


class AirBenchLoader(BaseDatasetLoader):
    """Loader for AirBench dataset"""
    
    def load_prompts(self) -> List[str]:
        """Load prompts from AirBench dataset"""
        
        # Check if using HuggingFace dataset
        if self.config.get('source') == 'huggingface':
            return self._load_huggingface_airbench()
        else:
            return self._load_local_airbench()
    
    def _load_huggingface_airbench(self) -> List[str]:
        """Load AIR-Bench-2024 dataset from HuggingFace"""
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("HuggingFace datasets library not available. Install with: pip install datasets")
        
        dataset_name = self.config.get('dataset_name', 'stanford-crfm/air-bench-2024')
        config_name = self.config.get('config_name', 'default')  # default, us, china, eu_comprehensive, etc.
        prompt_column = self.config.get('prompt_column', 'prompt')
        l4_sampling = self.config.get('l4_sampling', False)  # Sample one example from each L4 category
        l4_categories = self.config.get('l4_categories', None)  # Filter specific L4 categories
        
        try:
            self.console.print(f"📊 Loading AIR-Bench-2024 dataset from HuggingFace: {dataset_name}")
            
            # Load dataset from HuggingFace
            dataset = load_dataset(dataset_name, config_name, split='test')
            df = dataset.to_pandas()
            
            self.console.print(f"📊 Loaded {len(df)} rows from {config_name} configuration")
            
            # Apply L4 category filter if specified
            if l4_categories and 'l4-name' in df.columns:
                original_len = len(df)
                df = df[df['l4-name'].isin(l4_categories)]
                self.console.print(f"📋 Filtered by L4 categories {l4_categories}: {len(df)}/{original_len} prompts")
            
            # Apply L4 sampling strategy (one example from each L4 category)
            if l4_sampling and 'l4-name' in df.columns:
                original_len = len(df)
                df = df.groupby('l4-name').first().reset_index()
                self.console.print(f"🎯 L4 sampling: selected {len(df)} examples from {df['l4-name'].nunique()} L4 categories (from {original_len} total)")
            
            # Extract prompts from specified column
            if prompt_column not in df.columns:
                available_cols = ', '.join(df.columns)
                raise ValueError(f"Column '{prompt_column}' not found. Available columns: {available_cols}")
            
            prompts = df[prompt_column].tolist()
            self.dataset_data = df
            
            # Filter and sample
            valid_prompts = self.filter_prompts(prompts)
            sampled_prompts = self.apply_sampling(valid_prompts)
            
            # Store prompts and metadata
            self.prompts = sampled_prompts
            self.metadata = {
                'total_prompts': len(prompts),
                'valid_prompts': len(valid_prompts),
                'sampled_prompts': len(sampled_prompts),
                'prompt_column': prompt_column,
                'config_name': config_name,
                'dataset_name': dataset_name,
                'l4_sampling': l4_sampling,
                'l4_categories': l4_categories,
                'source': 'huggingface'
            }
            
            self.console.print(f"✅ Loaded {len(sampled_prompts)} prompts from AIR-Bench-2024")
            return sampled_prompts
            
        except Exception as e:
            self.console.print(f"❌ Error loading AIR-Bench-2024 dataset: {e}")
            raise
    
    def _load_local_airbench(self) -> List[str]:
        """Load AirBench dataset from local files"""
        
        file_path = self.config['file_path']
        prompt_column = self.config.get('prompt_column', 'prompt')
        category_filter = self.config.get('category_filter', None)
        
        try:
            # Load data based on file type
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle AirBench JSON structure
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Convert dict to DataFrame
                    df = pd.DataFrame(data.get('data', []))
                else:
                    raise ValueError("Unsupported JSON structure")
                
                self.dataset_data = df
                
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                self.dataset_data = df
                
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Check if prompt column exists
            if prompt_column not in df.columns:
                available_cols = ', '.join(df.columns)
                raise ValueError(f"Column '{prompt_column}' not found. Available columns: {available_cols}")
            
            # Apply category filter if specified
            if category_filter and 'category' in df.columns:
                original_len = len(df)
                df = df[df['category'].isin(category_filter)]
                self.console.print(f"📋 Filtered by categories {category_filter}: {len(df)}/{original_len} prompts")
            
            prompts = df[prompt_column].tolist()
            self.console.print(f"📊 Loaded AirBench dataset with {len(prompts)} prompts")
            
            # Filter and sample
            valid_prompts = self.filter_prompts(prompts)
            sampled_prompts = self.apply_sampling(valid_prompts)
            
            # Store prompts for reuse
            self.prompts = sampled_prompts
            self.metadata = {
                'total_prompts': len(prompts),
                'valid_prompts': len(valid_prompts),
                'sampled_prompts': len(sampled_prompts),
                'prompt_column': prompt_column,
                'category_filter': category_filter,
                'source': 'local'
            }
            
            self.console.print(f"✅ Loaded {len(sampled_prompts)} prompts from AirBench")
            return sampled_prompts
            
        except Exception as e:
            self.console.print(f"❌ Error loading AirBench dataset: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get AirBench dataset information"""
        info = {
            'type': 'airbench',
            'name': 'AirBench',
            'description': 'AI red-teaming benchmark dataset',
            'total_prompts': self.metadata.get('total_prompts', 0),
            'loaded_prompts': len(self.prompts),
            'prompt_column': self.metadata.get('prompt_column', 'prompt'),
            'source': self.metadata.get('source', 'local')
        }
        
        # Add HuggingFace-specific info if applicable
        if self.metadata.get('source') == 'huggingface':
            info.update({
                'dataset_name': self.metadata.get('dataset_name', ''),
                'config_name': self.metadata.get('config_name', ''),
                'l4_sampling': self.metadata.get('l4_sampling', False),
                'l4_categories': self.metadata.get('l4_categories', None),
                'hf_available': HF_DATASETS_AVAILABLE
            })
        else:
            info['category_filter'] = self.metadata.get('category_filter', None)
        
        return info
    
    def get_analysis_data(self) -> Optional[pd.DataFrame]:
        """Get AirBench dataset for analysis"""
        return self.dataset_data
    
    def get_categories(self) -> Dict[str, Any]:
        """Get category information from AirBench dataset"""
        if self.dataset_data is None:
            return {}
        
        categories = {}
        
        # For AIR-Bench-2024 hierarchical structure
        if 'l2-name' in self.dataset_data.columns:
            categories['l2_categories'] = self.dataset_data['l2-name'].value_counts().to_dict()
        
        if 'l3-name' in self.dataset_data.columns:
            categories['l3_categories'] = self.dataset_data['l3-name'].value_counts().to_dict()
        
        if 'l4-name' in self.dataset_data.columns:
            categories['l4_categories'] = self.dataset_data['l4-name'].value_counts().to_dict()
        
        # For legacy format
        if 'category' in self.dataset_data.columns:
            categories['categories'] = self.dataset_data['category'].value_counts().to_dict()
        
        if 'subcategory' in self.dataset_data.columns:
            categories['subcategories'] = self.dataset_data['subcategory'].value_counts().to_dict()
        
        return categories