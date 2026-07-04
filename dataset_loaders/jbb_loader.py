"""
JBB (Jailbreak Bench) Dataset Loader
====================================

Loads data from JBB dataset files and HuggingFace JBB-Behaviors dataset.
Supports both local files and direct HuggingFace dataset loading.
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


class JBBLoader(BaseDatasetLoader):
    """Loader for JBB (Jailbreak Bench) dataset"""
    
    def load_prompts(self) -> List[str]:
        """Load prompts from JBB dataset"""
        
        # Check if using HuggingFace dataset
        if self.config.get('source') == 'huggingface':
            return self._load_huggingface_jbb()
        else:
            return self._load_local_jbb()
    
    def _load_huggingface_jbb(self) -> List[str]:
        """Load JBB-Behaviors dataset from HuggingFace"""
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("HuggingFace datasets library not available. Install with: pip install datasets")
        
        dataset_name = self.config.get('dataset_name', 'JailbreakBench/JBB-Behaviors')
        config_name = self.config.get('config_name', 'behaviors')  # behaviors or judge_comparison
        split_name = self.config.get('split', 'harmful')  # harmful, benign, or test
        prompt_column = self.config.get('prompt_column', 'Goal')  # Use 'Goal' as default for JBB-Behaviors
        category_filter = self.config.get('category_filter', None)
        
        try:
            self.console.print(f"📊 Loading JBB-Behaviors dataset from HuggingFace: {dataset_name}")
            
            # Load dataset from HuggingFace
            dataset = load_dataset(dataset_name, config_name, split=split_name)
            df = dataset.to_pandas()
            
            self.console.print(f"📊 Loaded {len(df)} rows from {split_name} split")
            
            # Apply category filter if specified
            if category_filter and 'Category' in df.columns:
                original_len = len(df)
                df = df[df['Category'].isin(category_filter)]
                self.console.print(f"📋 Filtered by categories {category_filter}: {len(df)}/{original_len} behaviors")
            
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
                'split': split_name,
                'dataset_name': dataset_name,
                'config_name': config_name,
                'source': 'huggingface'
            }
            
            self.console.print(f"✅ Loaded {len(sampled_prompts)} prompts from JBB-Behaviors")
            return sampled_prompts
            
        except Exception as e:
            self.console.print(f"❌ Error loading JBB-Behaviors dataset: {e}")
            raise
    
    def _load_local_jbb(self) -> List[str]:
        """Load JBB dataset from local files"""
        
        file_path = self.config['file_path']
        prompt_column = self.config.get('prompt_column', 'prompt')
        
        try:
            # Determine file type and load accordingly
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    # List of prompts or objects
                    if isinstance(data[0], dict):
                        prompts = [item.get(prompt_column, '') for item in data]
                    else:
                        prompts = data
                elif isinstance(data, dict):
                    # Dictionary with prompts key
                    prompts = data.get('prompts', data.get('data', []))
                else:
                    raise ValueError("Unsupported JSON structure")
                
                self.dataset_data = pd.DataFrame({'prompt': prompts})
                
            elif file_path.endswith('.csv'):
                # CSV file
                df = pd.read_csv(file_path)
                self.dataset_data = df
                
                if prompt_column not in df.columns:
                    available_cols = ', '.join(df.columns)
                    raise ValueError(f"Column '{prompt_column}' not found. Available columns: {available_cols}")
                
                prompts = df[prompt_column].tolist()
                
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.console.print(f"📊 Loaded JBB dataset with {len(prompts)} prompts")
            
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
                'source': 'local'
            }
            
            self.console.print(f"✅ Loaded {len(sampled_prompts)} prompts from JBB")
            return sampled_prompts
            
        except Exception as e:
            self.console.print(f"❌ Error loading JBB dataset: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get JBB dataset information"""
        info = {
            'type': 'jbb',
            'name': 'JBB (Jailbreak Bench)',
            'description': 'Jailbreak benchmark dataset',
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
                'split': self.metadata.get('split', ''),
                'hf_available': HF_DATASETS_AVAILABLE
            })
        
        return info
    
    def get_analysis_data(self) -> Optional[pd.DataFrame]:
        """Get JBB dataset for analysis"""
        return self.dataset_data
    
    def get_categories(self) -> Dict[str, Any]:
        """Get category information from JBB dataset"""
        if self.dataset_data is None:
            return {}
        
        categories = {}
        
        # Get categories if available (JBB-Behaviors format)
        if 'Category' in self.dataset_data.columns:
            categories['categories'] = self.dataset_data['Category'].value_counts().to_dict()
        
        # Get behavior info if available
        if 'Behavior' in self.dataset_data.columns:
            categories['behaviors'] = len(self.dataset_data['Behavior'].unique())
        
        # Get source info if available
        if 'Source' in self.dataset_data.columns:
            categories['sources'] = self.dataset_data['Source'].value_counts().to_dict()
        
        return categories