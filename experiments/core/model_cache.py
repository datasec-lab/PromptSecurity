"""
Model Cache System for PromptSecurity Experiments

Provides LRU caching for model instances to avoid repeated loading of the same model,
particularly important for local models which are expensive to load.
"""

import time
import logging
from typing import Dict, Any, Tuple, Optional
import gc
import torch


logger = logging.getLogger(__name__)


class ModelCache:
    """
    LRU (Least Recently Used) cache for model instances.
    
    This cache is designed to optimize performance by avoiding repeated loading
    of the same model, which is particularly expensive for local models.
    """
    
    def __init__(self, max_cache_size: int = 3, max_memory_gb: float = 8.0):
        """
        Initialize the model cache.
        
        Args:
            max_cache_size: Maximum number of models to cache simultaneously
            max_memory_gb: Maximum memory usage in GB (security limit)
        """
        self._model_cache: Dict[str, Any] = {}  # {model_name: model_instance}
        self._model_parameters: Dict[str, Any] = {}  # {model_name: model_parameters}
        self._last_used: Dict[str, float] = {}  # {model_name: timestamp}
        self._config_paths: Dict[str, str] = {}  # {model_name: config_path}
        self._max_cache_size = max_cache_size
        
        # Security: Memory limits to prevent exhaustion attacks
        self._max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self._current_memory_usage = 0
        
        logger.debug(f"ModelCache initialized with max_cache_size={max_cache_size}, max_memory={max_memory_gb}GB")
    
    def _check_memory_limits(self) -> bool:
        """检查内存限制"""
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss
            
            if current_memory > self._max_memory_bytes:
                logger.warning(f"内存使用超出限制: {current_memory/1024/1024/1024:.2f}GB > {self._max_memory_bytes/1024/1024/1024:.2f}GB")
                return False
            
            return True
        except ImportError:
            # If psutil is not available, allow operation but log warning
            logger.warning("psutil not available, cannot check memory limits")
            return True
        except Exception as e:
            logger.warning(f"内存检查失败: {e}")
            return True
    
    def _estimate_model_memory(self, model_instance) -> int:
        """估算模型内存使用"""
        try:
            import sys
            if hasattr(model_instance, 'model') and hasattr(model_instance.model, 'parameters'):
                # For PyTorch models, estimate based on parameters
                total_params = sum(p.numel() for p in model_instance.model.parameters())
                # Rough estimate: 4 bytes per parameter (float32)
                return total_params * 4
            else:
                # Fallback: use sys.getsizeof
                return sys.getsizeof(model_instance)
        except Exception as e:
            logger.warning(f"无法估算模型内存: {e}")
            return 1024 * 1024 * 1024  # Default to 1GB estimate
    
    def get_model(self, model_name: str, config_path: str) -> Tuple[Any, Any]:
        """
        Get a model instance from cache or load it if not cached.
        
        Args:
            model_name: Name of the model
            config_path: Path to the model configuration file
            
        Returns:
            Tuple of (model_instance, model_parameters)
        """
        # Update last used time if model is already cached
        if model_name in self._model_cache:
            self._last_used[model_name] = time.time()
            logger.debug(f"ModelCache HIT: {model_name}")
            return self._model_cache[model_name], self._model_parameters[model_name]
        
        # Model not in cache, need to load it
        logger.info(f"ModelCache MISS: Loading {model_name} for the first time")
        return self._load_and_cache_model(model_name, config_path)
    
    def _load_and_cache_model(self, model_name: str, config_path: str) -> Tuple[Any, Any]:
        """
        Load a new model and add it to the cache.
        
        Args:
            model_name: Name of the model
            config_path: Path to the model configuration file
            
        Returns:
            Tuple of (model_instance, model_parameters)
        """
        # Security: Check memory limits before loading new model
        if not self._check_memory_limits():
            logger.warning("内存使用已达到限制，清理缓存")
            self._evict_oldest_model()
        
        # Check if we need to make room in the cache
        if len(self._model_cache) >= self._max_cache_size:
            self._evict_oldest_model()
        
        # Load the new model
        logger.info(f"🔄 Loading model: {model_name}")
        start_time = time.time()
        
        from models.loader import load_model
        model_instance, model_parameters = load_model(model_name)
        
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.2f}s: {model_name}")
        
        # Validate and log token parameters after successful loading
        self._validate_token_parameters_after_loading(model_name, model_parameters)
        
        # Cache the model
        self._model_cache[model_name] = model_instance
        self._model_parameters[model_name] = model_parameters
        self._last_used[model_name] = time.time()
        self._config_paths[model_name] = config_path
        
        logger.debug(f"ModelCache: Cached {model_name} (cache size: {len(self._model_cache)}/{self._max_cache_size})")
        
        return model_instance, model_parameters
    
    def _validate_token_parameters_after_loading(self, model_name: str, model_parameters: dict):
        """验证模型缓存后的token参数传递"""
        token_limit_param = None
        token_limit_value = None
        
        # 检查token限制参数
        if 'max_tokens' in model_parameters:
            token_limit_param = 'max_tokens'
            token_limit_value = model_parameters['max_tokens']
        elif 'max_new_tokens' in model_parameters:
            token_limit_param = 'max_new_tokens'
            token_limit_value = model_parameters['max_new_tokens']
        
        if token_limit_param and token_limit_value:
            logger.info(f"🎯 ModelCache: {model_name} 的 {token_limit_param} = {token_limit_value}")
        else:
            logger.warning(f"⚠️ ModelCache: {model_name} 缺少token限制参数")
            
        # 记录所有参数以便调试
        param_summary = {k: v for k, v in model_parameters.items() 
                        if k in ['temperature', 'max_tokens', 'max_new_tokens', 'top_p', 'do_sample']}
        if param_summary:
            logger.debug(f"📋 ModelCache: {model_name} 缓存的参数: {param_summary}")
    
    def _evict_oldest_model(self):
        """
        Remove the least recently used model from cache to make room for a new one.
        """
        if not self._last_used:
            return
        
        # Find the model that was used longest ago
        oldest_model = min(self._last_used, key=self._last_used.get)
        
        logger.info(f"🗑️  Evicting model from cache: {oldest_model}")
        
        # Remove from cache
        if oldest_model in self._model_cache:
            model_instance = self._model_cache[oldest_model]
            
            # Clean up GPU memory if it's a local model
            if hasattr(model_instance, 'model') and hasattr(model_instance.model, 'to'):
                try:
                    # Move model to CPU and delete
                    # Keep the attribute to avoid breaking in-flight users of the model instance.
                    # Eviction can happen while the instance is still referenced elsewhere.
                    model_instance.model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.debug(f"GPU memory cleaned for {oldest_model}")
                except Exception as e:
                    logger.warning(f"Failed to clean GPU memory for {oldest_model}: {e}")
            
            del self._model_cache[oldest_model]
        
        if oldest_model in self._model_parameters:
            del self._model_parameters[oldest_model]
        if oldest_model in self._last_used:
            del self._last_used[oldest_model]
        if oldest_model in self._config_paths:
            del self._config_paths[oldest_model]
        
        # Force garbage collection to free memory
        gc.collect()
        
        logger.debug(f"ModelCache: Evicted {oldest_model} (cache size: {len(self._model_cache)}/{self._max_cache_size})")
    
    def clear_cache(self):
        """
        Clear all models from the cache and free memory.
        """
        logger.info("🗑️  Clearing model cache")
        
        for model_name in list(self._model_cache.keys()):
            self._evict_oldest_model()
        
        self._model_cache.clear()
        self._model_parameters.clear()
        self._last_used.clear()
        self._config_paths.clear()
        
        # Enhanced GPU memory cleanup
        gc.collect()  # Python garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()      # Clear GPU cache
            torch.cuda.synchronize()      # Ensure operations complete
            
            # Log GPU memory status if available
            try:
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                logger.debug(f"GPU memory after cleanup - Allocated: {memory_allocated/1024/1024:.1f}MB, Reserved: {memory_reserved/1024/1024:.1f}MB")
            except Exception:
                pass  # Ignore memory status logging errors
        
        logger.info("✅ Model cache and GPU memory cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current cache state.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_models": list(self._model_cache.keys()),
            "cache_size": len(self._model_cache),
            "max_cache_size": self._max_cache_size,
            "cache_utilization": len(self._model_cache) / self._max_cache_size * 100,
            "last_used": {
                model: time.time() - timestamp 
                for model, timestamp in self._last_used.items()
            }
        }
    
    def is_model_cached(self, model_name: str) -> bool:
        """
        Check if a model is currently cached.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if the model is cached, False otherwise
        """
        return model_name in self._model_cache
    
    def get_cached_model_names(self) -> list:
        """
        Get the names of all currently cached models.
        
        Returns:
            List of cached model names
        """
        return list(self._model_cache.keys())


# Global model cache instance
# This ensures that the same cache is used across all parts of the application
_global_model_cache: Optional[ModelCache] = None


def get_global_model_cache() -> ModelCache:
    """
    Get the global model cache instance.
    
    Returns:
        The global ModelCache instance
    """
    global _global_model_cache
    if _global_model_cache is None:
        _global_model_cache = ModelCache()
    return _global_model_cache


def clear_global_model_cache():
    """
    Clear the global model cache.
    """
    global _global_model_cache
    if _global_model_cache is not None:
        _global_model_cache.clear_cache()
        _global_model_cache = None
