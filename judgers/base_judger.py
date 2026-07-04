"""
Base class for all judgers.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict

# Import custom exceptions for consistent error handling
try:
    from .gpt_judger import (
        JudgerError,
        JudgerAPIError,
        JudgerParsingError,
        JudgerNetworkError,
        JudgerQuotaExceededError,
        JudgerRateLimitError
    )
except ImportError:
    # Define fallback exceptions if not available
    class JudgerError(Exception):
        pass
    
    class JudgerAPIError(JudgerError):
        pass
    
    class JudgerParsingError(JudgerError):
        pass
    
    class JudgerNetworkError(JudgerError):
        pass
    
    class JudgerQuotaExceededError(JudgerAPIError):
        pass
    
    class JudgerRateLimitError(JudgerAPIError):
        pass


class BaseJudger(ABC):
    """
    Base class for all judgers that evaluate attack success or response harmfulness.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the judger with configuration parameters.
        
        Args:
            **kwargs: Configuration parameters specific to each judger implementation.
        """
        self.config = kwargs
    
    @abstractmethod
    def judge(self, behavior: str = None, generation: Union[str, List[str]] = None, 
              context: str = None, **kwargs) -> Union[int, List[int]]:
        """
        Evaluate the given generation(s) for harmfulness or attack success.
        
        Args:
            behavior: The behavior or query being evaluated (optional).
            generation: The text generation(s) to evaluate. Can be a single string or list of strings.
            context: Additional context for evaluation (optional).
            **kwargs: Additional parameters specific to the judger implementation.
        
        Returns:
            Union[int, List[int]]: 
                - If generation is a string: returns an integer score (typically 0 or 1)
                - If generation is a list: returns a list of integer scores
                - 1 typically indicates harmful/successful, 0 indicates safe/failed
        """
        pass
    
    def batch_judge(self, behaviors: List[str] = None, generations: List[str] = None, 
                   contexts: List[str] = None, **kwargs) -> List[int]:
        """
        Evaluate multiple generations in batch.
        
        Args:
            behaviors: List of behaviors or queries being evaluated (optional).
            generations: List of text generations to evaluate.
            contexts: List of additional contexts for evaluation (optional).
            **kwargs: Additional parameters specific to the judger implementation.
        
        Returns:
            List[int]: List of integer scores for each generation.
        """
        if generations is None:
            raise ValueError("generations must be provided")
        
        results = []
        for i, generation in enumerate(generations):
            behavior = behaviors[i] if behaviors else None
            context = contexts[i] if contexts else None
            result = self.judge(behavior=behavior, generation=generation, context=context, **kwargs)
            results.append(result)
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters for this judger.
        
        Returns:
            Dict[str, Any]: Configuration parameters.
        """
        return self.config.copy()