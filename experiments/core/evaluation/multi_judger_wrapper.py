"""
多Judger包装器 - 支持同时使用多个judger进行评估
"""

# Import custom exceptions for consistent error handling
try:
    from judgers.gpt_judger import (
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

class MultiJudger:
    """包装多个judger，返回所有judger的结果"""
    
    def __init__(self, judgers):
        """
        Args:
            judgers: judger实例列表
        """
        self.judgers = judgers
        # 创建包含所有judger名称的字符串
        judger_names = []
        for j in judgers:
            if hasattr(j, 'name'):
                judger_names.append(j.name)
            elif hasattr(j, 'judger_name'):
                judger_names.append(j.judger_name)
            elif hasattr(j, '__class__'):
                judger_names.append(j.__class__.__name__)
            else:
                judger_names.append(str(j))
        self.name = f"[{', '.join(judger_names)}]"
        self.judger_names = judger_names
        
    def judge(self, behavior, generation):
        """
        使用所有judger进行评判
        
        Returns:
            如果只有一个judger，返回单个结果
            如果有多个judger，返回结果列表
        
        Raises:
            JudgerError: If any judger fails, raises appropriate exception instead of returning None
        """
        results = []
        failed_judgers = []
        
        for i, judger in enumerate(self.judgers):
            try:
                result = judger.judge(behavior=behavior, generation=generation)
                results.append(result)
            except JudgerError:
                # Re-raise our custom exceptions without modification
                failed_judgers.append(f"Judger {i} ({getattr(judger, 'name', judger.__class__.__name__)})")
                raise
            except Exception as e:
                # Wrap any other exceptions in a general JudgerError
                judger_name = getattr(judger, 'name', judger.__class__.__name__)
                failed_judgers.append(f"Judger {i} ({judger_name})")
                raise JudgerError(f"Judger {judger_name} failed: {e}")
        
        # 总是返回列表，让调用方决定如何处理
        return results
    
    def batch_judge(self, behaviors, generations):
        """
        批量评判
        
        Args:
            behaviors: List of behaviors to judge
            generations: List of generations to judge
            
        Returns:
            List of result lists from all judgers
            
        Raises:
            JudgerError: If any judger fails on any sample
        """
        all_results = []
        for i, (behavior, generation) in enumerate(zip(behaviors, generations)):
            try:
                result = self.judge(behavior, generation)
                all_results.append(result)
            except JudgerError:
                # Re-raise with context about which sample failed
                raise
            except Exception as e:
                # Wrap unexpected errors with context
                raise JudgerError(f"MultiJudger batch evaluation failed on sample {i}: {e}")
        return all_results