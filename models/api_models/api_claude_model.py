# models/api_models/api_claude_model.py

from .api_base import APIModel
import anthropic
import httpx
import os

class APIClaudeModel(APIModel):
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key)
        
        # Handle proxy configuration if needed
        client_kwargs = {"api_key": api_key}
        
        # Check for proxy environment variables and create custom http_client if needed
        proxy_http = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        proxy_https = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        
        proxy_configured = False
        if proxy_http or proxy_https:
            # Create httpx client with proxy support
            try:
                # httpx.Client uses 'proxy' parameter, not 'proxies'
                # Use HTTPS proxy if available, otherwise HTTP proxy
                proxy_url = proxy_https or proxy_http
                
                http_client = httpx.Client(proxy=proxy_url)
                client_kwargs["http_client"] = http_client
                proxy_configured = True
            except Exception as e:
                print(f"Warning: Failed to configure proxy: {e}")
        
        try:
            self.client = anthropic.Anthropic(**client_kwargs)
        except TypeError as e:
            # Handle various proxy-related errors
            if any(keyword in str(e) for keyword in ["proxies", "http_client", "unexpected keyword argument"]):
                # Fallback: create client without any proxy configuration
                print(f"Warning: Anthropic client doesn't support proxy configuration in version {anthropic.__version__}. Creating client without proxy.")
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                raise e
                
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        # Filter out any unsupported parameters for Claude API
        # Note: max_tokens is already included and is the correct parameter name for Claude
        supported_params = {
            'max_tokens', 'temperature', 'top_p', 'top_k', 'stop_sequences',
            'stream', 'system', 'metadata', 'stop', 'tool_choice', 'tools'
        }
        
        # Add parameter name mapping for consistency
        if 'max_new_tokens' in kwargs and 'max_tokens' not in kwargs:
            kwargs = kwargs.copy()
            kwargs['max_tokens'] = kwargs.pop('max_new_tokens')
        
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            timeout=60,
            **filtered_kwargs
        )
        return response.content[0].text.strip()