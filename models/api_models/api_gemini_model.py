# models/api_models/api_gemini_model.py

from .api_base import APIModel
import google.generativeai as genai

class APIGeminiModel(APIModel):
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key)
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        # Extract generation configuration parameters
        generation_params = {}
        other_params = {}
        
        # Parameter name mappings for Gemini API compatibility
        param_mappings = {
            'max_tokens': 'max_output_tokens',
            'max_new_tokens': 'max_output_tokens',
            'max_length': 'max_output_tokens',
            'stop': 'stop_sequences',
            'stop_words': 'stop_sequences'
        }
        
        # Apply parameter mappings
        mapped_kwargs = {}
        for key, value in kwargs.items():
            mapped_key = param_mappings.get(key, key)
            mapped_kwargs[mapped_key] = value
        
        # Parameters that should go into GenerationConfig
        generation_config_params = {
            'temperature', 'top_p', 'top_k', 'max_output_tokens', 
            'candidate_count', 'stop_sequences'
        }
        
        # Split kwargs into generation config and other parameters
        for key, value in mapped_kwargs.items():
            if key in generation_config_params:
                generation_params[key] = value
            else:
                other_params[key] = value
        
        # Create generation config if we have relevant parameters
        generation_config = None
        if generation_params:
            try:
                generation_config = genai.types.GenerationConfig(**generation_params)
            except Exception as e:
                print(f"Warning: Failed to create GenerationConfig with params {generation_params}: {e}")
                # Try with a minimal config
                temperature = generation_params.get('temperature', 0.0)
                max_output_tokens = generation_params.get('max_output_tokens', 512)
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens
                )
        
        # Prepare arguments for generate_content
        generate_args = {"contents": prompt}
        if generation_config:
            generate_args["generation_config"] = generation_config
        
        # Add other parameters (like safety_settings)
        generate_args.update(other_params)
        
        try:
            # Add timeout to generation args if supported
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Gemini API request timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout
            
            try:
                response = self.model.generate_content(**generate_args)
            finally:
                signal.alarm(0)  # Cancel the alarm
            
            # Check if response has valid content
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
                    # finish_reason 2 means SAFETY, try with different safety settings
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                    ]
                    generate_args["safety_settings"] = safety_settings
                    response = self.model.generate_content(**generate_args)
            
            # Extract text safely
            try:
                if hasattr(response, 'text') and response.text:
                    return response.text.strip()
            except ValueError as ve:
                # Handle cases where response.text is not accessible due to safety filters
                if "finish_reason" in str(ve) and "2" in str(ve):
                    return "I can't provide that information as it violates content policy."
                
            # Try to extract from candidates
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    try:
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                return candidate.content.parts[0].text.strip()
                    except (ValueError, AttributeError):
                        # Handle safety-filtered candidates
                        if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
                            return "I can't provide that information as it violates content policy."
                        continue
            
            return "No valid response generated"
            
        except Exception as e:
            if "temperature" in str(e) or "GenerationConfig" in str(e):
                # Fallback: try with safe minimal config (NEVER without token limit!)
                print(f"Warning: Gemini API error with generation config: {e}. Trying with safe minimal config.")
                try:
                    # Always enforce 512 token limit as fallback
                    safe_config = genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=512
                    )
                    response = self.model.generate_content(prompt, generation_config=safe_config)
                    if hasattr(response, 'text'):
                        return response.text.strip()
                    return "No valid response generated"
                except Exception as e2:
                    raise e2
            else:
                raise e