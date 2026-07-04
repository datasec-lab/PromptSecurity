from openai import OpenAI
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# for gpt
def chatCompletion(model, messages, temperature, retry_times, round_sleep, fail_sleep, api_key, base_url=None):
    # Handle case where model is a model object instead of string
    if hasattr(model, 'generate'):
        # This is a model object, use it directly
        prompt_text = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt_text += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt_text += f"Assistant: {msg['content']}\n"
        
        try:
            model_output = model.generate(prompt=prompt_text, temperature=temperature)
            time.sleep(round_sleep)
            return model_output
        except Exception as e:
            print(f"Model generation failed: {e}")
            return "I cannot help with that request."
    
    # Original OpenAI API logic for string model names
    if base_url is None:
        client = OpenAI(
            api_key=api_key
            )
    else:
        client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    response = None
    try:
        response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
    except Exception as e:
        print(e)
        for retry_time in range(retry_times):
            retry_time = retry_time + 1
            print(f"{model} Retry {retry_time}")
            time.sleep(fail_sleep)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                break
            except:
                continue

    if response is None:
        print("All API calls failed, returning default response")
        return "I cannot help with that request."
        
    model_output = response.choices[0].message.content.strip()
    time.sleep(round_sleep)

    return model_output

# for claude
def claudeCompletion(model, max_tokens, temperature, prompt, retry_times, round_sleep, fail_sleep, api_key, base_url=None):
    # Handle case where model is a model object instead of string
    if hasattr(model, 'generate'):
        # This is a model object, use it directly
        try:
            model_output = model.generate(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
            time.sleep(round_sleep)
            return model_output
        except Exception as e:
            print(f"Model generation failed: {e}")
            return "I cannot help with that request."
    
    # Original Anthropic API logic for string model names
    if base_url is None:
        client = Anthropic(
            api_key=api_key
            )
    else:
        client = Anthropic(
            base_url=base_url,
            auth_token=api_key
            )
    
    completion = None
    try:
        completion = client.completions.create(
            model=model,
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"
            )
    except Exception as e:
        print(e)
        for retry_time in range(retry_times):
            retry_time = retry_time + 1
            print(f"{model} Retry {retry_time}")
            time.sleep(fail_sleep)
            try:
                completion = client.completions.create(
                model=model,
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
                prompt=prompt
                )
                break
            except:
                continue

    if completion is None:
        print("All API calls failed, returning default response")
        return "I cannot help with that request."
        
    model_output = completion.completion.strip()
    time.sleep(round_sleep)

    return model_output