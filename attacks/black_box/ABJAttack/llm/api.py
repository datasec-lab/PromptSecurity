import openai

def get_response(prompt, url, api_key, model_name):
    openai.api_base = url
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=0
        )
        return response.choices[0].message['content']
    except Exception as e:
        return str(e)



