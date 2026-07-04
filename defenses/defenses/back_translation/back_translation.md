# BackTranslation Defense
Defending LLMs against Jailbreaking Attacks via Backtranslation (ACL'24)
This defense method protects language models from harmful or jailbreaking prompts using a simple idea:  
it asks a secondary model (called the backtranslator) to guess the original prompt based on the model's response.  
If the guessed (backtranslated) prompt seems harmful and the model refuses to answer it, the original prompt is blocked too.

This approach helps catch prompts that are subtly phrased to bypass filters, while having minimal effect on safe prompts.

---

## Parameters

- **backtranslator_model**: The model used to infer the original prompt, such as `gpt-3.5-turbo` from OpenAI.
- **threshold**: Minimum log-likelihood value used to decide whether the guessed prompt is valid (default: `-2.0`).
- **return_new_response_anyway**: If `true`, returns the re-generated response from the backtranslated prompt regardless of refusal status.

---

**Note:**  
Make sure to set your `OPENAI_API_KEY` in the environment to use the backtranslator model.
