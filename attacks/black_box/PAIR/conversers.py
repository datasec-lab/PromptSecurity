from .language_models import ConfigLLM
import json

class AttackLM:
    """
    Generates adversarial prompts using a config-based LLM.
    Expects the config file to define the model_type, model_name, and possibly
    default generation parameters (like temperature).
    """
    def __init__(
        self,
        config_path: str,
        max_n_tokens: int,
        max_n_attack_attempts: int,
        category: str,
        attacker_template_name: str
    ):
        # Load your model from config
        self.llm = ConfigLLM(config_path)
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.category = category
        self.template_name = attacker_template_name

    def preprocess_conversation(self, convs_list, prompts_list):
        """
        Optionally manipulates conversation messages if you need to seed JSON or
        add special tokens. Return the final text that you'll pass into `generate()`.
        """
        # For demonstration, we keep the same approach of building an "init_message"
        init_message = '{\"improvement\": \"\",\"prompt\": \"'
        processed_convs = []

        for conv, prompt in zip(convs_list, prompts_list):
            # Append user prompt to conversation
            conv.append_message(conv.roles[0], prompt)
            # Append partial JSON “start” to conversation
            conv.append_message(conv.roles[1], init_message)
            # Convert to text that we'll pass to the LLM
            processed_convs.append(conv.to_openai_api_messages())

        return processed_convs, init_message

    def _generate_attack(self, processed_convs_list, init_message):
        """
        Generate attacks with retries and ensure the final output is valid JSON.
        """
        batchsize = len(processed_convs_list)
        valid_outputs = [None] * batchsize
        new_adv_prompts = [None] * batchsize

        indices_to_regenerate = list(range(batchsize))

        for attempt in range(self.max_n_attack_attempts):
            for i in indices_to_regenerate:
                input_text = ""
                for m in processed_convs_list[i]:
                    role = m["role"].lower()
                    content = m["content"]
                    input_text += f"{role.upper()}: {content}\n"

                raw_output = self.llm.generate(input_text)

                try:
                    # Use a dictionary and let 'json.dumps()' do the escaping
                    attack_dict = {
                        "improvement": "",
                        "prompt": raw_output
                    }
                    # Convert that dict to a JSON string
                    final_json_str = json.dumps(attack_dict)

                    # Test that the resulting JSON string can be re-parsed:
                    parsed_obj = json.loads(final_json_str)

                    # If we get here, it's valid JSON. Save it.
                    valid_outputs[i] = parsed_obj
                    new_adv_prompts[i] = final_json_str

                except Exception as e:
                    print(f"Error generating or parsing JSON for conversation {i}: {e}")

            indices_to_regenerate = [idx for idx, v in enumerate(valid_outputs) if v is None]
            if not indices_to_regenerate:
                break

        if any(o is None for o in valid_outputs):
            raise ValueError("Failed to generate valid JSON after "
                             f"{self.max_n_attack_attempts} attempts.")

        return valid_outputs, new_adv_prompts

    def get_attack(self, convs_list, prompts_list):
        """
        Main entry point for AttackLM to produce adversarial prompts.
        """
        # Build conversation messages + init JSON
        processed_convs_list, init_message = self.preprocess_conversation(
            convs_list, prompts_list
        )

        # Generate in a loop until valid
        valid_outputs, new_adv_prompts = self._generate_attack(
            processed_convs_list, init_message
        )

        # Update conversation so you can hold onto the final prompt
        for adv_prompt, conv in zip(new_adv_prompts, convs_list):
            conv.update_last_message(adv_prompt)

        return valid_outputs


class TargetLM:
    """
    For the model we are trying to jailbreak. Takes a config-based LLM,
    simply calls .generate() on each prompt to get the target's response.
    """
    def __init__(
        self,
        config_path: str,
        category: str,
        max_n_tokens: int,
        target_template_name: str
    ):
        self.llm = ConfigLLM(config_path)
        self.category = category
        self.max_n_tokens = max_n_tokens
        self.template_name = target_template_name

    def get_response(self, prompts_list):
        responses = []
        for prompt in prompts_list:
            out = self.llm.generate(prompt)
            responses.append(out)
        return responses


def load_attack_and_target_models(args):
    """
    Load attack and target models using configurations.
    """
    attacker_template_name = "gpt-4"  # Fallback if template not required
    target_template_name = "gpt-4"

    attackLM = AttackLM(
        config_path=args.attack_model_config,
        max_n_tokens=args.attack_max_n_tokens,
        max_n_attack_attempts=args.max_n_attack_attempts,
        category=args.category,
        attacker_template_name=attacker_template_name,
    )

    targetLM = TargetLM(
        config_path=args.target_model_config,
        category=args.category,
        max_n_tokens=args.target_max_n_tokens,
        target_template_name=target_template_name,
    )
    return attackLM, targetLM
