import os
from defenses.base_defended_model import BaseDefendedModel
from .model import LitellmModel
from jinja2 import Environment, FileSystemLoader
from .utils import route_templates, extract_and_eval_json

class PrimeGuardDefendedModel(BaseDefendedModel):

    def __init__(
        self,
        target_model,
        tokenizer,
        template_dir=os.path.join(os.path.dirname(__file__), "./templates"),
        cached_results=None,
        generate_kwargs=None,
        verbose=False
    ):
        super().__init__(target_model)
        self.tokenizer = tokenizer
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.cached_results = cached_results or []
        self.generate_kwargs = generate_kwargs or {}
        self.verbose = verbose

    def _generate(self, prompt, system_prompt=""):
        """Unified single-call generation through target_model with corrected sampling parameters."""
        if system_prompt:
            full_prompt = f"{system_prompt}\n{prompt}"
        elif hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            # Use chat template for instruction-tuned models that require it
            messages = [{"role": "user", "content": prompt}]
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            full_prompt = prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.target_model.device)

        # Explicitly set correct generation parameters to silence warnings
        generate_kwargs = {
            "do_sample": False,
            "temperature": self.generate_kwargs.get("temperature", 0.0),
            "max_new_tokens": self.generate_kwargs.get("max_new_tokens", 512),
            **self.generate_kwargs
        }

        input_length = inputs["input_ids"].shape[1]
        output = self.target_model.generate(**inputs, **generate_kwargs)
        return self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

    def route(self, prompt):
        cached_result = next(
            (item for item in self.cached_results if item["prompt"] == prompt), None
        )

        if cached_result:
            return cached_result

        system_instructions = self.env.get_template("oai_safety.j2").render()
        # vanilla_result = self._generate(prompt, system_prompt=system_instructions)

        # PrimeGuard Routing Decision
        routing_template = self.env.get_template("route_selection.j2")
        routing_rendered = routing_template.render(
            system_prompt=system_instructions, user_input=prompt
        )
        route_selection_output = self._generate(routing_rendered)

        final_conv, routes, system_checks, system_tips = route_templates(
            route_selection_outputs=[route_selection_output],
            prompts=[prompt],
            restrictive_system_instructions=system_instructions,
            env=self.env,
            no_violation_template_name="answer_utility.j2",
            potential_violation_template_name_first_turn="display_analysis.j2",
            potential_violation_template_name_second_turn="get_answer.j2",
            direct_violation_template_name="refusal.j2",
        )

        # Execute routed action
        conv = final_conv[0]
        if routes[0] == "potential_violation" and len(conv) == 3:
            messages = [
                {"role": "user", "content": conv[0]},
                {"role": "assistant", "content": conv[1]},
                {"role": "user", "content": conv[2]},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            final_output = self._generate(formatted)
        else:
            final_output = self._generate(conv[0])

        primeguard_answer = final_output
        reevaluation = "N/A"

        if routes[0] == "potential_violation":
            parsed_json = extract_and_eval_json(final_output)
            if parsed_json and "reevaluation" in parsed_json[0] and "final_response" in parsed_json[0]:
                reevaluation = parsed_json[0]["reevaluation"]
                primeguard_answer = parsed_json[0]["final_response"]

        result = {
            "prompt": prompt,
            "route": routes[0],
            # "vanilla_result": vanilla_result,
            "primeguard_result": primeguard_answer,
            "system_check": system_checks[0],
            "system_tip": system_tips[0],
            "reevaluation": reevaluation,
        }

        self.cached_results.append(result)

        if self.verbose:
            print("[PrimeGuard] Route Decision:", routes[0])
            print("[PrimeGuard] System Check:", system_checks[0])
            print("[PrimeGuard] System Tip:", system_tips[0])

        return result

    def generate(self, prompt: str, **kwargs) -> str:
        result = self.route(prompt)
        return result["primeguard_result"]
