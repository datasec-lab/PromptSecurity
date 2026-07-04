import re
from .system_prompts import get_judge_system_prompt
from .loggers import logger
from .language_models import ConfigLLM


class JudgeBase:
    def __init__(self, args):
        self.max_n_tokens = (
            args["judge_max_n_tokens"] if isinstance(args, dict) else getattr(args, "judge_max_n_tokens", None)
        )
        self.temperature = (
            args["judge_temperature"] if isinstance(args, dict) else getattr(args, "judge_temperature", None)
        )
        self.goal = (
            args["goal"] if isinstance(args, dict) else getattr(args, "goal", None)
        )
        self.target_str = (
            args["target_str"] if isinstance(args, dict) else getattr(args, "target_str", None)
        )
        self.system_prompt = get_judge_system_prompt(self.goal, self.target_str)


    def update_system_prompt(self, goal, target_str):
        """
        Dynamically update the judge's system prompt based on the new goal and target string.
        """
        self.goal = goal
        self.target_str = target_str
        self.system_prompt = get_judge_system_prompt(goal, target_str)
        logger.debug(f"Updated judge system prompt:\n{self.system_prompt}")

    def get_judge_prompt(self, attack_prompt, target_response):
        return f"[PROMPT]: {attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"

    def score(self, prompt_list, response_list):
        """
        Must return a list of numeric scores. Child classes must implement.
        """
        raise NotImplementedError

    def process_output(self, raw_output):
        """
        Looks for something like [[10]] in the raw output. Default fallback = 1
        """
        pattern = r'\[\[(\d+)\]\]'
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        if output is None:
            logger.warning(f"Error in processing judge output: {raw_output}")
            output = 1
        return output


class GPTJudge(JudgeBase):
    def __init__(self, args):
        super().__init__(args)
        self.llm = ConfigLLM(args["model_config"])

    def score(self, attack_prompt_list, target_response_list):
        scores = []
        for atk, resp in zip(attack_prompt_list, target_response_list):
            full_input = f"{self.system_prompt}\n{self.get_judge_prompt(atk, resp)}"
            raw_output = self.llm.generate(
                full_input,
                max_tokens=self.max_n_tokens,
                temperature=self.temperature,
            )
            s = self.process_output(raw_output)
            scores.append(s)
        return scores


class NoJudge(JudgeBase):
    """
    If you want a “no-op” judge that always gives a neutral score:
    """
    def __init__(self, args):
        super().__init__(args)
        self.llm = None

    def score(self, prompt_list, response_list):
        return [1] * len(prompt_list)
