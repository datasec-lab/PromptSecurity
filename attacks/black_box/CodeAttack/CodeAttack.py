import os
import re
from attacks.base_attack import BaseAttack


class CodeAttack(BaseAttack):
    def __init__(self, target_model, target_model_parameters, prompt_type="python_stack") -> None: # prompt_type="python_stack", "python_stack_plus", "python_list", "python_list_plus", "python_string", "python_string_plus", "C_string", "go_string"
        super().__init__(target_model)
        self.prompt_type = prompt_type
        self.prompt_name = f"code_{self.prompt_type}.txt"
        self.user_prompt = None
        self.target_model=target_model
        self.target_model_parameters = target_model_parameters


    def prompt(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, "prompt_templates", self.prompt_name)
        with open(filename, 'r') as file:
            prompt = file.read()
        return prompt


    def replace(self, filepath, replacement, replace_line=None) -> str:
        prompt_prefix = ""
        prompt_prefix_file = open(filepath, 'r')
        for i, line in enumerate(prompt_prefix_file):
            if replace_line is not None and i == replace_line:
                prompt_prefix += replacement
            else:
                prompt_prefix += line
        if replace_line > i:
            prompt_prefix += replacement
        return prompt_prefix


    def wrap(self, ori_query) -> str:
        """This function processes the queries depending on the chosen prompt
        type, and then injects them into the template. The template depends on
        the prompt type, too."""

        wrapped_input = None
        replacement_line = None

        if "python_stack" in self.prompt_name:
            # In a stack, elements are pushed in reverse order
            def format_and_push(input_string):
                # Split spaces and hyphens
                words = re.split(r'[\s\-]+', input_string)
                if len(words) == 1:
                    # handle the case where input is Chinese
                    # Split every character
                    words = list(words[0])
                    print(words)
                # Reverse order of words
                words = words[::-1]
                formatted_output = '\n'.join(f'    my_stack.append("{word}")'
                                             for word in words)
                return formatted_output
            wrapped_input = format_and_push(ori_query) + '\n'
        elif "python_list" in self.prompt_name:
            # In a list, elements are appended respecting their order
            def format_and_push(input_string):
                words = input_string.split()
                formatted_output = '\n'.join(f'    my_list.append("{word}")'
                                             for word in words)
                return formatted_output
            wrapped_input = format_and_push(ori_query) + '\n'
        elif "python_string" in self.prompt_name:
            wrapped_input = f"    my_string = \"{ori_query}\"\n"
        elif "C_string" in self.prompt_name:
            wrapped_input = f"    std::string my_string = \"{ori_query}\";\n"
            replacement_line = 10
        elif "go_string" in self.prompt_name:
            wrapped_input = f"        myQueue := \"{ori_query}\"\n"
            replacement_line = 17

        # Inject the wrapped input into the prompt template
        if "python" in self.prompt_name:
            # If the template is python, use python built-in format method to
            # inject the input replacing the placeholders
            prompt = self.prompt().format(wrapped_input=wrapped_input)
        else:
            # If the template is in a language other than python, use the
            # replace function from this class
            # TODO
            prompt = self.replace(f"./prompt_templates/{self.prompt_name}",
                                  wrapped_input,
                                  replacement_line)
        return prompt


    def attack(self, original_prompt: str, **model_parameters) -> (int, str):
        crafted_prompt = self.wrap(original_prompt)
        self.query_count = 1  # Count as 1 query for the crafted prompt
        return self.query_count, crafted_prompt

