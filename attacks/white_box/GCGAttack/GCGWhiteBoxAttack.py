import gc
import numpy as np
import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack
from .llm_attacks.minimal_gcg.string_utils import (
    HFChatSuffixManager,
    SuffixManager,
    load_conversation_template,
)
from .llm_attacks.minimal_gcg.opt_utils import (
    token_gradients,
    sample_control,
    get_logits,
    target_loss,
    get_filtered_cands,
)


class GCGWhiteBoxAttack(BaseAttack):
    """
    Implements the Greedy Coordinate Gradient (GCG) white-box attack.
    This attack iteratively adjusts an adversarial suffix (added to the user prompt)
    using gradient information computed from the target model.
    """

    def __init__(
        self,
        target_model,
        target_model_parameters,
        num_steps=500,
        user_prompt=None,
        adv_string_init=None,
        target=None,
        batch_size=512,
        topk=256,
        allow_non_ascii=False,
        template_name="auto",
    ):
        if user_prompt is None:
            raise ValueError("user_prompt must be provided in the attack config")
        if adv_string_init is None:
            raise ValueError("adv_string_init must be provided in the attack config")
        if target is None:
            raise ValueError("target must be provided in the attack config")

        self.target_model = target_model
        self.target_model_parameters = target_model_parameters
        self.num_steps = num_steps
        self.user_prompt = user_prompt
        self.adv_string_init = adv_string_init
        self.target = target
        self.batch_size = batch_size
        self.topk = topk
        self.allow_non_ascii = allow_non_ascii
        self.template_name = template_name

        self.query_count = 0

        # Use the target model's tokenizer and align prompt formatting with the
        # target chat template when available.  GCG optimizes token slices inside
        # the serialized prompt, so optimization-time and query-time formatting
        # must match.
        self.tokenizer = target_model.get_tokenizer()
        if (
            self.template_name in {"auto", "hf", "target", "target-chat"}
            and hasattr(target_model, "format_chat_prompt")
            and getattr(target_model, "supports_chat_template", lambda: False)()
        ):
            self.conv_template = None
            self.suffix_manager = HFChatSuffixManager(
                tokenizer=self.tokenizer,
                chat_formatter=target_model.format_chat_prompt,
                instruction=self.user_prompt,
                target=self.target,
                adv_string=self.adv_string_init,
            )
        else:
            self.conv_template = load_conversation_template(self.template_name)
            self.suffix_manager = SuffixManager(
                tokenizer=self.tokenizer,
                conv_template=self.conv_template,
                instruction=self.user_prompt,
                target=self.target,
                adv_string=self.adv_string_init,
            )

    def attack(self, original_prompt: str) -> (int, str):
        """
        Runs the white-box attack given a raw prompt.
        It returns a tuple (query_count, attacked_prompt) where:
         - query_count is the total number of model queries made,
         - attacked_prompt is the original prompt with an adversarial suffix appended.
        """
        # Update the SuffixManager with the provided original prompt
        self.suffix_manager.instruction = original_prompt

        adv_suffix = self.adv_string_init

        for step in range(self.num_steps):
            # Get the tokenized input (with current adversarial suffix)
            input_ids = self.suffix_manager.get_input_ids(adv_string=adv_suffix)
            input_ids = input_ids.to(self.target_model.device)

            # Compute the gradient (with respect to the control tokens)
            grad = token_gradients(
                self.target_model,
                input_ids,
                self.suffix_manager._control_slice,
                self.suffix_manager._target_slice,
                self.suffix_manager._loss_slice,
            )
            self.query_count += 1

            # Sample a batch of new candidate control tokens
            new_adv_suffix_toks = sample_control(
                self.suffix_manager.control_toks,
                grad,
                batch_size=self.batch_size,
                topk=self.topk,
                temp=1,
                not_allowed_tokens=None  # Optionally, pass non-ASCII tokens if needed
            )
            # Ensure candidate token sequences keep the same length (using filtering)
            new_adv_suffix = get_filtered_cands(
                self.tokenizer, new_adv_suffix_toks, filter_cand=True, curr_control=adv_suffix
            )

            # Compute loss for each candidate by replacing the control tokens in the input
            logits, ids = get_logits(
                model=self.target_model,
                tokenizer=self.tokenizer,
                input_ids=input_ids,
                control_slice=self.suffix_manager._control_slice,
                test_controls=new_adv_suffix,
                return_ids=True,
                batch_size=self.batch_size,
            )
            losses = target_loss(logits, ids, self.suffix_manager._target_slice)
            losses_np = losses.float().detach().cpu().numpy()
            best_idx = np.argmin(losses_np)
            best_new_adv_suffix = new_adv_suffix[best_idx]
            current_loss = losses_np[best_idx]

            # Update adversarial suffix
            adv_suffix = best_new_adv_suffix

            self.suffix_manager.adv_string = adv_suffix  # update internal state

            # Check if the attack is successful:
            # We use a simple criterion: if none of the test prefixes appear in the generation.
            gen_text = self.suffix_manager.generate_str(self.target_model)
            if not any(prefix in gen_text for prefix in self.suffix_manager.test_prefixes):
                print(gen_text)
                break

            gc.collect()
            torch.cuda.empty_cache()

        # The attacked prompt is simply the original prompt plus the final adversarial suffix.
        attacked_prompt = original_prompt + " " + adv_suffix
        return self.query_count, attacked_prompt
