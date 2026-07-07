# attacks/black_box/ABJAttack/pipeline/data_analysis.py

from ..utils.attack_prompt import PROMPT_LIST


def get_attack_prompt(attack_method: str, data_block: str, task: str = None) -> str:
    """
    Build the final Stage-2 attack prompt.

    Args:
        attack_method: Key from PROMPT_LIST (e.g. 'original_ABJ').
        data_block:    Plain-text persona block from data_to_block().
        task:          Original harmful query; only used for the
                       'abj_with_harmful_query' ablation variant.
    Returns:
        Formatted attack prompt ready to send to the target model.
    """
    template = PROMPT_LIST.get(attack_method, PROMPT_LIST['original_ABJ'])

    if attack_method == 'abj_with_harmful_query' and task is not None:
        return template.format(DATA=data_block, TASK=task)
    return template.format(DATA=data_block)
