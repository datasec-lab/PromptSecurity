# attacks/black_box/ABJAttack/pipeline/data_analysis.py

from ..utils.print_helper import print_attack
from ..utils.attack_prompt import PROMPT_LIST
from ..utils.random_split import split_sentence_randomly

def get_ABJ_attack_prompt(original_attack_prompt, task, character, feature):
    # This was the old logic
    attack_prompt = original_attack_prompt.format(TASK=task, CHARACTER=character, FEATURE=feature)
    # Optionally remove printing or keep it
    print_attack(attack_prompt)
    return attack_prompt

def get_modified_ABJ_attack_prompt(original_attack_prompt, character, feature, job):
    attack_prompt = original_attack_prompt.format(CHARACTER=character, FEATURE=feature, JOB=job)
    print_attack(attack_prompt)
    return attack_prompt

def get_adversarial_ABJ_attack_prompt(original_attack_prompt, s1, s2, w1, w2, character, feature):
    attack_prompt = original_attack_prompt.format(
        S1=s1,
        S2=s2,
        FIRST=w1,
        SECOND=w2,
        CHARACTER=character,
        FEATURE=feature
    )
    print_attack(attack_prompt)
    return attack_prompt

def get_attack_prompt(attack_method, task, character, feature, job):
    """
    Orchestrates which prompt type to use (original_ABJ, modified_ABJ, etc.)
    """
    if attack_method == 'modified_ABJ':
        return get_modified_ABJ_attack_prompt(
            PROMPT_LIST[attack_method], character, feature, job
        )
    elif attack_method == 'adversarial_ABJ':
        s1, s2, w1, w2 = split_sentence_randomly(task)
        return get_adversarial_ABJ_attack_prompt(
            PROMPT_LIST[attack_method], s1, s2, w1, w2, character, feature
        )
    else:
        # fallback to 'original_ABJ' or 'code_based_ABJ'
        return get_ABJ_attack_prompt(
            PROMPT_LIST[attack_method], task, character, feature
        )
