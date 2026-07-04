# main_utils.py
import numpy as np

def clean_attacks_and_convs(attack_list, convs_list):
    tmp = [(a, c) for (a, c) in zip(attack_list, convs_list) if a is not None]
    if not tmp:
        return [], []
    attack_list, convs_list = zip(*tmp)
    return list(attack_list), list(convs_list)

def prune(
    on_topic_scores,
    judge_scores,
    adv_prompt_list,
    improv_list,
    convs_list,
    target_response_list,
    extracted_attack_list,
    sorting_score,
    width
):
    """
    Original pruning logic. Sort all items by sorting_score (descending),
    keep only the top 'width' (and only if their score > 0).
    """
    idx_score = list(enumerate(sorting_score))
    np.random.shuffle(idx_score)
    idx_score.sort(key=lambda x: x[1], reverse=True)

    def get_first_k(seq_list):
        filtered = []
        for i, sc in idx_score:
            if sc > 0 and len(filtered) < width:
                filtered.append(seq_list[i])
        # fallback if everything is 0 or the list is empty
        if len(filtered) == 0 and len(seq_list) > 1:
            filtered = [seq_list[idx_score[0][0]], seq_list[idx_score[1][0]]]
        return filtered

    if judge_scores is not None:
        judge_scores = get_first_k(judge_scores)
    if target_response_list is not None:
        target_response_list = get_first_k(target_response_list)

    on_topic_scores = get_first_k(on_topic_scores)
    adv_prompt_list = get_first_k(adv_prompt_list)
    improv_list = get_first_k(improv_list)
    convs_list = get_first_k(convs_list)
    extracted_attack_list = get_first_k(extracted_attack_list)

    return (
        on_topic_scores,
        judge_scores,
        adv_prompt_list,
        improv_list,
        convs_list,
        target_response_list,
        extracted_attack_list
    )
