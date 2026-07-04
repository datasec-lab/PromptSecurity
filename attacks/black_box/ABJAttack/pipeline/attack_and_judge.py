from utils.clean_text import clean_text
from utils.print_helper import print_response_judgement

def attack_and_judge(target_model, assist_model, attack_prompt, judge_prompt, attack_rounds, columns, task, saver):

    response_results = []
    judgement_results = []

    # Perform multiple rounds of attack and judgement
    for i in range(attack_rounds):
        response = clean_text(target_model.generate_response(attack_prompt))
        judgement = clean_text(assist_model.generate_response(judge_prompt.format(PROMPT=response)))
        print_response_judgement(i, response, judgement)
        judgement_results.append(judgement)
        response_results.append(response)

    # Add data to the saver and save the results
    data_values = [task, attack_prompt] + response_results + judgement_results
    data_dict = dict(zip(columns, data_values))
    saver.add_and_save(data_dict)
