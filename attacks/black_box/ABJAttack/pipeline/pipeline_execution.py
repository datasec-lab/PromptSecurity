from tqdm import tqdm
from pipeline.data_analysis import get_attack_prompt
from pipeline.data_preparation import data_preparation
from pipeline.attack_and_judge import attack_and_judge

def pipeline_execution(df, attack_method, judge_prompt, saver, columns, target_model, assist_model, attack_rounds):
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        # data preparation
        task = row['goal']
        character, feature, job = data_preparation(task, assist_model)

        # data analysis
        attack_prompt = get_attack_prompt(attack_method, task, character, feature, job)

        # Perform multiple rounds of attack and judgement
        attack_and_judge(target_model, assist_model, attack_prompt, judge_prompt, attack_rounds, columns, task, saver)

    # Final save to ensure all data is saved
    saver.final_save()

