import argparse
from pipeline.pipeline_execution import pipeline_execution
from pipeline.pipeline_initialization import pipeline_initialization


parser = argparse.ArgumentParser(description='Process input parameters for generating text and judging harmfulness.')
parser.add_argument('--attack_method', type=str, default='modified_ABJ', help='[original_ABJ, modified_ABJ, code_based_ABJ, adversarial_ABJ]')
parser.add_argument('--target_model', type=str, default='gpt4', help='[qwen2, glm4, llama3, gpt3, gpt4, claude3_haiku]')
parser.add_argument('--assist_model', type=str, default='gpt4')
parser.add_argument('--dataset_dir', type=str, default='./dataset/harmful_behaviors.csv', help='Directory of the dataset')
parser.add_argument('--save_interval', type=int, default=1 * 1 * 10, help='Interval of saving CSV file')
parser.add_argument('--attack_rounds', type=int, default=3, help='Number of iteration rounds')
parser.add_argument('--target_model_cuda_id', type=str, default="cuda:0")
parser.add_argument('--assist_model_cuda_id', type=str, default="cuda:1")
args = parser.parse_args()



if __name__ == '__main__':
    initialize_data = pipeline_initialization(args)
    pipeline_execution(**initialize_data)