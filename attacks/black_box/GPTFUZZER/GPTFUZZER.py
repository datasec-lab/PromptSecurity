import os
import re
import pandas as pd
import sys
import random
import argparse
from attacks.base_attack import BaseAttack

from .gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from .gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)

from .gptfuzzer.fuzzer import GPTFuzzer
from .gptfuzzer.utils.predict import RoBERTaPredictor
from .gptfuzzer.llm import OpenAILLM


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(project_root)
from models import api_keys
from judgers import load_judger_from_config

class GPTFUZZER(BaseAttack):
    def __init__(self, target_model, target_model_parameters, mutate_model='gpt-4', energy=1, max_jailbreak=1, max_query=100, judge_model_config=None, **kwargs) -> None: # mutate model = 'gpt-3.5-turbo' or 'gpt-4'
        super().__init__(target_model)
        self.target_model=target_model
        self.target_model_parameters = target_model_parameters
        self.mutate_model = mutate_model
        self.energy = energy # 'The energy of the fuzzing process'
        self.max_jailbreak = max_jailbreak #'The maximum jailbreak number'
        self.max_query = max_query # 'The maximum number of queries'
        
        # Initialize judger (optional)
        self.judger = None
        if judge_model_config:
            try:
                self.judger = load_judger_from_config(judge_model_config)
            except Exception as e:
                print(f"Warning: Could not load judger from config {judge_model_config}: {e}")
                print("Falling back to RoBERTa predictor evaluation.")




    def attack(self, original_prompt: str, **model_parameters) -> (int, str):
        openai_model = OpenAILLM(self.mutate_model, api_keys.OPENAI_API_KEY) # mutate model
        
        # Use custom judger if available, otherwise use RoBERTa
        if self.judger:
            # Create a wrapper to make judger compatible with GPTFUZZER interface
            class JudgerWrapper:
                def __init__(self, judger):
                    self.judger = judger
                
                def predict(self, sequences):
                    """Convert sequences to judger format and return predictions."""
                    results = []
                    for seq in sequences:
                        try:
                            # Use the judger to evaluate the sequence
                            score = self.judger.judge(generation=seq)
                            # Convert to binary prediction (1 = harmful, 0 = safe)
                            pred = 1 if (score >= 0.5 if isinstance(score, (int, float)) else bool(score)) else 0
                            results.append(pred)
                        except Exception as e:
                            print(f"Judger prediction failed for sequence: {e}")
                            results.append(0)  # Default to safe
                    return results
            
            predictor = JudgerWrapper(self.judger)
        else:
            predictor = RoBERTaPredictor('hubert233/GPTFuzz', device='cuda:0')
        roberta_model = predictor

        project_root = os.path.dirname(__file__)
        seed_path = "./datasets/prompts/GPTFuzzer.csv"
        seed_path = os.path.join(project_root, seed_path)
        initial_seed = pd.read_csv(seed_path)['text'].tolist()

        questions = [original_prompt]
        fuzzer = GPTFuzzer(
            questions=questions,
            # target_model=openai_model,
            target=self.target_model,
            predictor=predictor,
            initial_seed=initial_seed,
            mutate_policy=MutateRandomSinglePolicy([
                OpenAIMutatorCrossOver(openai_model, temperature=0.0),
                # for reproduction only, if you want better performance, use temperature>0
                OpenAIMutatorExpand(openai_model, temperature=0.0),
                OpenAIMutatorGenerateSimilar(openai_model, temperature=0.0),
                OpenAIMutatorRephrase(openai_model, temperature=0.0),
                OpenAIMutatorShorten(openai_model, temperature=0.0)],
                concatentate=True,
            ),
            select_policy=MCTSExploreSelectPolicy(),
            energy=self.energy,
            max_jailbreak=self.max_jailbreak,
            max_query=self.max_query,
            generate_in_batch=False,
        )

        query_times, successful_prompts = fuzzer.run()
        # Choose a single prompt to return
        if successful_prompts:
            best_prompt = successful_prompts[0]          # or any selection rule you like
        else:
            best_prompt = original_prompt                # fall back to the original

        return query_times, best_prompt                   # �� return a *string*, not list





