# attacks/black_box/ABJAttack/utils/print_helper.py

def print_response_judgement(i, response, judgement):
    print(f"Response {i + 1}:\n", response)
    print("-" * 125)
    print(f"Judgement {i + 1}:\n", judgement)
    print("-" * 125)

def print_attack(attack_prompt):
    print("*" * 125)
    print("Attack Prompt:\n", attack_prompt)
    print("-" * 125)
