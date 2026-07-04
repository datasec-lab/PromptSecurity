# attacks/black_box/ABJAttack/utils/random_split.py

import random

def split_sentence_randomly(sentence: str):
    """
    Split a sentence into words and then randomly divides each word into two parts.
    Combine these parts into two separate sentences.
    """
    words = sentence.split()
    if len(words) < 2:
        # Minimal fallback
        return sentence, sentence, "", ""

    first_word, second_word = words[0], words[1]

    sentence1_parts, sentence2_parts = [], []
    for word in words:
        if len(word) > 1:
            split_point = random.randint(1, len(word) - 1)
            part1, part2 = word[:split_point], word[split_point:]
        else:
            part1, part2 = word, ' '
        sentence1_parts.append(part1)
        sentence1_parts.append(',')
        sentence2_parts.append(part2)
        sentence2_parts.append(',')

    sentence1_parts = sentence1_parts[:-1]
    sentence2_parts = sentence2_parts[:-1]
    sentence1 = f"[{''.join(sentence1_parts)}]"
    sentence2 = f"[{''.join(sentence2_parts)}]"

    return sentence1, sentence2, first_word, second_word
