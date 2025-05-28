import re
from datasets import load_dataset
from typing import Literal 

def extract_answer(text):
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return match.group(1)
    else:
        return "No matching answer."
    
def format_example_gsm8k(dataset, idx): 
    question = dataset[idx]["question"]
    answer = extract_answer(dataset[idx]["answer"])
    return question, answer

def gen_gsm8k_dataset(data_dir, phase: Literal["train", "test"]): 
    dataset = []
    origin_dataset = load_dataset(data_dir, "main")[phase]
    for i in range(origin_dataset.num_rows): 
        question, answer = format_example_gsm8k(origin_dataset, i)
        dataset.append((question, answer))
    return dataset
