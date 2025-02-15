import json
from utils import get_tool_dict 


def format_examples(dataset, idx):
    item = dataset[idx]
    query = item["question"]
    correct_answer = item["correct answer"]
    incorrect_answer = item["incorrect answer"]
    adv_texts = item["adv_texts"]
    return query, adv_texts, correct_answer, incorrect_answer

def gen_poisonrag_data(datapath): 
    dataset = []
    with open(datapath, "r") as f:
        data_json = json.load(f)
    
    for id in data_json.keys():
        example = format_examples(data_json, id)
        dataset.append(example)
    
    return dataset
