import json
from utils import get_tool_dict 


def format_examples(dataset, idx, tool_dict): 
    item = dataset[idx]
    available_tool_names = [item['User Tool']] + item["Attacker Tools"]
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    user_instruction = item["User Instruction"]
    agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    attacker_tools = item["Attacker Tools"]
    user_tools = item["User Tool"]
    attacker_instruction = item["Attacker Instruction"]
    
    return user_instruction, available_tool_names, available_tool_descriptions,  agent_scratchpad, attacker_instruction, user_tools, attacker_tools


def gen_injecagent_data(datapath, phase="train"): 
    dataset = []
    with open(datapath, "r") as f:
        data_json = json.load(f)
    tool_dict = get_tool_dict()
    
    for i in range(len(data_json)):
        example = format_examples(data_json, i, tool_dict)
        dataset.append(example)
    
    if phase == "train": 
        dataset = dataset[:int(len(dataset)*0.8)]
    else:
        dataset = dataset[int(len(dataset)*0.8):]
        
    return dataset