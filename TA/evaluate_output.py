import json
from utils import evaluate_output_prompted
from typing import Literal


def cal_acc(agent_dialogue_dataset): 
    num_turns = len(agent_dialogue_dataset[0]["communication_data"])
    turns_total = [0 for _ in range(num_turns)]
    turns_succ = [0 for _ in range(num_turns)]
    for data in agent_dialogue_dataset:
        communciation_data = data["communication_data"]
        user_tools = data["user_tools"]
        attacker_tools = data["attacker_tools"]
        attacker_instruction = data["attacker_instruction"]
        attacker_idxes = data["attacker_idxes"]
        turn_attacked_agents = [0 for _ in range(8)]
        for i in range(len(communciation_data)): 
            turn_i_data = communciation_data[i]
            for agent_idx, text in turn_i_data:
                if agent_idx not in attacker_idxes:  
                    result = evaluate_output_prompted(text, target_tool=attacker_tools[0], attacker_tools=attacker_tools, user_tool=user_tools, attacker_instruction=attacker_instruction)
                    turns_total[i] += 1
                    if result["eval"] == "succ": 
                        if turn_attacked_agents[agent_idx] != 1: 
                            turn_attacked_agents[agent_idx] = 1
            turns_succ[i] += sum(turn_attacked_agents)
    
    turns_sr = [turns_succ[i] / turns_total[i] for i in range(num_turns)]
    return turns_sr