import random
import json
import pickle
import asyncio
from typing import Literal
from sentence_transformers import SentenceTransformer
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import numpy as np 
from agent_prompts import SYS_PROMPT, ATTACKER_SYS_PROMPT, USER_PROMPT
from utils import get_tool_dict
from get_tool_attack_data import gen_injecagent_data
from agents import AgentGraph
import os


def generate_directed_graph_dataset(num_nodes, sparsity, num_graphs):
    if not (0 <= sparsity <= 1):
        raise ValueError("Sparsity must be a value between 0 and 1.")
    
    graph_dataset = []
    
    for _ in range(num_graphs):
        adj_matrix = np.random.rand(num_nodes, num_nodes) <= sparsity
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = adj_matrix.astype(int)
        
        graph_dataset.append(adj_matrix)
    
    return graph_dataset


def generate_agent_graph_dataset(num_nodes, sparsity, num_graphs, num_attackers): 
    assert num_attackers < num_nodes
    dataset = []
    adj_matrices = generate_directed_graph_dataset(num_nodes, sparsity, num_graphs)
    for adj_m in adj_matrices: 
        random_attacker_idxes = random.sample(range(0, num_nodes), num_attackers)
        normal_prompt = "Your are agent_{}. " + SYS_PROMPT
        attacker_prompt = "You are agent_{}. " + ATTACKER_SYS_PROMPT

        data = {}
        data["adj_matrix"] = adj_m
        data["attacker_idxes"] = random_attacker_idxes
        data["normal_prompt"] = normal_prompt
        data["attacker_prompt"] = attacker_prompt
        dataset.append(data)
    return dataset


def generate_initial_data(ag_data, example): 
    new_data = {}
    adj_m = ag_data["adj_matrix"]
    attacker_idxes = ag_data["attacker_idxes"]
    normal_prompt = ag_data["normal_prompt"]
    attacker_prompt = ag_data["attacker_prompt"]
    agent_system_prompts = []
    for idx in range(adj_m.shape[0]): 
        if idx in attacker_idxes: 
            agent_idx_prompt = attacker_prompt.format(idx)
        else:
            agent_idx_prompt = normal_prompt.format(idx)
        agent_system_prompts.append(agent_idx_prompt)
    new_data["adj_matrix"] = adj_m
    new_data["attacker_idxes"] = attacker_idxes
    new_data["system_prompts"] = agent_system_prompts
    new_data["user_instruction"] = example[0]
    new_data["available_tool_names"] = example[1]
    new_data["available_tool_descriptions"] = example[2]
    new_data["agent_scratchpad"] = example[3]
    new_data["attacker_instruction"] = example[4]
    new_data["user_tools"] = example[5]
    new_data["attacker_tools"] = example[6]
    
    return new_data


async def generate_graph_dataset(args): 
    cases_dataset = gen_injecagent_data(args.dataset_path, phase=args.phase)
    ag_dataset = generate_agent_graph_dataset(num_nodes=args.num_nodes, sparsity=args.sparsity, num_graphs=args.num_graphs, num_attackers=args.num_attackers)
    initial_dataset = []
    for agent_graph in tqdm(ag_dataset, desc="Generate meta data"):
        for case in cases_dataset: 
            initial_data = generate_initial_data(agent_graph, case)
            initial_dataset.append(initial_data)
    
    random.shuffle(initial_dataset)
    sampled_initial_dataset = initial_dataset[:args.samples]
    # return sampled_initial_dataset
    final_dataset = []
    for d in tqdm(sampled_initial_dataset, desc="Generate communication data"): 
        adj_m = d["adj_matrix"]
        attacker_idxes = d["attacker_idxes"]
        system_prompts = d["system_prompts"]
        user_instruction = d["user_instruction"]
        available_tool_names = d["available_tool_names"]
        available_tool_descriptions = d["available_tool_descriptions"]
        agent_scratchpad = d["agent_scratchpad"]
        case = (user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad)
        ag = AgentGraph(adj_m, system_prompts, attacker_idxes, model_type=args.model_type)
        communication_data = []
        initial_responses = await ag.afirst_generate(case)
        communication_data.append(initial_responses)
        for _ in range(args.num_dialogue_turns): 
            responses = await ag.are_generate()
            communication_data.append(responses)
        d["communication_data"] = communication_data
        d["adj_matrix"] = d["adj_matrix"].tolist()
        final_dataset.append(d)

    with open(args.save_filepath, "w") as file:
        json.dump(final_dataset, file, indent=None) 

    return final_dataset



if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Experiments that generate dataset")

        parser.add_argument("--dataset_path", type=str, default="./datasets/attack_unsucc_data.json", help="The path to store the dataset")
        parser.add_argument("--dataset", type=str, default="tool_attack")
        parser.add_argument("--num_nodes", type=int, default=8)
        parser.add_argument("--sparsity", type=float, default=1.0, help="Sparsity of the edges (0 to 1), where higher values indicate denser graphs. 1 represents complete graph.")
        parser.add_argument("--num_graphs", type=int, default=20)
        parser.add_argument("--num_attackers", type=int, default=4)
        parser.add_argument("--num_dialogue_turns", type=int, default=3)
        parser.add_argument("--samples", type=int, default=500)
        parser.add_argument("--save_dir", type=str, default="./agent_graph_dataset")
        parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
        parser.add_argument("--phase", type=str, default="train")
        parser.add_argument("--save_filepath", type=str)

        args = parser.parse_args()

        args.save_dir = os.path.join(args.save_dir, args.dataset, args.phase)
        if not os.path.exists(args.save_dir): 
            os.makedirs(args.save_dir)
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_filepath = os.path.join(args.save_dir, f"{current_time_str}-dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}.json")

        return args

    args = parse_arguments()
    dataset = asyncio.run(generate_graph_dataset(args))
    print(len(dataset))