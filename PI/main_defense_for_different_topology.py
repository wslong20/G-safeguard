import os
from model import MyGAT
from agents import AgentGraphWithDefense, AgentGraph
from tqdm import tqdm
import json
import random
import numpy as np
import torch
from utils import get_sentence_embedding
from einops import rearrange
from torch_scatter import scatter_mean
import argparse 
from datetime import datetime
import asyncio
import copy
import time
from utils import get_adj_matrix
os.environ["BASE_URL"] = 'https://api2.aigcbest.top/v1'
os.environ["OPENAI_API_KEY"] = 'sk-iz4cyOsIWbpvsbunhwMXfnQ18UBSYj8484RUuawUdEqTMcig'

def response2embeddings(responses): 
    embeddings = [None for _ in range(len(responses))]
    for agent_idx, agent_response in responses: 
        embeddings[agent_idx] = get_sentence_embedding(agent_response)
    
    embeddings = np.array(embeddings)
    return embeddings


def embeddings2graph(embeddings, adj_matrix):
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()))
    edge_attr = torch.tensor(np.array(embeddings))[:, edge_index[1]]  # 仅仅针对统一回复的情况

    x = edge_attr[0, :]
    x = scatter_mean(x, edge_index[1], dim=0, dim_size=len(embeddings[0]))
    edge_attr = edge_attr.transpose(0, 1)
    return x, edge_index, edge_attr


async def no_defense_communication(ag: AgentGraph, qa_data, num_dialogue_turns): 
    communication_data = []
    initial_responses = await ag.afirst_generate(qa_data)
    communication_data.append(initial_responses)
    for _ in range(num_dialogue_turns): 
        responses = await ag.are_generate()
        communication_data.append(responses)
    return communication_data


async def defense_communication(ag:AgentGraphWithDefense, gnn: MyGAT, qa_data, adj_m: np.ndarray,  num_dialogue_turns): 
    communication_data = []
    response_embeddings = []
    initial_responses = await ag.afirst_generate(qa_data)
    embeddings = response2embeddings(initial_responses)
    response_embeddings.append(embeddings)
    x, edge_index, edge_attr = embeddings2graph(response_embeddings, adj_m)
    predicts = torch.sigmoid(gnn(x, edge_index, edge_attr).squeeze(-1))>=0.5
    for idx, predict in enumerate(predicts): 
        if predict == 1: 
            ag.agents[idx].set_role("attacker")
    communication_data.append(initial_responses)

    for _ in range(num_dialogue_turns): 
        responses = await ag.are_generate()
        embeddings = response2embeddings(responses)
        response_embeddings.append(embeddings)
        x, edge_index, edge_attr = embeddings2graph(response_embeddings, adj_m)
        predicts = torch.sigmoid(gnn(x, edge_index, edge_attr).squeeze(-1))>=0.5
        for idx, predict in enumerate(predicts): 
            if predict == 1: 
                ag.agents[idx].set_role("attacker")
        communication_data.append(responses)

    return communication_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train GAT")

    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--graph_type", type=str, choices=["random", "chain", "tree", "star"], default="chain")
    parser.add_argument("--gnn_checkpoint_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./result")
    parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
    parser.add_argument("--samples", type=int, default=60)

    args = parser.parse_args()


    if args.dataset == "mmlu": 
        args.dataset_path = "./agent_graph_dataset/mmlu/test/dataset.json"
    elif args.dataset == "csqa": 
        args.dataset_path = "./agent_graph_dataset/csqa/test/dataset.json"
    elif args.dataset == "gsm8k": 
        args.dataset_path = "./agent_graph_dataset/gsm8k/test/dataset.json"
    else: 
        raise Exception(f"Unknown dataset {args.dataset}")
    
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.graph_type)

    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_no_defense = f"{current_time_str}-no_defense-model_type_{args.model_type}.json"
    filename_defense = f"{current_time_str}-defense-model_type_{args.model_type}.json"
    args.save_path_no_defense = os.path.join(args.save_dir, filename_no_defense)
    args.save_path_with_defense = os.path.join(args.save_dir, filename_defense)

    return args


async def main(): 
    args = parse_arguments()
    filepath = args.dataset_path
    graph_type = args.graph_type
    with open(filepath, "r") as f:
        dataset = json.load(f)
    dataset_len = len(dataset)
    dataset = dataset[-args.samples:]
    num_dialogue_turns = len(dataset[0]["communication_data"])-1


    gnn = MyGAT(in_channels=384, hidden_channels=1024, out_channels=1, heads=8, edge_dim=(3, 384))
    state_dict = torch.load(args.gnn_checkpoint_path, map_location=torch.device('cpu'))
    gnn.load_state_dict(state_dict)

    final_dataset_nd = []
    final_dataset_wd = []
    for d in tqdm(dataset): 
        if graph_type == "random": 
            adj_m = np.array(d["adj_matrix"])
        elif graph_type in ["chain", "tree", "star"]: 
            adj_m = get_adj_matrix(graph_type, len(d["adj_matrix"]))
        else:
            raise Exception(f"Unknown graph type: {graph_type}! Can only be one of [random, chain, tree, star]")
        attacker_idxes = d["attacker_idxes"]
        system_prompts = d["system_prompts"]
        qa_data_origin = d["question"], d["correct_answer"], d["wrong_answer"]
        wrong_answer = random.choice(qa_data_origin[2]) if qa_data_origin[2] else None
        qa_data = (qa_data_origin[0], qa_data_origin[1], wrong_answer)

        try:
            agnd = AgentGraph(adj_m, system_prompts, attacker_idxes, model_type=args.model_type)  # agent graph no defense
            agwd = AgentGraphWithDefense(adj_m, system_prompts, attacker_idxes, model_type=args.model_type)  # agent graph with defense
            
            communication_data_no_defense = await no_defense_communication(agnd, qa_data, num_dialogue_turns)
            communication_data_defense = await defense_communication(agwd, gnn, qa_data, adj_m, num_dialogue_turns)
        except Exception as e: 
            print(e)
            continue
            
        d_nd = copy.deepcopy(d)
        d_wd = copy.deepcopy(d)
        d_nd["communication_data"] = communication_data_no_defense
        d_wd["communication_data"] = communication_data_defense

        final_dataset_nd.append(d_nd)
        final_dataset_wd.append(d_wd)
    
    with open(args.save_path_no_defense, "w") as file:
        json.dump(final_dataset_nd, file, indent=None) 
    with open(args.save_path_with_defense, "w") as file:
        json.dump(final_dataset_wd, file, indent=None) 


if __name__ == "__main__": 
    asyncio.run(main())


