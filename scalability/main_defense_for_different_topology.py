import os
from model import MyGAT
from agents import AgentGraphWithDefense, AgentGraph
from tqdm import tqdm
import json
import random
import numpy as np
import torch
from gen_graph import get_sentence_embedding
from einops import rearrange
from torch_scatter import scatter_mean
import argparse 
from datetime import datetime
import asyncio
import copy
import time
from utils import get_adj_matrix


def communication_info_2_graph(communication_data, adj_matrix, num_agent):
    adj_matrix_np = np.array(adj_matrix)
    np.fill_diagonal(adj_matrix_np, 1)

    # edge_embedding
    edge_index = adj_matrix_np.nonzero()

    # edge_index = np.array(edge_index)
    communication_embeddings = [[] for _ in range(len(communication_data))]
    for i in range(len(communication_data)):
        for src_idx, tgt_idx in zip(edge_index[0], edge_index[1]):
            communication_embeddings[i].append(get_sentence_embedding(communication_data[i][tgt_idx][src_idx]))  # 这里和gat的实现有点冲突

    edge_index = torch.tensor(np.array(edge_index))
    edge_attr = torch.tensor(np.array(communication_embeddings))
    x = edge_attr[0, :]
    x = scatter_mean(x, edge_index[1], dim=0, dim_size=num_agent)
    edge_attr = edge_attr.transpose(0, 1)

    return x, edge_index, edge_attr

async def no_defense_communication(ag: AgentGraph, query, context, num_dialogue_turns): 
    communication_data = []
    initial_responses = await ag.afirst_generate(query, context)
    communication_data.append(initial_responses)
    for _ in range(num_dialogue_turns): 
        responses = await ag.are_generate()
        communication_data.append(responses)
    return communication_data


async def defense_communication(ag:AgentGraphWithDefense, gnn: MyGAT, query, context, adj_m: np.ndarray,  num_dialogue_turns): 
    communication_data = []
    identified_attackers = []
    inner_communication_data = []
    initial_responses = await ag.afirst_generate(query, context)
    inner_communication_data.append(copy.deepcopy(ag.communication_info_matrix))
    x, edge_index, edge_attr = communication_info_2_graph(inner_communication_data, adj_matrix=adj_m, num_agent=len(adj_m))
    predicts = torch.sigmoid(gnn(x, edge_index, edge_attr).squeeze(-1))>=0.5
    for idx, predict in enumerate(predicts): 
        if predict == 1: 
            ag.agents[idx].set_role("attacker")
    communication_data.append(initial_responses)
    identified = []
    for idx, predict in enumerate(predicts): 
            if predict == 1: 
                ag.agents[idx].set_role("attacker")
                if idx not in identified: 
                    identified.append(idx)
    identified_attackers.append(copy.deepcopy(identified))
    for _ in range(num_dialogue_turns):
        responses = await ag.are_generate()
        inner_communication_data.append(copy.deepcopy(ag.communication_info_matrix))
        x, edge_index, edge_attr = communication_info_2_graph(inner_communication_data, adj_matrix=adj_m, num_agent=len(adj_m))
        predicts = torch.sigmoid(gnn(x, edge_index, edge_attr).squeeze(-1))>=0.5
        for idx, predict in enumerate(predicts): 
            if predict == 1: 
                ag.agents[idx].set_role("attacker")
                if idx not in identified: 
                    identified.append(idx)
        communication_data.append(responses)
        identified_attackers.append(copy.deepcopy(identified))

    return communication_data, identified_attackers


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train GAT")

    parser.add_argument("--dataset_path", type=str, default="./agent_graph_dataset_large/memory_attack/test/dataset.json", help="Save path of the dataset")
    parser.add_argument("--graph_type", type=str, choices=["random", "chain", "tree", "star"], default="random")
    parser.add_argument("--gnn_checkpoint_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./result")
    parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()

    normalized_path = os.path.normpath(args.dataset_path)
    parts = normalized_path.split(os.sep)
    dataset = parts[-2]
    args.save_dir = os.path.join(args.save_dir, dataset, args.graph_type)

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
        query = d["query"]
        context = d["adv_texts"]

        try:
            agnd = AgentGraph(adj_m, system_prompts, attacker_idxes, model_type=args.model_type)  # agent graph no defense
            agwd = AgentGraphWithDefense(adj_m, system_prompts, attacker_idxes, model_type=args.model_type)  # agent graph with defense
            
            communication_data_no_defense = await no_defense_communication(agnd, query, context, num_dialogue_turns)
            communication_data_defense, identified_attackers = await defense_communication(agwd, gnn, query, context, adj_m, num_dialogue_turns)
        except Exception as e: 
            print(e)
            continue
            
        d_nd = copy.deepcopy(d)
        d_wd = copy.deepcopy(d)
        d_nd["communication_data"] = communication_data_no_defense
        d_wd["communication_data"] = communication_data_defense
        d_wd["identified_attackers"] = identified_attackers
        final_dataset_nd.append(d_nd)
        final_dataset_wd.append(d_wd)
    
    with open(args.save_path_no_defense, "w") as file:
        json.dump(final_dataset_nd, file, indent=None) 
    with open(args.save_path_with_defense, "w") as file:
        json.dump(final_dataset_wd, file, indent=None) 


if __name__ == "__main__": 
    asyncio.run(main())


