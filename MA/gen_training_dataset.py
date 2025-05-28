import os 
import json
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def gen_model_training_set(language_dataset, embedding_model, save_path): 
    dataset = []
    for meta_data in tqdm(language_dataset, desc="Generate training data"): 
        adj_matrix = meta_data["adj_matrix"]
        attacker_idxes = meta_data["attacker_idxes"]
        system_prompts = meta_data["system_prompts"]
        communication_data = meta_data["communication_data"]

        
        adj_matrix_np = np.array(adj_matrix)
        labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix)) ])
        system_prompts_embedding = []
        for i in range(len(system_prompts)): 
            system_prompts_embedding.append(embedding_model.encode(system_prompts[i]))
        system_prompts_embedding = np.array(system_prompts_embedding)

        # edge_embedding
        edge_index = adj_matrix_np.nonzero()
        edge_index = np.array(edge_index)
        communication_embeddings = [[] for _ in range(len(adj_matrix))]
        for i in range(len(communication_data)):
            turn_i_data = communication_data[i]
            turn_i_embeddings = [None] * len(turn_i_data)
            for agent_idx, c_data in turn_i_data: 
                i_turns_agent_idx_embedding = embedding_model.encode(c_data)
                turn_i_embeddings[agent_idx] = i_turns_agent_idx_embedding
            for agent_idx in range(len(turn_i_embeddings)): 
                communication_embeddings[agent_idx].append(turn_i_embeddings[agent_idx])
        
        communication_embeddings = np.array(communication_embeddings)
        edge_attr = np.array(communication_embeddings[edge_index[1]], copy=True)  # edge_index[1]表示目的节点， 因为每个人说的话要作为目的节点传到目的节点
        
        data = {}
        data["adj_matrix"] = adj_matrix_np
        data["features"] = system_prompts_embedding
        data["labels"] = labels    
        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        
        dataset.append(data)
        
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    meta_dataset = "memory_attack"
    data_dir = "./agent_graph_dataset/{}/train/dataset.json".format(meta_dataset)
    save_dir = f"./ModelTrainingSet/{meta_dataset}"
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "dataset.pkl")
    embedding_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_dir)
    with open(data_dir, 'r') as file:
        language_dataset = json.load(file)
    
    gen_model_training_set(language_dataset, embedding_model, save_path)
    with open(save_path, "rb") as f: 
        loaded_dataset = pickle.load(f)    
    print(len(loaded_dataset))
    print(loaded_dataset[0])