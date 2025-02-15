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
        np.fill_diagonal(adj_matrix_np, 1)
        labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix)) ])
        system_prompts_embedding = []
        for i in range(len(system_prompts)): 
            system_prompts_embedding.append(embedding_model.encode(system_prompts[i]))
        system_prompts_embedding = np.array(system_prompts_embedding)

        # edge_embedding
        edge_index = adj_matrix_np.nonzero()
        edge_index = np.array(edge_index)
        communication_embeddings = [[] for _ in range(len(communication_data))]
        for i in range(len(communication_data)):
            for src_idx, tgt_idx in zip(edge_index[0], edge_index[1]):
                communication_embeddings[i].append(embedding_model.encode(communication_data[i][tgt_idx][src_idx]))
        
        edge_attr = np.array(communication_embeddings).transpose(1, 0, 2)
        
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
    data_dir = "./agent_graph_dataset/{}/dataset.json".format(meta_dataset)
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