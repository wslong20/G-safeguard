import torch
from typing import Literal
import pickle
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.transforms import to_sparse_tensor

class AgentGraphDataset(Dataset): 
    def __init__(self, root, transform=None, phase: Literal["train", "val"]="train"):
        super().__init__()
        with open(root, "rb") as f:
            origin_dataset = pickle.load(f)
        origin_dataset_len = len(origin_dataset)
        if phase == "train": 
            self.dataset = origin_dataset[:int(origin_dataset_len*0.8)]
        elif phase == "val":
            self.dataset = origin_dataset[int(origin_dataset_len*0.8):]
        else:
            raise Exception(f"Unknown phase {phase}")
    
    def len(self):
        return len(self.dataset)
    
    def get(self, idx):
        origin_data = self.dataset[idx]
        origin_data["adj_matrix"]
        x = torch.tensor(origin_data["features"])
        y = torch.tensor(origin_data["labels"], dtype=torch.long)
        edge_index = torch.tensor(origin_data["edge_index"])
        edge_attr = torch.tensor(origin_data["edge_attr"])
        
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = len(x)
        
        return data
    