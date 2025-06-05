from gat_with_attr_conv import GATwithEdgeConv
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange


class DiaglogueEmbeddingProcessModules(nn.Module):
    def __init__(self, aggr_type, edge_dim, max_turns=3, add_time_emb=False):
        super().__init__()
        self.aggr_type = aggr_type
        self.add_time_emb = add_time_emb 

    def forward(self, diag_emb: torch.Tensor): 
        if self.aggr_type == "last":
            emb = diag_emb[:, -1, :]
        elif self.aggr_type == "mean":
            emb = diag_emb.mean(dim=1)
        # TODO: other method
        else: 
            raise Exception("Not a correct method of aggregation!")
        
        return emb


class MyGAT(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, concat=True, edge_dim=None, num_layers=2, dropout=0.2, residual=False, aggr_type="mean", add_time_emb=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout
    
        self.heads = heads
        self.head_channels = hidden_channels // heads
        self.hidden_channels = self.head_channels * heads

        max_turns, edge_dim = self.edge_dim
        self.convs = nn.ModuleList()
        conv1 = GATwithEdgeConv(in_channels, self.head_channels, heads=heads, concat=concat, edge_dim=edge_dim, residual=residual)
        self.convs.append(conv1)
        for i in range(num_layers-1):
            conv_i = GATwithEdgeConv(self.hidden_channels, self.head_channels, heads=heads, concat=concat, edge_dim=hidden_channels, residual=residual)
            self.convs.append(conv_i)
        
        self.diag_emb_proc = DiaglogueEmbeddingProcessModules(aggr_type, edge_dim, max_turns, add_time_emb)

        self.out = nn.Linear(self.hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr): 
        edge_attr = self.diag_emb_proc(edge_attr)

        for i in range(self.num_layers): 
            x, edge_attr = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)

        return x


if __name__ == "__main__":
    x = torch.randn((2, 100))
    edge_index = torch.tensor([[0], [1]])
    model = MyGAT(100, 200, 1, heads=2, concat=True)
    y = model(x, edge_index)
    print(y)