import torch
from torch import nn
import torch.nn.functional as F

class Tower(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int) -> None:
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), 
                       nn.ReLU(), 
                       nn.BatchNorm1d(dims[i+1])]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return F.normalize(self.net(x), dim=-1) # L2归一化，方便使用点积近似余弦相似度
    
class TwoTower(nn.Module):
    def __init__(self, user_dim:int, item_dim:int, hidden_dims:list =[256, 128], embed_dim:int =64):
        super().__init__()
        self.UserTower = Tower(user_dim, hidden_dims, embed_dim)
        self.ItemTower = Tower(item_dim, hidden_dims, embed_dim)
        self.temperature = nn.Parameter(torch.ones(1)*0.07) # Learnalble Parameter
    
    def forward(self, user_feature: torch.tensor, item_feature: torch.tensor)->torch.tensor:
        u_emb = self.UserTower(user_feature)
        i_emb = self.ItemTower(item_feature)
        return u_emb, i_emb
    
    def similarity(self, u:torch.tensor, v:torch.tensor):
        return torch.sum(u*v, dim=-1)   # 点积相似度
