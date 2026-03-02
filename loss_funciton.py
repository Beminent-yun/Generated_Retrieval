import torch
from torch import nn
import torch.nn.functional as F

def InfoNCE(u:torch.tensor, v:torch.tensor, temperature:int):
    """
    In-Batch InfoNCE Loss
    (同一Batch内其他样本作为负例)
    """
    logits = torch.matmul(u, v.T)/temperature
    
    labels = torch.arange(len(u), device=u.device)
    
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))/2
    
    return loss
    