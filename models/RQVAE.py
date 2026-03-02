import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple, List

class VectorQuantizer(nn.Module):
    """
    单层向量量化
    input: 
        - Dense Vector z: [B, D]
    output:
        - Quantized Vector z_q: 最近码字索引 indices
    """
    def __init__(self, codebook_size:int, embed_dim:int, commitment_cost:float):
        super().__init__()
        self.codebook_size = codebook_size  # K
        self.embed_dim = embed_dim  # D: Dimension of embedding vector
        self.commitment_cost = commitment_cost  # beta
        
        self.codebook = nn.Embedding(codebook_size, embed_dim)  # [K, D]
        
        nn.init.uniform_(self.codebook.weight, -1/codebook_size, 1/codebook_size)
    
    def forward(self, z: torch.tensor):
        """
        input:
            - z: [B, D]
        output:
            - z_q: [B, D]
            - loss: scalar [B,]
        """
        # 1. Compute all distances between z and all codebook vector
        # 矩阵乘法优化：||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        z_sq = (z**2).sum(dim=-1, keepdim=True) # [B, 1]
        e_sq = (self.codebook.weight**2).sum(dim=-1)    # [K,]
        ze = z @ self.codebook.weight.transpose(-2, -1) # [B, D]@[D,K] -> [B, K]
        
        distances = z_sq + e_sq - 2*ze  # [B, K]    (make use of broadcast)
        
        # 2. find the closest codebook vector
        indices = distances.argmin(dim=-1)  # [B,]
        z_q = self.codebook(indices)    # 码字 [B, D]

        # 3. Calculate VQ loss
        # codebook loss: "固定"z, 让码字向z靠近
        codebook_loss = F.mse_loss(z_q, z.detach())
        # commitment_loss: "固定"码字z_q, 让z向码字靠近
        commitment_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + commitment_loss
        
        # 4. Straight-Through Estimator: 梯度垂直技巧
        # 前向用 z_q, 反向传播时将 z_q 用 z 替代
        z_q = z + (z_q - z).detach()
        
        return z_q, vq_loss, indices
    
    
class VectorQuantizerEMA(nn.Module):
    """
    单层 VQ, 使用EMA(指数移动平均)更新码本
    
    -为什么用 EMA 而不是梯度更新码本：
        直接用梯度更新码本，码本向量之间会互相竞争
        EMA 用统计量更新，更稳定，不容易 collapse
    
    EMA 更新公式：
        N_k = decay × N_k + (1 - decay) × n_k
        m_k = decay × m_k + (1 - decay) × sum(z_k)
        e_k = m_k / N_k
    其中：
        N_k: 码字k被分配到的次数（指数平滑）
        m_k: 被分配到码字k的向量之和（指数平滑）
        e_k: 码字 k 的向量 （= 分配到它的所有向量的加权平均）
    """
    def __init__(self,
                 codebook_size:int,     # K = 256
                 embed_dim:int,         # d: latent dim
                 decay:float = 0.99,    # EMA衰减率，越大越记忆过去
                 epsilon:float = 1e-5,  # 防止除以0
                 commitment_weight:float = 0.25  # commitment loss 的权重
                 ):
        super().__init__()
        self.codebook_size = codebook_size  # K
        self.embed_dim = embed_dim  # d
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_weight = commitment_weight
        
        # 码本：K个d维向量
        # 用 register_buffer 而不是 nn.Parameter，因为EMA更新不走梯度，但需要随模型保存
        self.register_buffer('codebook', torch.randn(codebook_size, embed_dim))
        self.register_buffer('ema_count', torch.zeros(codebook_size))
        self.register_buffer('ema_weight', torch.randn(codebook_size, embed_dim))
        
        # 归一化码本初始向量（让初始码本分布平均）
        self.codebook = F.normalize(self.codebook, dim=-1)
        self.ema_weight = self.codebook.clone()
        
    def forward(self, z:torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Input:
            - z [B, d]: encoder输出或上一层的残差
        Output:
            - z_q [B, d]: 量化后的向量 (最近码字)
            - codes [B, ]: 每个样本被分配到的码字索引
            - loss: 量化损失
        """
        B, d = z.shape
        
        # vq 用欧式距离计算：||z - e||² = ||z||² + ||e||² - 2 z·e^T 
        # 等价于先normalized, 再用 2(1 - cosine similarity) <- 对语义embedding效果更好
        z_norm = F.normalize(z, dim=-1) # [B, d]
        cb_norm = F.normalize(self.codebook, dim=-1)    # [K, d]
        
        # 余弦相似度矩阵
        similarity = z_norm @ cb_norm.T # [B, K]
        
        # 取最大相似度对应的码字索引（等价与最小距离）
        code_ids = similarity.argmax(dim=-1)   # [B,]
        
        z_q = self.codebook[code_ids]   # [B, d]
        
        # EMA更新码本（只在训练时）
        if self.training:
            with torch.no_grad():
                # 统计每个码字被分配了多少次
                # one-hot: [B, K], 每行第code_ids[i]位为1
                one_hot = F.one_hot(code_ids, self.codebook_size).float()
                
                # n_k: 这个batch内码字k被分配的次数
                n_k = one_hot.sum(dim=0)    # [K,]
                
                # sum_k: 被分配到的码字 k 的所有 z 的加权求和
                sum_k = one_hot.T @ z   # [B, K] @ [B, d] -> [K, d]
                
                # EMA更新
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * n_k
                self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * sum_k
                # 用EMA统计量更新码本；显式扩展维度避免广播形状歧义
                # ema_count: [K] -> [K, 1] so division matches ema_weight [K, d]
                self.codebook = self.ema_weight / (self.ema_count.unsqueeze(-1) + self.epsilon)
            
        # Calculate Weight
        # Commitment Loss: 让encoder的输出靠近码本(z靠近z_q)
        # z_q.detach(): 不让梯度流向码本(码本用EMA更新，不用梯度)
        loss = self.commitment_weight * F.mse_loss(z, z_q.detach())

        # Straight-Through Estimator (STE)
        # 前向传播用 z_q（量化后的值）
        # 反向传播时梯度直接绕过量化操作，从 z_q 流向 z
        # 原理：z_q - z_q.detach() + z = z（前向），但梯度从 z_q 流向 z
        z_q = z + (z_q - z).detach()    # [B, d]
        
        return z_q, code_ids, loss
                

    @property
    def codebook_utilization(self) -> float:
        """
        码本利用率：有多少码字被实际使用过
        
        判断标准：ema_count > 1 (至少被分配过1次)
        利用率低（<50%）说明在发生 codebook collapse
        """
        return (self.ema_count > 1).float().mean().item()
    
    
    
    
class ResidualQuantizer(nn.Module):
    """
    Residual Quantizer
    每层量化当前残差，然后把残差传给下一层
    """
    def __init__(self, num_layer:int, 
                 codebook_size:int, 
                 embed_dim:int, 
                 decay:float=0.99,
                 commitment_cost:int=0.25):
        super().__init__()
        self.num_layer = num_layer  # L
        
        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(
                codebook_size,
                embed_dim,
                decay=decay,
                commitment_weight=commitment_cost,
            ) for _ in range(num_layer)
        ])
    
    def forward(self, z:torch.tensor):
        """
        z: original vector [B, D]
        Return:
            - z_q: [B, D]   Reconstructed/ Quantized Vector
            - total_loss: Total VQ Loss
            - all_indices: [B, L] 每层的code索引，即 Semantic ID
        """
        residual = z
        z_q_total = torch.zeros_like(z)
        total_loss = 0.0
        all_indices = []
        for quantizer in self.quantizers:
            z_q_i, indices_i, vq_loss_i = quantizer(residual)
            # avoid in-place on residual to keep autograd versions consistent
            residual = residual - z_q_i.detach()
            total_loss += vq_loss_i
            all_indices.append(indices_i)   # [B,]
            # avoid in-place accumulation on a tensor that participates in autograd
            z_q_total = z_q_total + z_q_i
        all_indices = torch.stack(all_indices, dim=1)  # 拼接成[B, L]
        
        return z_q_total, total_loss, all_indices
    
    @torch.inference_mode()
    def encode(self, z):
        """
        推理时只需要索引，不需要梯度
        """
        _, _, semantic_indices = self(z)
        return semantic_indices # [B, L]
    
    @torch.inference_mode()
    def decode_from_indices(self, indices):
        """
        从 Semantic ID 还原量化向量 （用于推荐时的向量还原）
        Returns:
            - z_q: [B, D]
        """
        # indices [B, L]
        z_q = torch.zeros(indices.shape[0], 
                          self.quantizers[0].embed_dim, 
                          device=indices.device)
        for i, quantizer in enumerate(self.quantizers):
            z_q += quantizer.codebook[indices[:, i]]
        
        return z_q  # [B, D]

    @property
    def utilization_per_layer(self) -> List[float]:
        return [q.codebook_utilization for q in self.quantizers]
        



class RQVAE(nn.Module):
    def __init__(self, input_dim:int,   # dimension of item embedding   
                 hidden_dim:int,    # dimension of neural netword hidden layer
                 latent_dim:int,    # dimension of latent vector of VAE 
                 num_layers:int,    # num of RQ layers
                 codebook_size:int, # size/dimension of codebook vector
                 decay:float = 0.99,
                 commitment_cost:float = 0.25,
                 dropout_rate:float = 0.1
                 ):
        super().__init__()
        
        # Encoder: item_embedding -> latent vector z
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # RQ: latent vector z -> quantized vector z_q
        self.rq = ResidualQuantizer(num_layers, codebook_size, latent_dim, decay, commitment_cost)
        
        # Decoder: quantized vector z_q -> reconstructed item_embedding
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x:torch.tensor)->torch.tensor:
        """
        x: item_embedding [B, input_dim]
        """
        z = self.encoder(x) # [B, input_dim] -> [B, latent_dim]
        z_q, vq_loss, semantic_ids = self.rq(z)
        x_recon = self.decoder(z_q) # [B, latent_dim] -> [B, input_dim]
        
        return x_recon, vq_loss, semantic_ids
    
    
    def compute_loss(self, x_recon, x, vq_loss):
        
        # Reconstruction Loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Total Loss
        total_loss = recon_loss + vq_loss
        
        return total_loss, {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss
        }
    
    
    @torch.inference_mode()
    def generate_semantic_ids(self, x):
        """
        Interface: item_embedding -> semantic_ids
        """
        self.eval()
        z = self.encoder(x)
        semantic_ids = self.rq.encode(z)
        return semantic_ids
        
    
    