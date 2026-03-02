import torch
from torch import nn
import torch.nn.functional as F


class PositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()


class CausalTransformer(nn.Module):
    def __init__(self, 
                 vocab_size:int,
                 d_model:int = 128,
                 num_head:int = 4,
                 num_layers:int = 4,
                 dim_ffn:int = 512,
                 max_seq_len:int = 200,
                 dropout_rate:float = 0.1,
                 num_rq_layers:int = 3,
                 codebook_size:int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_rq_layers = num_rq_layers
        self.codebook_size = codebook_size
        self.CODE_OFFSET = 3
        
        # Token Embedding   token 表示
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Position Encoding  -> 区分每一个token
        self.pos_enc = PositionEncoding(d_model, max_seq_len, dropout_rate)
        # rq position encoding  -> 区分每一层
        self.rq_pos_emb = nn.Embedding(num_rq_layers, d_model)
        
        # Transformer Decoder-only (Causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_head,
            dim_feedforward=dim_ffn,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True # pre-LayerNorm, 训练更稳定
        )
        self.transformer = nn.TransformerDecoder(encoder_layer, num_layers)
        
        # 输出头/投影头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重绑定 (输入emb & 输出头共享权重，减少训练参数)
        self.lm_head.weight = self.token_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _make_causal_mask(self, seq_len:int, device) -> torch.tensor:
        """
        上三角 causal mask 
        [[False,  True,  True],
        [False, False,  True],
        [False, False, False]]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
        
    def _make_rq_position_ids(self, seq_len:int, device) -> torch.tensor:
        """
        生成层间 position embedding/ID -> 区分层间ID/位置
        序列:      [BOS, c0_item#0, c1_item#0, c2_item#0, c0_item#1, ...]
        pos_emb:   [0,     0,          1,      2,          0, .      ...]
        """
        positions = [0] # BOS位置为0
        L = self.num_rq_layers  # codebook 个数/RQ层数
        remaining = seq_len - 1 # 除去BOS的序列长度
        for i in range(remaining):
            positions.append(i % L)
        return positions
        
    def forward(self, input_ids:torch.tensor,   # 输入序列: [B, T]
                attention_mask: torch.tensor    # 掩码M: [B, T]
                )->torch.tensor:
        """
        返回 logits [B, T, ,vocab_size]
        """
        
        B, T = input_ids.shape
        device = input_ids.device
        
        # x = token_emb + pos_emb   # [B, T, D] + [B, T, D] = [B, T, D]
        x = self.pos_enc(self.token_emb(input_ids))
        
        # 加入层间位置嵌入
        rq_pos_id = self._make_rq_position_ids(T, device)   # [T, D]
        x = x + self.rq_pos_emb(rq_pos_id).unsqueeze(0) # [B, T, D] + [1, T, D]
        
        # Causal mask (将t+1及其之后填充为-inf, True的位置即为-inf)
        causal_mask = self._make_causal_mask(T, device) # [T, T]
        # Padding mask (将<pad>位置填充为-inf, True的位置即为-inf)
        key_padding_mask = (attention_mask==0)  # [B, T]
        
        # Transformer(decoder-only, 用 x 作为memory和target)
        out = self.transformer(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )   # [B, T, D]
        
        logits = self.lm_head(out)  # [B, T, vocab_size]
        
        return logits
    
    def compute_loss(self,input_ids: torch.tensor,  # [B, T]
                     attention_mask:torch.tensor,   # [B, T]
                     target_ids:torch.tensor    # [B, L] 下一个item(目标)的语义ID
                     )->dict:
        """
        Teacher forcing 训练损失
        预测最后L个位置的tokens (即预测下一个item的语义ID)
        """
        logits = self(input_ids, attention_mask)    # [B, T, vocab_size]
        
        # 获取目标item的L个token
        # 即输入ids的最后L个位置的预测ids = 估计目标item的L个ids (?)
        pred_logits = logits[:, -self.num_rq_layers:, :]    # [B, L, vocab_size]
        
        loss = F.cross_entropy(
            pred_logits.reshape(-1, self.vocab_size),   # [B*L, vocab_size]
            target_ids.reshape(-1))
        
        # Compute Accuracy
        pred_ids = pred_logits.argmax(-1)   # [B, L]
        acc = (pred_ids == target_ids).all(dim=1).float().mean()
        
        return {
            "loss": loss,
            "acc": acc
        }
        
    @torch.infernece_mode()
    def generate_beam(self, input_ids:torch.tensor, # [B, T]
                      attention_mask:torch.tensor,  # [B, T]
                      beam_size:int = 20,
                      num_rq_layers:int = 3
                      )->torch.tensor:
        """
        Beam Search Generate Semantic ID
        Returns:
            - Semantic ID: [B, beam_size, L]   Top-{beam_size} candidated semantic ids, sorted by probability
        """
        self.eval()
        B, T = input_ids.shape
        device = input_ids.device
        L = num_rq_layers
        
        # 初始Logits
        logits = self(input_ids, attention_mask)    # [B, T, vocab_size]
        last_logits = logits[:, -1, :]  # [B, V]    预测的第1个id
        
        # 取 top-{beam_size} 个候选
        top_probs, top_ids = torch.topk(
            F.log_softmax(last_logits, dim=-1), beam_size, dim=-1
        )   # [B, beam_size]
        
        # beams: [B, beam_size, L]  当前已经生成的token序列
        beams = torch.zeros((B, beam_size, L), dtype=torch.long, device=device)
        beams_scores = top_probs
        beams[:, :, 0] = top_ids
        
        # 逐步生成后续 token (L-1 步)
        for step in range(1, L):
            # 对每个beam，扩展输入序列
            # 简化：取最高分beam继续拓展(greedy per-beam)
            for b in range(beam_size):
                ext_ids = torch.cat([
                    input_ids,
                    beams[:, b, :step]
                ], dim=1)   # [B, T+step]
                
                ext_mask = torch.cat([
                    attention_mask,
                    torch.ones(B, step, dtype=torch.long, device=device)
                ])
                
                logits_b = self(ext_ids, ext_mask)  # [B, T+step, vocab_size]
                next_logit = logits_b[:, -1, :] # [B, V]
                next_log_prob = F.log_softmax(next_logit, dim=-1)
                
                # 取最佳 token
                best_prob, best_id = next_log_prob.max(dim=-1)  # [B,]
                beams[:, b, step] = best_id
                beams_scores[:, b] += best_prob
                
        # 按分数排序
        sorted_idx = beams_scores.argsort(dim=-1, descending=True)  # [B, beam_size]
        beams = beams.gather(
            1, sorted_idx.unsqueeze(-1).expand(-1, -1, L)
        )
        
        return beams