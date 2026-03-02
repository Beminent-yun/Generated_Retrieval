from Amazon_Dataset import get_rec_loaders
import argparse
import pickle
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from metrics import hr_at_k, ndcg_at_k
from typing import Dict, List, Tuple
import numpy as np
from models.Transformer import CausalTransformer


def build_sid_to_item(semantic_ids:np.ndarray) -> Dict[tuple, List[int]]:
    """
    构建 Lookup Table {(c0, c1, c2): [item_id, ...]}
    为什么不是List而是单个int
        两个不同的item可能拥有相同的语义ID (发生碰撞)
        虽然概率极低(256^3 >> num_items)，但需要处理
    """
    sid2item = {}
    for item_id, sid in enumerate(semantic_ids):
        sid = tuple(sid)
        if sid not in sid2item:
            sid2item[sid] = []
        sid2item[sid].append(item_id)
    return sid2item


def calculate_metrics(
    recommended: List[int],
    target:int,
    topk:List[int]
)->Dict[str, float]:
    """
    对单个用户计算所有K值的指标：
    Returns: {"HR@1": ..., "NDCG@1": ..., "HR@5": ..., ...}
    """
    metrics = {}
    for k in topk:
        metrics[f"HR@{k}"] = hr_at_k(recommended, target, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(recommended, target, k)
    return metrics


def print_metrics(
    metrics:Dict[str, float],
    topk: List[int],
    prefix: str = ""
):
    """
    格式化打印评估指标
    """
    parts = []
    for k in topk:
        parts.append(
            f"HR@{k}={metrics[f'HR@{k}']:.4f} "
            f"NDCG@{k}={metrics[f'NDCG@{k}']:.4f}"
        )
    line = " | ".join(parts)
    print(f'{prefix} | {line}'if prefix else line)

    

def beam_to_candidate(
    beams: torch.tensor,    # [batch_size, beam_size, L]
    sid2item: Dict[tuple, List[int]],
    code_offset:int = 3
)-> List[List[int]]:
    """
    把 Beam Search 输出结果转换成推荐 item 列表
    Input: 
        - beams: [batch_size, beam_size, num_rq_layers], token ID(含code_offset)
    Output:
        - List[List[int]], 外层->用户，内层->推荐列表（按beam search结果排序）
    """
    num_user, beam_size, L = beams.shape
    all_candidates = []
    
    for u in range(num_user):
        candidates = []
        seen = set()
        
        for beam_idx in range(beam_size):
            # token ID -> raw code
            raw_codes = tuple(
                beams[u, beam_idx, l].item() - code_offset for l in range(L)
            )
            # 查表
            items = sid2item.get(raw_codes, [])
            for item in items:
                if item not in seen:
                    candidates.append(item)
                    seen.add(item)
        all_candidates.append(candidates)
    
    return all_candidates


@torch.inference_mode()
def evaluate(
    model: CausalTransformer,
    loader: DataLoader,
    sid2item: Dict[tuple, List[int]],
    topk: List[int],
    beam_size:int,
    device:str,
    split:str = 'val'
) -> Dict[str, float]:
    """
    在给定DataLoader上完整评估
    流程：
        1. Beam Search 生成候选语义ID
        2. 查表转换成候选 item 列表
        3. 逐用户计算 HR@K 和 NDCG@K
        4. 返回所有指标的均值
    params:
        - split:日志前缀，'val'或'test'
    Returns:
        {"HR@1": 0.032, "NDCG@1": 0.032,
       "HR@5": 0.098, "NDCG@5": 0.071, ...}
    """
    
    print(f"Using {device}..")
    
    model.eval()
    
    # 初始化累计指标
    total_metrics = {f"HR@{k}":0.0 for k in topk}
    total_metrics.update({f"NDCG@{k}":0.0 for k in topk})
    total_num_users = 0
    
    for batch in tqdm(loader, desc=f"[{split}]", leave=False):
        input_ids = batch['input_ids'].to(device)   # [B, T]
        attention_mask = batch['attention_mask'].to(device) # [B, T]
        target_items = batch['target_item'].squeeze(-1).tolist()   # List[int]
        
        num_user = input_ids.size(0)
        
        # Beam Search: 生成候选语义ID
        beams = model.generate_beam(
            input_ids=input_ids,
            attention_mask=attention_mask,
            beam_size=beam_size
        )
        
        # 转换成推荐列表
        all_candidates = beam_to_candidate(beams, sid2item)
        
        # 逐用户计算指标并累计
        for u in range(num_user):
            user_metrics = calculate_metrics(
                recommended=all_candidates[u],
                target=target_items[u],
                topk=topk
                )
            for key, val in user_metrics.items():
                total_metrics[key] += val
        total_num_users += num_user
    
    # 取均值
    for key in total_metrics:
        total_metrics[key] /= total_num_users
    
    return total_metrics


# 加载 checkpoint 评估
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--beam_size', type=int, default=20)
    args = parser.parse_args()
    
    # 加载checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt['config']
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"From Epoch {ckpt['epoch']}")
    
    # Loading Data
    with open(config['data_path'], 'rb') as f:
        data = pickle.load(f)
    semantic_ids = np.load(config['sid_path'])
    sid2item = build_sid_to_item(semantic_ids)
    
    # Build DataLoader
    _, val_loader, test_loader, vocab_size = get_rec_loaders(
        data=data,
        semantic_ids=semantic_ids,
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        num_rq_layers=config['num_rq_layers'],
        num_workers=config['num_workers']
    )
    loader = test_loader if args.split == 'test' else val_loader
    
    # initialize model
    max_tokens = 1 + config['max_seq_len'] * config['num_rq_layers']
    model = CausalTransformer(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        num_head=config['num_head'],
        num_layers=config['num_layers'],
        dim_ffn=config['dim_feedforward'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=max_tokens+config['num_rq_layers'],
        num_rq_layers=config['num_rq_layers'],
        codebook_size=config['codebook_size']
    ).to(device)
    
    model.load_state_dict(ckpt['model_state'])
    
    metrics = evaluate(
        model=model,
        loader=loader,
        sid2item=sid2item,
        topk=config['topk'],
        beam_size=args.beam_size,
        device=device,
        split=args.split.capitalize()
    )
    
    print(f"\n{args.split.capitalize()} 结果：")
    print_metrics(metrics, config['topk'])
    

if __name__ == "__main__":
    main()