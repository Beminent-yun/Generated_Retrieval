import os
import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import random
from torch.utils.data import DataLoader

# 优先级：CUDA GPU > Apple MPS > CPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
os.makedirs('./model', exist_ok=True)


def worker_init_fn(worker_id):
    """
    DataLoader 每个 worker 进程的随机种子初始化函数。

    PyTorch 多进程加载时，所有 worker 默认共享相同的 numpy/random 全局随机状态，
    可能导致同一 epoch 内不同 worker 产生相同的随机数序列，违反 i.i.d. 假设。
    此函数为每个 worker 分配独立的、由主进程种子派生的确定性随机种子，
    兼顾可复现性与采样多样性。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class RecommenderTrainer:
    def __init__(self, model, dataset, optimizer, device=device):
        """
        通用推荐模型训练器，负责训练循环、排序指标评估与模型持久化。

        支持功能：
        - 多进程 DataLoader（含 pin_memory 加速 CPU->GPU 传输）
        - 梯度裁剪防止梯度爆炸
        - 基于 NDCG@K 的 Early Stopping
        - 按 NDCG@K 自动保存最优模型到 ./model/best_model.pth

        注意：Early Stopping 触发后仅停止训练，不会自动加载最优权重。
        如需使用最佳模型，训练结束后请手动执行：
            model.load_state_dict(torch.load('./model/best_model.pth'))
            
        Args:
            model     : 待训练的 PyTorch 模型，输出形状应为 (B, 1) 或 (B,) 的 logits
            dataset   : BaseRecommenderDataset 子类实例，需支持 set_stage() 切换
            optimizer : PyTorch 优化器
            device    : 计算设备，默认自动选择
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_fn = nn.BCEWithLogitsLoss()  # 接受原始 logits，内部自动 sigmoid，数值更稳定
        self.device = device

        self.best_ndcg = -np.inf  # 历史最优 NDCG@K，用于判断是否保存模型
        self.patience = 5         # 连续多少个 epoch NDCG 未提升则触发 Early Stopping
        self.counter = 0          # 未提升计数器

    def train_one_epoch(self, train_loader):
        """
        执行一个完整的训练 epoch，返回平均 batch loss。

        Args:
            train_loader : 已配置好的训练 DataLoader（由 fit() 负责构建并传入）
        """
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc='Training'):
            X, y = batch['X'].to(self.device), batch['y'].to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(X)

            # squeeze(-1)：仅压缩最后一维，将 (B,1) -> (B,)，避免 batch_size=1 时
            # squeeze() 将张量压成标量导致 loss 维度不匹配
            loss = self.loss_fn(y_pred.squeeze(-1), y)
            total_loss += loss.item()
            loss.backward()

            # 梯度裁剪：将所有参数的梯度 L2 范数限制在 5.0 以内，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

        return total_loss / len(train_loader)

    def evaluate_ranking(self, test_loader, k=10):
        """
        基于预存储负样本的排序评估，计算 HR@K 和 NDCG@K。

        测试集结构说明：
          - 负样本在 dataset.__init__ 时已由 perform_test_negative_sampling 固定写入，
            每位用户的数据严格按「1 条正样本 + num_test_negatives 条负样本」连续排列。
          - 因此 test_loader 必须 shuffle=False（由 fit() 构建时保证），
            否则正样本不在每组第 0 位，评估结果将完全错误。
          - 将 batch 内预测分数 reshape 为 (Users, group_size)，
            取第 0 列（正样本得分）与整行比较，统计排名。

        排名计算（严格比较）：
          rank = #{items with score > pos_score} + 1
          当负样本与正样本得分相同时，正样本排名靠前（乐观），
          这是学术界的常见 tie-breaking 策略。

        HR@K   = 1 if rank <= K else 0
        NDCG@K = 1/log2(rank+1) if rank <= K else 0

        Args:
            test_loader : 由 fit() 构建的测试 DataLoader，
                          batch_size 须为 group_size 的整数倍且 shuffle=False
            k           : 截断位置，默认 10
        """
        self.model.eval()
        group_size = 1 + self.dataset.num_test_negatives
        hrs, ndcgs = [], []

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc='Evaluating'):
                X = batch['X'].to(self.device)
                # squeeze(-1)：仅压缩最后一维 (B,1) -> (B,)，
                # 避免 batch_size=1 时 squeeze() 将张量压成 0 维标量
                preds = self.model(X).squeeze(-1)

                # 最后一个 batch 若无法构成完整的 group_size 组则截断，防止 reshape 报错
                if preds.shape[0] % group_size != 0:
                    valid_len = (preds.shape[0] // group_size) * group_size
                    preds = preds[:valid_len]
                    if preds.shape[0] == 0:
                        continue

                # reshape 为 (num_users_in_batch, group_size)
                preds = preds.view(-1, group_size)

                # 正样本得分：取第 0 列，保持 (Users, 1) 形状以触发行广播
                pos_scores = preds[:, 0].unsqueeze(1)

                # 统计得分严格大于正样本的负样本数，+1 得到正样本排名
                hits = (preds > pos_scores).sum(dim=1) + 1

                hr = (hits <= k).float()
                # NDCG = 1/log2(rank+1)，仅当命中（rank<=k）时有贡献，否则为 0
                ndcg = (1.0 / torch.log2(hits.float() + 1.0)) * hr

                hrs.extend(hr.cpu().numpy().tolist())
                ndcgs.extend(ndcg.cpu().numpy().tolist())

        return np.mean(hrs), np.mean(ndcgs)

    def fit(self, epochs=20, batch_size=1024, num_workers=4):
        """
        完整训练流程：每个 epoch 执行训练 + 测试集排序评估，
        并基于 NDCG@K 进行 Early Stopping 与最优模型保存。

        负采样策略说明：
          - 静态负采样（dynamic_negative=False）：
            负样本在 dataset.__init__ 时一次性生成，所有 epoch 共用相同的负样本集。
            实现简单，但模型后期容易"记住"固定负样本，泛化能力较弱。
          - 动态负采样（dynamic_negative=True）：
            负样本在 dataset.__getitem__ 中每次调用时实时随机生成，
            等价于每个 epoch 都面对不同的负样本，模型需持续区分更多物品，
            通常能取得更好的泛化效果（NCF、BPR 等论文的标准做法）。

        Loader 构建策略：
          - test_loader 在循环外只构建一次：测试集数据固定，无需每 epoch 重建。
          - train_loader 每个 epoch 重新构建：触发新一轮随机 shuffle；
            动态负采样模式下还能确保每次 __getitem__ 产生不同的负样本。

        注意：Early Stopping 触发后不会自动加载最优权重，
        训练结束后可通过以下代码恢复最佳模型：
            model.load_state_dict(torch.load('./model/best_model.pth'))

        Args:
            epochs      : 最大训练轮数
            batch_size  : 训练 batch 大小
            num_workers : DataLoader 并行进程数，建议设为 CPU 核心数
        """
        # 测试集结构固定（1 正 + num_test_negatives 负），在循环外只构建一次
        # batch_size 须为 group_size 的整数倍；shuffle=False 保证分组顺序不被打乱
        group_size = 1 + self.dataset.num_test_negatives
        self.dataset.set_stage('test')
        test_loader = DataLoader(
            self.dataset,
            batch_size=group_size * 20,
            shuffle=False
        )

        for epoch in range(epochs):
            # 每 epoch 重建 train_loader：触发新 shuffle，动态负采样时产生新负样本
            self.dataset.set_stage('train')
            train_loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                # pin_memory 将数据锁页在 CPU 内存，加速 CPU->GPU 的异步传输
                pin_memory=True
            )

            train_loss = self.train_one_epoch(train_loader)
            hr, ndcg = self.evaluate_ranking(test_loader, k=10)

            print(f"Epoch: {epoch} | Loss: {train_loss:.4f} | HR@K: {hr:.4f} | NDCG@K: {ndcg:.4f}")

            if ndcg > self.best_ndcg:
                self.best_ndcg = ndcg
                torch.save(self.model.state_dict(), "./model/best_model.pth")
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early Stopping Triggered.")
                    break