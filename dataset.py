import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod


class BaseRecommenderDataset(ABC, Dataset):
    """
    推荐系统数据集的抽象基类，继承自 ABC 与 PyTorch Dataset。

    主要职责：
      - 数据加载与合并（_load_and_merge，抽象方法）
      - 低活跃用户过滤（filter_users）
      - 时间感知的数据集划分（split_data），支持 LOO 和 GTS 两种模式
      - 负样本采样（训练集动态/静态，测试集固定）
      - 特征编码与预处理（preprocessing，抽象方法）
      - 多阶段数据切换（set_stage：train / val / test）

    设计说明：
      _load_and_merge 和 preprocessing 为抽象方法，子类通过实现这两个方法
      来决定"加载哪些表"与"使用哪些特征"，从而支持不同类型的模型：
        - 纯协同过滤模型（MF/BPR）：只加载 ratings，特征仅 user_id + item_id
        - 特征交叉模型（FM/DeepFM）：合并多张表，使用 side information

    推荐的类层次结构：
      BaseRecommenderDataset（抽象基类，通用流程）
        └── MovieLensBaseDataset（MovieLens 中间基类，封装 ML 特有逻辑）
              ├── MovieLensCFDataset（纯协同过滤，只加载 ratings）
              └── MovieLensDataset  （特征交叉模型，加载三表合并）

    子类至少需实现：_load_and_merge, preprocessing,
      perform_negative_sampling, perform_test_negative_sampling,
      _prepare_dynamic_sampling, __getitem__
    """
    def __init__(self, data_path, split_mode='GTS', min_inter=5, num_negatives=0, num_test_negatives=99, dynamic_negative=False) -> None:
        """
        Args:
            data_path          : 原始数据文件所在目录
            split_mode         : 划分模式，'LOO'（留一法）或 'GTS'（全局时间分割）
            min_inter          : 用户最少交互次数阈值，低于此值的用户将被过滤
            num_negatives      : 每条正样本对应的训练负样本数（0 表示不采样）
            num_test_negatives  : 评估时每条正样本对应的测试负样本数（默认 99，共 100 候选）
            dynamic_negative   : 若为 True，则在线动态生成训练负样本，否则离线静态生成
        """
        self.data_path = data_path
        self.split_mode = split_mode
        self.min_inter = min_inter
        self.num_negatives = num_negatives
        self.num_test_negatives = num_test_negatives
        self.dynamic_negative = dynamic_negative

        self.full_df = self._load_and_merge()
        filtered_df = self.filter_users(self.full_df)
        self.train_df, self.val_df, self.test_df = self.split_data(filtered_df)

        # 静态离线负采样：在 DataLoader 迭代前一次性生成负样本并拼入 DataFrame
        if self.num_negatives > 0 and not self.dynamic_negative:
            self.train_df = self.perform_negative_sampling(self.train_df, self.num_negatives)

        # 测试集固定负采样：保证每位用户的评估候选集大小一致（1 正 + num_test_negatives 负）
        if self.num_test_negatives > 0:
            self.test_df = self.perform_test_negative_sampling(self.test_df, self.num_test_negatives)

        # feature_encoder 必须在 preprocessing 之前初始化，因为 is_train=True 时会向其写入编码器
        self.feature_encoder = {}

        # preprocessing 按顺序调用：先 train（fit + transform），再 val/test（transform only）
        # 纯内存操作，直接持有 {'X': ndarray, 'y': ndarray} 字典，避免反复 IO
        self.train_data = self.preprocessing(self.train_df, is_train=True)
        self.val_data = self.preprocessing(self.val_df, is_train=False)
        self.test_data = self.preprocessing(self.test_df, is_train=False)

        # 动态负采样所需的辅助数据结构（编码后的用户-物品交互集合）必须在 preprocessing 之后构建
        if self.dynamic_negative:
            self._prepare_dynamic_sampling()

        self.set_stage('train')
        
    @abstractmethod
    def _load_and_merge(self): pass

    def filter_users(self, df) -> pd.DataFrame:
        """过滤交互次数不足 min_inter 的低活跃用户，减少冷启动噪声。"""
        user_counts = df.groupby('user_id').size()
        active_users = user_counts[user_counts >= self.min_inter].index
        return df[df['user_id'].isin(active_users)].copy()

    def split_data(self, df):
        """
        按时间顺序划分训练/验证/测试集，支持两种策略：
          - LOO （Leave-One-Out）：每位用户按时间排序后，
            最后一条交互为测试，倒数第二为验证，其余为训练。
            数据泄露风险低，但测试集仅含单条正样本。
          - GTS（Global Temporal Split）：以整体时间轴的 80% / 90% 分位点切分，
            前 80% 为训练，80%-90% 为验证，后 10% 为测试。
            模拟真实生产中按时间推进的评估场景。
        """
        df = df.sort_values('timestamp')
        if self.split_mode == 'LOO':
            df['rank'] = df.groupby('user_id')['timestamp'].rank(method='first', ascending=False)
            test = df[df['rank'] == 1].copy()
            val = df[df['rank'] == 2].copy()
            train = df[df['rank'] > 2].copy()
        elif self.split_mode == 'GTS':
            t_train_end = df['timestamp'].quantile(0.8)
            t_val_end = df['timestamp'].quantile(0.9)
            train = df[df['timestamp'] <= t_train_end].copy()
            val = df[(df['timestamp'] > t_train_end) & (df['timestamp'] <= t_val_end)].copy()
            test = df[df['timestamp'] > t_val_end].copy()
        return train, val, test
    
    @abstractmethod
    def preprocessing(self, df, is_train=True): pass

    def set_stage(self, stage='train'):
        """切换当前激活的数据集分区，控制 __len__ 和 __getitem__ 的数据来源。"""
        stages = {'train': self.train_data, 'val': self.val_data, 'test': self.test_data}
        self.active_data = stages[stage]

    @abstractmethod
    def perform_negative_sampling(self, df, num_negatives=4): pass

    @abstractmethod
    def perform_test_negative_sampling(self, df, num_negatives=99): pass

    @abstractmethod
    def _prepare_dynamic_sampling(self): pass

    def __len__(self):
        # active_data 是由 preprocessing 产生的非空字典，理论上不会为空；
        # 此处保留防御性检查以兼容子类可能返回空字典的情况。
        if not self.active_data:
            return 0
        base_len = len(next(iter(self.active_data.values())))
        # 动态负采样模式下，训练集的逻辑长度需扩展为正样本数 × (1 + 负样本数)
        # __getitem__ 通过整除/取模将虚拟索引映射回真实正样本行
        if self.dynamic_negative and self.active_data is self.train_data:
            return base_len * (1 + self.num_negatives)
        return base_len

    def __getitem__(self, idx):
        # 具体索引逻辑由子类实现，支持动态/静态两种负采样模式
        pass

class MovieLensBaseDataset(BaseRecommenderDataset):
    """
    MovieLens 数据集的中间基类，封装所有 MovieLens 特有的实现逻辑：
    负采样、动态负采样辅助结构、特征编码辅助方法、__getitem__。

    子类只需实现两个方法，分别决定数据加载策略与特征策略：
      - _load_and_merge() : 选择加载哪些表
      - preprocessing()   : 选择使用哪些特征列

    目前提供两个具体子类：
      - MovieLensCFDataset  : 只加载 ratings，特征仅 user_id + movie_id（纯协同过滤）
      - MovieLensDataset    : 加载三表合并，特征含 side information（特征交叉模型）
    """

    def perform_negative_sampling(self, df, num_negatives=4):
        """
        离线静态训练负采样：为每条正样本采样 num_negatives 条负样本。

        负样本排除条件：
          1. 该物品在全量数据（full_df）中已被该用户交互过（避免假负样本）
          2. 本轮为同一正样本已采样过该物品（保证每条正样本的负样本集内部唯一）

        最终正负样本混洗后返回，label=1 表示正样本，label=0 表示负样本。
        """
        all_item_ids = self.full_df['movie_id'].unique()
        # 使用 full_df 构建交互集合，覆盖 train/val/test 全部分区，最大限度避免假负样本
        user_inter_dict = self.full_df.groupby('user_id')['movie_id'].apply(set).to_dict()

        df = df.copy()
        df['label'] = 1
        neg_rows = []
        for row in df.itertuples(index=False):
            user_id = row.user_id
            interacted_items = user_inter_dict[user_id]
            # 用 set 追踪已采样的负 ID，防止同一正样本产生重复负样本
            sampled_neg_ids = set()
            while len(sampled_neg_ids) < num_negatives:
                neg_id = np.random.choice(all_item_ids)
                if neg_id not in interacted_items and neg_id not in sampled_neg_ids:
                    sampled_neg_ids.add(neg_id)
                    neg_row_dict = row._asdict()
                    neg_row_dict['movie_id'] = neg_id
                    neg_row_dict['label'] = 0
                    if 'rating' in neg_row_dict:
                        neg_row_dict['rating'] = 0
                    neg_rows.append(neg_row_dict)

        neg_df = pd.DataFrame(neg_rows)
        sampled_df = pd.concat([df, neg_df], ignore_index=True)
        return sampled_df.sample(frac=1.0).reset_index(drop=True)

    def perform_test_negative_sampling(self, df, num_negatives=99):
        """
        离线固定测试负采样：为每条正样本构建大小为 (1 + num_negatives) 的候选集。

        输出行顺序严格保证：正样本行在前，紧跟其 num_negatives 条负样本行。
        这与 evaluate_ranking 中 preds.view(-1, group_size) 后取 preds[:, 0]
        作为正样本得分的假设完全一致，顺序不可打乱。

        负样本去重：排除用户已交互物品 + 排除组内已采样的负 ID。
        """
        all_item_ids = self.full_df['movie_id'].unique()
        user_inter_dict = self.full_df.groupby('user_id')['movie_id'].apply(set).to_dict()

        df = df.copy()
        df['label'] = 1
        eval_rows = []
        for row in df.itertuples(index=False):
            # 正样本始终排在每组的第一位
            eval_rows.append(row._asdict())
            user_id = row.user_id
            interacted_items = user_inter_dict[user_id]
            sampled_neg_ids = set()
            while len(sampled_neg_ids) < num_negatives:
                neg_id = np.random.choice(all_item_ids)
                if neg_id not in interacted_items and neg_id not in sampled_neg_ids:
                    sampled_neg_ids.add(neg_id)
                    neg_row_dict = row._asdict()
                    neg_row_dict['movie_id'] = neg_id
                    neg_row_dict['label'] = 0
                    if 'rating' in neg_row_dict:
                        neg_row_dict['rating'] = 0
                    eval_rows.append(neg_row_dict)

        return pd.DataFrame(eval_rows)

    def _encode_features(self, df, is_train, sparse_features, rating_threshold=4):
        """
        将 DataFrame 中的稀疏特征列编码为整数矩阵，供子类 preprocessing 调用。

        编码规则：
          - is_train=True  : fit_transform，编码从 1 开始（+1 偏移，0 保留为 padding），
                             将 LabelEncoder 写入 self.feature_encoder。
          - is_train=False : 使用训练集 LabelEncoder 做 transform，
                             OOV（未知类别）编码为 0（padding index）。
                             ⚠️ 模型 Embedding 层须设置 padding_idx=0。

        标签规则：
          - df 含 'label' 列（静态负采样后）→ 直接使用。
          - 动态负采样训练集 → 全部初始化为 1.0，负样本由 __getitem__ 实时生成。
          - 其他情况 → rating >= rating_threshold 二值化为隐式反馈标签。

        Args:
            df                : 待编码的 DataFrame
            is_train          : 是否为训练阶段（决定 fit 还是 transform）
            sparse_features   : 需要编码的特征列名列表
            rating_threshold  : 正样本 rating 阈值，默认 4

        Returns:
            {'X': np.ndarray (N, len(sparse_features)), 'y': np.ndarray (N,)}
        """
        processed_x = []
        for feature in sparse_features:
            if is_train:
                le = LabelEncoder()
                encoded_vals = le.fit_transform(df[feature].astype(str)) + 1  # 0 保留为 padding
                self.feature_encoder[feature] = le
            else:
                le = self.feature_encoder[feature]
                feature_vals = df[feature].astype(str).values
                known_mask = np.isin(feature_vals, le.classes_)
                encoded_vals = np.zeros(len(df), dtype=np.int32)  # OOV 默认编码为 0（padding）
                if known_mask.any():
                    encoded_vals[known_mask] = le.transform(feature_vals[known_mask]) + 1
            processed_x.append(encoded_vals.reshape(-1, 1))

        X_array = np.hstack(processed_x).astype(np.int32)

        if 'label' in df.columns:
            y_array = df['label'].values.astype(np.float32)
        elif is_train and self.dynamic_negative:
            # 动态负采样：train_df 全为正样本，负样本在 __getitem__ 中实时生成
            y_array = np.ones(len(df), dtype=np.float32)
        else:
            y_array = (df['rating'].values >= rating_threshold).astype(np.float32)

        return {'X': X_array, 'y': y_array}

    def _prepare_dynamic_sampling(self):
        """
        构建动态负采样所需的辅助数据结构，在 preprocessing 完成后调用。

        主要工作：
          1. 记录编码后的物品 ID 上界（max_item_id），用于均匀随机采样
          2. 将 full_df 中已知用户/物品转换为编码 ID，
             构建 user_inter_dict（编码用户 -> 编码物品 set），
             __getitem__ 在线采样时用于 O(1) 排除已交互物品

        ⚠️ valid_df 只含训练编码器可见的用户和物品，val/test 中的新物品
        不在编码空间内，极少数情况下可能产生跨分区假负样本，大规模数据下可忽略。
        """
        le_movie = self.feature_encoder['movie_id']
        # 编码从 1 开始（+1 偏移），合法物品 ID 范围为 [1, max_item_id]
        self.max_item_id = len(le_movie.classes_)

        le_user = self.feature_encoder['user_id']
        # 过滤掉 encoder 中不存在的用户/物品，防止 transform 时报 ValueError
        valid_users = self.full_df['user_id'].astype(str).isin(le_user.classes_)
        valid_movies = self.full_df['movie_id'].astype(str).isin(le_movie.classes_)
        valid_df = self.full_df[valid_users & valid_movies]

        encoded_users = le_user.transform(valid_df['user_id'].astype(str)) + 1
        encoded_movies = le_movie.transform(valid_df['movie_id'].astype(str)) + 1

        # 构建编码后的用户交互集合，O(1) 集合查找支撑在线负采样效率
        self.user_inter_dict = {}
        for u, m in zip(encoded_users, encoded_movies):
            if u not in self.user_inter_dict:
                self.user_inter_dict[u] = set()
            self.user_inter_dict[u].add(m)

        # 记录特征矩阵 X 中 user_id 和 movie_id 的列下标，供 __getitem__ 索引
        self.user_col_idx = 0
        self.movie_col_idx = 1

    def __getitem__(self, idx):
        """
        动态负采样模式下的索引逻辑（仅训练阶段）：
          __len__ 将训练集大小扩展为 N * (1 + num_negatives)，
          idx 通过整除映射到真实正样本行（real_idx），
          每组的第 0 个 idx 返回正样本，其余在线生成负样本。

        非动态模式或非训练阶段：直接按 idx 切片返回预处理好的特征与标签。
        """
        if self.dynamic_negative and self.active_data is self.train_data:
            # 将虚拟扩展索引还原为真实正样本行号
            real_idx = idx // (1 + self.num_negatives)
            is_positive = (idx % (1 + self.num_negatives)) == 0

            # 复制特征行，避免 in-place 修改共享的 numpy 数组
            X_np = np.copy(self.active_data['X'][real_idx])

            if is_positive:
                y_val = 1.0
            else:
                user_id = X_np[self.user_col_idx]
                interacted_items = self.user_inter_dict.get(user_id, set())

                # 拒绝采样：O(1) 标量整数生成，循环期望次数 ≈ N/(N - |interacted|) ≈ 1
                while True:
                    neg_id = np.random.randint(1, self.max_item_id + 1)
                    if neg_id not in interacted_items:
                        break

                X_np[self.movie_col_idx] = neg_id
                y_val = 0.0

            X = torch.from_numpy(X_np).long()
            y = torch.tensor(y_val, dtype=torch.float32)
            return {'X': X, 'y': y}

        # 静态模式或 val/test 阶段：直接返回预处理好的数据
        X = torch.from_numpy(self.active_data['X'][idx]).long()
        y = torch.tensor(self.active_data['y'][idx], dtype=torch.float32)
        return {'X': X, 'y': y}


class MovieLensCFDataset(MovieLensBaseDataset):
    """
    MovieLens 1M 纯协同过滤版本。

    - 只加载 ratings.dat，不读取用户和电影的 side information
    - 特征仅使用 user_id 和 movie_id
    - 适配 MF、BPR、NCF（GMF/MLP）等 ID-only 模型
    - 内存占用低，无宽表冗余，适合大规模快速实验

    注意：所有稀疏特征编码从 1 开始（+1 偏移），0 保留为 padding index。
    模型 Embedding 层须设置 padding_idx=0，且 num_embeddings = 类别数 + 1。
    """

    def _load_and_merge(self) -> pd.DataFrame:
        """只读取 ratings.dat，不做任何表连接。"""
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        return pd.read_csv(
            os.path.join(self.data_path, 'rating.dat'),
            sep='::', names=rnames, engine='python'
        )

    def preprocessing(self, df, is_train=True):
        """特征仅用 user_id + movie_id，调用基类通用编码逻辑。"""
        return self._encode_features(df, is_train, sparse_features=['user_id', 'movie_id'])


class MovieLensDataset(MovieLensBaseDataset):
    """
    MovieLens 1M 特征交叉模型版本。

    - 加载 ratings / users / movies 三张表并合并为宽表
    - 特征使用 user_id, movie_id, gender, age, occupation（含 side information）
    - 适配 FM、DeepFM、Wide&Deep 等需要特征交叉的模型

    注意：所有稀疏特征编码从 1 开始（+1 偏移），0 保留为 padding index。
    模型 Embedding 层须设置 padding_idx=0，且 num_embeddings = 类别数 + 1。
    """

    def _load_and_merge(self) -> pd.DataFrame:
        """读取 rating / user / movie 三张表并按 user_id、movie_id 连接为宽表。"""
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(os.path.join(self.data_path, 'rating.dat'), sep='::', names=rnames, engine='python')
        unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
        users = pd.read_csv(os.path.join(self.data_path, 'user.dat'), sep='::', names=unames, engine='python')
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_csv(os.path.join(self.data_path, 'movies.dat'), sep='::', names=mnames, engine='python')
        return ratings.merge(users, on='user_id').merge(movies, on='movie_id')

    def preprocessing(self, df, is_train=True):
        """特征含 side information，调用基类通用编码逻辑。"""
        return self._encode_features(
            df, is_train,
            sparse_features=['user_id', 'movie_id', 'gender', 'age', 'occupation']
        )


# ---------------------------------------------------------------------------
# MovieLens 100K
# ---------------------------------------------------------------------------

class MovieLens100KBaseDataset(MovieLensBaseDataset):
    """
    MovieLens 100K 数据集的中间基类。

    与 ML-1M 的主要格式差异：
      - 分隔符：u.data 用 \\t，u.user / u.item 用 |（ML-1M 全用 ::）
      - u.user 字段顺序：user_id | age | gender | occupation | zip
        （ML-1M 为 user_id | gender | age | occupation | zip，age/gender 对调）

    fold 参数说明（如何处理自带的 5 折划分）：
      fold=None（默认）：加载 u.data 全量数据，走父类 split_data 的时序划分（GTS/LOO）。
                         适合需要时序评估的场景，与 ML-1M 流程完全一致。
      fold=1~5 ：直接加载官方预切分的 u{fold}.base / u{fold}.test，
                  绕过时序 split_data，适合复现基于官方折的 benchmark 结果。
                  ⚠️ 官方折是按用户分层随机抽取的 80/20 随机划分，不保证时序性；
                  val 集由 base 按时间戳最后 10% 切出（原始官方折不含 val）。

    无论 fold 取何值，full_df 始终来自 u.data，用于负样本采样时的全量交互排除。
    """

    def __init__(self, data_path, fold=None, **kwargs):
        """
        Args:
            data_path : 数据文件目录（包含 u.data、u.user、u1.base 等）
            fold      : 预定义折编号（1-5），None 表示使用全量数据走时序划分
            **kwargs  : 透传给 BaseRecommenderDataset（split_mode、min_inter 等）
        """
        # 必须在 super().__init__ 前赋值，因为父类 __init__ 会调用 split_data
        self.fold = fold
        super().__init__(data_path, **kwargs)

    def split_data(self, df):
        """fold 模式下加载预定义折文件；否则走父类时序划分。"""
        if self.fold is not None:
            return self._load_fold_split()
        return super().split_data(df)

    def _load_fold_split(self):
        """
        加载官方预切分的第 fold 折：
          - u{fold}.base → 按时间戳 90/10 切为 train / val
          - u{fold}.test → 直接作为 test
        """
        if not 1 <= self.fold <= 5:
            raise ValueError(f"fold 须为 1~5 的整数，当前值：{self.fold}")

        col_names = ['user_id', 'movie_id', 'rating', 'timestamp']
        base_df = pd.read_csv(
            os.path.join(self.data_path, f'u{self.fold}.base'),
            sep='\t', names=col_names
        )
        test_df = pd.read_csv(
            os.path.join(self.data_path, f'u{self.fold}.test'),
            sep='\t', names=col_names
        )
        # 从 base 中按时间戳切出最后 10% 作为 val，其余为 train
        base_df = base_df.sort_values('timestamp')
        t_val_start = base_df['timestamp'].quantile(0.9)
        train_df = base_df[base_df['timestamp'] <= t_val_start].copy()
        val_df   = base_df[base_df['timestamp'] >  t_val_start].copy()
        return train_df, val_df, test_df


class MovieLens100KCFDataset(MovieLens100KBaseDataset):
    """
    MovieLens 100K 纯协同过滤版本。

    - 只加载 u.data，不使用任何 side information
    - 特征仅 user_id + movie_id
    - 适配 MF、BPR、NCF 等 ID-only 模型
    """

    def _load_and_merge(self) -> pd.DataFrame:
        """只读 u.data（\\t 分隔），不做表连接。"""
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        return pd.read_csv(
            os.path.join(self.data_path, 'u.data'),
            sep='\t', names=rnames
        )

    def preprocessing(self, df, is_train=True):
        """特征仅 user_id + movie_id，调用基类通用编码逻辑。"""
        return self._encode_features(df, is_train, sparse_features=['user_id', 'movie_id'])


class MovieLens100KDataset(MovieLens100KBaseDataset):
    """
    MovieLens 100K 特征交叉模型版本。

    - 合并 u.data + u.user，使用用户侧 side information
    - 特征：user_id, movie_id, gender, age, occupation
    - 适配 FM、DeepFM 等特征交叉模型

    注意：
      u.user 字段顺序为 user_id | age | gender | occupation | zip（| 分隔），
      与 ML-1M 的 user_id :: gender :: age :: occupation :: zip（:: 分隔）不同。
      所有稀疏特征编码从 1 开始（+1 偏移），0 保留为 padding index。
    """

    def _load_and_merge(self) -> pd.DataFrame:
        """合并 u.data（\\t 分隔）与 u.user（| 分隔）。"""
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(
            os.path.join(self.data_path, 'u.data'),
            sep='\t', names=rnames
        )
        # ⚠️ ML-100K u.user 字段为 age | gender，与 ML-1M 的 gender | age 顺序相反
        unames = ['user_id', 'age', 'gender', 'occupation', 'zip']
        users = pd.read_csv(
            os.path.join(self.data_path, 'u.user'),
            sep='|', names=unames
        )
        return ratings.merge(users, on='user_id')

    def preprocessing(self, df, is_train=True):
        """特征含用户 side information，调用基类通用编码逻辑。"""
        return self._encode_features(
            df, is_train,
            sparse_features=['user_id', 'movie_id', 'gender', 'age', 'occupation']
        )




