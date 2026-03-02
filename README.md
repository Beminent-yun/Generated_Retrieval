# Rec_Project 快速上手（低流量版）

## 1) 克隆 & 环境
- 建议先推送不含数据的仓库（已在 .gitignore 忽略 datasets/、模型文件等）。
- 服务器上执行：
  ```bash
  git clone <your-repo-url>
  cd Rec_Project
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt  # GPU 机器可在安装 torch 前设置 CUDA 源
  ```

## 2) 数据准备（优先低流量路径）
### 推荐：直接传输已预处理文件
- 本地已有的 `datasets/processed/beauty.pkl` 约 151MB，可直接上传到服务器同一路径，避免下载 3.5GB 原始数据：
  ```bash
  # 在本地执行
  scp datasets/processed/beauty.pkl <user>@<server>:~/Rec_Project/datasets/processed/
  ```

### 需要从零构建（高流量）
- 需下载两份原始文件：
  - `datasets/Beauty_and_Personal_Care.jsonl.gz` ≈ 2.8GB（交互数据）
  - `datasets/meta_Beauty_and_Personal_Care.jsonl.gz` ≈ 0.7GB（元数据，可用 `python main.py` 下载）
- 放置到 `datasets/` 后运行预处理生成 `beauty.pkl`：
  ```bash
  python Amazon_Data.py
  # 输出: datasets/processed/beauty.pkl
  ```
- 若带宽紧张，优先只上传/下载 `beauty.pkl`，跳过原始文件。

## 3) 训练 RQ-VAE 生成语义 ID
- 依赖文件：`datasets/processed/beauty.pkl`
- 运行：
  ```bash
  python train_rqvae.py
  ```
- 结果：
  - Sentence-BERT 会首次下载模型（≈80MB，自动缓存）。
  - 生成 `datasets/processed/item_embeddings.npy`、`datasets/processed/semantic_ids.npy`
  - 模型权重保存在 `checkpoints/rqvae/best_model.pt`

## 4) 流量与缓存提示
- 数据与模型已在 `.gitignore` 中忽略，推送仓库时不会上传大文件。
- 复用缓存：保持 `~/.cache`（或自定义 `HF_HOME`）以避免重复下载 Sentence-BERT。
- 如需再次下载大文件，建议使用 `scp/rsync` 从已有副本拷贝，优先传输体积较小的 `beauty.pkl`。

## 5) 其他
- MovieLens 100K/1M 原始文件已包含在 `datasets/ml-100k/` 内，可用于快速功能验证，无需额外下载。
- 若遇到依赖问题，确保已按 `requirements.txt` 完整安装；下载工具使用 `rich` 的进度条，已在 `utils.py` 修复导入。
