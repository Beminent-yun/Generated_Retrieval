## CausalTransformer 的完整数学原理

按照数据流的顺序，逐层推导。

---

## 符号定义

```
B    : batch size（批大小）
T    : 序列长度（max_tokens = 69）
d    : 模型维度（d_model = 128）
h    : 注意力头数（nhead = 4）
d_h  : 每个头的维度 = d/h = 128/4 = 32
d_ff : FFN 中间维度（dim_feedforward = 512）
V    : 词表大小（vocab_size = 259）
L    : Transformer 层数（num_layers = 4）
N_rq : RQ 层数（num_rq_layers = 3）
```

---

## 第一层：输入表示

### Token Embedding

输入是一个整数序列 $\mathbf{x} \in \mathbb{Z}^{B \times T}$，每个整数是 token ID。

通过 Embedding 矩阵映射到连续向量：

$$
\mathbf{E} *{tok} \in \mathbb{R}^{V \times d}, \quad \mathbf{H}^{tok} = \mathbf{E}* {tok}[\mathbf{x}] \in \mathbb{R}^{B \times T \times d}
$$

对应代码：

```python
self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
x = self.token_emb(input_ids)   # (B, T, d)
```

### 正弦位置编码

让模型感知序列中的绝对位置。对位置 $t$，维度 $k$：

$$
PE(t, 2k)   = \sin\left(\frac{t}{10000^{2k/d}}\right)
$$

$$
PE(t, 2k+1) = \cos\left(\frac{t}{10000^{2k/d}}\right)
$$

写成矩阵：$\mathbf{P} \in \mathbb{R}^{T \times d}$，加到 token embedding 上：

$$
\mathbf{H}^{pos} = \text{Dropout}(\mathbf{H}^{tok} + \mathbf{P}) \in \mathbb{R}^{B \times T \times d}
$$

### 层内位置编码（RQ-VAE 特有）

区分同一个 item 内部的第几个 code（c1/c2/c3）：

$$
\mathbf{E} *{rq} \in \mathbb{R}^{N* {rq} \times d}, \quad \text{层内位置ID序列：} \mathbf{r} = [0, \underbrace{0,1,2}_{\text{item} *1}, \underbrace{0,1,2}* {\text{item}_2}, ...] \in \mathbb{Z}^T
$$

$$
\mathbf{H}^{(0)} = \mathbf{H}^{pos} + \mathbf{E}_{rq}[\mathbf{r}] \in \mathbb{R}^{B \times T \times d}
$$

这是第一个 Transformer Block 的输入，也是 **三种位置信息的叠加** ：

```
H^(0)[b,t,:] = token_emb(x[b,t])     ← 这是什么token
             + PE(t)                   ← 在整个序列的第几位
             + rq_pos_emb(r[t])        ← 在当前item的第几层
```

---

## 第二层：Transformer Block（重复 L=4 次）

每个 Block 包含两个子层，都使用  **Pre-LayerNorm** （`norm_first=True`）。

输入记为 $\mathbf{H} \in \mathbb{R}^{B \times T \times d}$，输出同形状。

### 子层一：Masked Multi-Head Self-Attention

**Step 1：LayerNorm**

$$
\tilde{\mathbf{H}} = \text{LayerNorm}(\mathbf{H}) \in \mathbb{R}^{B \times T \times d}
$$

**Step 2：线性投影，生成 Q、K、V**

对第 $i$ 个注意力头（$i = 1,...,h$）：

$$
\mathbf{Q}_i = \tilde{\mathbf{H}} \mathbf{W}_i^Q \in \mathbb{R}^{B \times T \times d_h}
$$

$$
\mathbf{K}_i = \tilde{\mathbf{H}} \mathbf{W}_i^K \in \mathbb{R}^{B \times T \times d_h}
$$

$$
\mathbf{V}_i = \tilde{\mathbf{H}} \mathbf{W}_i^V \in \mathbb{R}^{B \times T \times d_h}
$$

其中 $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \in \mathbb{R}^{d \times d_h}$。

**Step 3：计算注意力分数，加 Causal Mask**

$$
\mathbf{A}_i = \frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_h}} + \mathbf{M} \in \mathbb{R}^{B \times T \times T}
$$

其中 $\mathbf{M}$ 是因果掩码（上三角为 $-\infty$）：

$$
M_{ts} = \begin{cases} 0 & \text{if } t \geq s \ -\infty & \text{if } t < s \end{cases}
$$

用矩阵展开（$T=6$ 时）：

```
         s=0  s=1  s=2  s=3  s=4  s=5
t=0  [   0   -∞   -∞   -∞   -∞   -∞  ]
t=1  [   0    0   -∞   -∞   -∞   -∞  ]
t=2  [   0    0    0   -∞   -∞   -∞  ]
t=3  [   0    0    0    0   -∞   -∞  ]
t=4  [   0    0    0    0    0   -∞  ]
t=5  [   0    0    0    0    0    0  ]

-∞ 经过 softmax 后变为 0 → 位置 t 看不到未来
```

**Step 4：Softmax + 加权求和**

$$
\hat{\mathbf{A}}_i = \text{softmax}(\mathbf{A}_i) \in \mathbb{R}^{B \times T \times T}
$$

$$
\text{head}_i = \hat{\mathbf{A}}_i \mathbf{V}_i \in \mathbb{R}^{B \times T \times d_h}
$$

**Step 5：拼接所有头，线性变换**

$$
\text{MultiHead}(\tilde{\mathbf{H}}) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \mathbf{W}^O \in \mathbb{R}^{B \times T \times d}
$$

其中 $\mathbf{W}^O \in \mathbb{R}^{d \times d}$。

**Step 6：残差连接**

$$
\mathbf{H}' = \mathbf{H} + \text{Dropout}(\text{MultiHead}(\text{LayerNorm}(\mathbf{H})))
$$

---

### 子层二：Point-wise Feed-Forward Network

**Step 1：Pre-LayerNorm + 两层线性 + 激活**

$$
\text{FFN}(\mathbf{H}') = \text{GELU}(\mathbf{H}' \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
$$

其中 $\mathbf{W} *1 \in \mathbb{R}^{d \times d* {ff}}, \mathbf{W} *2 \in \mathbb{R}^{d* {ff} \times d}$，逐位置独立计算。

展开维度变化：

```
(B, T, 128) → W1 → (B, T, 512) → GELU → (B, T, 512) → W2 → (B, T, 128)
```

**Step 2：残差连接**

$$
\mathbf{H}^{out} = \mathbf{H}' + \text{Dropout}(\text{FFN}(\text{LayerNorm}(\mathbf{H}')))
$$

---

### 单个 Block 的完整公式

$$
\mathbf{H}^{(l)} = f^{(l)}(\mathbf{H}^{(l-1)})
$$

其中：

$$
\mathbf{H}^{'(l)} = \mathbf{H}^{(l-1)} + \text{MHA}(\text{LN}(\mathbf{H}^{(l-1)}), \mathbf{M})
$$

$$
\mathbf{H}^{(l)} = \mathbf{H}^{'(l)} + \text{FFN}(\text{LN}(\mathbf{H}^{'(l)}))
$$

叠加 $L=4$ 层后得到 $\mathbf{H}^{(4)} \in \mathbb{R}^{B \times T \times d}$。

---

## 第三层：输出头

### LM Head（权重绑定）

把每个位置的隐向量映射回词表概率：

$$
\mathbf{Z} = \mathbf{H}^{(L)} \mathbf{E}_{tok}^T \in \mathbb{R}^{B \times T \times V}
$$

注意这里的 $\mathbf{E}_{tok}^T$ 就是 token embedding 矩阵的转置，**权重绑定**意味着：

```python
self.lm_head.weight = self.token_emb.weight
# lm_head 和 token_emb 共享同一个参数矩阵
# 参数量节省：减少 V × d = 259 × 128 = 33,152 个参数
```

数学上这等价于：

$$
P(x_t = v \mid x_{<t}) = \frac{\exp(\mathbf{h}_t \cdot \mathbf{e} *v)}{\sum* {v'=0}^{V-1} \exp(\mathbf{h} *t \cdot \mathbf{e}* {v'})}
$$

其中 $\mathbf{h}_t = \mathbf{H}^{(L)}[:,t,:]$ 是位置 $t$ 的隐向量，$\mathbf{e}_v$ 是 token $v$ 的 embedding。

---

## 第四层：训练目标

### Teacher Forcing 下的损失

取最后 $N_{rq}=3$ 个位置的 logits，对比 target_ids：

$$
\mathcal{L} = -\frac{1}{N_{rq}} \sum_{k=1}^{N_{rq}} \log P(c_k^* \mid \mathbf{x}, c_1^ *, ..., c_{k-1}^* )
$$

展开三项：

$$
\mathcal{L} = -\frac{1}{3}\left[\log P(c_1^* \mid \mathbf{x}) + \log P(c_2^* \mid \mathbf{x}, c_1^*) + \log P(c_3^* \mid \mathbf{x}, c_1^ *, c_2^* )\right]
$$

其中每一项：

$$
\log P(c_k^* \mid ...) = \log \text{softmax}(\mathbf{Z} *{T-N* {rq}+k-1})[c_k^*]
$$

```
序列位置与loss的对应关系（T=10, N_rq=3为例）：

位置: 0    1    2    3    4    5    6    7    8    9
      BOS  48   15   91  206   70   14  102  207   36
                                                   ↑
                                              最后一个历史token

位置7的输出 → 预测 c1*（用 logits[:,7,:] 计算loss）
位置8的输出 → 预测 c2*（用 logits[:,8,:] 计算loss）
位置9的输出 → 预测 c3*（用 logits[:,9,:] 计算loss）
```

---

## 第五层：自回归推理（Beam Search）

推理时不用 Teacher Forcing，改为自回归生成。

### 数学定义

目标是找到概率最大的语义ID序列：

$$
\hat{\mathbf{c}} = \arg\max_{(c_1, c_2, c_3)} P(c_1, c_2, c_3 \mid \mathbf{x})
$$

由链式法则分解：

$$
P(c_1, c_2, c_3 \mid \mathbf{x}) = P(c_1 \mid \mathbf{x}) \cdot P(c_2 \mid \mathbf{x}, c_1) \cdot P(c_3 \mid \mathbf{x}, c_1, c_2)
$$

### Beam Search 的数学过程

维护 beam_size 条路径，每条路径保存 **对数概率之和** （避免连乘下溢）：

$$
\text{score}(c_1, ..., c_k) = \sum_{i=1}^{k} \log P(c_i \mid \mathbf{x}, c_1, ..., c_{i-1})
$$

 **Step 1** ：生成 $c_1$，取 top beam_size 候选：

$$
\text{Beams} *1 = \text{TopK}* {beam_size}\left[\log P(c_1 \mid \mathbf{x})\right]
$$

 **Step 2** ：对每条 beam，扩展生成 $c_2$：

$$
\text{score}(c_1^{(b)}, c_2) = \text{score}(c_1^{(b)}) + \log P(c_2 \mid \mathbf{x}, c_1^{(b)})
$$

取各 beam 的最优 $c_2$：

$$
c_2^{(b)} = \arg\max_{c_2} P(c_2 \mid \mathbf{x}, c_1^{(b)})
$$

 **Step 3** ：同理得 $c_3^{(b)}$。

 **最终结果** ：beam_size 条候选路径按 score 排序：

$$
{(c_1^{(1)}, c_2^{(1)}, c_3^{(1)}), (c_1^{(2)}, c_2^{(2)}, c_3^{(2)}), ...}
$$

---

## 整体数学流水线

```
输入: x ∈ Z^{B×T}（token ID矩阵）
  ↓
H^(0) = E_tok[x] + PE + E_rq[r]     ∈ R^{B×T×d}
  ↓
for l = 1,...,L=4:
  H'^(l) = H^(l-1) + MHA(LN(H^(l-1)), M)
  H^(l)  = H'^(l)  + FFN(LN(H'^(l)))
  ↓
Z = H^(L) · E_tok^T                  ∈ R^{B×T×V}
  ↓
训练: L = CrossEntropy(Z[:,-3:,:], target_ids)
推理: Beam Search on P(c|x) = softmax(Z[:,-1,:])
```

---

## 参数量复盘（和代码完全对应）

```
组件                    参数量公式                    具体值
──────────────────────────────────────────────────────────────
Token Embedding         V × d                        259×128 = 33,152
RQ Pos Embedding        N_rq × d                     3×128   =    384
Pos Encoding            无参数（固定正弦）              0

单个Block：
  Attn Q/K/V proj      3 × d × d                    3×128²  = 49,152
  Attn O proj          d × d                         128²    = 16,384
  Attn LayerNorm       2 × d                         2×128   =    256
  FFN W1               d × d_ff                      128×512 = 65,536
  FFN W2               d_ff × d                      512×128 = 65,536
  FFN LayerNorm        2 × d                         2×128   =    256
  单层合计                                                    197,120

4个Block                4 × 197,120                         788,480

LM Head                 权重绑定，不额外计算                      0
──────────────────────────────────────────────────────────────
总参数量                33,152 + 384 + 788,480             = 822,016
                        ≈ 0.82M                            ✅ 与之前计算一致
```

---

## 一句话总结

> 模型的数学本质是：用三种位置信息（绝对位置 $PE$、层内位置 $E_{rq}$、token语义 $E_{tok}$）叠加构造输入表示，经过 $L$ 层带因果掩码的 Multi-Head Attention（保证位置 $t$ 只能看到 $s \leq t$ 的信息），输出每个位置的隐向量，最后通过权重绑定的 LM Head 映射到词表概率分布。训练用 Teacher Forcing 最大化真实 token 的对数概率，推理用 Beam Search 在自回归条件概率的乘积空间里搜索最优语义ID序列。
>
