# 并行策略对比：DP、FSDP、TP、PP

> 对应 lecture_07（`lectures/lecture_07.py` 中的 `data_parallelism`、`tensor_parallelism`、`pipeline_parallelism` 三节）和讲义第 8 章（p.39–46）。重点是 FSDP 和 TP 的区别。下文“第 x 行”都指 `lecture_07.py` 的行号，另有明确文件名的除外。
>
> 相关笔记：[数据并行的逐步优化](数据并行的逐步优化.md)、[DDP、Sharded Optimizer、FSDP 的通信操作](DDP_ShardedOptimizer_FSDP的通信操作.md)、[FSDP 的四种实现](FSDP的四种实现Basic_Async_Prefetch_MixedPrecision.md)

---

## 0. 核心：计算要求数据和权重出现在同一个 rank 上

每个矩阵乘法 y = x · W 都要求 x 和 W 在同一个 rank 上。各种并行策略的区别就在于：切开什么、什么留在原地、计算时搬什么。

| 策略 | 切开什么 | 什么留在原地 | 训练中通信什么 |
|---|---|---|---|
| DDP | 数据 | 数据和激活；权重在每个 rank 上各有一整份 | 计算时什么都不搬；backward 后对梯度做 all-reduce |
| FSDP | 数据，以及权重的存储 | 数据和激活 | 计算前 all-gather 权重；backward 时对梯度做 reduce-scatter |
| TP | 每一层的权重 | 权重 | 激活（all-gather 或 all-reduce） |
| PP | 按层切成几段 | 每个 rank 负责的那几层 | 激活（相邻 rank 之间 send / recv） |

讲义 p.39–40 对这几种策略的定义：

- **DP**：batch 切到多个设备上，每个设备算自己那份数据的梯度，然后对梯度求平均。
- **FSDP**：在 DP 的基础上，把优化器状态、梯度、权重也切开以节省显存；forward 和 backward 时需要从其他设备 gather 权重分片。
- **TP**：权重矩阵沿输入维度或输出维度切开，每个设备算自己分片对应的激活，再对激活做 reduce 或 gather。
- **PP**：模型按层切成几个 stage，每个 stage 在不同的设备上运行。
- **EP**：MoE 模型的 expert 分到不同设备上，每个设备只算自己那个 expert 的输出。第 8 章只分析 DP、FSDP、TP 及其组合。

---

## 1. lecture_07 的三种并行方式

| 方式 | 图上怎么画 | 讲义中的原话 |
|---|---|---|
| data_parallelism | Data 方块被切开 | "each rank gets a slice of the data" |
| tensor_parallelism | 一条**竖线**穿过每一层，Data 不切 | "each rank gets part of each layer, transfer all data/activations" |
| pipeline_parallelism | 一条**横线**在 layer 1 和 layer 2 之间，Data 不切 | "each rank gets subset of layers, transfer all data/activations" |

### 1.1 data_parallelism（第 375–436 行）

- 数据按 batch 切开，每个 rank 只拿自己那段（第 407–410 行）。
- 每个 rank 持有全部参数（第 413 行），各自有一个优化器（第 414 行）。
- backward 之后对每个参数的梯度做 all-reduce 求平均（第 429 行）。
- 各 rank 的 loss 不同，因为数据不同；梯度经过 all-reduce 后相同，所以参数始终保持一致（第 382–385 行）。
- 这一节最后写着 "Next time: FSDP/ZeRO: use all-gather and reduce-scatter to avoid holding all parameters in memory"（第 387 行），也就是说 FSDP 被当作数据并行的延伸。

### 1.2 tensor_parallelism：竖线（第 439–482 行）

- **竖线穿过每一层**：每层的权重矩阵都被切开，每个 rank 拿到每层的一片。代码切的是输出维度；每个 rank 上每层的权重形状是 `num_dim × local_num_dim`（第 459 行）。图上只画了一条线，也就是切成 2 份；代码里 `world_size = 4`，切成 4 份。
- **Data 不切**：所有 rank 都拿到同一份完整数据（第 450 行）。
- **每一层的数据流**（以 2 个 rank 为例）：

```text
rank 0：完整 x × W 的左半列 → 激活的左半 ┐
                                        ├ all-gather → 两边都拿到完整激活 → 进入下一层
rank 1：完整 x × W 的右半列 → 激活的右半 ┘
```

- 第 465 行算出部分激活（`batch_size × local_num_dim`），第 472 行 all-gather，第 475 行沿特征维拼接，得到 `batch_size × num_dim` 的完整激活。
- **通信**：传的是**激活**，每一层都要做一次集合操作；权重始终不动。
- **特点**：所有 rank 同时计算，没有 PP 那样的空闲等待；但每一层都要通信，频率很高，所以 TP 通常放在同一台机器内、带宽很高的 GPU 之间。
- backward 在讲义里留作练习（第 479 行）。

### 1.3 pipeline_parallelism：横线（第 484–536 行）

- **横线在 layer 1 和 layer 2 之间**：按层切开，每一层本身保持完整。`world_size = 2` 时，rank 0 持有 layer 0–1，rank 1 持有 layer 2–3（第 501–504 行）。
- **Data 不切**：所有 rank 都加载了完整数据（第 495–496 行），但只有 rank 0 真正使用它，把它切成 micro-batch（第 510–512 行）。其他 rank 只分配缓冲区，用来接收上一个 rank 发来的激活（第 513–515 行）。
- **数据流**：

```text
rank 0：[layer 0 → layer 1] ──send 激活──▶ rank 1：[layer 2 → layer 3]
```

- **通信**：只在切口处传**激活**，是相邻两个 rank 之间的点对点 send / recv（第 520、530 行），不是集合操作。
- **气泡（bubble）**：rank 1 必须等 rank 0 算完才能开始，这段时间它是空闲的。代码把 batch 切成 4 个 micro-batch 来缓解：

```text
rank 0：[mb1][mb2][mb3][mb4]
rank 1：     [mb1][mb2][mb3][mb4]
```

- rank 1 处理 mb1 时，rank 0 已经在处理 mb2，两个 rank 同时在工作。切 micro-batch 只是为了在时间上错开，并不是数据并行，因为每个 micro-batch 都要依次经过所有 rank。
- 讲义注明这份代码没有做通信和计算的重叠（第 532 行）。backward 同样留作练习（第 534 行）。

### 1.4 怎么区分：看 Data 有没有被切开

- **DP、FSDP 切的是数据**：属于数据并行。
- **TP、PP 切的是模型，数据不切**：属于模型并行。lecture_07 里面竖线和横线的那两张图都是模型并行，不是数据并行。

---

## 2. TP 的“权重不动，交换激活”：矩阵例子

### 2.1 按列切（column parallel，切输出维度）

```text
x = [1, 2]          W = [1 2 3 4]      完整结果 y = x·W = [11, 14, 17, 20]
                       [5 6 7 8]

rank 0 只持有 W 的第 1–2 列 [1 2; 5 6] → x·W0 = [11, 14]
rank 1 只持有 W 的第 3–4 列 [3 4; 7 8] → x·W1 = [17, 20]

all-gather：rank 0 发出 [11, 14]，rank 1 发出 [17, 20]
            两边都得到 [11, 14 | 17, 20] = y
```

- **为什么每个 rank 算出的就是最终结果**：y 的第 j 列只依赖 x 和 W 的第 j 列。所以 rank 0 算出的 `[11, 14]` 就是 y 前两列的最终值，不是近似值，也不需要再相加。
- **为什么还要交换**：下一层的每个输出都要用到 y 的全部特征，而每个 rank 只有其中几列，所以每层算完都要 all-gather 一次。
- **权重为什么不动**：每个 rank 自始至终只用自己那几列 W，从不需要别人的权重。
- 讲义 §8.4 的写法（式 45）：`xW = all-gather({x W⁽ⁱ⁾})`，其中 W⁽ⁱ⁾ 的形状是 `(D, D_FF / N_TP)`。

### 2.2 按行切（row parallel，切输入维度）

按行切 W 时，每个 rank 还要把输入 x 也按特征切开，只用对应的那一段。

```text
rank 0 持有 W 的第 1 行 [1 2 3 4]，只用 x 的第 1 个特征 1 → 1 × [1 2 3 4] = [1, 2, 3, 4]
rank 1 持有 W 的第 2 行 [5 6 7 8]，只用 x 的第 2 个特征 2 → 2 × [5 6 7 8] = [10, 12, 14, 16]

all-reduce 求和：[1, 2, 3, 4] + [10, 12, 14, 16] = [11, 14, 17, 20] = y
```

- 每个 rank 算出的是 y 的**一部分和**，形状和 y 一样，所以要 **all-reduce 求和**，而不是拼接。
- 讲义 §8.4 的写法（式 46）：`xW = all-reduce({x⁽ⁱ⁾ W⁽ⁱ⁾})`，其中 W⁽ⁱ⁾ 的形状是 `(D / N_TP, D_FF)`，x⁽ⁱ⁾ 的形状是 `(B, D / N_TP)`。

| | 按列切 | 按行切 |
|---|---|---|
| 切 W 的哪一维 | 输出维度 | 输入维度 |
| 每个 rank 需要的输入 | 完整的 x | x 的对应一段特征 |
| 每个 rank 得到 | y 的几列，是最终值 | 和 y 同形状的部分和 |
| 怎么得到完整的 y | all-gather 拼接 | all-reduce 求和 |

### 2.3 讲义 §8.4 的 FFN 配置：先按列切，再按行切

讲义（p.44，式 47–51）对 FFN 用的配置是：W1、W2 按列切，W3 按行切。按列切得到的激活正好是按特征切开的，而这正是按行切的 W3 需要的输入，所以中间那次 all-gather 可以省掉：

```text
x（完整，各 rank 相同）
 → 按列切的 W1、W2：rank i 得到 x1、x2 的第 i 组特征（不做 all-gather）
 → 逐元素运算 f(x1) * x2：各自在自己那组特征上做，不需要通信
 → 按行切的 W3：只需要第 i 组输入特征，正好就在手上 → 得到部分和 y⁽ⁱ⁾
 → all-reduce 求和 → 完整的 y（各 rank 相同）
```

lecture_07 的版本每层都按列切，所以每层都要 all-gather。这种配置的 backward 怎么写，是 `tp_calcs (a)` 的内容，留给自己推导。

---

## 3. FSDP 和 TP 的对比

### 3.1 同一个例子换成 FSDP

现在有 2 个样本：s1 = `[1, 2]`，s2 = `[3, 4]`。rank 0 只处理 s1，rank 1 只处理 s2，每个 rank 平时只存 W 的一半。

```text
计算前：all-gather 权重 → 两边都有完整的 W
rank 0：s1 · W = [1, 2] · W = [11, 14, 17, 20]
rank 1：s2 · W = [3, 4] · W = [23, 30, 37, 44]
```

每个 rank 算出的是**自己样本**的完整输出，激活留在本地，不需要交换。

### 3.2 存法可以完全一样，不同的是分工

一个 rank 需要哪些权重，由它负责算什么决定，和它平时存了什么无关。

- **TP**：数据不切，rank r 负责算**全部样本**的**第 r 组输出特征**。算这组特征只需要 W 的对应几列，正好就是它存的那一片，所以不用 gather。
- **FSDP**：数据切开。rank r 负责算**自己那些样本**的**全部输出特征**，因为后面的层和 loss 都要用完整的激活，而别的 rank 没有这些样本，帮不上忙。全部输出特征需要 W 的所有列，所以要先 gather。

作业里的 FSDP 实现，存法就和 TP 按列切一样：`fully_sharded_data_parallel_training.py` 第 69 行 `torch.chunk(..., dim=0)` 切的是 Linear 权重的第 0 维。cs336 的 Linear 权重形状是 `d_out × d_in`，所以第 0 维就是输出特征，也就是例子里的“列”。

```text
W 按列切：rank 0 存第 1–2 列，rank 1 存第 3–4 列（两种方案存法相同）

TP（数据不切，两个 rank 都有 s1 和 s2）
  rank 0：[s1; s2] · W 的第 1–2 列 = [11 14; 23 30]  ← 两个样本的前两个特征
  rank 1：[s1; s2] · W 的第 3–4 列 = [17 20; 37 44]  ← 两个样本的后两个特征
  all-gather 激活 → 两边都得到完整的 2×4 输出

FSDP（数据切开，rank 0 只有 s1，rank 1 只有 s2）
  rank 0 如果不 gather，只能算出 s1 的前两个特征 [11, 14]
  s1 的后两个特征 [17, 20] 需要 s1（只在 rank 0）× W 的第 3–4 列（只在 rank 1）
  → 没有任何一个 rank 同时拥有这两样东西
```

这时只能搬动其中一样：

- **把权重搬到数据所在的 rank**：all-gather 权重。这是 FSDP 的做法，对应作业实现第 71 行注册的 forward pre-hook `gather_weight_pre_hook`；Linear 在 backward 前还要再 gather 一次（第 74 行）。
- **把数据搬到权重所在的 rank**：让每个 rank 都拿到全部样本，算完再交换激活。这是 TP 的做法。

DDP 则把权重永久复制到每个 rank，计算时什么都不用搬，代价是显存。FSDP 可以看成“沿用 DDP 的分工，换一种省显存的存法”：存法和分工对不上，所以计算前要临时把权重拼回来。

### 3.3 完整对比

| | DDP | FSDP | TP（lecture_07 的版本） |
|---|---|---|---|
| 数据 | 切开 | 切开 | 不切（同一个 TP 组内） |
| 权重存储 | 每个 rank 一整份 | 每个 rank 一片 | 每个 rank 每层一片 |
| 每个 rank 算什么 | 自己的样本 × 完整的 W | 自己的样本 × 完整的 W（计算前拼回） | 全部样本 × 自己那片 W |
| forward 的通信 | 无 | all-gather 权重 | 每层之后 all-gather 激活 |
| backward 的通信 | all-reduce 梯度 | 需要权重的层再 all-gather 一次权重；用本 rank 的数据算出完整形状的梯度后，做 reduce-scatter | 交换激活的梯度（具体见 `tp_calcs (a)`） |
| 最终拿到的梯度 | 全部数据的梯度，覆盖整个权重 | 全部数据的梯度中，自己那片权重对应的部分 | 见 `tp_calcs (a)` |
| step 更新什么 | 全部权重 | 自己的权重分片 | 自己那片权重 |
| 激活 | 留在本地 | 留在本地 | 每层都要交换 |
| 什么不动 | 权重（已复制） | 激活 | 权重 |

### 3.4 FSDP backward 中的梯度：先“本地、完整形状”，再“全局、只有分片”

backward 时先算出的“完整梯度”，“完整”指的是**形状**：它覆盖整个权重矩阵的每个元素，形状和完整的 W 一样。但它只来自**本 rank 的数据分片**，只是全部数据梯度的一个部分和。

| 步骤 | 每个 rank 手上的梯度 | 覆盖哪些权重 | 来自哪些数据 |
|---|---|---|---|
| 1. all-gather 权重后做 backward | 本地梯度 G_r | 整个权重（形状完整） | 只有本 rank 的数据 |
| 2. reduce-scatter：先逐元素求和，再按分片切开 | 总梯度的第 r 片 | 只有自己那片权重 | 全部数据 |
| 3. 除以 N（作业实现在 `finish_gradient_synchronization` 里做） | 平均梯度的第 r 片 | 只有自己那片权重 | 全部数据 |
| 4. optimizer step | 用这片梯度更新自己的权重分片，以及对应的那片优化器状态 | — | — |

讲义 p.43（式 38–44）的原话："the inputs to the reduce-scatter are partial sums on the full weights, while the outputs are full sums on the sharded weights."（reduce-scatter 的输入是完整权重上的部分和，输出是分片权重上的完整和。）

接着用前面的例子。对 y = x · W，权重梯度是 dW = xᵀ · dy，和 W 形状相同（2×4）。为了好算，假设两个样本传回的 dy 都是 `[1, 2, 3, 4]`；实际中每个样本的 dy 一般不同，过程完全一样。

```text
rank 0（只有 s1 = [1, 2]）：G0 = [1; 2] · [1 2 3 4] = [1 2 3 4; 2 4 6 8]
rank 1（只有 s2 = [3, 4]）：G1 = [3; 4] · [1 2 3 4] = [3 6 9 12; 4 8 12 16]
  ↑ 形状都是完整的 2×4，但各自只包含自己样本的贡献

reduce-scatter：先逐元素求和  G0 + G1 = [4 8 12 16; 6 12 18 24]  ← 全部数据的梯度
                再按列切开    rank 0 拿第 1–2 列 [4 8; 6 12]
                              rank 1 拿第 3–4 列 [12 16; 18 24]
除以 N = 2：    rank 0 得到 [2 4; 3 6]，rank 1 得到 [6 8; 9 12]
step：          rank 0 用它更新自己存的 W 第 1–2 列，rank 1 更新第 3–4 列
```

和 DDP 对比：DDP 做 all-reduce，每个 rank 都拿到整个权重的总梯度，因为它要更新全部权重；FSDP 的每个 rank 只更新自己那片权重，所以 reduce-scatter 拿到自己那片总梯度就够了。

### 3.5 TP 不切的数据从哪来

- **第一层的输入，也就是原始 batch：一直持有，不靠通信。** lecture_07 第 450 行，每个进程启动时就拿到同一份完整数据。实际系统中，同一个 TP 组里的 rank 读取同一批数据，或者由组内一个 rank 读取后广播给其他 rank。
- **之后每一层的输入：靠通信获得。** 第 k 层的输入就是第 k−1 层的输出，而每个 rank 只算出了其中一部分，要通过通信拼出完整的输入（lecture_07 的版本用 all-gather）。讲义写的 "transfer all data/activations" 指的就是这一步。

```text
原始 batch（每个 rank 各有一份）
 → 第 0 层：完整输入 × 自己那片 W0 → 部分激活 → all-gather → 完整激活
 → 第 1 层：完整输入 × 自己那片 W1 → 部分激活 → all-gather → 完整激活
 → ...
```

“TP 不切数据”说的是**同一个 TP 组内部**。实际训练中 TP 通常和 DP 或 FSDP 一起用（讲义 p.44："In practice, FSDP is often combined with ... tensor parallelism"）：数据先在不同的组之间切开，同一个 TP 组里的 rank 共享同一份数据切片。

### 3.6 容易混淆的几点

1. **画竖线的那张图是 TP，不是 FSDP。** FSDP 的存储方式和它一样（每个 rank 存每层的一片），但 FSDP 还会切开数据，计算前把权重拼回来。如果给 FSDP 画图，Data 方块也要切开。
2. **FSDP all-gather 的是权重，不是梯度。** 梯度在 backward 时做 reduce-scatter。
3. **FSDP 计算之后 reduce 的是梯度，不是激活。** FSDP 中激活从不跨 rank 传输；每个 rank 只需要自己样本的激活，用来算自己样本的 loss 和 backward。
4. **lecture_07 的 TP 和 PP 都不是数据并行。** 两张图里的 Data 都没有被切开，它们都属于模型并行。
5. **TP 的交换不一定是 all-gather。** 按列切得到的是几列最终值，用 all-gather 拼接；按行切得到的是部分和，用 all-reduce 求和。
6. **FSDP backward 先算出的“完整梯度”只是形状完整。** 它只来自本 rank 的数据；reduce-scatter 之后，才得到全部数据的梯度，而且每个 rank 只拿到自己那片，正好用来更新自己的权重分片。

---

## 4. 和作业的关系

| 内容 | 作业 2 中的位置 |
|---|---|
| DDP → optimizer state sharding → FSDP | 第 5–7 章，需要写代码实现 |
| TP、PP | 不需要实现 |
| 通信原语的 ring 耗时 | §8.1（p.40–41），`alternate_ring_all_reduce` |
| DP 的计算量和通信时间 | §8.2（p.41–42），`data_parallel_calcs` |
| FSDP 的计算量和通信时间 | §8.3（p.42–44），`fsdp_calcs` |
| TP（包括写出上面 FFN 配置的 backward） | §8.4（p.44–45），`tp_calcs` |
| FSDP 和 TP 组合的 2D 并行 | p.46，`fsdp_tp_calcs` |
| PP、EP | 只在 p.40 给了定义，没有题目 |

“搬权重”和“搬激活”哪个更划算，第 8 章的 `fsdp_calcs` 和 `tp_calcs` 会让你定量比较，留给自己推导。
