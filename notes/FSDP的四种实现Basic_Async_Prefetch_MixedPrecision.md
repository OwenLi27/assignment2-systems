# FSDP 四个版本的 forward / backward 机制：基础版 → Async → Prefetch → MixedPrecision

> 对应讲义第 7 章 (p.38–39) 和 `cs336_systems/fully_sharded_data_parallel_training.py` 里的 `FSDP`、`AsyncFSDP`、`PrefetchFSDP`、`MixedPrecisionFSDP`，四个类逐层继承。
>
> 相关笔记：[数据并行的逐步优化](数据并行的逐步优化.md)、[5.3.2 同步 vs 异步](DDP的sync和async机制.md)

---

## 0. 四个版本的区别

| 版本 | 在上一版基础上改了什么 | 效果 |
|---|---|---|
| `FSDP` | —（所有通信都是同步的） | 没有任何通信和计算重叠 |
| `AsyncFSDP` | backward 中的 reduce-scatter 和 norm 的 all-reduce 改成异步，到 finish 时才统一 wait | backward 里的梯度通信可以和计算重叠 |
| `PrefetchFSDP` | forward 中的 all-gather 改成**提前两层**异步发起 | forward 里的权重 gather 也可以和计算重叠 |
| `MixedPrecisionFSDP` | 传了 `compute_dtype` 时：分片先转成低精度再 gather，计算也用低精度；梯度转回 FP32 再做 reduce-scatter | 权重的通信量减半，计算更快；主权重和优化器状态仍然是 FP32 |

---

## 1. 四个版本共用的部分

**平时每个 rank 只保存什么**：每个 Linear 或 Embedding 权重的一片（FP32 分片），存在 `weight.data` 里。完整形状存在 `weight._full_shape`。norm 不切分，每个 rank 存完整的一份。

**各个 hook 的注册位置和作用**（以基础版为准）：

| hook | 注册在哪 | 什么时候触发 | 做什么 |
|---|---|---|---|
| `gather_weight_pre_hook` | 每个切分层的 forward pre-hook；**另外只给 Linear** 注册成 backward pre-hook | 这一层 forward 前 / Linear 的 backward 前 | 把分片保存到 `weight._saved_fp32`；用 `get_communication_shard` 取出要通信的张量，在它上面分配完整形状的缓冲区，all-gather 后换进 `weight.data` |
| `forward_post_hook` | 每个切分层的 forward hook | 这一层 forward 后 | 把 `weight.data` 换回分片；完整权重没有引用了，会被释放 |
| `backward_post_hook` | 切分层 weight 上的 post-accumulate hook | 这个权重的完整梯度累加好之后 | 调用 `reduce_scatter_grad` 得到（分片梯度缓冲区、handle）；把 grad 设为 None；把 data 换回分片；登记到待处理列表 |
| `unsharded_grad_hook` | norm 参数上的 post-accumulate hook | norm 的梯度累加好之后 | 调用 `all_reduce_grad` 得到 handle，登记到待处理列表 |
| `finish_gradient_synchronization` | 由 adapter 在 optimizer step 前调用 | backward 结束之后 | 有 handle 的先 wait；再除以 N；切分参数把缓冲区赋给 `.grad`；最后清空列表 |

**`get_communication_shard(weight)`**：返回这次 all-gather 要发出去的张量。基类直接返回 `weight.data`（FP32 分片）。基类的同步 gather 和 `PrefetchFSDP` 的异步预取都通过它取张量，并且在它上面分配缓冲区，所以缓冲区的 dtype 会自动跟着它走。`MixedPrecisionFSDP` 只需要重写这一个方法，两条 gather 路径就都会改用低精度。

**为什么 Embedding 不注册 backward pre-hook**：Embedding 的 backward 只需要用到 token 索引，不需要权重的值。而且它的输入是整数、不需要梯度，给它注册 full backward pre-hook 会触发警告，测试开了 `filterwarnings("error")`，会直接失败。

---

## 2. 基础版 `FSDP`：所有通信都是同步的

### forward（逐层，以第 i 层为例）

1. **pre-hook**：同步 all-gather 第 i 层的权重。在通信完成之前，计算只能停下来等。
2. 用完整权重计算第 i 层。
3. **post-hook**：换回分片，完整权重被释放。

### backward（按相反的顺序逐层）

1. **只有 Linear**：backward pre-hook 再同步 all-gather 一次。forward 之后完整权重已经释放了，而计算输入的梯度需要它。
2. 计算激活梯度和完整的权重梯度。
3. **post-accumulate hook**：同步 reduce-scatter，得到本 rank 那片梯度的总和；然后换回分片，把结果登记到待处理列表。
4. **norm**：同步 all-reduce。

backward 全部结束后，`finish` 统一做除法和赋值。

### 时间线

下面以 4 层的 `ToyFSDPModel` 为例：E = embedding，L1、L2 = 两个 Linear，H = lm_head。G 表示 all-gather，B 表示 backward 计算，RS 表示 reduce-scatter，`....` 表示计算在等通信。时间长度只是示意。

```text
forward
t       : 0       4       8       12      16      20      24      28
compute : ......[E=====]......[L1====]......[L2====]......[H=====]
comm    : [GE==]        [GL1=]        [GL2=]        [GH==]

backward (Linear 要先 gather; Embedding 不用)
t       : 0       4       8       12      16      20      24      28      32      36
compute : ......[BH====]............[BL2===]............[BL1===]......[BE====]......
comm    : [GH==]        [RSH=][GL2=]        [RSL2][GL1=]        [RSL1]        [RSE=]
```

**问题**：每一次通信，计算都要停下来等。

---

## 3. `AsyncFSDP`：backward 的梯度通信改成异步

**改动**：只重写了 `reduce_scatter_grad` 和 `all_reduce_grad` 两个方法，都加上 `async_op=True`，并把 handle 返回出去。另外重写了 `finish`，用来清空对完整梯度的引用。

- **forward**：和基础版完全一样。
- **backward**：hook 里发起 reduce-scatter 之后立刻返回，计算接着做下一层的 backward，通信在后台进行。到 `finish` 里才统一 wait。

```text
backward (异步 RS)
t       : 0       4       8       12      16      20      24      28      32      36
compute : ......[BH====]............[BL2===]............[BL1===][BE====]......
comm    : [GH==]        [RSH=][GL2=]        [RSL2][GL1=]        [RSL1]  [RSE=]
```

**观察**：在这个小例子里，只有 BE 和 RSL1 重叠上了。原因是每个 Linear 在 backward 前都要**同步** gather，而通信通道一次只执行一个集合操作，这次 gather 只能排在前面那次 RS 后面等着。图里假设通信一次只能执行一个：在 GPU 上，同一个进程组的 NCCL 集合操作在同一条 stream 上依次执行。真正的 FSDP 实现还会在 backward 里预取，先发起下一层的 gather，再发起当前层的 RS。讲义没有要求这一点。

---

## 4. `PrefetchFSDP`：forward 的 gather 提前两层发起

### 4.1 核心规则：提前两层

讲义 p.39 原文："only start gathering after the layer two before the current one has completed its forward pass."（只有在当前层往前数第二层的 forward 完成之后，才开始 gather 当前层。）

同一条规则可以从两个角度来理解：

| 视角 | 说法 |
|---|---|
| 从“被 gather 的层”来看 | 第 i 层的 gather，在第 i−2 层的 forward **结束时**发起 |
| 从“刚算完的层”来看（代码就是这么写的） | 第 p 层的 forward 一结束，就在它的 `forward_post_hook` 里发起第 **p+2** 层的 gather |

**为什么是 2**：

- **提前 1 层**（第 i−1 层结束时才 gather 第 i 层）：gather 刚发起，第 i 层就要开始算了，没有任何计算可以遮住这次通信。
- **提前 2 层**：第 i 层的 gather 和第 i−1 层的计算同时进行。等第 i 层开始时，权重通常已经准备好了。
- **提前更多层**：同时存在的完整权重会更多，显存就省不下来了。

### 4.2 第一次迭代：先记录执行顺序

这时还不知道各层的执行顺序，所以**不预取**，所有层都走父类的同步 gather，和基础版一样。各层的 pre-hook 会依次把自己追加进 `execute_seq`。`forward` 结束后，把 `recorded` 设成 True，并建好“层 → 位置”的字典 `layer_position_map`。

之所以要实际记录，而不是直接用模块的注册顺序，是因为两者可能不一致。比如 Transformer 的 FFN，注册顺序是 w1、w2、w3，实际调用顺序是 w1、w3、w2。

### 4.3 第二次迭代起：完整的事件顺序

各层的位置：0 = E，1 = L1，2 = L2，3 = H。

| 事件 | 发生在哪个方法 | 做什么 | 此时存在的完整权重 |
|---|---|---|---|
| forward 开始 | `forward` | 对第 0、1 层调用 `start_prefetch`（异步，立刻返回） | GE、GL1 正在通信 |
| E 的 pre-hook | `gather_weight_pre_hook` | E 在 `prefetch_dict` 里，取出来 wait，保存分片，换成完整权重 | E（正在使用）、L1（预取中） |
| E 计算 | — | — | 同上 |
| E 的 post-hook | `forward_post_hook` | 换回分片，释放 E；p=0，所以预取第 **2** 层（L2） | L1（已就绪）、L2（预取中） |
| L1 的 pre-hook | `gather_weight_pre_hook` | wait（GL1 早已完成，几乎不用等），然后换入 | L1（正在使用）、L2 |
| L1 的 post-hook | `forward_post_hook` | 释放 L1；p=1，所以预取第 **3** 层（H） | L2、H |
| L2 的 pre-hook / post-hook | 同上 | wait 后换入；释放；p+2=4 超出范围，不再预取 | H |
| H 的 pre-hook / post-hook | 同上 | wait 后换入；释放 | 无 |

**同时存在的完整权重最多只有 2 份**：一份正在使用，一份预取中或已就绪。forward 结束时，`prefetch_dict` 一定是空的，因为每个被预取的层都会被执行到。

### 4.4 时间线（第二次迭代起的 forward）

```text
t       : 0       4       8       12      16      20
compute : ......[E=====][L1====][L2====][H=====]
comm    : [GE==][GL1=]  [GL2=]  [GH==]
```

- GE 和 GL1 在 forward 一开始就发起。
- GL2 在 E 结束时发起（p=0 → 预取第 2 层），和 L1 的计算重叠。
- GH 在 L1 结束时发起（p=1 → 预取第 3 层），和 L2 的计算重叠。
- 只有第一层要等通信完成（t=0–3），之后每一层开始时，权重都已经准备好了。forward 的耗时从基础版的 28 降到 19。

### 4.5 backward

和 `AsyncFSDP` 一样：backward 时 Linear 按需同步 gather，reduce-scatter 是异步的。

### 4.6 为什么不会出现数据竞争

- 预取在后台**读的是分片**，而分片只会在 optimizer step 时被修改，那已经是 forward 结束之后的事了。
- 预取的**缓冲区**要等 pre-hook 里 wait 完成之后才会被换进 `weight.data`。
- 各 rank 的执行顺序相同，预取规则也是确定的，所以各 rank 发起异步 all-gather 的顺序一致。

---

## 5. `MixedPrecisionFSDP`：先转低精度再通信，主权重保持 FP32

### 5.1 讲义的要求

- p.38: "While it is necessary to store master weights in FP32 (any values that are repeatedly accumulated into are sensitive to precision), the weights do not need to be used in FP32 … we may as well convert even before the weight is communicated to save on bandwidth."（主权重必须用 FP32 保存，因为反复累加的值对精度很敏感；但计算时不必用 FP32，而且可以在通信之前就转成低精度，节省带宽。）
- p.39: "When compute_dtype is provided, cast the weights to that dtype before communicating or using them for compute, while keeping master weights and optimizer updates in FP32."（传了 compute_dtype 时，在通信和计算之前把权重转成这个精度；主权重和优化器更新保持 FP32。）

### 5.2 改动：只重写了两个方法

| 方法 | 做什么 |
|---|---|
| `get_communication_shard(weight)` | `compute_dtype` 是 None 时，原样返回 FP32 分片；否则返回转成 `compute_dtype` 的分片 |
| `reduce_scatter_grad(weight, full_grad)` | 如果梯度不是 FP32，先转成 FP32，再用 super 调用父类的方法，也就是 `AsyncFSDP` 的异步版本 |

不传 `compute_dtype` 时，它的行为和 `PrefetchFSDP` 完全一样，所以 fp32 和 fp16 的测试都可以用这一个类来跑。

### 5.3 一层的数据在各个阶段是什么精度（以 compute_dtype = fp16 为例）

| 阶段 | 张量 | 精度 |
|---|---|---|
| 平时 | 本 rank 的分片（`weight.data`，也就是主权重） | FP32 |
| forward 前（同步 gather 或预取） | 要通信的分片：`get_communication_shard` 转换精度后的副本 | fp16 |
| | all-gather 出的完整权重，换进 `weight.data` | fp16 |
| forward 计算 | 用完整权重计算 | fp16 |
| forward 后 | `weight.data` 换回分片 | FP32 |
| backward 前（只有 Linear） | 再 gather 一次完整权重 | fp16 |
| backward 计算 | 完整梯度（Linear 和 Embedding 都是） | fp16 |
| 梯度累加后 | 梯度先转成 FP32，再做 reduce-scatter，结果写进分片梯度缓冲区 | FP32 |
| | `weight.data` 换回分片 | FP32 |
| finish | 缓冲区除以 N 后赋给 `.grad`，和 data 的 dtype 一致 | FP32 |
| optimizer step | 更新分片，也就是主权重 | FP32 |
| norm 的权重和梯度 | 不做任何精度转换 | FP32 |
| `gather_full_params` | 拼接的是 FP32 分片 | FP32 |

### 5.4 为什么这样设计

- **先转换精度再 gather**：通信量减半。精度转换是逐元素进行的，所以“先把分片转成 fp16 再 gather”得到的值，和测试对照组“在完整权重上转成 fp16”完全相同。
- **梯度转回 FP32 再做 reduce-scatter**：
  - 输出缓冲区是按 FP32 分片分配的，而 `reduce_scatter_tensor` 要求输入和输出的 dtype 一致；
  - 对照组是在全部数据上一次算出梯度；这里是各 rank 各算一半再求和，用 FP32 求和可以尽量减少额外的舍入误差。测试的容差只有 1e-4。
- **主权重和优化器状态保持 FP32**：每一步的更新量都要累加到参数上，用低精度累加会丢失很小的更新。
- **norm 不做精度转换**：和测试对照组保持一致，而且 norm 很小，转换精度省不了多少。

### 5.5 为什么阶段 D 只需要改两处

这得益于前面几个阶段的设计：

- **两条 gather 路径都通过 `get_communication_shard` 取张量**，所以只要重写这一个方法，同步 gather 和预取就都会改用低精度。
- **`reduce_scatter_grad` 把完整梯度作为参数显式传入**，所以子类可以先转换精度，再交给父类。转换后的 FP32 张量会被 `AsyncFSDP` 放进 `ongoing_communication_params`，在异步通信完成前一直保留引用。如果像基类早期那样在方法内部直接读 `weight.grad`，转换后的张量就传不进去：`.grad` 赋值时会检查它的 dtype 和形状是否和 `.data` 一致。

### 5.6 一个可以再加强的地方

预取时，传给异步 all-gather 的是转换精度后生成的临时 fp16 张量，现在没有地方保留它的引用，只能依赖 handle 自己持有。测试在 Gloo 上是稳定通过的。如果想更稳妥，可以把它也存进 `prefetch_dict`，等 wait 之后再释放。

---

## 6. 汇总

| 版本 | forward 的 all-gather | backward 的 all-gather（只有 Linear） | reduce-scatter 和 norm 的 all-reduce | 和计算重叠的通信 | 重写或新增的方法 |
|---|---|---|---|---|---|
| `FSDP` | 同步，用到时才做 | 同步，用到时才做 | 同步 | 无 | — |
| `AsyncFSDP` | 同步，用到时才做 | 同步，用到时才做 | 异步，finish 里 wait | backward：RS 和后面层的计算 | `reduce_scatter_grad`、`all_reduce_grad`、`finish` |
| `PrefetchFSDP` | 异步，**提前两层发起** | 同步，用到时才做 | 同 Async | 再加上 forward：gather 和前一层的计算 | `gather_weight_pre_hook`、`forward_post_hook`、`forward`；新增 `start_prefetch` |
| `MixedPrecisionFSDP` | 同 Prefetch，但用 `compute_dtype` 通信（通信量减半） | 同左，用 `compute_dtype` | 梯度先转成 FP32 再做 RS；norm 保持 FP32 | 同 Prefetch | `get_communication_shard`、`reduce_scatter_grad` |
