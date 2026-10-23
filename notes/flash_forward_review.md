# FlashAttention forward (a)/(b)：实现检查与精度分析

验证日期：2026-09-28。本文记录该日期的实现状态与测试结果。

## 检查对象

- PyTorch 实现：`FlashForwardFunc`，初始版本文件名为 `flash_forward_pytorch.py`。
- Triton 实现：`FlashForwardTriton`，初始版本文件名为 `flash_forward_triton.py`。

以下问题和代码位置对应初始版本；同日的精度修订及其验证结果见末节。

## 验证环境与方法

- 项目虚拟环境：PyTorch 2.11.0，CUDA 不可用，MPS 可用，未安装 Triton。
- 两个文件均通过 `python -m py_compile`。这只能证明 Python 语法有效，不能证明 Triton JIT 编译或 CUDA 执行成功。
- 官方测试：在单次 Python 进程中把 `tests.test_attention.get_flashattention_autograd_function_pytorch` 绑定到 `FlashForwardFunc`，再通过 pytest 执行 `test_flash_forward_pass_pytorch`。结果：1 passed, 5 deselected。
- 额外使用官方 `_attention_and_lse` 作为参考，分别在 CPU、MPS 检查输出 O 和保存的 logsumexp L，容差为 atol=rtol=1e-4。
- 检查作业 PDF 第 26–27 页的 flash_forward (a)/(b) 要求，以及本地 Triton 源码中 dot 操作数类型的检查。

## 已通过的数值检查

以下均为 float32、batch=2、非 causal；每组在 CPU 和 MPS 上分别通过。

| N_q | N_k | D | 结果 |
|---|---|---|---|
| 32 | 32 | 16 | O、L 均通过 |
| 64 | 96 | 32 | O、L 均通过 |
| 128 | 128 | 64 | O、L 均通过 |

这些额外测试中，O 最大绝对误差不超过 4.18e-7，L 不超过 4.77e-7。由此可确认：在这些输入下，PyTorch 版的 online softmax 最大值更新、分母重缩放、未归一化输出累计、最终归一化及 LSE 计算正确。

## 初始版本的问题

### 1. PyTorch 版不支持长度 16：已复现

位置：`flash_forward_pytorch.py:6`、第 17–25 行。

固定 B_q=B_k=32，rearrange 要求序列长度能被 32 整除。当 N_q=16 或 N_k=16 时直接抛出 EinopsError。CPU 和 MPS 均复现：

- (N_q,N_k,D)=(16,32,16)
- (N_q,N_k,D)=(32,16,16)

作业第 26 页允许维度为至少 16 的二次幂，因此 16 是题目范围内的边界，而不只是额外扩展需求。需要重新检查 tile 大小与短序列之间的关系。

### 2. PyTorch 版没有保存 O：已检查

位置：`flash_forward_pytorch.py:58`。

ctx.save_for_backward 只保存 Q、K、V、logsumexp，没有保存输出 O。作业 (a) 明确要求保存 L、Q、K、V、O。现有官方前向测试仅检查 O 和保存的 L，因此仍会通过；此遗漏不影响当前前向结果，但不满足为后续 backward 保存状态的要求。

### 3. Triton 版的补零 key 会参与 softmax：静态分析与数学复核

位置：`flash_forward_triton.py:79`–89。

tl.load 的 boundary_check 防止越界读取，padding_option="zero" 让越界 K/V 变成 0；但它不会自动将相应 score 排除出 softmax。越界 key 的 score=0，exp(0) 仍计入分母。

固定 K_TILE_SIZE=32 时，N_k=16 就会触发；此输入也在题目允许范围内。一般情况下，N_k 不是 32 的倍数也有此问题。

用 CPU 独立复核该数学行为（不是执行 Triton kernel）：令 Q=0、16 个有效 K=0、有效 V 全为 1。正确 O=1，L=ln(16)≈2.772589；加入 16 个补零 K/V 后，O=0.5，L=ln(32)≈3.465736。需要区分“读内存安全”与“softmax 中只计入有效 key”。

### 4. Triton 版半精度输入有 dot 类型不一致问题：静态检查

位置：`flash_forward_triton.py:89`，tl.dot(p, V_tile)。

scores 与 p 为 float32；若 V 输入为 float16 或 bfloat16，V_tile 保留该输入类型。本地 Triton `python/triton/language/semantic.py:1461` 对此类 dot 要求两个操作数 dtype 一致，作业第 27 页也专门提示乘 V 前的 P 类型转换。因此半精度路径存在编译错误风险；未在 GPU 上复现。float32 输入不触发这一类型不一致。

## 不应混淆的问题与验证边界

- 初始版本均未使用 `is_causal`，因此不支持 `causal=True`。causal masking 属于后续 (c) 的要求。
- 两版 backward 均抛出 NotImplementedError，符合当前只实现 forward 的阶段要求，未验证梯度。
- 初始验证时，`tests/adapters.py` 的两个 getter 尚未实现，PyTorch 测试通过进程内临时绑定 getter 执行。
- Triton 版未验证 JIT 编译、GPU 数值误差、不同硬件的 dot 精度和性能。不能把 Python 语法检查或 CPU 数学复核等同于 GPU 测试通过。
- 初始检查未覆盖 FP16/BF16 的完整数值稳定性，也未验证额外 head 维度等题目接口之外的输入。

## 结论

PyTorch 版通过官方前向用例和额外的可整除分块测试，但有短序列边界与保存状态问题；Triton 版存在上述可定位的边界/类型问题，仍需要 CUDA 环境执行验证。精度修订及其验证结果见下节。

## 精度专项修订（2026-09-28）

精度修订版加入了 `p.to(V_tile.dtype)`，并接入 attention 测试 adapter；初始版本的问题清单不能直接视为修订后的状态。

Triton 实现的精度处理如下：

- `m`、`l`、`O_tile` 保持 FP32，P 在乘 V 前转换为与 V 相同的类型。softmax 分母仍用转换前的 FP32 P 计算。
- 先重缩放 `O_tile`，再用 `tl.dot(..., acc=O_tile, out_dtype=tl.float32)` 累加 PV。
- 归一化后，显式将输出转为 `O_block_ptr.type.element_ty` 再写回。原本 store 也会隐式转换；显式转换让舍入位置更清楚。
- 将 HBM 中的 logsumexp 缓冲区改为 FP32，避免输入为 FP16/BF16 时损失归一化统计量精度。
- 两个 dot 显式指定 `input_precision="ieee"`，避免 FP32 输入默认采用 TF32 的精度风险。这是精度优先的基线，可能降低 FP32 路径吞吐量；不是性能最优结论。

精度测试 `tests/test_flash_forward_precision.py` 覆盖：FP32/FP16/BF16 × 两种输入幅度，共 6 例，比较输出与保存的 LSE 和 FP64 参考值。使用可整除的长度，专门隔离精度问题，不覆盖 tail/causal masking。

修订后的验证结果（2026-09-28）：

- 两个文件的 `py_compile` 通过；这不等同于 Triton JIT 编译通过。
- 验证环境没有 CUDA，6 例精度测试均跳过，尚未完成 GPU 数值验证。
- 完整 pytest：1 passed、10 skipped、9 failed。通过项是 PyTorch 前向；失败项均最终来自未实现接口的 `NotImplementedError`，分别是：
  - `tests/test_attention.py::test_flash_backward_pytorch`
  - `tests/test_ddp.py::test_DistributedDataParallel[ToyModel]`
  - `tests/test_ddp.py::test_DistributedDataParallel[ToyModelWithTiedWeights]`
  - `tests/test_fsdp.py::test_fsdp_correctness[fp32]`
  - `tests/test_fsdp.py::test_fsdp_correctness[fp16]`
  - `tests/test_fsdp.py::test_fsdp_gradient_sync[fp32]`
  - `tests/test_fsdp.py::test_fsdp_gradient_sync[fp16]`
  - `tests/test_sharded_optimizer.py::test_sharded_optimizer[ToyModel]`
  - `tests/test_sharded_optimizer.py::test_sharded_optimizer[ToyModelWithTiedWeights]`

精度修订未覆盖尾块 key masking、causal masking 和 backward。N_k=16 的补零 key 问题仍需单独处理。
