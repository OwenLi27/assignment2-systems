import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint

# num_layers for this model is 32
d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length))

# Fuse as much torch.compile will allow
block = torch.compile(block, fullgraph=True)
x = torch.randn((4, context_length, d_model), requires_grad=True)

# Now logs the number of bytes saved
total_size_bytes = 0
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter): # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

# # Run forward pass, saving for backward
# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     y = block(x)
#     print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB")   
"""
Total size of saved tensors in single TransformerBlock: 3651.31 MiB
"""

# naive four block layers without checkpointing
# def four_blocks(x):
#     x = block(x)
#     x = block(x)
#     x = block(x)
#     x = block(x)
#     return x

# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     y = four_blocks(x)
    
# print(f"Total size of saved tensors in four TransformerBlocks: {total_size_bytes / (1024**2):.2f} MiB")

"""
Total size of saved tensors in four TransformerBlocks: 14605.25 MiB
"""

def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
# checkpoint throws out all the saved tensors until the backward pass
# when getting to the checkpointed block in the backward pass,
# it reruns a forward pass to produce the saved tensors,
# then completes normal backward pass.
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)
print(f"Total size of saved tensors in four TransformerBlocks with checkpointing: {total_size_bytes / (1024**2):.2f} MiB")

"""
Total size of saved tensors in four TransformerBlocks with checkpointing: 160.00 MiB
160 MiB 不是完整训练的峰值 activation memory, 也不能直接说训练显存降低了 91 倍。
当前统计的是原始 forward 结束后长期留下的 saved tensors, 真正执行backward 时, checkpoint
会重新执行当前 region 的两个block, 短暂重新生成大约两个 block 的内部 
residuals(这时内存峰值会显著大于160MB), 然后完成该 region 的 backward并释放它们。

因此完整过程:
  1. Forward 结束：长期保存约 160 MiB。
  2. Backward 到第二个 region: 重新生成两个 block 的
     residuals。
  3. 完成第二个 region 的 backward并释放。
  4. Backward 到第一个 region: 再次局部重计算。
  5. 完成第一个 region 的 backward并释放。
"""