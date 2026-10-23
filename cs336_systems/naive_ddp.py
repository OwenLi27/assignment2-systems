import torch
import torch.distributed as dist


class NaiveDDP(torch.nn.Module):
    # 接收传入的model
    def __init__(self, model):
        super().__init__()
        self.module = model
        # rank0 发送 其他rank接收模型参数
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)
        # 广播完成后，所有进程的模型参数和缓冲区都与rank0保持一致

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        CPU这边
        1. backward: CPU 把一串 backward kernel 提交到计算 stream，这些 kernel 会写出 grad。
        2. 同步 all-reduce (async_op=False)，NCCL 在 CPU 上做了三件事：
            - 让 NCCL stream 等计算 stream。原因是 all-reduce 要读 grad，必须等 backward 把 grad 写完才能开始。这一步保证的是"输入已经就绪"。
            - 把 all-reduce 提交到 NCCL stream。
            - 因为是同步调用，再让计算 stream 等 NCCL stream 上的 all-reduce 完成。这一步保证的是"输出已经就绪"。
        做完这三件事，函数就返回了。这时 GPU 上的 all-reduce 可能还没开始。
        3. 除法：CPU 把除法 kernel 提交到计算 stream，然后立刻返回。

        GPU 这边:
        CPU        : [提交bwd][提交all-reduce，立刻返回][提交除法][可以去做别的事了]
        计算stream :  [bwd ...............][  等待（第2步插入的等待指令） ][除法][optimizer…]
        NCCL stream:               [      all-reduce      ]
               ↑ NCCL stream 先等 bwd 写完 grad      ↑ 计算 stream 再等 all-reduce 完成
        """
        for param in self.module.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad, async_op=False)
                param.grad /= dist.get_world_size()
