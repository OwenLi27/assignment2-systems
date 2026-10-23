import torch
import torch.distributed as dist
import torch._utils as tu


class FlatGradientDDP(torch.nn.Module):
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
        all_grads = [param.grad for param in self.module.parameters() if param.requires_grad]
        all_grads = tu._flatten_dense_tensors(all_grads)
        dist.all_reduce(all_grads, async_op=False)
        all_grads /= dist.get_world_size()
        # 将扁平化的梯度重新分配回各个参数
        all_grads = tu._unflatten_dense_tensors(all_grads, [param.grad for param in self.module.parameters() if param.requires_grad])
        for param, grad in zip([param for param in self.module.parameters() if param.requires_grad], all_grads):
            param.grad.copy_(grad)
