import torch
import torch.distributed as dist


class OverlapDDP(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.module = model
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)
        self.handles = []
        self.world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.hook)

    def hook(self, param):
        param.grad /= self.world_size
        handle = dist.all_reduce(param.grad, async_op=True)
        self.handles.append(handle)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles = []
