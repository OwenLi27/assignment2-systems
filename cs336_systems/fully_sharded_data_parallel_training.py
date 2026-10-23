import torch
import torch.distributed as dist

from cs336_basics.model import Embedding, Linear

# FSDP基本实现
class FSDP(torch.nn.Module):

    # 获取通信用的权重分片，MixedPrecisionFSDP会改写
    def get_communication_shard(self, weight):
        return weight.data

    # forward/backward 之前all_gather所有rank的权重
    def gather_weight_pre_hook(self, module, args):
        module.weight._saved_fp32 = module.weight.data
        communication_data = self.get_communication_shard(module.weight)
        gathered_weight = communication_data.new_empty(module.weight._full_shape)
        dist.all_gather_into_tensor(gathered_weight, communication_data)
        module.weight.data = gathered_weight

    # forward 之后恢复原来的权重
    def forward_post_hook(self, module, args, output):
        module.weight.data = module.weight._saved_fp32

    # reduce-scatter (synchronous in basic implementation)
    def reduce_scatter_grad(self, weight, full_grad):
        tensor = torch.empty_like(weight._saved_fp32)
        dist.reduce_scatter_tensor(tensor, full_grad, async_op=False)
        return tensor, None

    # backward 之后恢复原来的权重
    def backward_post_hook(self, weight):
        tensor, handle = self.reduce_scatter_grad(weight, weight.grad)
        weight.grad = None
        weight.data = weight._saved_fp32
        self.sharded_params_temp.append((weight, tensor, handle))

    # all reduce (synchronous in basic implementation)
    def all_reduce_grad(self, param):
        dist.all_reduce(param.grad, async_op=False)
        return None

    # 给不切分的参数(norm) 同步梯度
    def unsharded_grad_hook(self, param):
        handle = self.all_reduce_grad(param)
        self.unsharded_params_temp.append((param, handle))

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # 确保rank切分前的权重在所有rank上都是一致的
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        self.unsharded_params = []
        self.sharded_params = set()

        self.sharded_params_temp = []
        self.unsharded_params_temp = []

        for mod in self.module.modules():
            # Linear or Embedding 需要切分权重
            if isinstance(mod, (Embedding, Linear)):
                self.sharded_params.add(mod.weight)
                # 每个rank只保留一部分权重
                mod.weight._full_shape = mod.weight.shape
                chunk = torch.chunk(mod.weight.data, self.world_size, dim=0)[self.rank].clone()
                mod.weight.data = chunk
                mod.register_forward_pre_hook(self.gather_weight_pre_hook)
                mod.register_forward_hook(self.forward_post_hook)
                if isinstance(mod, Linear):
                    mod.register_full_backward_pre_hook(self.gather_weight_pre_hook)
                mod.weight.register_post_accumulate_grad_hook(self.backward_post_hook)
        for param in self.module.parameters():
            if param in self.sharded_params:
                continue
            if not param.requires_grad:
                continue
            self.unsharded_params.append(param)
            param.register_post_accumulate_grad_hook(self.unsharded_grad_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for weight, tensor, handle in self.sharded_params_temp:
            if handle is not None:
                handle.wait()
            tensor /= self.world_size
            weight.grad = tensor
        for param, handle in self.unsharded_params_temp:
            if handle is not None:
                handle.wait()
            param.grad /= self.world_size
        self.sharded_params_temp.clear()
        self.unsharded_params_temp.clear()

    def gather_full_params(self):
        full_params = {}
        for param_name, param in self.module.named_parameters():
            if param in self.sharded_params:
                output_tensor = param.data.new_empty(param._full_shape)
                dist.all_gather_into_tensor(output_tensor, param.data)
                full_params[param_name] = output_tensor
            else:
                full_params[param_name] = param.data
        return full_params

# Add async communication for reduce_scatter_grad/all_reduce
class AsyncFSDP(FSDP):

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__(module, compute_dtype)
        self.ongoing_communication_params = []

    def reduce_scatter_grad(self, weight, full_grad):
        tensor = torch.empty_like(weight._saved_fp32)
        handle = dist.reduce_scatter_tensor(tensor, full_grad, async_op=True)
        self.ongoing_communication_params.append(full_grad)
        return tensor, handle

    def all_reduce_grad(self, param):
        handle = dist.all_reduce(param.grad, async_op=True)
        return handle

    def finish_gradient_synchronization(self):
        super().finish_gradient_synchronization()
        self.ongoing_communication_params.clear()

class PrefetchFSDP(AsyncFSDP):
    # 在AsyncFSDP的基础上，给forward的all-gather加上“提前两层”的预取
    # 第一次forward (recorded=False)，通过只记录执行顺序，不预取
    # 1. forward 不预取 直接执行模型
    # 2. 每层pre-hook: gather_weight_pre_hook 本层追加进execute_seq。没有预取，走父类同步gather
    # 3. 每层post-hook: forward_post_hook 换回自己rank的分片 释放完整权重（父类forward_post_hook）。还没有完整顺序，不预取
    # 4. forward结束：建立layer_position_map，（layer到pos的映射）。设置recorded为True。
    # 之后每次forward (recorded=True)
    # 1. forward开始：异步预取0,1层(他们前面没有1-2层)
    # 2. 第p层pre-hook：从从prefetch_dict 取出缓冲区，handle -> wait -> 保存分片 -> 换成完整权重
    # 3. 第p层post-hook：换回分片(释放完整权重) -> 异步预取第p+2层
    # 4. forward结束 prefetch_dict为空，同时存在的完整权重最多两层，当前用的 和 预取的第p+2层
    # backward(每次都一样)：Linear 的 backward pre-hook 也调用 gather_weight_pre_hook，
    # 这时没有预取，走父类的同步 gather；梯度的 reduce-scatter 沿用 AsyncFSDP 的异步实现


    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__(module, compute_dtype)
        self.execute_seq = []
        self.recorded = False
        self.layer_position_map = {}
        self.prefetch_dict = {}

    def start_prefetch(self, module):
        if module in self.prefetch_dict:
            return
        communication_data = self.get_communication_shard(module.weight)
        tensor = communication_data.new_empty(module.weight._full_shape)
        handle = dist.all_gather_into_tensor(tensor, communication_data, async_op=True)
        self.prefetch_dict[module] = (tensor, handle)

    def gather_weight_pre_hook(self, module, args):
        # 第一次forward时记录实际模块执行顺序
        if not self.recorded and module not in self.execute_seq:
            self.execute_seq.append(module)
        # 后面forward时，如果模块已经在预取字典中，等待预取完成并替换权重
        if module in self.prefetch_dict:
            tensor, handle = self.prefetch_dict.pop(module)
            handle.wait()
            module.weight._saved_fp32 = module.weight.data
            module.weight.data = tensor
        # 没有预取：第一次 forward，或者 backward 时 Linear 重新gather，都走父类的同步 gather
        else:
            super().gather_weight_pre_hook(module, args)

    def forward_post_hook(self, module, args, output):
        super().forward_post_hook(module, args, output)
        # 如果已经记录了执行顺序(说明已经完成了第一次forward)，预取第p+2层，提前两层发起gather
        if self.recorded:
            p = self.layer_position_map[module]
            if p+2 < len(self.execute_seq):
                next_module = self.execute_seq[p+2]
                self.start_prefetch(next_module)

    def forward(self, *args, **kwargs):
        # 如果已经记录了执行顺序，提前预取前两个模块
        if self.recorded:
            self.start_prefetch(self.execute_seq[0])
            if len(self.execute_seq) > 1:
                self.start_prefetch(self.execute_seq[1])
        output = super().forward(*args, **kwargs)
        # 第一次 forward 结束：顺序已经在 pre-hook 里记录好。这里建立层 -> 位置的映射，再设置标志。
        # 标志必须在调用模型之后才设置，否则pre-hook不会记录了
        if not self.recorded:
            for i, module in enumerate(self.execute_seq):
                self.layer_position_map[module] = i
            self.recorded = True
        return output


class MixedPrecisionFSDP(PrefetchFSDP):
    # 获取用于通信的权重分片，考虑混合精度
    def get_communication_shard(self, weight):
        if self.compute_dtype is None:
            return weight.data
        return weight.data.to(self.compute_dtype)

    # 如果full_grad不是 FP32，就先转成 FP32
    def reduce_scatter_grad(self, weight, full_grad):
        if full_grad.dtype != torch.float32:
            full_grad = full_grad.to(torch.float32)
        return super().reduce_scatter_grad(weight, full_grad)

