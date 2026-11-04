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
        module.weight.data = module.weight._saved_fp32  # 释放点：换回分片后，参数不再引用 gather 出的完整权重。Embedding 的完整权重在这里就被释放。
        # NOTE Linear 的不会：autograd 为 backward 保存了它的视图，要到这一层的 backward 结束才释放。

    # reduce-scatter (synchronous in basic implementation)
    def reduce_scatter_grad(self, weight, full_grad):
        tensor = torch.empty_like(weight._saved_fp32)
        dist.reduce_scatter_tensor(tensor, full_grad, async_op=False)
        return tensor, None

    # backward 之后恢复原来的权重
    def backward_post_hook(self, weight):
        tensor, handle = self.reduce_scatter_grad(weight, weight.grad)
        weight.grad = None  # 释放点：去掉参数对完整梯度的引用。基础版的 reduce-scatter 已同步完成，完整梯度在这里就被释放；
        # Async 及其子类还在通信，完整梯度由 ongoing_communication_params 继续持有，到 finish 才释放。
        weight.data = weight._saved_fp32  # 释放点：换回分片，释放 backward 前重新 gather 的完整权重(NOTE 只有 Linear 有这个步骤，对应forward_post_hook里的Linear层不能释放)
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
        self.hook_handles = []
        self.sharded_params = set()

        self.sharded_params_temp = []
        self.unsharded_params_temp = []

        for mod in self.module.modules():
            # Linear or Embedding 需要切分权重
            if isinstance(mod, (Embedding, Linear)):
                self.sharded_params.add(mod.weight)
                # 每个rank只保留一部分权重
                mod.weight._full_shape = mod.weight.shape
                # 权重释放点
                # 1. 张量分两层：张量对象记录形状、stride、偏移量。真正的内存在底层的 storage 里。多个张量(比如视图)可以共享同一个 storage。
                #    storage 靠引用计数管理：最后一个引用它的张量消失时，内存立刻释放，不用等垃圾回收。
                # 2. torch.chunk 返回的是视图，和完整权重共用同一个 storage；.clone() 才会新分配一块 1/N 大小的 storage，并把这一片的数据拷过去。
                # 3. 给 mod.weight.data 赋值不会换掉参数对象(optimizer 和 hook 拿到的还是同一个对象)，只是把参数内部指向的 storage 换成分片的那块。
                #    完整 storage 随之失去最后一个引用，在这一行就被释放。(CUDA 上内存回到 PyTorch 的缓存分配器：
                #    memory_allocated 下降，memory_reserved 不变)
                # 4. 如果不 clone，参数会通过视图继续引用整块完整 storage，显存一点都没省。
                # 前提：没有别的引用指向完整权重。其他变量、它的视图、autograd 为 backward 保存的张量，都会让它继续存活。
                # 验证：赋值前后看 mod.weight.untyped_storage().nbytes()，应从完整大小变成约 1/N。
                chunk = torch.chunk(mod.weight.data, self.world_size, dim=0)[self.rank].clone()
                mod.weight.data = chunk
                self.hook_handles.append(mod.register_forward_pre_hook(self.gather_weight_pre_hook))
                self.hook_handles.append(mod.register_forward_hook(self.forward_post_hook))
                if isinstance(mod, Linear):
                    self.hook_handles.append(mod.register_full_backward_pre_hook(self.gather_weight_pre_hook))
                self.hook_handles.append(mod.weight.register_post_accumulate_grad_hook(self.backward_post_hook))
        for param in self.module.parameters():
            if param in self.sharded_params:
                continue
            if not param.requires_grad:
                continue
            self.unsharded_params.append(param)
            self.hook_handles.append(param.register_post_accumulate_grad_hook(self.unsharded_grad_hook))

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
        # 释放点：清空临时列表，丢掉对 handle 和分片梯度缓冲区的引用。分片梯度此时已交给 .grad，由 optimizer 的 zero_grad 释放。
        # NOTE：每次 backward 后都必须调用 finish，否则这两个列表会跨步累积。
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

    # remove_hooks:丢弃这个实例之前调用，让模型 del 后能立刻释放内存，避免引用环泄漏
    # 原理：
    #   hook 注册的是绑定方法 self.xxx，形成"参数 → hook → self → 模型 → 参数"的引用环；其中 post-accumulate-grad hook 形成的环连 gc.collect() 都回收不了。
    #   对 register_* 返回的 handle 调用 remove() 删掉 hook，环断开后靠引用计数立刻释放。
    # 先调用 finish：等完还在进行的异步通信，避免释放正在被读写的张量(PrefetchFSDP 会另外处理 prefetch_dict)。
    # 用法：只在同一进程里还要再建模型时需要(比如 benchmark 依次比较各版本)：
    #       先 remove_hooks()，再 del 包装器、optimizer、output 和 loss。调用后这个实例不能再用来训练。
    #       不能放进 __del__：泄漏存在时对象不会被销毁，__del__ 根本不会执行。
    def remove_hooks(self):
        self.finish_gradient_synchronization()
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.sharded_params_temp.clear()
        self.unsharded_params_temp.clear()

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
        self.ongoing_communication_params.clear()  # 释放点：所有 handle 已在父类的 finish 里 wait 完，这里释放全部完整梯度。

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
            tensor, handle = self.prefetch_dict.pop(module)  # 释放点：从 prefetch_dict 取出，去掉字典对缓冲区的引用。wait 之后缓冲区换进 .data，之后在 forward_post_hook 里释放。
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
            if p + 2 < len(self.execute_seq):
                next_module = self.execute_seq[p + 2]
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

    def remove_hooks(self):
        for tensor, handle in self.prefetch_dict.values():
            handle.wait()
        self.prefetch_dict.clear()
        super().remove_hooks()


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

