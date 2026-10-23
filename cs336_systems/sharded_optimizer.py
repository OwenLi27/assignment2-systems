from typing import Any
import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.kwargs = kwargs
        # 全局参数和rank的映射关系
        self.param_rank_map = {}
        # 全局参数计数器 用来分配rank 策略A
        self.counter = 0
        # 全局负载列表 用来分配rank 策略B
        self.rank_load = [0] * self.world_size
        # 构造期间暂存本 rank 的参数组
        self.current_rank_params = []
        # 内层优化器
        self.inner_optimizer = None
        # super().__init__ 构造期间调用下面的 add_param_group 方法。
        # 内层优化器初始化在下面，所以构造期间的参数会暂存到 current_rank_params
        super().__init__(params, defaults=kwargs)
        self.inner_optimizer = optimizer_cls(self.current_rank_params, **self.kwargs)
        # 清空
        self.current_rank_params = []


    # 分配策略A：按参数个数轮流分配，参数个数均衡但不保证参数内张量元素个数均衡。
    # 举例
    # ┌────────────────────────┬──────────┬──────────┐
    # │ 参数                   │ 元素个数 │ 轮流分给 │
    # ├────────────────────────┼──────────┼──────────┤
    # │ P0 no_grad_fixed_param │ 2        │ rank 0   │
    # ├────────────────────────┼──────────┼──────────┤
    # │ P1 fc1.weight          │ 100      │ rank 1   │
    # ├────────────────────────┼──────────┼──────────┤
    # │ P2 fc2.fc.weight       │ 500      │ rank 0   │
    # ├────────────────────────┼──────────┼──────────┤
    # │ P3 fc2.fc.bias         │ 50       │ rank 1   │
    # ├────────────────────────┼──────────┼──────────┤
    # │ P4 fc3.weight          │ 500      │ rank 0   │
    # └────────────────────────┴──────────┴──────────┘
    # 结果
    # ┌──────────────────────────┬────────┬────────┐
    # │                          │ rank 0 │ rank 1 │
    # ├──────────────────────────┼────────┼────────┤
    # │ 张量个数                 │ 3      │ 2      │
    # ├──────────────────────────┼────────┼────────┤
    # │ 元素总数                 │ 1002   │ 150    │
    # ├──────────────────────────┼────────┼────────┤
    # │ 其中需要存 m、v 的元素数  │ 1000   │ 100    │
    # └──────────────────────────┴────────┴────────┘
    def rotation_assign_rank(self, param):
        assigned_rank = self.counter % self.world_size
        self.param_rank_map[param] = assigned_rank
        self.counter += 1
        return assigned_rank

    # 分配策略B：按参数大小轮流分配，保证每个 rank 的参数张量元素总量大致均衡
    def load_balance_assign_rank(self, param):
        # 找到当前负载最小的 rank
        assigned_rank = self.rank_load.index(min(self.rank_load))
        self.param_rank_map[param] = assigned_rank
        # 更新负载：用参数元素个数作为负载
        self.rank_load[assigned_rank] += param.numel()
        return assigned_rank

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        self_params_list = []
        for param in param_group['params']:
            assigned_rank = self.load_balance_assign_rank(param)
            if assigned_rank == self.rank:
                self_params_list.append(param)
        self_param_group = param_group.copy()
        self_param_group['params'] = self_params_list
        # 有内层优化器 -> 训练中调用 给 内层优化器
        if self.inner_optimizer is not None:
            self.inner_optimizer.add_param_group(self_param_group)
        # 否则 -> 构造期间调用 给 构造期间的暂存列表
        else:
            self.current_rank_params.append(self_param_group)

    def step(self, closure=None, **kwargs):
        # 内部的优化器只更新当前rank负责的参数
        loss = self.inner_optimizer.step(closure, **kwargs)
        # 1. step后进行参数同步：每个 rank 都对每个参数调用 broadcast，顺序一致，src = 负责该参数的 rank。
        # 2. 一次 broadcast 只有 src 发送，其他 rank 接收并覆盖本地旧值（集合操作按调用顺序配对，只让发送方或只让接收方调用都会卡住）。
        # 3. 发送方一定先完成自己的 step 才进入这个循环，所以不会发出旧数据；先到的 rank 会在这里等 step 最慢的那个rank。
        # - 发送方发出的是新数据：每个 rank 内部是按顺序执行的，发送方做完自己的 step，才会进入广播循环。在 GPU 上，NCCL 发起通信前还会先等本 rank 的计算 stream。
        # - 当前在本地用的是 Gloo，没有 stream，靠的是 CPU 按顺序执行代码。
        # - 先到的 rank 要等：一次集合操作，必须等所有 rank 都调用了才能完成。所以先进入广播循环的 rank 会停在某次 broadcast 上，等最慢的 rank 赶上来。
        # rank 0 : [step (fast)]...... wait for rank 1 ......[P0][P1][P2][P3][P4]
        # rank 1 : [step (slow)............................][P0][P1][P2][P3][P4]
        for param, assigned_rank in self.param_rank_map.items():
            dist.broadcast(param.data, src=assigned_rank)
        return loss
