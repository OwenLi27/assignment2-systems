import cs336_basics.model
import cs336_basics.optimizer
import cs336_basics.nn_utils
import argparse
import numpy as np
import torch
import timeit
import json

from contextlib import nullcontext

# 完成e2e 基准测试
# 两份随机long张量生成:1.输入的token列表 2.用来计算loss的target token列表
# 使用mode 超参数控制测试覆盖的进度范围 forward/backward/optimizer
# warmup 循环 同时每一步后synchronize GPU

import torch._functorch.config
torch._functorch.config.donated_buffer = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=512)

    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=10)

    parser.add_argument("--mode", type=str, choices=["forward", "backward", "optimizer"], required=True)
    parser.add_argument("--output_file", type=str, default="benchmark_results.jsonl")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")

    parser.add_argument("--compile", action="store_true", default=False)

    return parser.parse_args()

def benchmark_model(mode, model, input, target, dtype, device, optimizer=None):
    context = nullcontext()
    if dtype is not None and dtype == torch.bfloat16:
        context = torch.autocast(device_type=device.type, dtype=dtype)
    with context:
        # forward
        logits = model(input)
        if mode == "forward":
            return 
        loss = cs336_basics.nn_utils.cross_entropy(logits, target)
    # backward
    loss.backward()
    if mode == "backward":
        return

    # optimizer step
    optimizer.step() 

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    # 初始化模型和优化器
    model = cs336_basics.model.BasicsTransformerLM(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_length=args.context_length,
        vocab_size=10000  # 假设词汇表大小为10000
    ).to(device)
    if args.compile:
        model = torch.compile(model)

    optimizer = cs336_basics.optimizer.AdamW(model.parameters())
    dtype = None
    if args.precision == "bf16":
        dtype = torch.bfloat16

    # 生成随机输入和目标张量
    input_tensor = torch.randint(0, 10000, (args.batch_size, args.context_length), dtype=torch.long).to(device)
    target_tensor = torch.randint(0, 10000, (args.batch_size, args.context_length), dtype=torch.long).to(device)

    # 预热阶段
    for _ in range(args.warmup_steps):
        optimizer.zero_grad()
        synchronize(device)
        benchmark_model(args.mode, model, input_tensor, target_tensor, dtype, device, optimizer)
        synchronize(device)

    # 基准测试阶段
    outputs = []
    for _ in range(args.benchmark_steps):
        optimizer.zero_grad()
        synchronize(device)
        start_time = timeit.default_timer()
        benchmark_model(args.mode, model, input_tensor, target_tensor, dtype, device, optimizer)
        synchronize(device)
        elapsed_time = timeit.default_timer() - start_time
        outputs.append(elapsed_time)
        print(f"Step time: {elapsed_time:.6f} seconds")
    avg_time = np.mean(outputs)
    std_time = np.std(outputs, ddof=0)
    print(f"Average time per step: {avg_time:.6f} seconds, Std Dev: {std_time:.6f} seconds")
    with open(args.output_file, "a") as f:
        json.dump({
            "config": {
                "d_model": args.d_model,
                "d_ff": args.d_ff,
                "batch_size": args.batch_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "context_length": args.context_length,
                "mode": args.mode,
                "warmup_steps": args.warmup_steps,
                "benchmark_steps": args.benchmark_steps,
                "device": device.type,
                "precision": args.precision,
                "torch_compile": args.compile
            },
            "avg_time": avg_time,
            "std_time": std_time
        }, f)
        f.write("\n")
