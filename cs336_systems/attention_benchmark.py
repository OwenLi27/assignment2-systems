import json
import time
import torch

# import torch._functorch.config
from cs336_basics.model import scaled_dot_product_attention

# torch._functorch.config.donated_buffer = False

# scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
if torch.backends.mps.is_available():
    device = torch.device("mps")

def synchronize():
    if device == torch.device("mps"):
        torch.mps.synchronize()
    elif device == torch.device("cuda"):
        torch.cuda.synchronize()
    else:
        pass
def get_memory_usage():
    if device == torch.device("mps"):
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    elif device == torch.device("cuda"):
        return torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        return None
def empty_cache():
    if device == torch.device("mps"):
        torch.mps.empty_cache()
    elif device == torch.device("cuda"):
        torch.cuda.empty_cache()
    else:
        pass

def benchmark_attention(d_model: int,
                        seq: int,
                        batch_size: int = 8,
                        num_iters: int = 100):
    try:
        empty_cache()
        Q = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True)

        # 进入warmup阶段
        print("Warming up...")
        for _ in range(10):
            out = scaled_dot_product_attention(Q, K, V)
            # 选用最简单的loss方法获得标量 不需要cross entropy精确loss
            loss = out.sum()
            loss.backward()
            # 梯度清零 防止梯度累加
            Q.grad = None
            K.grad = None
            V.grad = None

        empty_cache()
        synchronize()

        # 进入benchmark forward阶段
        print("Benchmarking forward pass...")
        start_time = time.perf_counter()
        for _ in range(num_iters):
            out = scaled_dot_product_attention(Q, K, V)
            synchronize()
        fwd_time = (time.perf_counter() - start_time)/num_iters
        print(f"Forward pass time: {fwd_time*1000:.2f} ms")

        empty_cache()
        out = scaled_dot_product_attention(Q, K, V)
        loss = out.sum()
        synchronize()
        mem_usage = get_memory_usage()

        # 进入benchmark backward阶段
        print("Benchmarking backward pass...")
        start_time = time.perf_counter()
        for _ in range(num_iters):
            loss.backward(retain_graph=True)
            Q.grad = None
            K.grad = None
            V.grad = None
            synchronize()
        bwd_time = (time.perf_counter() - start_time)/num_iters
        print(f"Backward pass time: {bwd_time*1000:.2f} ms")

        return {
            "d_model": d_model,
            "seq_len": seq,
            "fwd_time_ms": fwd_time*1000 if fwd_time is not None else None,
            "bwd_time_ms": bwd_time*1000 if bwd_time is not None else None,
            "estimated_memory_mb": mem_usage,
            "status": "OK"
        }

    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "int_max" in error_msg or "dims larger" in error_msg:
            return {
                "d_model": d_model,
                "seq_len": seq,
                "fwd_time_ms": None,
                "bwd_time_ms": None,
                "estimated_memory_mb": None,
                "status": "OOM/Error"
            }
        else:
            raise

if __name__ == "__main__":
    d_model = [16, 32, 64, 128]
    seq = [256, 1024, 4096, 8192, 16384]

    for d in d_model:
        for s in seq:
            result = benchmark_attention(d_model=d, seq=s)
            with open ("attention_basic_benchmark_results.jsonl", "a") as f:
                json.dump(result, f)
                f.write("\n")
