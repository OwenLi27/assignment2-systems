import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import argparse
import json


def set_up(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, required=True, choices=["gloo", "nccl"])
    p.add_argument("--output_file_path", type=str, required=True)
    return p.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def distributed_allreduce_worker(rank,
                                 world_size,
                                 tensor_length_params,
                                 backend,
                                 output_file_path):
    set_up(rank, world_size, backend)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    for tensor_length in tensor_length_params:
        tensor = torch.rand(1, tensor_length, dtype=torch.float32, device=device)
        # warmup 5 times
        for _ in range(5):
            dist.all_reduce(tensor, async_op=False)

        results = []
        # measure 10 times
        for _ in range(10):
            synchronize(device)
            start_time = timeit.default_timer()
            dist.all_reduce(tensor, async_op=False)
            synchronize(device)
            end_time = timeit.default_timer()
            step_time = end_time - start_time
            results.append(step_time)

        all_results = [None for _ in range(world_size)]
        dist.all_gather_object(all_results, results)
        if rank == 0:
            avg_time = sum([sum(sublist) for sublist in all_results]) / (len(all_results) * len(all_results[0]))
            standard_deviation = (sum((x - avg_time) ** 2 for sublist in all_results for x in sublist) / (len(all_results) * len(all_results[0]))) ** 0.5
            print(f"num_process: {world_size}, "
                  f"tensor_length: {tensor_length}, "
                  f"average time: {avg_time:.6f} seconds, "
                  f"standard deviation: {standard_deviation:.6f}")

            with open(output_file_path, "a") as f:
                json.dump({
                    "num_process": world_size,
                    "tensor_length": tensor_length,
                    "average_time": avg_time,
                    "standard_deviation": standard_deviation,
                    "backend": backend,
                    "dtype": str(tensor.dtype),
                    "warmup_runs": 5,
                    "measure_runs": 10,
                    "raw_data": all_results,
                    "data_size(MB)": tensor.numel() * tensor.element_size()/1e6,
                    "device": str(device)
                }, f)
                f.write("\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    tensor_length_params = [int(1e6/4), int(1e7/4), int(1e8/4), int(1e9/4)]
    num_process_params = [2, 4, 6]
    args = parse_args()
    backend = args.backend
    output_path = args.output_file_path
    for num_process in num_process_params:
        if backend == "nccl" and num_process > torch.cuda.device_count():
            print(f"Skipping num_process={num_process} as it exceeds available CUDA devices ({torch.cuda.device_count()})")
            continue
        mp.spawn(distributed_allreduce_worker,
                 args=(num_process, tensor_length_params, backend, output_path),
                 nprocs=num_process,
                 join=True)
