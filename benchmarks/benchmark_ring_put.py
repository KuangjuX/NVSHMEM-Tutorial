import os
import time
import torch
import torch.distributed as dist
import _nvshmem_pybind_cpp as nvshmem_ops # The name comes from setup.py

os.environ["NVSHMEM_REMOTE_TRANSPORT"] = "none"

def bootstrap_nvshmem():
    """
    Initializes torch.distributed and then uses it to bootstrap NVSHMEM.
    """
    # 1. 从环境变量获取 rank 和 world_size
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 2. 【关键步骤】将当前进程绑定到指定的 GPU
    #    这可以防止所有进程都挤在 GPU 0 上
    print(f"[Rank {rank}] Setting device to cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # 3. 初始化 PyTorch 的分布式通信组
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # 4. 使用 PyTorch DDP 来同步 NVSHMEM 的 unique_id
    if rank == 0:
        # Rank 0 创建 unique_id
        unique_id = nvshmem_ops.get_unique_id()
    else:
        unique_id = None

    unique_ids = [None] * world_size
    dist.all_gather_object(unique_ids, unique_id, group = dist.group.WORLD)

    # 5. 使用接收到的 unique_id 初始化 NVSHMEM
    nvshmem_ops.init_with_unique_id(unique_ids[0], rank, world_size)
    
    print(f"[Rank {rank}] NVSHMEM initialized successfully.")
    
    # 确保所有进程都完成了初始化再继续
    dist.barrier()

    return rank, world_size


def benchmark_put_bandwidth(rank, world_size, size_bytes=1024*1024*128, n_warmup=10, n_reps=100):
    if world_size < 2:
        if rank == 0:
            print("This benchmark requires at least 2 processes. Skipping.")
        return
    
    if rank == 0:
        print(f"NVSHMEM_DISABLE_P2P: {os.environ.get('NVSHMEM_DISABLE_P2P')}")

    sbyte_tensor = nvshmem_ops.alloc_symmetric(size_bytes)
    rbyte_tensor = nvshmem_ops.alloc_symmetric(size_bytes)

    send_tensor = sbyte_tensor.view(torch.float32)
    recv_tensor = rbyte_tensor.view(torch.float32)

    num_elems = size_bytes // 4
    
    threads_per_block = 1024
    num_blocks = (num_elems + threads_per_block - 1) // threads_per_block

    if rank == 0:
        print(f"num_elems: {num_elems}, num_blocks: {num_blocks}, threads_per_block: {threads_per_block}")

    for _ in range(n_warmup):
        recv_tensor.zero_()
        nvshmem_ops.launch_ring_put_block(send_tensor, recv_tensor, num_blocks, threads_per_block)
        nvshmem_ops.barrier_all()
    
    torch.cuda.synchronize()
    nvshmem_ops.barrier_all()


    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_reps):
        nvshmem_ops.launch_ring_put_block(send_tensor, recv_tensor, num_blocks, threads_per_block)
        nvshmem_ops.barrier_all()

    end_event.record()
    torch.cuda.synchronize()
    nvshmem_ops.barrier_all()

    if rank == 0:
        elapsed_time_ms = start_event.elapsed_time(end_event)
        total_time_s = elapsed_time_ms / 1000.0
        # 在 ring benchmark 中，每个 GPU 发送一次数据
        total_data_bytes = size_bytes * n_reps 
        # 带宽通常使用 10^9 (GB/s)，而不是 1024^3 (GiB/s)
        bandwidth_gbps = (total_data_bytes / total_time_s) / 1e9 
        print(f"Bandwidth: {bandwidth_gbps:.2f} GB/s")


    # nvshmem_ops.free_symmetric(send_tensor)
    # nvshmem_ops.free_symmetric(recv_tensor)

    
def main():
    # Set the current CUDA device based on LOCAL_RANK provided by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Step 1: Bootstrap NVSHMEM using torchrun's environment
    rank, world_size = bootstrap_nvshmem()

    # Step 2: Run the performance test
    benchmark_put_bandwidth(rank, world_size, 8192)

    # # Step 3: Finalize NVSHMEM
    nvshmem_ops.finalize()
    print(f"[Rank {rank}] Finalized and exiting.")


if __name__ == "__main__":
    main()