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
        # uid_list = nvshmem_ops.get_unique_id()
        # print(f"[Rank {rank}] Unique ID: {uid_list}")
        # uid_tensor = torch.tensor(uid_list, dtype=torch.uint8).cuda()
        unique_id = nvshmem_ops.get_unique_id()
    else:
        # 其他 rank 创建一个同样大小的空 tensor 来接收
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
    """
    Measures point-to-point put bandwidth between rank 0 and rank 1.
    """
    if world_size < 2:
        if rank == 0:
            print("This benchmark requires at least 2 processes. Skipping.")
        return

    # Allocate a symmetric buffer on all PEs
    # This buffer can be a source for puts and a destination for gets from other PEs
    symmetric_buffer = nvshmem_ops.alloc_symmetric(size_bytes)
    
    # Rank 0 will send, Rank 1 will receive
    src_pe = 0
    dst_pe = 1
    
    # Barrier to ensure all allocations are complete before communication
    nvshmem_ops.barrier_all()

    if rank == src_pe:
        # I am the sender. Create a local tensor to be the source of data.
        # Initialize it with some values.
        local_src_tensor = torch.ones(size_bytes, dtype=torch.uint8, device='cuda')
        
        print(f"\n--- Starting Put Bandwidth Benchmark ---")
        print(f"Message Size: {size_bytes / 1024**2:.2f} MB")
        print(f"Sender (PE {src_pe}) -> Receiver (PE {dst_pe})")
        print(f"Warmup iterations: {n_warmup}")
        print(f"Timed iterations: {n_reps}")

        # Warmup runs
        for _ in range(n_warmup):
            nvshmem_ops.put_blocking(symmetric_buffer, local_src_tensor, dst_pe)

        # Timed runs
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(n_reps):
            nvshmem_ops.put_blocking(symmetric_buffer, local_src_tensor, dst_pe)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # Calculate and print results
        duration = end_time - start_time
        total_data_gb = (size_bytes * n_reps) / 1024**3
        bandwidth_gbps = total_data_gb / duration
        
        print("\n--- Results ---")
        print(f"Total time for {n_reps} puts: {duration:.4f} seconds")
        print(f"Bandwidth: {bandwidth_gbps:.2f} GB/s")

    # Barrier to ensure the benchmark is complete before finalizing
    nvshmem_ops.barrier_all()


def main():
    # Set the current CUDA device based on LOCAL_RANK provided by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Step 1: Bootstrap NVSHMEM using torchrun's environment
    rank, world_size = bootstrap_nvshmem()

    # Step 2: Run the performance test
    benchmark_put_bandwidth(rank, world_size)

    # # Step 3: Finalize NVSHMEM
    nvshmem_ops.finalize()
    print(f"[Rank {rank}] Finalized and exiting.")


if __name__ == "__main__":
    main()