import os
import sys
import inspect
from pathlib import Path
import torch
import torch.distributed as dist

# Find nvshmem_tutorial module without installing it.
sys.path.append(Path(__file__).parent.parent)  # Equal to using PYTHONPATH

from nvshmem_tutorial import NvshmemBuffer


def init_dist():
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    # This print is helpful for debugging
    # print(f"ip = {ip}, port = {port}, world_size = {world_size}, rank = {rank}, local_rank={local_rank}")

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": world_size,
        "rank": rank,
    }
    if "device_id" in sig.parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)

    print(f"Rank {rank} initialized successfully on device cuda:{local_rank}.")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(world_size))),
    )


def benchmark_all_gather(nvshmem_buffer: NvshmemBuffer, rank: int, world_size: int):
    """
    Benchmark and compare the performance of NCCL and NVSHMEM all-gather.
    """
    if world_size <= 1:
        if rank == 0:
            print("Skipping benchmark because world_size is 1.")
        return

    # --- Benchmark Settings ---
    # Test tensor sizes from 4KB to 128MB
    tensor_sizes_kb = [
        4,
        16,
        64,
        256,
        1024,
        2 * 1024,
        4 * 1024,
        8 * 1024,
        16 * 1024,
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1024 * 1024,
        2048 * 1024,
    ]
    dtype = torch.bfloat16
    warmup_iters = 10
    benchmark_iters = 50

    if rank == 0:
        print("\n" + "=" * 95)
        print(f" All-Gather Benchmark (world_size={world_size}, dtype={dtype})")
        print(
            f" Warmup Iterations: {warmup_iters}, Benchmark Iterations: {benchmark_iters}"
        )
        print("=" * 95)
        # Print table header
        header = f"{'Size (KB)':>12s} | {'NCCL Time (ms)':>18s} | {'NVSHMEM Time (ms)':>18s} | {'NCCL BW (GB/s)':>18s} | {'NVSHMEM BW (GB/s)':>18s}"
        print(header)
        print("-" * len(header))

    for size_kb in tensor_sizes_kb:
        num_elements = (size_kb * 1024) // dtype.itemsize
        tensor = torch.randn(num_elements, dtype=dtype, device="cuda")
        tensor_bytes = tensor.numel() * tensor.element_size()

        # --- Prepare output tensors ---
        # Use the same tensor_list to reduce allocations.
        output_list = [torch.empty_like(tensor) for _ in range(world_size)]

        # --- NCCL Benchmark ---
        # Warm-up
        for _ in range(warmup_iters):
            dist.all_gather(output_list, tensor, group=dist.group.WORLD)

        # Measurement
        torch.cuda.synchronize()  # Ensure all previous GPU work is done
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        dist.barrier()

        '''
        Torch Programming Paradigm:
            1. Sync is inevitable because end_event cannot be launched on the same stream as nccl.

            2. handle.wait() will yield the main thread, while cudaDeviceSynchronize will spin,
            just makes the device wait for the completion of previous operations of all streams.

            3. async_all_gather + handle.wait() == sync_all_gather. In this case, 
            handle.wait() inserts an event on NCCL stream and calls cudaStreamWaitEvent()
            to make the default stream to wait for NCCL stream.

            4. When handle.wait() or sync_all_gather returns, it doesn't mean the communication is
            finished. It still needs to call torch.cuda.synchronize()/torch.current_stream().synchronize() 
            to ensure the communication is finished. current_stream.synchronize() is also valid
            because handle.wait() makes the default stream to wait for NCCL stream. `event.synchronize()`
            is also valid.

            5. Related Discussions: https://github.com/pytorch/pytorch/issues/68112
        '''

        # Async All Gather
        start_event.record()
        for _ in range(benchmark_iters):
            handle = dist.all_gather(output_list, tensor, group=dist.group.WORLD, async_op=True)
        torch.cuda.synchronize()    # Replace handle.wait() with torch.cuda.synchronize()
        end_event.record()
        end_event.synchronize()  # Wait for the benchmarked work to finish

        # Sync All Gather
        # start_event.record()
        # for _ in range(benchmark_iters):
        #     dist.all_gather(output_list, tensor, group=dist.group.WORLD)
        # end_event.record()

        # torch.cuda.synchronize()
        # end_event.synchronize()
        nccl_time_ms = start_event.elapsed_time(end_event) / benchmark_iters

        # --- NVSHMEM Benchmark ---
        # Warm-up
        for _ in range(warmup_iters):
            nvshmem_buffer.all_gather(output_list, tensor, async_op=False)

        # Measurement(Async)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(benchmark_iters):
            nvshmem_buffer.all_gather(output_list, tensor, async_op=True)
        torch.cuda.synchronize()
        end_event.record()

        torch.cuda.synchronize()
        nvshmem_time_ms = start_event.elapsed_time(end_event) / benchmark_iters

        # --- Calculate Bandwidth and Report Results ---
        # Bandwidth formula: (world_size - 1) * tensor_size / time
        # Each rank sends its data to (world_size - 1) other ranks.
        bus_bandwidth_bytes = (world_size - 1) * tensor_bytes

        nccl_bw_gbps = bus_bandwidth_bytes / (nccl_time_ms / 1000) / 1e9
        nvshmem_bw_gbps = bus_bandwidth_bytes / (nvshmem_time_ms / 1000) / 1e9

        if rank == 0:
            # Print table row
            row = (
                f"{size_kb:>12d} | {nccl_time_ms:>18.4f} | {nvshmem_time_ms:>18.4f} | "
                f"{nccl_bw_gbps:>18.2f} | {nvshmem_bw_gbps:>18.2f}"
            )
            print(row)

        # Barrier to ensure all ranks finish before starting the next size
        dist.barrier()


if __name__ == "__main__":
    rank, world_size, group = init_dist()
    print(f"Global rank = {rank}, world_size = {world_size}")

    buffer_size = 4 * 1024 * 1024 * 1024

    if rank == 0:
        print(f"Buffer size = {buffer_size/1024/1024} MB")

    if world_size <= 8:
        buffer = NvshmemBuffer(group, rank, world_size, buffer_size, 0, False)
    else:
        buffer = NvshmemBuffer(
            group, rank, world_size, buffer_size, 2 * buffer_size, False
        )

    # Run the benchmark
    benchmark_all_gather(buffer, rank, world_size)

    dist.destroy_process_group()
    if rank == 0:
        print("Benchmark finished successfully.")
