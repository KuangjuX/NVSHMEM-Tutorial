import os
import time
import torch
import torch.distributed as dist
import inspect
from nvshmem_tutorial import (
    get_unique_id,
    init_with_unique_id,
    nvshmem_alloc_tensor,
    nvshmem_free_tensor,
    nvshmem_barrier,
    nvshmem_put_tensor,
    NvshmemBuffer,
)


# def init_nvshmem():
#     """
#     Initializes torch.distributed and then uses it to bootstrap NVSHMEM.
#     """
#     # os.environ["NVSHMEM_REMOTE_TRANSPORT"] = "none"

#     rank = int(os.environ["RANK"])
#     local_rank = int(os.environ["LOCAL_RANK"])
#     world_size = int(os.environ["WORLD_SIZE"])

#     print(f"[Rank {rank}] Setting device to cuda:{local_rank}")
#     torch.cuda.set_device(local_rank)

#     dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

#     if rank == 0:
#         unique_id = get_unique_id()
#     else:
#         unique_id = None

#     unique_ids = [None] * world_size
#     dist.all_gather_object(unique_ids, unique_id, group=dist.group.WORLD)

#     init_with_unique_id(unique_ids[0], rank, world_size, False)

#     print(f"[Rank {rank}] NVSHMEM initialized successfully.")

#     dist.barrier()

#     return rank, world_size


def init_dist():
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    print(f"ip = {ip}, port = {port}, world_size = {world_size}, rank = {rank}")

    # sig 用于获取 dist.init_process_group 函数的签名信息
    # 通过检查函数签名中是否包含 "device_id" 参数，来判断当前 PyTorch 版本是否支持该参数
    # 这样做是为了保证代码在不同版本的 PyTorch 中都能正常运行
    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": world_size,
        "rank": rank,
    }
    if "device_id" in sig.parameters:
        # noinspection PyTypeChecker
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)

    print(f"Rank {local_rank} initialized successfully.")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(world_size))),
    )


def init_nvshmem():
    """
    Initializes torch.distributed and then uses it to bootstrap NVSHMEM for multi-node execution.
    """
    os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"

    # These are set by the torchrun launcher
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank = rank // int(os.environ["LOCAL_WORLD_SIZE"])

    print(f"[Node {node_rank}, Rank {rank}] Setting device to cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # torch.distributed is used for bootstrapping NVSHMEM's connection info
    # dist.init_process_group(backend="gloo")
    rank, world_size, group = init_dist()

    if rank == 0:
        unique_id = get_unique_id()
    else:
        unique_id = None

    nvshmem_unique_ids = [None] * world_size
    dist.all_gather_object(nvshmem_unique_ids, unique_id, group=group)

    print(f"[Node {node_rank}, Rank {rank}] all gather object.")

    # All ranks now have the same unique_id from rank 0
    init_with_unique_id(nvshmem_unique_ids[0], rank, world_size, False)

    print(f"[Node {node_rank}, Rank {rank}] NVSHMEM initialized successfully.")
    dist.barrier()
    return rank, world_size, local_rank, node_rank


def benchmark_nvshmem_put_throughput(
    rank, world_size, size_bytes, num_warmup=10, num_trials=100
):
    """
    Benchmark NVSHMEM put_tensor throughput.
    The original name was misleading; this measures throughput, not async behavior itself.
    """
    if world_size < 2:
        if rank == 0:
            print(
                "This benchmark requires at least 2 processes. Skipping NVSHMEM test."
            )
        return None

    # Allocate symmetric buffer
    buffer = nvshmem_alloc_tensor(size_bytes, 128)
    # Create local tensor for data
    # local_tensor = torch.ones(size_bytes, dtype=torch.uint8, device="cuda")
    local_buffer = nvshmem_alloc_tensor(size_bytes, 128)
    local_buffer.fill_(1)

    local_rank = 0
    remote_rank = 1

    # Use CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    nvshmem_barrier()

    if rank == local_rank:
        # Warmup
        for _ in range(num_warmup):
            nvshmem_put_tensor(buffer, local_buffer, size_bytes, remote_rank)
        # Ensure warmup is complete before starting benchmark
        torch.cuda.synchronize()

        # Actual benchmark
        nvshmem_barrier()  # Synchronize processes before starting the timer
        start_event.record()

        for _ in range(num_trials):
            nvshmem_put_tensor(buffer, local_buffer, size_bytes, remote_rank)

        end_event.record()
        # Wait for all operations to complete
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        total_time_s = total_time_ms / 1000.0
        avg_time_ms = total_time_ms / num_trials
        # Bandwidth calculation: total data / total time
        bandwidth_gbps = (size_bytes * num_trials) / (total_time_s * 1e9)

        result = {
            "method": "NVSHMEM_PUT_THROUGHPUT",
            "size_bytes": size_bytes,
            "avg_time_ms": avg_time_ms,
            "bandwidth_gbps": bandwidth_gbps,
            "num_trials": num_trials,
        }
    else:
        # Remote rank just needs to exist and have the symmetric buffer available.
        # It also participates in barriers.
        nvshmem_barrier()  # Match the barrier on the sending rank
        result = None

    nvshmem_free_tensor(buffer)
    nvshmem_free_tensor(local_buffer)
    nvshmem_barrier()

    return result


def benchmark_nvshmem_buffer_send_throughput(
    rank, world_size, size_bytes, num_warmup=10, num_trials=100
):
    """
    Benchmark NVSHMEM buffer send throughput.
    """
    if world_size < 2:
        if rank == 0:
            print(
                "This benchmark requires at least 2 processes. Skipping NVSHMEM buffer send test."
            )
        return None

    buffer = NvshmemBuffer(dist.group.WORLD, rank, world_size, 1024 * 1024 * 1024, 0)

    sender_rank = 0
    receiver_rank = 1

    result = None

    # Warmup
    if rank == sender_rank:

        tensor = torch.ones(size_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(num_warmup):
            buffer.send(tensor, receiver_rank)

        # Wait for warmup sends to complete on the sender side
        torch.cuda.synchronize()

        # Synchronize with receiver before starting the benchmark
        dist.barrier()

        # Actual benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_trials):
            buffer.send(tensor, receiver_rank)
        end_event.record()
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        total_time_s = total_time_ms / 1000.0
        avg_time_ms = total_time_ms / num_trials
        bandwidth_gbps = (size_bytes * num_trials) / (total_time_s * 1e9)

        result = {
            "method": "NVSHMEM_BUFFER_SEND_THROUGHPUT",
            "size_bytes": size_bytes,
            "avg_time_ms": avg_time_ms,
            "bandwidth_gbps": bandwidth_gbps,
            "num_trials": num_trials,
        }

    if rank == receiver_rank:
        recv_tensor = torch.zeros(size_bytes, dtype=torch.uint8, device="cuda")

        for _ in range(num_warmup):
            buffer.recv(recv_tensor, sender_rank)
        torch.cuda.synchronize()

        dist.barrier()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_trials):
            buffer.recv(recv_tensor, sender_rank)
        end_event.record()
        torch.cuda.synchronize()

    dist.barrier()
    return result


def benchmark_nccl_p2p_throughput(
    rank, world_size, size_bytes, num_warmup=10, num_trials=100
):
    """
    Benchmark NCCL point-to-point throughput using send/recv.
    """
    if world_size < 2:
        if rank == 0:
            print("This benchmark requires at least 2 processes. Skipping NCCL test.")
        return None

    sender_rank = 0
    receiver_rank = 1
    result = None

    # Use CUDA events for accurate timing on the sender
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # All ranks participate in the barrier to ensure setup is complete
    dist.barrier()

    if rank == sender_rank:
        tensor = torch.ones(size_bytes, dtype=torch.uint8, device="cuda")

        # Warmup
        for _ in range(num_warmup):
            dist.send(tensor, dst=receiver_rank)
        # Wait for warmup sends to complete on the sender side
        torch.cuda.synchronize()

        # Synchronize with receiver before starting the benchmark
        dist.barrier()

        # Actual benchmark
        start_event.record()
        for _ in range(num_trials):
            dist.send(tensor, dst=receiver_rank)
        end_event.record()

        # IMPORTANT: Wait for the final send operation to complete before stopping the timer
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        total_time_s = total_time_ms / 1000.0
        avg_time_ms = total_time_ms / num_trials
        bandwidth_gbps = (size_bytes * num_trials) / (total_time_s * 1e9)

        result = {
            "method": "NCCL_P2P_THROUGHPUT",
            "size_bytes": size_bytes,
            "avg_time_ms": avg_time_ms,
            "bandwidth_gbps": bandwidth_gbps,
            "num_trials": num_trials,
        }

    elif rank == receiver_rank:
        recv_tensor = torch.zeros(size_bytes, dtype=torch.uint8, device="cuda")

        # Warmup
        for _ in range(num_warmup):
            dist.recv(recv_tensor, src=sender_rank)
        # CRITICAL FIX: Wait for warmup receives to complete
        torch.cuda.synchronize()

        # Synchronize with sender before they start the benchmark
        dist.barrier()

        # Actual benchmark: receive the data
        for _ in range(num_trials):
            dist.recv(recv_tensor, src=sender_rank)

        # CRITICAL FIX: Ensure all receive operations are finished.
        # Without this, the benchmark is invalid as the receiver might exit
        # before data transfer is complete.
        torch.cuda.synchronize()

    # Final barrier to ensure both ranks have finished their work before exiting
    dist.barrier()
    return result


def run_bandwidth_comparison(rank, world_size):
    """
    Run bandwidth comparison between NVSHMEM and NCCL for different data sizes.
    """
    # Test different data sizes (in bytes)
    test_sizes = [
        1024,  # 1 KB
        1024 * 16,  # 16 KB
        1024 * 64,  # 64 KB
        1024 * 256,  # 256 KB
        1024 * 1024,  # 1 MB
        1024 * 1024 * 4,  # 4 MB
        1024 * 1024 * 16,  # 16 MB
        1024 * 1024 * 64,  # 64 MB
        1024 * 1024 * 256,  # 256 MB
        1024 * 1024 * 1024,  # 1 GB
    ]

    results = []

    for size_bytes in test_sizes:
        if rank == 0:
            print(f"\n=== Testing size: {size_bytes / (1024*1024):.2f} MB ===")

        # Test NVSHMEM
        nvshmem_result = benchmark_nvshmem_put_throughput(rank, world_size, size_bytes)
        if nvshmem_result and rank == 0:
            results.append(nvshmem_result)
            print(
                f"NVSHMEM Put: {nvshmem_result['bandwidth_gbps']:.2f} GB/s, "
                f"avg_time: {nvshmem_result['avg_time_ms']:.3f} ms"
            )

        # Test NVSHMEM buffer
        # nvshmem_buffer_result = benchmark_nvshmem_buffer_send_throughput(
        #     rank, world_size, size_bytes
        # )
        # if nvshmem_buffer_result and rank == 0:
        #     results.append(nvshmem_buffer_result)
        #     print(
        #         f"CUDA IPC Buffer Send: {nvshmem_buffer_result['bandwidth_gbps']:.2f} GB/s, "
        #         f"avg_time: {nvshmem_buffer_result['avg_time_ms']:.3f} ms"
        #     )

        # Test NCCL
        nccl_result = benchmark_nccl_p2p_throughput(rank, world_size, size_bytes)
        if nccl_result and rank == 0:
            results.append(nccl_result)
            print(
                f"NCCL P2P:         {nccl_result['bandwidth_gbps']:.2f} GB/s, "
                f"avg_time: {nccl_result['avg_time_ms']:.3f} ms"
            )

            # Calculate speedup
            if nvshmem_result:
                speedup = (
                    nvshmem_result["bandwidth_gbps"] / nccl_result["bandwidth_gbps"]
                )
                print(f"NVSHMEM Speedup:  {speedup:.2f}x")

    return results


def main():
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)

    # rank, world_size = init_nvshmem()

    rank, world_size, _, _ = init_nvshmem()

    if rank == 0:
        print("Starting Intra-node P2P Bandwidth Comparison: NVSHMEM vs NCCL")
        print(f"World size: {world_size}")
        print("=" * 80)

    results = run_bandwidth_comparison(rank, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
