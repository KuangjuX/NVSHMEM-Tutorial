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
)


def get_rdma_rank(rank: int):
    return rank // 8


def get_nvl_rank(rank: int):
    return rank % 8


def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_nodes * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in sig.parameters:
        # noinspection PyTypeChecker
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(num_local_ranks * num_nodes))),
    )


def init_nvshmem():
    """
    Initializes torch.distributed and then uses it to bootstrap NVSHMEM for multi-node execution.
    """
    # CRITICAL CHANGE: Enable UCX for network transport
    # 'ucx' is the recommended transport for InfiniBand/RoCE and high-speed Ethernet.
    # os.environ["NVSHMEM_REMOTE_TRANSPORT"] = "ucx"
    os.environ["NVSHMEM_DISABLE_P2P"] = "1"
    os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"
    os.environ["NVSHMEM_MAX_TEAMS"] = "7"
    os.environ["NVSHMEM_DEBUG"] = "INFO"

    # These are set by the torchrun launcher
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank = rank // int(os.environ["LOCAL_WORLD_SIZE"])

    print(f"[Node {node_rank}, Rank {rank}] Setting device to cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # torch.distributed is used for bootstrapping NVSHMEM's connection info
    # dist.init_process_group(backend="gloo")
    rank, world_size, group = init_dist(local_rank, 8)

    # Rank 0 creates a unique ID and broadcasts it to all other ranks
    if get_rdma_rank(rank) == 0:
        unique_id = get_unique_id()
    else:
        unique_id = None

    nvshmem_unique_ids = [None] * world_size
    dist.all_gather_object(nvshmem_unique_ids, unique_id, group=group)

    print(f"[Node {node_rank}, Rank {rank}] all gather object.")

    # All ranks now have the same unique_id from rank 0
    init_with_unique_id(nvshmem_unique_ids[get_nvl_rank(rank)], rank, world_size)

    print(f"[Node {node_rank}, Rank {rank}] NVSHMEM initialized successfully.")
    dist.barrier()
    return rank, world_size, local_rank, node_rank


def run_inter_node_test(rank, world_size, node_rank, size_bytes=1024):
    """
    Performs a put/get operation between two ranks on different nodes.
    """
    if world_size < 2 or os.environ.get("NNODES", "1") == "1":
        if rank == 0:
            print("Skipping inter-node test: requires at least 2 nodes.")
        return

    # --- Test 1: PUT from Node 0 to Node 1 ---

    # Define producer (on node 0) and consumer (on node 1)
    # This ensures we are testing cross-node communication
    producer_pe = 0
    consumer_pe = world_size // 2  # e.g., rank 8 in a 16-GPU setup

    if rank == 0:
        print("\n--- Running Inter-Node PUT Test ---")
        print(f"Producer: Rank {producer_pe} (on Node 0)")
        print(f"Consumer: Rank {consumer_pe} (on Node 1)")

    # All PEs allocate a symmetric buffer
    buffer = nvshmem_alloc_tensor((size_bytes,), dtype=torch.uint8)
    nvshmem_barrier()

    # Producer (rank 0) creates data and PUTs it to the consumer's buffer
    if rank == producer_pe:
        local_tensor = torch.full(
            (size_bytes,), fill_value=42, dtype=torch.uint8, device="cuda"
        )
        print(f"[Rank {rank}] Putting tensor to Rank {consumer_pe}'s buffer...")
        nvshmem_put_tensor(buffer, local_tensor, consumer_pe)

    # Barrier ensures the PUT is complete before the consumer tries to read it
    nvshmem_barrier()

    # Consumer (rank 8) copies the data from its own symmetric buffer to a local tensor and verifies
    if rank == consumer_pe:
        remote_tensor = torch.zeros(size_bytes, dtype=torch.uint8, device="cuda")
        remote_tensor.copy_(buffer)  # Direct copy from symmetric memory

        print(f"[Rank {rank}] Copied from my symmetric buffer. Verifying...")
        expected_sum = 42 * size_bytes
        actual_sum = remote_tensor.sum().item()

        if actual_sum == expected_sum:
            print(f"SUCCESS (PUT): Data verified on Rank {rank}. Sum is correct.")
        else:
            print(
                f"FAILURE (PUT): Data mismatch on Rank {rank}. Expected sum {expected_sum}, got {actual_sum}."
            )

    nvshmem_free_tensor(buffer)
    nvshmem_barrier()


def main():
    rank, world_size, local_rank, node_rank = init_nvshmem()

    # Run the specific inter-node test
    run_inter_node_test(rank, world_size, node_rank)

    dist.barrier()
    if rank == 0:
        print("\nAll tests finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
