import os
import time
import torch
import torch.distributed as dist
from nvshmem_tutorial import (
    get_unique_id,
    init_with_unique_id,
    nvshmem_alloc,
    nvshmem_free,
    nvshmem_barrier,
    nvshmem_get_mem,
    nvshmem_put_mem,
)

os.environ["NVSHMEM_REMOTE_TRANSPORT"] = "none"


def bootstrap_nvshmem():
    """
    Initializes torch.distributed and then uses it to bootstrap NVSHMEM.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"[Rank {rank}] Setting device to cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if rank == 0:
        unique_id = get_unique_id()
    else:
        unique_id = None

    unique_ids = [None] * world_size
    dist.all_gather_object(unique_ids, unique_id, group=dist.group.WORLD)

    init_with_unique_id(unique_ids[0], rank, world_size)

    print(f"[Rank {rank}] NVSHMEM initialized successfully.")

    dist.barrier()

    return rank, world_size


def put(rank, world_size, size_bytes=1024):
    if world_size < 2:
        if rank == 0:
            print("This benchmark requires at least 2 processes. Skipping.")
        return

    buffer = nvshmem_alloc(size_bytes, 128)

    local_rank = 0
    remote_rank = 1

    nvshmem_barrier()

    if rank == local_rank:
        local_tensor = torch.ones(size_bytes, dtype=torch.uint8, device="cuda")
        nvshmem_put_mem(buffer, local_tensor, size_bytes, remote_rank)
        nvshmem_barrier()

    if rank == remote_rank:
        remote_tensor = torch.zeros(size_bytes, dtype=torch.uint8, device="cuda")
        nvshmem_get_mem(remote_tensor, buffer, size_bytes, local_rank)
        print(remote_tensor)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank, world_size = bootstrap_nvshmem()

    put(rank, world_size)


if __name__ == "__main__":
    main()
