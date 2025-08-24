import os
import sys
import inspect
from pathlib import Path

sys.path.append(Path(__file__).parent.parent)

import torch

from nvshmem_tutorial import NvshmemBuffer
import torch.distributed as dist


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


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = 8

    rank, world_size, group = init_dist(local_rank, num_local_ranks)

    buffer = NvshmemBuffer(group, rank, world_size, 1024 * 1024, 1024 * 1024)

    if rank == 0:
        print(f"buffer.group_size = {buffer.group_size}")
        print(f"buffer.num_nvl_bytes = {buffer.num_nvl_bytes}")
        print(f"buffer.num_rdma_bytes = {buffer.num_rdma_bytes}")
