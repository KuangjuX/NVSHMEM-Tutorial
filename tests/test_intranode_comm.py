import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent)

import torch

from nvshmem_tutorial import NvshmemBuffer
import torch.distributed as dist

if __name__ == "__main__":
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    print(f"[Rank {rank}] Setting device to cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # Initialize dist 
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Initialize NVSHMEM buffer
    buffer = NvshmemBuffer(dist.group.WORLD, rank, world_size, 1024*1024*128, 0)
