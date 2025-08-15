import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent)

import torch

from nvshmem_tutorial import NvshmemBuffer
import torch.distributed as dist


def test_intranode_all_to_all(buffer: NvshmemBuffer):
    input_tensor = torch.randn(1024 * 1024 * 128, dtype=torch.float32, device="cuda")
    output_tensor = torch.zeros(1024 * 1024 * 128, dtype=torch.float32, device="cuda")
    input_split_sizes = torch.randint(
        1, 1024 * 1024 * 128, (buffer.group_size,), dtype=torch.int32, device="cuda"
    )
    output_split_sizes = torch.randint(
        1, 1024 * 1024 * 128, (buffer.group_size,), dtype=torch.int32, device="cuda"
    )
    buffer.intranode_all_to_all(
        input_tensor, output_tensor, input_split_sizes, output_split_sizes
    )
    torch.testing.assert_close(input_tensor, output_tensor)


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"[Rank {rank}] Setting device to cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # Initialize dist
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=local_rank
    )

    # Initialize NVSHMEM buffer
    buffer = NvshmemBuffer(dist.group.WORLD, rank, world_size, 1024 * 1024 * 128, 0)

    if rank == 0:
        print(f"Number of SMs on device: {buffer.get_num_device_sms()}")

    dist.destroy_process_group()
