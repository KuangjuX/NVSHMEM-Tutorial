import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent)

import torch

from nvshmem_tutorial import NvshmemBuffer
import torch.distributed as dist


def test_all_gather(nvshmem_buffer: NvshmemBuffer):
    """
    Test all-gather communication.
    """
    tensor = torch.randn(1024, dtype=torch.float32, device="cuda")
    tensor_list = [torch.zeros_like(tensor) for _ in range(nvshmem_buffer.group_size)]
    nvshmem_buffer.intranode_all_gather(tensor_list, tensor, async_op=True)

    ref_tensor_list = [torch.zeros_like(tensor) for _ in range(buffer.group_size)]
    dist.all_gather(ref_tensor_list, tensor, group=dist.group.WORLD)

    for i in range(buffer.group_size):
        print(f"tensor_list[{i}] = {tensor_list[i]}")
        print(f"ref_tensor_list[{i}] = {ref_tensor_list[i]}")
        torch.testing.assert_close(tensor_list[i], ref_tensor_list[i])


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

    test_all_gather(buffer)

    dist.destroy_process_group()
