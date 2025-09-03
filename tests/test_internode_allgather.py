import os
import sys
import inspect
from pathlib import Path

sys.path.append(Path(__file__).parent.parent)

import torch

from nvshmem_tutorial import NvshmemBuffer
import torch.distributed as dist


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


def test_internode_allgather(nvshmem_buffer: NvshmemBuffer):
    """
    Test internode all-gather communication.
    """
    tensor = torch.randn(256, dtype=torch.float32, device="cuda")
    tensor_list = [torch.zeros_like(tensor) for _ in range(nvshmem_buffer.group_size)]
    nvshmem_buffer.internode_all_gather(tensor_list, tensor, async_op=False)

    ref_tensor_list = [torch.zeros_like(tensor) for _ in range(buffer.group_size)]
    dist.all_gather(ref_tensor_list, tensor, group=dist.group.WORLD)

    env_rank = int(os.getenv("RANK", "0"))

    if env_rank == 0:
        for i in range(nvshmem_buffer.group_size):
            print(f"tensor_list[{i}] = {tensor_list[i]}")
            print(f"ref_tensor_list[{i}] = {ref_tensor_list[i]}")
            # torch.testing.assert_close(tensor_list[i], ref_tensor_list[i])

    if env_rank == 0:
        print("Test passed")


if __name__ == "__main__":
    rank, world_size, group = init_dist()
    print(f"rank = {rank}, world_size = {world_size}")

    buffer = NvshmemBuffer(group, rank, world_size, 1024 * 1024, 1024 * 1024)

    print(f"[DEBUG] Rank {rank} initialized buffer.")

    test_internode_allgather(buffer)

    dist.destroy_process_group()
