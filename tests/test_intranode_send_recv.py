#!/usr/bin/env python3
"""
Test script for intranode send/recv functionality.
"""

import os
import torch
import torch.distributed as dist
from nvshmem_tutorial import NvshmemBuffer


def init_distributed():
    """Initialize distributed environment."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, local_rank, world_size


def test_basic_send_recv(rank, world_size):
    """Test basic send/recv functionality."""
    if world_size < 2:
        print("Need at least 2 processes for send/recv test")
        return False

    print(f"[Rank {rank}] Starting basic send/recv test")

    # Initialize buffer
    buffer_size = 1024 * 1024  # 1MB
    buffer = NvshmemBuffer(dist.group.WORLD, rank, world_size, buffer_size, 0)

    test_size = 1024  # 1KB test data

    if rank == 0:
        # Sender
        send_tensor = torch.arange(test_size, dtype=torch.uint8, device="cuda")
        print(f"[Rank {rank}] Sending data: {send_tensor[:10]}...")

        buffer.send(send_tensor, 1)
        print(f"[Rank {rank}] Send completed")

    elif rank == 1:
        # Receiver
        recv_tensor = torch.zeros(test_size, dtype=torch.uint8, device="cuda")
        print(f"[Rank {rank}] Waiting to receive data...")

        buffer.recv(recv_tensor, 0)
        print(f"[Rank {rank}] Received data: {recv_tensor[:10]}...")

        # Verify data
        expected = torch.arange(test_size, dtype=torch.uint8, device="cuda")
        if torch.equal(recv_tensor, expected):
            print(f"[Rank {rank}] ✓ Data verification passed!")
            return True
        else:
            print(f"[Rank {rank}] ✗ Data verification failed!")
            return False

    dist.barrier()
    return True


def test_bidirectional_send_recv(rank, world_size):
    """Test bidirectional send/recv."""
    if world_size < 2:
        print("Need at least 2 processes for bidirectional test")
        return False

    print(f"[Rank {rank}] Starting bidirectional send/recv test")

    buffer_size = 1024 * 1024  # 1MB
    buffer = NvshmemBuffer(dist.group.WORLD, rank, world_size, buffer_size, 0)

    test_size = 512

    if rank == 0:
        # Send to rank 1, then receive from rank 1
        send_tensor = torch.full((test_size,), rank, dtype=torch.uint8, device="cuda")
        recv_tensor = torch.zeros(test_size, dtype=torch.uint8, device="cuda")

        print(f"[Rank {rank}] Sending data to rank 1")
        buffer.send(send_tensor, 1)

        print(f"[Rank {rank}] Receiving data from rank 1")
        buffer.recv(recv_tensor, 1)

        expected = torch.full((test_size,), 1, dtype=torch.uint8, device="cuda")
        if torch.equal(recv_tensor, expected):
            print(f"[Rank {rank}] ✓ Bidirectional test passed!")
            return True
        else:
            print(f"[Rank {rank}] ✗ Bidirectional test failed!")
            return False

    elif rank == 1:
        # Receive from rank 0, then send to rank 0
        send_tensor = torch.full((test_size,), rank, dtype=torch.uint8, device="cuda")
        recv_tensor = torch.zeros(test_size, dtype=torch.uint8, device="cuda")

        print(f"[Rank {rank}] Receiving data from rank 0")
        buffer.recv(recv_tensor, 0)

        print(f"[Rank {rank}] Sending data to rank 0")
        buffer.send(send_tensor, 0)

        expected = torch.full((test_size,), 0, dtype=torch.uint8, device="cuda")
        if torch.equal(recv_tensor, expected):
            print(f"[Rank {rank}] ✓ Bidirectional test passed!")
            return True
        else:
            print(f"[Rank {rank}] ✗ Bidirectional test failed!")
            return False

    dist.barrier()
    return True


def test_multiple_sizes(rank, world_size):
    """Test send/recv with multiple data sizes."""
    if world_size < 2:
        print("Need at least 2 processes for multiple sizes test")
        return False

    print(f"[Rank {rank}] Starting multiple sizes test")

    buffer_size = 1024 * 1024  # 1MB
    buffer = NvshmemBuffer(dist.group.WORLD, rank, world_size, buffer_size, 0)

    test_sizes = [64, 256, 1024, 4096, 16384]  # Various sizes

    success_count = 0

    for size in test_sizes:
        if rank == 0:
            # Sender
            send_tensor = torch.randint(
                0, 256, (size,), dtype=torch.uint8, device="cuda"
            )
            buffer.send(send_tensor, 1)
            print(f"[Rank {rank}] Sent {size} bytes")

        elif rank == 1:
            # Receiver
            recv_tensor = torch.zeros(size, dtype=torch.uint8, device="cuda")
            buffer.recv(recv_tensor, 0)
            print(f"[Rank {rank}] Received {size} bytes")
            success_count += 1

        dist.barrier()

    if rank == 1:
        print(
            f"[Rank {rank}] ✓ Multiple sizes test: {success_count}/{len(test_sizes)} passed"
        )
        return success_count == len(test_sizes)

    return True


def benchmark_send_recv_latency(rank, world_size):
    """Benchmark send/recv latency."""
    if world_size < 2:
        print("Need at least 2 processes for latency benchmark")
        return

    print(f"[Rank {rank}] Starting latency benchmark")

    buffer_size = 1024 * 1024  # 1MB
    buffer = NvshmemBuffer(dist.group.WORLD, rank, world_size, buffer_size, 0)

    test_sizes = [64, 256, 1024, 4096]
    num_trials = 1000

    for size in test_sizes:
        if rank == 0:
            tensor = torch.ones(size, dtype=torch.uint8, device="cuda")

            # Warmup
            for _ in range(10):
                buffer.send(tensor, 1)

            dist.barrier()

            # Benchmark
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(num_trials):
                buffer.send(tensor, 1)
            end_event.record()

            torch.cuda.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            avg_latency_us = (total_time_ms * 1000) / num_trials

            print(
                f"[Rank {rank}] Size: {size:4d} bytes, Latency: {avg_latency_us:.2f} μs"
            )

        elif rank == 1:
            tensor = torch.zeros(size, dtype=torch.uint8, device="cuda")

            # Warmup
            for _ in range(10):
                buffer.recv(tensor, 0)

            dist.barrier()

            # Benchmark
            for _ in range(num_trials):
                buffer.recv(tensor, 0)

        dist.barrier()


def main():
    """Main test function."""
    rank, local_rank, world_size = init_distributed()

    if rank == 0:
        print("Starting intranode send/recv tests")
        print("=" * 50)

    tests = [
        ("Basic Send/Recv", test_basic_send_recv),
        ("Bidirectional Send/Recv", test_bidirectional_send_recv),
        ("Multiple Sizes", test_multiple_sizes),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        if rank == 0:
            print(f"\n--- {test_name} ---")

        try:
            result = test_func(rank, world_size)
            if result:
                passed += 1
                if rank == 0:
                    print(f"✓ {test_name} passed")
            else:
                if rank == 0:
                    print(f"✗ {test_name} failed")
        except Exception as e:
            if rank == 0:
                print(f"✗ {test_name} crashed: {e}")

        dist.barrier()

    # Run benchmark
    if rank == 0:
        print(f"\n--- Latency Benchmark ---")
    benchmark_send_recv_latency(rank, world_size)

    if rank == 0:
        print("\n" + "=" * 50)
        print(f"Test Results: {passed}/{total} tests passed")
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
