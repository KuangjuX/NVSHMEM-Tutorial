import torch
import pytest
import nvshmem_tutorial


def test_tma_copy_uint8():
    """Test TMA copy with uint8 tensors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create input tensor on GPU with uint8 dtype
    size = (1024, 1024)
    input_tensor = torch.randint(0, 256, size, dtype=torch.uint8, device="cuda")

    # Create output tensor on GPU
    output_tensor = torch.zeros_like(input_tensor)

    # Perform TMA copy
    nvshmem_tutorial.tma_copy(input_tensor, output_tensor)

    # Verify the copy was successful
    torch.testing.assert_close(input_tensor, output_tensor)


if __name__ == "__main__":
    test_tma_copy_uint8()
    print("Test passed!")
