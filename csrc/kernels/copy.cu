#include "kernels/copy.cuh"

#include <ATen/cuda/CUDAContext.h>

namespace nvshmem_tutorial::kernels {

template <typename DType, const int kChunkSize>
__global__ void tma_copy_kernel(const DType* input, DType* output,
                                int total_bytes) {
  // Declare shared memory array for storing data chunks during transfer
  // Ensure proper alignment for TMA operations
  __shared__ alignas(128) DType smem[kChunkSize];

  // Shared memory barrier pointer for synchronizing TMA operations
  __shared__ uint64_t mbar_ptr;
  int tid = threadIdx.x;

  // Initialize memory barrier - only thread 0 performs initialization
  if (tid == 0) {
    mbarrier_init(&mbar_ptr, 1);
  }

  // Synchronize all threads in the block after barrier initialization
  __syncthreads();

  // Initialize phase for barrier synchronization
  uint32_t phase = 0;

  // Process data in chunks across multiple thread blocks
  // Each block handles chunks at stride intervals to distribute work
  for (int offset = blockIdx.x * kChunkSize; offset < total_bytes;
       offset += gridDim.x * kChunkSize) {
    // Calculate input and output pointers for current chunk
    const DType* input_ptr = input + offset;
    DType* output_ptr = output + offset;

    // Calculate the number of bytes to copy for this chunk
    // TODO(Kuangjux): Determine the copy bytes more precisely for edge cases
    int copy_bytes = sizeof(DType) * kChunkSize;

    // Only thread 0 initiates TMA operations to avoid race conditions
    if (tid == 0) {
      // Launch a TMA (Tensor Memory Access) load operation from global memory
      // (input_ptr) to shared memory (smem).
      tma_load_1d(input_ptr, smem, &mbar_ptr, copy_bytes);
      // Arrive at the memory barrier and expect a transaction of copy_bytes to
      // synchronize the transfer.
      mbarrier_arrive_and_expect_tx(&mbar_ptr, copy_bytes);
    }

    // Wait for TMA load operation to complete using the barrier
    mbarrier_wait(&mbar_ptr, phase);

    // Store data from shared memory back to global memory
    if (tid == 0) {
      tma_store_1d(smem, output_ptr, copy_bytes);
    }

    // Wait for TMA store operation to complete
    tma_store_wait<0>();

    // Synchronize all threads before processing next chunk
    __syncthreads();
  }
}

template <typename DType, const int kChunkSize>
void tma_copy_host(const DType* input, DType* output, int total_bytes,
                   cudaStream_t stream) {
  // Calculate grid size based on total bytes and chunk size
  int num_chunks = (total_bytes + kChunkSize * sizeof(DType) - 1) /
                   (kChunkSize * sizeof(DType));
  int grid_size =
      std::min(num_chunks, 256);  // Limit grid size to avoid too many blocks

  // Calculate shared memory size
  size_t smem_size = kChunkSize * sizeof(DType);

  // Setup launch configuration
  SETUP_LAUNCH_CONFIG(grid_size, 256, stream);

  // Set shared memory for TMA if needed
  auto kernel = tma_copy_kernel<DType, kChunkSize>;
  SET_SHARED_MEMORY_FOR_TMA(kernel);

  // Launch the kernel
  LAUNCH_KERNEL(&cfg, kernel, input, output, total_bytes);
}

void tma_copy(const torch::Tensor& input, torch::Tensor& output) {
  // Validate input tensors
  HOST_ASSERT(input.is_cuda() && output.is_cuda());
  HOST_ASSERT(input.dtype() == output.dtype());
  HOST_ASSERT(input.numel() == output.numel());
  HOST_ASSERT(input.dtype() == torch::kUInt8);

  // Get tensor properties
  int total_bytes = input.numel() * input.element_size();

  // Use the current CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // Call TMA copy with uint8 data type
  constexpr int kChunkSize = 4096;  // 4KB chunks for uint8
  tma_copy_host<uint8_t, kChunkSize>(input.data_ptr<uint8_t>(),
                                     output.data_ptr<uint8_t>(), total_bytes,
                                     stream);
}

}  // namespace nvshmem_tutorial::kernels
