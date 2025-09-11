#include "buffer.cuh"
#include "launch.cuh"
#include "ptx_wrapper.cuh"
#include "sync.cuh"
#include "utils.hpp"

#include <torch/torch.h>

namespace nvshmem_tutorial {

namespace kernels {
template <typename DType, const int kChunkSize>
__global__ void tma_load_kernel(const DType* input, DType* output,
                                int total_bytes) {
  // Declare shared memory array for storing data chunks during transfer
  extern __shared__ DType smem[kChunkSize];

  // Shared memory barrier pointer for synchronizing TMA operations
  __shared__ uint64_t mbar_ptr;
  int tid = threadIdx.x;

  // Initialize memory barrier - only thread 0 performs initialization
  if (tid == 0) {
    mbarrier_init(&mbar_ptr, 1);
  }

  // Synchronize all threads in the block after barrier initialization
  __syncthreads();

  // Process data in chunks across multiple thread blocks
  // Each block handles chunks at stride intervals to distribute work
  for (offset = blockIdx.x * kChunkSize; offset < total_bytes;
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
void tma_load_host(const DType* input, DType* output, int total_bytes,
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
  SET_SHARED_MEMORY_FOR_TMA(tma_load_kernel<DType, kChunkSize>);

  // Launch the kernel
  LAUNCH_KERNEL(&cfg, tma_load_kernel<DType, kChunkSize>, input, output,
                total_bytes);
}

}  // namespace kernels

void Buffer::intranode_send(const torch::Tensor& tensor, int rank) {
  if (!tensor.is_cuda()) {
    throw std::runtime_error("intranode_send expects CUDA tensor");
  }

  if (buffer_ptrs_[nvl_rank_] == nullptr) {
    throw std::runtime_error("Local NVLink buffer not allocated");
  }

  CUDA_CHECK(cudaMemcpy(buffer_ptrs_[rank], tensor.data_ptr(), tensor.nbytes(),
                        cudaMemcpyDeviceToDevice));
}

void Buffer::intranode_recv(torch::Tensor& tensor, int rank) {
  if (!tensor.is_cuda()) {
    throw std::runtime_error("intranode_recv expects CUDA tensor");
  }

  if (buffer_ptrs_[nvl_rank_] == nullptr) {
    throw std::runtime_error("Local NVLink buffer not allocated");
  }

  CUDA_CHECK(cudaMemcpy(tensor.data_ptr(), buffer_ptrs_[nvl_rank_],
                        tensor.nbytes(), cudaMemcpyDeviceToDevice));
}

void Buffer::intranode_all_gather(std::vector<torch::Tensor>& tensor_list,
                                  const torch::Tensor& tensor, bool async_op) {
  if (!tensor.is_cuda()) {
    throw std::runtime_error("intranode_all_gather expects CUDA tensor");
  }

  if (static_cast<int>(tensor_list.size()) != num_nvl_ranks_) {
    throw std::runtime_error("tensor_list size must match num_nvl_ranks");
  }

  int64_t num_bytes = tensor.nbytes();

  if (buffer_ptrs_[nvl_rank_] == nullptr) {
    throw std::runtime_error("Local NVLink buffer not allocated");
  }

  // Copy tensor to local NVLink buffer
  CUDA_CHECK(cudaMemcpyAsync(buffer_ptrs_[nvl_rank_], tensor.data_ptr(),
                             num_bytes, cudaMemcpyDeviceToDevice,
                             comm_stream_));

  sync::barrier(barrier_signal_ptrs_gpu_, rank_, num_ranks_, comm_stream_);

  for (int pe = 0; pe < num_nvl_ranks_; ++pe) {
    if (buffer_ptrs_[pe] == nullptr) {
      throw std::runtime_error("buffer_ptrs_[pe] is nullptr");
    }

    CUDA_CHECK(cudaMemcpyAsync(tensor_list[pe].data_ptr(), buffer_ptrs_[pe],
                               num_bytes, cudaMemcpyDeviceToDevice,
                               comm_stream_));
  }

  if (!async_op) {
    cudaStreamSynchronize(comm_stream_);
  }
}

void Buffer::intranode_all_to_all(torch::Tensor input, torch::Tensor output,
                                  torch::Tensor input_split_sizes,
                                  torch::Tensor output_split_sizes,
                                  bool async_op) {
  if (!input.is_cuda() || !output.is_cuda()) {
    throw std::runtime_error("intranode_all_to_all expects CUDA tensors");
  }
  if (!input_split_sizes.is_cuda() || !output_split_sizes.is_cuda()) {
    throw std::runtime_error("split_sizes must be CUDA tensors");
  }
  if (input_split_sizes.numel() != num_local_pes_ ||
      output_split_sizes.numel() != num_local_pes_) {
    throw std::runtime_error("split_sizes length must match num_local_pes");
  }

  // Launch CUDA kernel for intranode all-to-all communication
  // intranode::launch_intranode_all_to_all(
  //     input.data_ptr(), output.data_ptr(),
  //     input_split_sizes.data_ptr<int64_t>(),
  //     output_split_sizes.data_ptr<int64_t>(), buffer_ptrs_gpu_, local_pe_,
  //     num_local_pes_, num_device_sms_, comm_stream_);

  if (!async_op) {
    cudaStreamSynchronize(comm_stream_);
  }
}
}  // namespace nvshmem_tutorial