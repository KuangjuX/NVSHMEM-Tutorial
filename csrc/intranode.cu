#include "buffer.cuh"
#include "sync.cuh"
#include "utils.hpp"

#include <torch/torch.h>

namespace nvshmem_tutorial {

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