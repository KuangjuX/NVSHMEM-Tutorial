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

void Buffer::intranode_all_gather(torch::Tensor& output_tensor,
                                  const torch::Tensor& tensor, bool async_op) {
  if (!tensor.is_cuda()) {
    throw std::runtime_error("intranode_all_gather expects CUDA tensor");
  }

  if (static_cast<int>(output_tensor.nbytes() / tensor.nbytes()) != num_nvl_ranks_) {
    throw std::runtime_error("output_tensor size must match num_nvl_ranks");
  }

  if (buffer_ptrs_[nvl_rank_] == nullptr) {
    throw std::runtime_error("Local NVLink buffer not allocated");
  }

  int prev_rank = (nvl_rank_ - 1 + num_nvl_ranks_) % num_nvl_ranks_;
  int next_rank = (nvl_rank_ + 1) % num_nvl_ranks_;
  int64_t num_bytes = tensor.nbytes();
  int64_t output_size = output_tensor.nbytes();
  char *output_ptr = static_cast<char*>(output_tensor.data_ptr());
  auto curr_sig_flag_ptr = reinterpret_cast<CUdeviceptr>(static_cast<char*>(buffer_ptrs_[nvl_rank_]) + output_size);
  auto next_sig_flag_ptr = reinterpret_cast<CUdeviceptr>(static_cast<char*>(buffer_ptrs_[next_rank]) + output_size);
  auto curr_ack_flag_ptr = reinterpret_cast<CUdeviceptr>(static_cast<char*>(buffer_ptrs_[nvl_rank_]) + output_size + sizeof(int));
  auto prev_ack_flag_ptr = reinterpret_cast<CUdeviceptr>(static_cast<char*>(buffer_ptrs_[prev_rank]) + output_size + sizeof(int));
  void *init_dst_slot_ptr = static_cast<char*>(buffer_ptrs_[next_rank]) + nvl_rank_ * num_bytes;

  // Copy input into output slot.
  void *output_dst_slot_ptr = static_cast<char*>(output_ptr) + nvl_rank_ * num_bytes;
  CUDA_CHECK(cudaMemcpyAsync(output_dst_slot_ptr, tensor.data_ptr(),
                             num_bytes, cudaMemcpyDeviceToDevice,
                             comm_streams_[prev_rank]));

  // Send input to next rank.
  CUDA_CHECK(cudaMemcpyAsync(init_dst_slot_ptr, tensor.data_ptr(), 
                             num_bytes, cudaMemcpyDeviceToDevice, 
                             comm_streams_[nvl_rank_]));

  // Run the ring for world_size - 1 steps.
  for (int step = 1; step < num_nvl_ranks_; ++step) {
    // Write sig to next rank.
    cuStreamWriteValue32(comm_streams_[nvl_rank_], next_sig_flag_ptr, step - 1, 0);

    // Wait signal from prev rank.
    cuStreamWaitValue32(comm_streams_[nvl_rank_], curr_sig_flag_ptr, step - 1, 0);
    cuStreamWaitValue32(comm_streams_[prev_rank], curr_sig_flag_ptr, step - 1, 0);

    // Send the received data to next rank.
    uint64_t offset = ((nvl_rank_ - step + num_nvl_ranks_) % num_nvl_ranks_) * num_bytes;
    void *nvl_src_slot_ptr = static_cast<char*>(buffer_ptrs_[nvl_rank_]) + offset;
    void *nvl_dst_slot_ptr = static_cast<char*>(buffer_ptrs_[next_rank]) + offset;
    CUDA_CHECK(cudaMemcpyAsync(nvl_dst_slot_ptr, nvl_src_slot_ptr,
                               num_bytes, cudaMemcpyDeviceToDevice,
                               comm_streams_[nvl_rank_]));

    // Copy the received data to output tensor, overlapped with NVLink transfer.
    void *output_dst_slot_ptr = static_cast<char*>(output_ptr) + offset;
    CUDA_CHECK(cudaMemcpyAsync(output_dst_slot_ptr, nvl_src_slot_ptr,
                               num_bytes, cudaMemcpyDeviceToDevice,
                               comm_streams_[prev_rank]));
      
    // Write ack to prev rank.
    cuStreamWriteValue32(comm_streams_[nvl_rank_], prev_ack_flag_ptr, step, 0);

    // Wait ack from next rank.
    cuStreamWaitValue32(comm_streams_[nvl_rank_], curr_ack_flag_ptr, step, 0);
  }

  if (!async_op) {
    cudaStreamSynchronize(comm_streams_[prev_rank]);
    cudaStreamSynchronize(comm_streams_[nvl_rank_]);
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
  //     num_local_pes_, num_device_sms_, comm_streams_[nvl_rank_]);

  if (!async_op) {
    cudaStreamSynchronize(comm_streams_[nvl_rank_]);
  }
}
}  // namespace nvshmem_tutorial
