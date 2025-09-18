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

  // Double buffer
  uint64_t ping = 0, pong = 1, offset = 0, num_bytes = tensor.nbytes();
  uint32_t prev_rank = (nvl_rank_ - 1 + num_nvl_ranks_) % num_nvl_ranks_;
  uint32_t next_rank = (nvl_rank_ + 1) % num_nvl_ranks_;
  const uint64_t num_flags = 4;   // Use 4 flags: sig, ack, local_copy_start, local_copy_end
  char *output_ptr = static_cast<char*>(output_tensor.data_ptr());
  char *curr_base_ptr = static_cast<char*>(buffer_ptrs_[nvl_rank_]);
  char *next_base_ptr = static_cast<char*>(buffer_ptrs_[next_rank]);
  char *prev_base_ptr = static_cast<char*>(buffer_ptrs_[prev_rank]);
  char *curr_data_base_ptr = curr_base_ptr + num_flags * sizeof(int);
  char *next_data_base_ptr = next_base_ptr + num_flags * sizeof(int);
  auto curr_sig_flag_ptr = reinterpret_cast<CUdeviceptr>(curr_base_ptr);
  auto next_sig_flag_ptr = reinterpret_cast<CUdeviceptr>(next_base_ptr);
  auto curr_ack_flag_ptr = reinterpret_cast<CUdeviceptr>(curr_base_ptr + sizeof(int));
  auto prev_ack_flag_ptr = reinterpret_cast<CUdeviceptr>(prev_base_ptr + sizeof(int));
  auto local_copy_start_ptr = reinterpret_cast<CUdeviceptr>(curr_base_ptr + 2 * sizeof(int));
  auto local_copy_end_ptr = reinterpret_cast<CUdeviceptr>(curr_base_ptr + 3 * sizeof(int));
  void *output_dst_slot_ptr = output_ptr + nvl_rank_ * num_bytes;
  void *nvl_src_slot_ptr = NULL, *nvl_dst_slot_ptr = NULL;

  // Copy input to NVLink buffer.
  nvl_src_slot_ptr = curr_data_base_ptr + ping * num_bytes;
  CUDA_CHECK(cudaMemcpyAsync(nvl_src_slot_ptr, tensor.data_ptr(), 
                             num_bytes, cudaMemcpyDeviceToDevice, 
                             comm_streams_[nvl_rank_]));

  // Run the ring for (world_size - 1) steps.
  for (uint32_t step = 0; step < num_nvl_ranks_ - 1; ++step) {
    // Write ack to prev_rank to ack the last step data.
    cuStreamWriteValue32(comm_streams_[nvl_rank_], prev_ack_flag_ptr, tag + step - 1, 0);

    // Wait ack from next_rank to ack the last step data.
    cuStreamWaitValue32(comm_streams_[nvl_rank_], curr_ack_flag_ptr, tag + step - 1,
                        CU_STREAM_WAIT_VALUE_EQ);

    // comm_streams_[nvl_rank_] tells comm_streams_[prev_rank] to start local copy.
    cuStreamWriteValue32(comm_streams_[nvl_rank_], local_copy_start_ptr, tag + step, 0);
    cuStreamWaitValue32(comm_streams_[prev_rank], local_copy_start_ptr, tag + step,
                        CU_STREAM_WAIT_VALUE_EQ);

    // Send this step data to next_rank through NVLink.
    nvl_src_slot_ptr = curr_data_base_ptr + ping * num_bytes;
    nvl_dst_slot_ptr = next_data_base_ptr + pong * num_bytes;
    CUDA_CHECK(cudaMemcpyAsync(nvl_dst_slot_ptr, nvl_src_slot_ptr,
                               num_bytes, cudaMemcpyDeviceToDevice,
                               comm_streams_[nvl_rank_]));

    // Copy received data into output slot, overlapped with NVLink transfer.
    offset = ((nvl_rank_ - step + num_nvl_ranks_) % num_nvl_ranks_) * num_bytes;
    output_dst_slot_ptr = output_ptr + offset;
    CUDA_CHECK(cudaMemcpyAsync(output_dst_slot_ptr, nvl_src_slot_ptr,
                               num_bytes, cudaMemcpyDeviceToDevice,
                               comm_streams_[prev_rank]));

    // Write signal to next_rank to notify data has been sent.
    cuStreamWriteValue32(comm_streams_[nvl_rank_], next_sig_flag_ptr, tag + step, 0);

    // Wait for signal from prev_rank to notify data has been sent.
    cuStreamWaitValue32(comm_streams_[nvl_rank_], curr_sig_flag_ptr, tag + step,
                        CU_STREAM_WAIT_VALUE_EQ);

    // comm_streams_[prev_rank] tells comm_streams_[nvl_rank_] local copy ends.
    cuStreamWriteValue32(comm_streams_[prev_rank], local_copy_end_ptr, tag + step, 0);
    cuStreamWaitValue32(comm_streams_[nvl_rank_], local_copy_end_ptr, tag + step,
                        CU_STREAM_WAIT_VALUE_EQ);

    ping = 1 - ping;
    pong = 1 - pong;
  }

  // Copy last step received data from NVLink buffer into output tensor.
  nvl_src_slot_ptr = curr_data_base_ptr + ping * num_bytes;
  output_dst_slot_ptr = output_ptr + next_rank * num_bytes;
  CUDA_CHECK(cudaMemcpyAsync(output_dst_slot_ptr, nvl_src_slot_ptr,
                             num_bytes, cudaMemcpyDeviceToDevice,
                             comm_streams_[nvl_rank_]));
  
  // Avoid using the same sync value among different calls.
  // Use a prime number to enlarge the cycle as much as possible.
  tag += PRIME_TAG_STRIDE;

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
