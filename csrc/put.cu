#include "utils.hpp"

#include <torch/extension.h>

__global__ void set_and_shift_kernel(float* send_data, float* recv_data,
                                     int num_elems, int mype, int npes) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  /* set the corresponding element of send_data */
  if (thread_idx < num_elems) send_data[thread_idx] = mype;

  int peer = (mype + 1) % npes;
  /* Every thread in block 0 calls nvshmemx_float_put_block. Alternatively,
  every thread can call shmem_float_p, but shmem_float_p has a disadvantage
  that when the destination GPU is connected via IB, there will be one rma
  message for every single element which can be detrimental to performance.
  And the disadvantage with shmem_float_put is that when the destination GPU is
  p2p connected, it cannot leverage multiple threads to copy the data to the
  destination GPU. */
  int block_offset = blockIdx.x * blockDim.x;
  nvshmemx_float_put_block(recv_data + block_offset, send_data + block_offset,
                           min(blockDim.x, num_elems - block_offset),
                           peer); /* All threads in a block call the API
                           with the same arguments */
}

float launch_ring_put_block(torch::Tensor send_tensor,
                            torch::Tensor recv_tensor, int num_blocks,
                            int threads_per_block) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  float* send_data = send_tensor.data_ptr<float>();
  float* recv_data = recv_tensor.data_ptr<float>();

  int num_elems = send_tensor.numel();

  dim3 block_dim(threads_per_block, 1, 1);
  dim3 grid_dim(num_blocks, 1, 1);

  set_and_shift_kernel<<<grid_dim, block_dim, 0, stream>>>(
      send_data, recv_data, num_elems, mype, npes);
  nvshmemx_barrier_all_on_stream(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  return 0;
}