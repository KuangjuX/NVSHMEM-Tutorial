#pragma once

#include "utils.hpp"

namespace nvshmem_tutorial::internode {

template <typename Element>
__global__ void internode_all_to_all(
    Element* input_data,      // Local input data
    Element* output_data,     // Local output data
    int* input_split_sizes,   // Size of data to send to each peer
    int* output_split_sizes,  // Size of data to receive from each peer
    int* input_offsets,       // Offset in input for each peer's data
    int* output_offsets,      // Offset in output for each peer's data
    int num_peers,            // Number of peers
    int my_rank               // My rank
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Send data to all peers (including self)
  for (int peer = 0; peer < num_peers; ++peer) {
    int send_size = input_split_sizes[peer];
    if (send_size > 0) {
      Element* src = input_data + input_offsets[peer];

      if (peer == my_rank) {
        // Local copy for self
        Element* dst = output_data + output_offsets[my_rank];
        for (int i = tid; i < send_size; i += blockDim.x * gridDim.x) {
          dst[i] = src[i];
        }
      } else {
        // Use NVSHMEM put to send data to remote peer
        for (int i = tid; i < send_size; i += blockDim.x * gridDim.x) {
          nvshmem_put(&output_data[output_offsets[my_rank] + i], &src[i], 1,
                      peer);
        }
      }
    }
  }

  // Ensure all puts are completed
  if (tid == 0) {
    nvshmem_quiet();
  }
  __syncthreads();

  // Barrier to ensure all peers have finished sending
  if (tid == 0) {
    nvshmem_barrier_all();
  }
  __syncthreads();
}
}  // namespace nvshmem_tutorial::internode
