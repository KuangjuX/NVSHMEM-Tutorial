#pragma once
#include "utils.hpp"

namespace nvshmem_tutorial::intranode {

template <typename Element>
__global__ void intranode_all_to_all(
    Element** peer_buffer_ptrs,  // Array of pointers to each peer's buffer
    Element* input_data,         // Local input data
    Element* output_data,        // Local output data
    int* input_split_sizes,      // Size of data to send to each peer
    int* output_split_sizes,     // Size of data to receive from each peer
    int* input_offsets,          // Offset in input for each peer's data
    int* output_offsets,         // Offset in output for each peer's data
    int num_peers,               // Number of peers in the node
    int local_rank               // Local rank within the node
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Copy data to peer buffers (scatter phase)
  for (int peer = 0; peer < num_peers; ++peer) {
    if (peer == local_rank) continue;  // Skip self

    int send_size = input_split_sizes[peer];
    if (send_size > 0) {
      Element* peer_buffer = peer_buffer_ptrs[peer];
      Element* src = input_data + input_offsets[peer];

      // Each thread copies a portion of the data
      for (int i = tid; i < send_size; i += blockDim.x * gridDim.x) {
        peer_buffer[local_rank * send_size + i] = src[i];
      }
    }
  }

  __syncthreads();

  // Copy data from peer buffers to local output (gather phase)
  for (int peer = 0; peer < num_peers; ++peer) {
    if (peer == local_rank) {
      // Copy local data directly
      int local_size = input_split_sizes[local_rank];
      Element* src = input_data + input_offsets[local_rank];
      Element* dst = output_data + output_offsets[local_rank];

      for (int i = tid; i < local_size; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
      }
    } else {
      // Copy from peer buffer
      int recv_size = output_split_sizes[peer];
      if (recv_size > 0) {
        Element* peer_buffer = peer_buffer_ptrs[peer];
        Element* dst = output_data + output_offsets[peer];

        for (int i = tid; i < recv_size; i += blockDim.x * gridDim.x) {
          dst[i] = peer_buffer[peer * recv_size + i];
        }
      }
    }
  }
}
}  // namespace nvshmem_tutorial::intranode