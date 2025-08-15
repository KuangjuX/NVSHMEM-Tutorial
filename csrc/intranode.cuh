#pragma once
#include "utils.hpp"

#include <torch/torch.h>

namespace nvshmem_tutorial::intranode {

template <typename Element>
__global__ void ke_intranode_all_to_all(
    Element** peer_buffer_ptrs,  // Array of pointers to each peer's buffer -
                                 // 通过CUDA IPC映射的各个peer的内存指针
    Element* input_data,         // Local input data
    Element* output_data,        // Local output data
    int* input_split_sizes,      // Size of data to send to each peer
    int* output_split_sizes,     // Size of data to receive from each peer
    int* input_offsets,          // Offset in input for each peer's data
    int* output_offsets,         // Offset in output for each peer's data
    int num_peers,               // Number of peers in the node
    int local_rank               // Local rank within the node
) {
  // peer_buffer_ptrs的作用：
  // 1. 这是一个指针数组，每个元素指向一个peer的GPU内存缓冲区
  // 2. 通过CUDA IPC (Inter-Process
  // Communication)，不同进程可以直接访问彼此的GPU内存
  // 3. peer_buffer_ptrs[i] 就是第i个peer的GPU内存地址，可以直接读写
  // 4. 这样避免了通过CPU或网络进行数据传输，实现真正的GPU-to-GPU直接内存访问

  // 为什么必须要peer_buffer_ptrs：
  // - 在intranode通信中，所有GPU都在同一个节点上，通过NVLink或PCIe连接
  // - CUDA IPC允许不同进程的GPU直接访问彼此的内存，这比CPU中转要快得多
  // - 每个peer都有自己独立的内存空间，我们需要知道这些内存的具体地址才能访问
  // - peer_buffer_ptrs就是存储这些地址的数组，是实现零拷贝通信的关键

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 第一阶段：将数据写入目标peer的缓冲区
  for (int peer = 0; peer < num_peers; ++peer) {
    int send_size = input_split_sizes[peer];
    if (send_size > 0) {
      Element* src = input_data + input_offsets[peer];

      if (peer == local_rank) {
        // 本地拷贝：直接从input拷贝到output
        Element* dst = output_data + output_offsets[local_rank];
        for (int i = tid; i < send_size; i += blockDim.x * gridDim.x) {
          dst[i] = src[i];
        }
      } else {
        // 远程写入：通过IPC直接写入peer的GPU内存
        // peer_buffer_ptrs[peer] 指向目标peer的缓冲区起始地址
        Element* peer_buffer = peer_buffer_ptrs[peer];
        // 计算在peer缓冲区中的写入位置，使用local_rank作为偏移避免冲突
        Element* dst = peer_buffer + output_offsets[local_rank];

        for (int i = tid; i < send_size; i += blockDim.x * gridDim.x) {
          dst[i] = src[i];
        }
      }
    }
  }

  // 确保所有写入操作完成
  __syncthreads();

  // 第二阶段：从各个peer的缓冲区读取数据
  for (int peer = 0; peer < num_peers; ++peer) {
    if (peer != local_rank) {  // 本地数据已在第一阶段处理
      int recv_size = output_split_sizes[peer];
      if (recv_size > 0) {
        // 通过IPC直接从peer的GPU内存读取数据
        Element* peer_buffer = peer_buffer_ptrs[peer];
        Element* src = peer_buffer + output_offsets[peer];
        Element* dst = output_data + output_offsets[peer];

        for (int i = tid; i < recv_size; i += blockDim.x * gridDim.x) {
          dst[i] = src[i];
        }
      }
    }
  }
}

template <typename Element>
void launch_intranode_all_to_all(Element* input_data, Element* output_data,
                                 int64_t* input_split_sizes,
                                 int64_t* output_split_sizes,
                                 void** peer_buffer_ptrs, int local_rank,
                                 int num_peers, int num_sms,
                                 cudaStream_t stream) {
  // 计算grid和block配置
  const int block_size = 256;
  const int grid_size = num_sms;

  // 启动CUDA kernel
  // ke_intranode_all_to_all<<<grid_size, block_size, 0, stream>>>(
  //     input_data, output_data, input_split_sizes, output_split_sizes,
  //     reinterpret_cast<Element**>(peer_buffer_ptrs), local_rank, num_peers);

  // 检查kernel启动错误
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace nvshmem_tutorial::intranode