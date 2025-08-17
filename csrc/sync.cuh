#pragma once

#include "launch.cuh"
#include "ptx_wrapper.cuh"
#include "utils.hpp"

namespace nvshmem_tutorial::sync {

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int** barrier_signal_ptrs,
                                              int rank) {
  auto thread_id = static_cast<int>(threadIdx.x);

  // For non-sync-only cases, the memory operations by other threads in the
  // block must be visible to the `sys` scope
  if constexpr (not kSyncOnly) {
    memory_fence();
    __syncthreads();
  }

  // Add self-ranks, sub other ranks
  if (thread_id < kNumRanks) {
    atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
    atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
  }

  DEVICE_ASSERT(kNumRanks <= blockDim.x);

  // Check timeout
  auto start_time = clock64();
  while (true) {
    auto value = thread_id < kNumRanks
                     ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id)
                     : 0;
    if (__all_sync(0xffffffff, value <= 0)) break;

    if (clock64() - start_time > NUM_TIMEOUT_CYCLES && thread_id < kNumRanks) {
      printf(
          "DeepEP timeout check failed: rank = %d, thread = %d, value = %d)\n",
          rank, thread_id, value);
      trap();
    }
  }
  __syncthreads();
}

template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {
  barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks,
             cudaStream_t stream) {
  // #define BARRIER_LAUNCH_CASE(ranks)                                \
//   LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank); \
//   break

  //   SETUP_LAUNCH_CONFIG(1, 32, stream);
  //   SWITCH_RANKS(BARRIER_LAUNCH_CASE);
  // #undef BARRIER_LAUNCH_CASE

  switch (num_ranks) {
    case 2:
      barrier<2><<<1, 32, 0, stream>>>(barrier_signal_ptrs, rank);
      break;
    case 4:
      barrier<4><<<1, 32, 0, stream>>>(barrier_signal_ptrs, rank);
      break;
    case 8:
      barrier<8><<<1, 32, 0, stream>>>(barrier_signal_ptrs, rank);
      break;
    default:
      std::cerr << "Unsupported number of ranks: " << num_ranks << std::endl;
      throw std::runtime_error("Unsupported number of ranks");
  }
}

}  // namespace nvshmem_tutorial::sync