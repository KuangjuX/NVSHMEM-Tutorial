#pragma once

#include "launch.cuh"
#include "ptx_wrapper.cuh"
#include "utils.hpp"

namespace nvshmem_tutorial::sync {

/**
 * @brief GPU device function implementing a distributed barrier synchronization
 * across multiple ranks
 *
 * This function implements a barrier synchronization mechanism that allows
 * multiple GPU ranks to synchronize with each other using shared memory
 * pointers. Each rank signals its completion and waits for all other ranks to
 * complete before proceeding.
 *
 * @tparam kNumRanks Number of ranks participating in the barrier (compile-time
 * constant)
 * @tparam kSyncOnly If true, only performs synchronization without memory fence
 * operations
 *
 * @param barrier_signal_ptrs 2D array of pointers where
 * barrier_signal_ptrs[i][j] points to the signal location for rank i to
 * communicate with rank j
 * @param rank The current rank ID of this GPU process
 *
 * Algorithm:
 * 1. Optional memory fence to ensure visibility of prior memory operations
 * 2. Each thread signals its own rank's completion and decrements other ranks'
 * counters
 * 3. Wait until all other ranks have signaled completion (all values <= 0)
 * 4. Timeout detection to prevent infinite waiting in case of failures
 */
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

  // Signal completion: add to own rank's counter, subtract from other ranks'
  // counters This creates a symmetric signaling pattern where each rank both
  // signals and waits
  if (thread_id < kNumRanks) {
    atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
    atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
  }

  DEVICE_ASSERT(kNumRanks <= blockDim.x);

  // Polling phase: wait for all other ranks to signal completion
  auto start_time = clock64();
  while (true) {
    // Read the signal value for this thread's corresponding rank
    auto value = thread_id < kNumRanks
                     ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id)
                     : 0;
    // All threads check if their respective values are <= 0 (completion
    // condition)
    if (__all_sync(0xffffffff, value <= 0)) break;

    // Timeout detection to prevent deadlock situations
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
__forceinline__ __global__ void barrier(int** barrier_signal_ptrs, int rank) {
  barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

inline void barrier(int** barrier_signal_ptrs, int rank, int num_ranks,
                    cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                \
  LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank); \
  break

  SETUP_LAUNCH_CONFIG(1, 32, stream);
  SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace nvshmem_tutorial::sync
