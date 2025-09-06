#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace nvshmem_tutorial {
namespace intranode_sync {

/**
 * GPU kernel for signaling completion using atomic operations
 */
__global__ void signal_completion(int* signal_ptr, int value) {
  atomicExch(signal_ptr, value);
}

/**
 * GPU kernel for waiting on a signal using atomic operations
 */
__global__ void wait_for_signal(int* signal_ptr, int expected_value) {
  // Busy wait until signal matches expected value
  while (atomicAdd(signal_ptr, 0) != expected_value) {
    // Brief pause to reduce memory pressure
    __nanosleep(100);
  }
}

/**
 * GPU kernel for resetting a signal
 */
__global__ void reset_signal(int* signal_ptr) { atomicExch(signal_ptr, 0); }

/**
 * Launch signal completion kernel
 */
inline void launch_signal_completion(int* signal_ptr, int value,
                                     cudaStream_t stream) {
  signal_completion<<<1, 1, 0, stream>>>(signal_ptr, value);
}

/**
 * Launch wait for signal kernel
 */
inline void launch_wait_for_signal(int* signal_ptr, int expected_value,
                                   cudaStream_t stream) {
  wait_for_signal<<<1, 1, 0, stream>>>(signal_ptr, expected_value);
}

/**
 * Launch reset signal kernel
 */
inline void launch_reset_signal(int* signal_ptr, cudaStream_t stream) {
  reset_signal<<<1, 1, 0, stream>>>(signal_ptr);
}

}  // namespace intranode_sync
}  // namespace nvshmem_tutorial
