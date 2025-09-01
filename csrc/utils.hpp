#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cuda/pipeline>

// #ifdef ENABLE_RDMA
//   #define NUM_MAX_NVL_PEERS 8
// #else
//   #define NUM_MAX_NVL_PEERS 4
// #endif

#define NUM_MAX_NVL_PEERS 8

#define NUM_MAX_RDMA_PEERS 20

#define FINISHED_SUM_TAG 1024

#ifndef ENABLE_FAST_DEBUG
  #define NUM_CPU_TIMEOUT_SECS 100
  #define NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s
#else
  #define NUM_CPU_TIMEOUT_SECS 10
  #define NUM_TIMEOUT_CYCLES 20000000000ull  // 20G cycles ~= 10s
#endif

#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define CUDA_CHECK(stmt)                                                    \
  do {                                                                      \
    cudaError_t result = (stmt);                                            \
    if (cudaSuccess != result) {                                            \
      fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
              cudaGetErrorString(result));                                  \
      exit(-1);                                                             \
    }                                                                       \
  } while (0)

#ifndef STATIC_ASSERT
  #define STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

#define HOST_ASSERT(cond)                                                    \
  do {                                                                       \
    if (not(cond)) {                                                         \
      printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, \
             #cond);                                                         \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)

#define DEVICE_ASSERT(cond)                                                  \
  do {                                                                       \
    if (not(cond)) {                                                         \
      printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, \
             #cond);                                                         \
      asm("trap;");                                                          \
    }                                                                        \
  } while (0)
