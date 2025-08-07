#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_CHECK(stmt)                                                          \
    do                                                                            \
    {                                                                             \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result)                                                \
        {                                                                         \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)