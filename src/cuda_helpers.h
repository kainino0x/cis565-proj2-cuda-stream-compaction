#pragma once

#include <cstdio>

#if DEBUG
#define CHECK_ERROR(msg) (checkCUDAError((msg), __FILE__, __func__, __LINE__))
#else
#define CHECK_ERROR(msg)
#endif


/// Check for CUDA errors.
inline void checkCUDAError(
        const char *msg, const char *file, const char *func, int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "%s:%d: %s\n    CUDA Error: %s - %s\n",
                file, line, func, msg, cudaGetErrorString(err));
        throw;
    }
}
