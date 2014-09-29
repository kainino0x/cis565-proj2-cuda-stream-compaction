#pragma once

#include <cstdio>

#define DEBUG 1

#if DEBUG && defined(__func__)
#define CHECK_ERROR(msg) (checkCUDAError((msg), __FILE__, __func__, __LINE__))
#elif DEBUG
#define CHECK_ERROR(msg) (checkCUDAError((msg), __FILE__, "", __LINE__))
#else
#define CHECK_ERROR(msg)
#endif


/// Check for CUDA errors.
inline void checkCUDAError(
        const char *msg, const char *file, const char *func, int line)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "%s:%d: %s\n    CUDA Error: %s - %s\n",
                file, line, func, msg, cudaGetErrorString(err));
        throw;
    }
}

/// cudaMalloc + copy to GPU
template<typename S>
inline S *mallocopy(const int len, const S *in)
{
    S *out;
    cudaMalloc(&out, len * sizeof(S));
    cudaMemcpy(out, in, len * sizeof(S), cudaMemcpyHostToDevice);
    return out;
}
