#include <cmath>

#include "compaction.h"
#include "cuda_helpers.h"

float timing = 0;


inline int ilog2(int x)
{
    int lg = 0;
    while (x >>= 1)
    {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x)
{
    return ilog2(x - 1) + 1;
}


__global__ void prefix_sum_naive_inner(
        const int len, const int bs, const int d, const int *in, int *out)
{
    const int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int k1 = ((k / bs) - (1 << d)) * bs + (bs - 1);
    if (k < len) {
        if (k1 >= 0) {
            out[k] = in[k] + in[k1];
        } else {
            out[k] = in[k];
        }
    }
}

int *prefix_sum_naive(const int len, const int *in)
{
    dim3 BS((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 TPB(BLOCK_SIZE);

    // Output host array
    int *out = new int[len];

    // Create two arrays and alternate between them when calculating
    int *dev_arrs[2];
    cudaMalloc(&dev_arrs[0], len * sizeof(int));
    cudaMalloc(&dev_arrs[1], len * sizeof(int));
    CHECK_ERROR("malloc");

    // Copy input values to GPU
    cudaMemcpy(dev_arrs[0], in, len * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_ERROR("input memcpy");

    // Do each of the log(n) steps as a separate kernel call
    int iout = 0;  // init to i_in then it gets flipped
    const int dmax = ilog2ceil(len);
#if 1
    cudaEvent_t ev0;
    cudaEvent_t ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
#endif
    for (int d = 0; d < dmax; ++d) {
        iout ^= 1;
        prefix_sum_naive_inner<<<BS, TPB>>>(
                len, 1, d, dev_arrs[iout ^ 1], dev_arrs[iout]);
    }
#if 1
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    float t;
    cudaEventElapsedTime(&t, ev0, ev1);
    timing += t;
#endif
    CHECK_ERROR("prefix_sum_naive_inner");

    // Copy the result value back to the CPU
    out[0] = 0;
    cudaMemcpy(&out[1], dev_arrs[iout], (len - 1) * sizeof(int),
            cudaMemcpyDeviceToHost);
    CHECK_ERROR("result memcpy");

    cudaFree(dev_arrs[0]);
    cudaFree(dev_arrs[1]);
    CHECK_ERROR("free");

    return out;
}


__global__ void prefix_sum_inner_shared(
        const int len, const int blockdmax, const int *in, int *out)
{
    __shared__ int tmp[2][BLOCK_SIZE];

    const int t = threadIdx.x;
    const int boff = blockIdx.x * blockDim.x;
    const int k = boff + t;

    // Copy input to shared memory
    tmp[0][t] = in[k];
    __syncthreads();

    // Execute as much as we can totally within shared memory...
    int iout = 0;
    for (int d = 0; d < blockdmax; ++d) {
        iout ^= 1;
        int *tmpout = tmp[iout];
        int *tmpin = tmp[iout ^ 1];
        if (k < len) {
            int t1 = t - (1 << d);
            if (t1 >= 0) {
                tmpout[t] = tmpin[t] + tmpin[t1];
            } else {
                tmpout[t] = tmpin[t];
            }
        }
        __syncthreads();
    }
    if (k < len) {
        out[k] = tmp[iout][t];
    }

    // And the rest needs to be done globally after this completes.
}

/// Prefix sum implementation using shared memory.
/// Currently this is kind of dumb: it takes CPU memory, copies it,
/// then allocates more CPU memory and copies the result back.
/// Later I'll need to factor some of that out so I can actually use this
/// algorithm in a bigger GPU pipeline.
int *prefix_sum(const int len, const int *in)
{
    const dim3 BS((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 TPB(BLOCK_SIZE);

    // Output host array
    int *out = new int[len];

    // Create arrays for input and output
    int *dev_arrs[2];
    cudaMalloc(&dev_arrs[0], len * sizeof(int));
    cudaMalloc(&dev_arrs[1], len * sizeof(int));
    CHECK_ERROR("malloc");

    // Copy input values to GPU
    cudaMemcpy(dev_arrs[0], in, len * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_ERROR("input memcpy");

    // Do what we can with shared memory
#if 1
    cudaEvent_t ev0;
    cudaEvent_t ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
#endif
    const int blockdmax = ilog2(BLOCK_SIZE);
    prefix_sum_inner_shared<<<BS, TPB>>>(len, blockdmax, dev_arrs[0], dev_arrs[1]);

    // Finish off globally
    int iout = 1;  // init to i_in then it gets flipped
    const int dmax = ilog2ceil((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    for (int d = 0; d < dmax; ++d) {
        iout ^= 1;
        prefix_sum_naive_inner<<<BS, TPB>>>(
                len, BLOCK_SIZE, d, dev_arrs[iout ^ 1], dev_arrs[iout]);
    }
#if 1
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    float t;
    cudaEventElapsedTime(&t, ev0, ev1);
    timing += t;
#endif
    CHECK_ERROR("prefix_sum_inner");

    // Copy the result value back to the CPU
    out[0] = 0;
    cudaMemcpy(&out[1], dev_arrs[iout], (len - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_ERROR("result memcpy");

    cudaFree(dev_arrs[0]);
    cudaFree(dev_arrs[1]);
    CHECK_ERROR("free");

    return out;
}
