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
        const int len, const int bs, const int d, const float *in, float *out)
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

float *prefix_sum_naive(const int len, const float *in)
{
    dim3 BC((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 TPB(BLOCK_SIZE);

    // Output host array
    float *out = new float[len];

    // Create two arrays and alternate between them when calculating
    float *dev_arrs[2];
    cudaMalloc(&dev_arrs[0], len * sizeof(float));
    cudaMalloc(&dev_arrs[1], len * sizeof(float));
    CHECK_ERROR("malloc");

    // Copy input values to GPU
    cudaMemcpy(dev_arrs[0], in, len * sizeof(float), cudaMemcpyHostToDevice);
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
        prefix_sum_naive_inner<<<BC, TPB>>>(
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
    cudaMemcpy(&out[1], dev_arrs[iout], (len - 1) * sizeof(float),
            cudaMemcpyDeviceToHost);
    CHECK_ERROR("result memcpy");

    cudaFree(dev_arrs[0]);
    cudaFree(dev_arrs[1]);
    CHECK_ERROR("free");

    return out;
}


__global__ void prefix_sum_eff_inner(const int len, const float *in, float *out)
{
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int offset = 1;

    {
        int ai = 2 * thid;
        int bi = ai + 1;
        temp[ai] = in[ai];
        temp[bi] = in[bi];
    }

    // "Build sum in place up the tree"
    for (int d = len >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // "Clear the last element"
    __syncthreads();
    if (thid == 0) {
        temp[len - 1] = 0;
    }

    // "Traverse down tree & build scan"
    for (int d = 1; d < len; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    // "Write results to device memory"
    __syncthreads();
    {
        int ai = 2 * thid;
        int bi = ai + 1;
        out[ai] = temp[ai];
        out[bi] = temp[bi];
    }
}

/// Prefix sum implementation using shared memory.
/// Currently this is kind of dumb: it takes CPU memory, copies it,
/// then allocates more CPU memory and copies the result back.
/// Later I'll need to factor some of that out so I can actually use this
/// algorithm in a bigger GPU pipeline.
float *prefix_sum_eff(const int len, const float *in)
{
    const int _bs = BLOCK_SIZE * 2;
    const int _bc = (len + _bs - 1) / _bs;
    const dim3 BC(_bc);
    const dim3 TPB(BLOCK_SIZE);
    const int lenplus = _bc * _bs;

    // Output host array
    float *out = new float[len];

    // Create arrays for input and output
    float *dev_arrs[2];
    cudaMalloc(&dev_arrs[0], lenplus * sizeof(float));
    cudaMalloc(&dev_arrs[1], lenplus * sizeof(float));
    CHECK_ERROR("malloc");

    // Copy input values to GPU : 0
    cudaMemcpy(dev_arrs[0], in, len * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_ERROR("input memcpy");

    // TODO
    prefix_sum_eff_inner<<<BC, TPB, _bs * sizeof(float)>>>(lenplus, dev_arrs[0], dev_arrs[1]);
    CHECK_ERROR("prefix_sum_eff_inner");

    // Copy the result value back to the CPU
    cudaMemcpy(out, dev_arrs[1], len * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_ERROR("result memcpy");

    cudaFree(dev_arrs[0]);
    cudaFree(dev_arrs[1]);
    CHECK_ERROR("free");

    return out;
}
