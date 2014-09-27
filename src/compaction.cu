#include <cmath>

#include "compaction.h"
#include "cuda_helpers.h"

#define BLOCK_SIZE 256


__global__ void prefix_sum_naive_inner(const int len, const int d, const int *in, int *out)
{
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    int k1 = k - (1 << d);
    if (k < len) {
        if (k1 >= 0) {
            out[k] = in[k] + in[k1];
        } else {
            out[k] = in[k];
        }
    }
}

int *prefix_sum_naive(const int *in, const int len)
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
    cudaMemcpy(dev_arrs[1], in, len * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_ERROR("input memcpy");

    // Do each of the log(n) steps as a separate kernel call
    int iout = 0;
    const int dmax = ceil(log2((float) len));
    for (int d = 0; d < dmax; ++d) {
        iout = d & 1;
        prefix_sum_naive_inner<<<BS, TPB>>>(
                len, d, dev_arrs[iout ^ 1], dev_arrs[iout]);
    }
    CHECK_ERROR("prefix_sum_naive_inner");

    // Copy the result value back to the CPU
    out[0] = 0;
    cudaMemcpy(&out[1], dev_arrs[iout], (len - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_ERROR("result memcpy");

    return out;
}


__global__ void prefix_sum_inner(const int len, const int dmax, const int *in, int *out)
{
    __shared__ int tmp[2][BLOCK_SIZE];

    int t = threadIdx.x;
    int k = (blockIdx.x * blockDim.x) + t;

    // Copy `in` into `tmp[1]`
    tmp[1][t] = in[k];
    __syncthreads();

    int iout = 0;
    for (int d = 0; d < dmax; ++d) {
        iout = d & 1;  // 0 1 0 1 ...
        int *tmpout = tmp[iout];
        int *tmpin = tmp[iout ^ 1];
        if (k < len) {
            int k1 = k - (1 << d);
            if (k1 >= 0) {
                tmpout[k] = tmpin[k] + tmpin[k1];
            } else {
                tmpout[k] = tmpin[k];
            }
        }
        __syncthreads();
    }

    out[k] = tmp[iout][t];
}

int *prefix_sum(const int *in, const int len)
{
    if (len > BLOCK_SIZE) {
        throw;
    }

    const dim3 BS((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 TPB(BLOCK_SIZE);

    // Output host array
    int *out = new int[len];

    // Create arrays for input and output
    int *dev_in, *dev_out;
    cudaMalloc(&dev_in, len * sizeof(int));
    cudaMalloc(&dev_out, len * sizeof(int));
    CHECK_ERROR("malloc");

    // Copy input values to GPU
    cudaMemcpy(dev_in, in, len * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_ERROR("input memcpy");

    const int dmax = ceil(log2((float) len));
    prefix_sum_inner<<<BS, TPB>>>(len, dmax, dev_in, dev_out);

    // Copy the result value back to the CPU
    out[0] = 0;
    cudaMemcpy(&out[1], dev_out, (len - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_ERROR("result memcpy");

    return out;
}
