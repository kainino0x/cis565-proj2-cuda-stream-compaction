#include <cmath>

#include "compaction.h"
#include "cuda_helpers.h"

#define BLOCK_SIZE 256


__global__ void prefix_sum_naive_inner(int len, int d, int *in, int *out)
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

int *prefix_sum_naive(int *in, int len)
{
    dim3 BS((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 TPB(BLOCK_SIZE);  // TODO fix this

    // Output host array
    int *out = new int[len];

    // Create two arrays and alternate between them when calculating
    int *dev_arrs[2];
    cudaMalloc(&dev_arrs[0], len * sizeof(int));
    cudaMalloc(&dev_arrs[1], len * sizeof(int));
    CHECK_ERROR("malloc");

    cudaMemcpy(dev_arrs[1], in, len * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_ERROR("input memcpy");

    int iout = 0;
    const int dmax = ceil(log2((float) len));
    for (int d = 0; d < dmax; ++d) {
        iout = d & 1;
        prefix_sum_naive_inner<<<BS, TPB>>>(
                len, d, dev_arrs[iout ^ 1], dev_arrs[iout]);
    }
    CHECK_ERROR("prefix_sum_naive_inner");

    out[0] = 0;
    cudaMemcpy(&out[1], dev_arrs[iout], (len - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_ERROR("result memcpy");

    return out;
}
