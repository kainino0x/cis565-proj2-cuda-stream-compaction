#include <cmath>
#include <thrust/copy.h>

#include "compaction.h"
#include "cuda_helpers.h"

#define TIMING 1
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
        const int len, const int bs, const int d, const T *in, T *out)
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

void prefix_sum_naive(const int len, T *dev_in, T *dev_out)
{
    dim3 BC((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 TPB(BLOCK_SIZE);

    // Create two arrays and alternate between them when calculating
    T *dev_arrs[2];
    dev_arrs[0] = dev_in;
    cudaMalloc(&dev_arrs[1], len * sizeof(T));
    CHECK_ERROR("malloc");

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

    // Copy the result value into the output array
    const T zero = 0;
    cudaMemcpy(dev_out, &zero, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_out[1], dev_arrs[iout], (len - 1) * sizeof(T), cudaMemcpyDeviceToDevice);
    CHECK_ERROR("result memcpy");

    cudaFree(dev_arrs[1]);
    CHECK_ERROR("free");
}


__global__ void prefix_sum_inner_shared(
        const int len, const int blockdmax, const T *in, T *out)
{
    __shared__ T tmp[2][BLOCK_SIZE];

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
        T *tmpout = tmp[iout];
        T *tmpin = tmp[iout ^ 1];
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

void prefix_sum(const int len, T *dev_in, T *dev_out)
{
    const dim3 BC((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 TPB(BLOCK_SIZE);

    // Create arrays for input and output
    T *dev_arrs[2];
    dev_arrs[0] = dev_in;
    cudaMalloc(&dev_arrs[1], len * sizeof(T));
    CHECK_ERROR("malloc");

    // Do what we can with shared memory
#if TIMING
    cudaEvent_t ev0;
    cudaEvent_t ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
#endif
    const int blockdmax = ilog2(BLOCK_SIZE);
    prefix_sum_inner_shared<<<BC, TPB>>>(len, blockdmax, dev_arrs[0], dev_arrs[1]);

    // Finish off globally
    int iout = 1;  // init to i_in then it gets flipped
    const int dmax = ilog2ceil((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    for (int d = 0; d < dmax; ++d) {
        iout ^= 1;
        prefix_sum_naive_inner<<<BC, TPB>>>(
                len, BLOCK_SIZE, d, dev_arrs[iout ^ 1], dev_arrs[iout]);
    }
#if TIMING
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    float t;
    cudaEventElapsedTime(&t, ev0, ev1);
    timing += t;
#endif
    CHECK_ERROR("prefix_sum_inner");

    // Copy the result value into the output array
    const T zero = 0;
    cudaMemcpy(dev_out, &zero, sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(&dev_out[1], dev_arrs[iout], (len - 1) * sizeof(T), cudaMemcpyDeviceToDevice);
    CHECK_ERROR("result memcpy");

    cudaFree(dev_arrs[1]);
    CHECK_ERROR("free");
}


__global__ void prefix_sum_eff_inner(const int len, const T *in, T *out)
{
    extern __shared__ T temp[];
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
            T t = temp[ai];
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

void prefix_sum_eff(const int len, T *dev_in, T *dev_out)
{
    const int _bs = BLOCK_SIZE * 2;
    if (len > _bs) {
        return;
    }
    const int _bc = (len + _bs - 1) / _bs;
    const dim3 BC(_bc);
    const dim3 TPB(BLOCK_SIZE);
    const int lenplus = _bc * _bs;

    // Create arrays for input and output
    T *dev_arrs[2];
    dev_arrs[0] = dev_in;
    cudaMalloc(&dev_arrs[1], lenplus * sizeof(T));
    CHECK_ERROR("malloc");

#if TIMING
    cudaEvent_t ev0;
    cudaEvent_t ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
#endif
    prefix_sum_eff_inner<<<BC, TPB, _bs * sizeof(T)>>>(lenplus, dev_arrs[0], dev_arrs[1]);
#if TIMING
    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);
    float t;
    cudaEventElapsedTime(&t, ev0, ev1);
    timing += t;
#endif
    CHECK_ERROR("prefix_sum_eff_inner");

    // Copy the result value into the output array
    cudaMemcpy(dev_out, dev_arrs[1], (len - 1) * sizeof(T), cudaMemcpyDeviceToDevice);
    CHECK_ERROR("result memcpy");

    cudaFree(dev_arrs[0]);
    cudaFree(dev_arrs[1]);
    CHECK_ERROR("free");
}


__global__ void scatter_inner(const int len, const T *in, int *out)
{
    const int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k < len) {
        out[k] = in[k] == 0 ? 0 : 1;
    }
}

void scatter(const int len, const T *dev_in, int *dev_out)
{
    const dim3 BC((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 TPB(BLOCK_SIZE);

    int *dev_tmp;
    cudaMalloc(&dev_tmp, len * sizeof(int));

    scatter_inner<<<BC, TPB>>>(len, dev_in, dev_tmp);
    prefix_sum(len, dev_tmp, dev_out);

    cudaFree(dev_tmp);
}


__global__ void compact_inner(const int len, const T *in, T *out, int *indices)
{
    const int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k < len) {
        int i = indices[k];
        int v = in[k];
        if (v != 0) {
            out[i] = v;
        }
    }
}

int compact(const int len, const T *dev_in, int *dev_out)
{
    dim3 BC((len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 TPB(BLOCK_SIZE);

    int *dev_tmp;
    cudaMalloc(&dev_tmp, len * sizeof(int));

    scatter(len, dev_in, dev_tmp);

    int finallen;
    cudaMemcpy(&finallen, &dev_tmp[len - 1], sizeof(int), cudaMemcpyDeviceToHost);

    compact_inner<<<BC, TPB>>>(len, dev_in, dev_out, dev_tmp);

    cudaFree(dev_tmp);
    return finallen + 1;
}


struct is_nonzero
{
    __host__ __device__ bool operator()(const T x)
    {
        return x != 0;
    }
};

int compact_thrust(const int len, const T *dev_in, int *dev_out)
{
    int *end = thrust::copy_if(dev_in, dev_in + len, dev_out, is_nonzero());
    return end - dev_out;
}
