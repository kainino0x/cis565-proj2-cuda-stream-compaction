#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "cuda_helpers.h"
#include "prefix_sum.h"
#include "compaction.h"

#define MAXVAL 5
#define ITERS 5
static int MAXLEN = 1024 * 1024 * 64;
static int LEN = 1000000;


bool test_equality(const int len, const T *a, const T *b)
{
    for (int i = 0; i < len; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

void test_impl(const int len, const T *in, const T *exp,
        void (*f)(const int, T *, T *))
{
    T *dev_in = mallocopy(len, in);
    T *dev_out;
    cudaMalloc(&dev_out, len * sizeof(T));

    f(len, dev_in, dev_out);

    T *out = new T[len];
    cudaMemcpy(out, dev_out, len * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(dev_in);
    cudaFree(dev_out);

    //for (int i = 0; i < len; ++i) { printf("%2d ",  in[i]); } printf("\n");
    //for (int i = 0; i < len; ++i) { printf("%2d ", out[i]); } printf("\n");
    if (out != NULL && test_equality(len, exp, out)) {
        printf("pass\n");
    } else {
        printf("FAIL\n");
    }
}

int main()
{
    srand(0);

    T *in = new T[MAXLEN];
    for (int i = 0; i < MAXLEN; ++i) { in[i] = rand() % MAXVAL; }
    T *exp = new T[MAXLEN];

    T *dev_in, *dev_out;
	cudaMalloc(&dev_in , MAXLEN * sizeof(T));
	cudaMalloc(&dev_out, MAXLEN * sizeof(T));
	CHECK_ERROR("malloc");

#if 1
    int *scatterout = new int[MAXLEN];
    prefix_sum_cpu(LEN, in, exp);

    printf("prefix_sum_naive:\n");
    test_impl(LEN, in, exp, prefix_sum_naive);

    printf("prefix_sum:\n");
    test_impl(LEN, in, exp, prefix_sum);

    //printf("prefix_sum_eff:\n");
    //test_impl(LEN, in, exp, prefix_sum_eff);

    printf("scatter_cpu:\n");
    scatter_cpu(LEN, in, exp);
    //for (int i = 0; i < LEN; ++i) { printf("%2d ", in[i]); } printf("\n");
    //for (int i = 0; i < LEN; ++i) { printf("%2d ", scatterout[i]); } printf("\n");

    printf("scatter:\n");
    {
        cudaMemcpy(dev_in, in, LEN * sizeof(T), cudaMemcpyHostToDevice);
        scatter(LEN, dev_in, dev_out);
        cudaMemcpy(scatterout, dev_out, LEN * sizeof(T), cudaMemcpyDeviceToHost);

		printf(test_equality(LEN, exp, scatterout) ? "pass\n" : "FAIL\n");
    }
    //for (int i = 0; i < LEN; ++i) { printf("%2d ", scatterout[i]); } printf("\n");

    printf("compact:\n");
    {
        cudaMemcpy(dev_in, in, LEN * sizeof(T), cudaMemcpyHostToDevice);
        int len = compact(LEN, dev_in, dev_out);
        cudaMemcpy(exp, dev_out, LEN * sizeof(T), cudaMemcpyDeviceToHost);
        //for (int i = 0; i < len; ++i) { printf("%2d ", exp[i]); } printf("\n");
    }

    printf("compact_thrust:\n");
    {
        cudaMemcpy(dev_in, in, LEN * sizeof(T), cudaMemcpyHostToDevice);
        int len = compact_thrust(LEN, dev_in, dev_out);
        cudaMemcpy(scatterout, dev_out, LEN * sizeof(T), cudaMemcpyDeviceToHost);

		printf(test_equality(len, exp, scatterout) ? "pass\n" : "FAIL\n");
        //for (int i = 0; i < len; ++i) { printf("%2d ", scatterout[i]); } printf("\n");
    }

    cudaFree(dev_out);
#endif

    for (LEN = 256; LEN <= MAXLEN; LEN *= 2) {
#if 0
#   if __linux__
        struct timespec ts1, ts2;
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        for (int i = 0; i < ITERS; ++i) {
            prefix_sum_cpu(LEN, in, exp);
        }
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        double t1 = ts1.tv_sec * 1e3 + ts1.tv_nsec * 1e-6;
        double t2 = ts2.tv_sec * 1e3 + ts2.tv_nsec * 1e-6;
        printf("cpu,%d,%d,%e\n", 0, LEN, (t2 - t1) / ITERS);
#   endif

        cudaMemcpy(dev_in, in, LEN * sizeof(T), cudaMemcpyHostToDevice);
        timing = 0;
        for (int i = 0; i < ITERS; ++i) {
            prefix_sum_naive(LEN, dev_in, dev_out);
        }
        printf("naive,%d,%d,%e\n", BLOCK_SIZE, LEN, timing / ITERS);

        cudaMemcpy(dev_in, in, LEN * sizeof(T), cudaMemcpyHostToDevice);
        timing = 0;
        for (int i = 0; i < ITERS; ++i) {
            prefix_sum(LEN, dev_in, dev_out);
        }
        printf("shared,%d,%d,%e\n", BLOCK_SIZE, LEN, timing / ITERS);

        cudaMemcpy(dev_in, in, LEN * sizeof(T), cudaMemcpyHostToDevice);
        timing = 0;
        for (int i = 0; i < ITERS; ++i) {
            prefix_sum_eff(LEN, dev_in, dev_out);
        }
        printf("eff,%d,%d,%e\n", BLOCK_SIZE, LEN, timing / ITERS);
#endif

#if 0
        cudaMemcpy(dev_in, in, LEN * sizeof(T), cudaMemcpyHostToDevice);
        timing = 0;
        for (int i = 0; i < ITERS; ++i) {
            compact(LEN, dev_in, dev_out);
        }
        printf("compact,%d,%d,%e\n", BLOCK_SIZE, LEN, timing / ITERS);

        cudaMemcpy(dev_in, in, LEN * sizeof(T), cudaMemcpyHostToDevice);
        timing = 0;
        for (int i = 0; i < ITERS; ++i) {
            compact_thrust(LEN, dev_in, dev_out);
        }
        printf("compact_thrust,%d,%d,%e\n", BLOCK_SIZE, LEN, timing / ITERS);
#endif
    }

    delete[] exp;
    delete[] in;
}
