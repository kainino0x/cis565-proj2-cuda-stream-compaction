#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "prefix_sum.h"
#include "compaction.h"

#define MAXVAL 5
#define ITERS 5
static int MAXLEN = 1024 * 1024 * 64;
static int LEN = 16;


bool test_equality(const int len, const float *a, const float *b)
{
    for (int i = 0; i < len; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

void test_impl(const int len, const float *in, const float *exp,
        float *(*f)(const int, const float*))
{
    float *out = f(len, in);
    for (int i = 0; i < len; ++i) { printf("%2.0f ",  in[i]); } printf("\n");
    for (int i = 0; i < len; ++i) { printf("%2.0f ", out[i]); } printf("\n");
    if (test_equality(len, exp, out)) {
        printf("pass\n");
    } else {
        printf("FAIL\n");
    }
}

int main()
{
    srand(0);

    float *in = new float[MAXLEN];
    for (int i = 0; i < MAXLEN; ++i) { in[i] = rand() % MAXVAL; }
    float *exp = new float[MAXLEN];

    prefix_sum_cpu(LEN, in, exp);

    printf("prefix_sum_naive:\n");
    test_impl(LEN, in, exp, prefix_sum_naive);

    printf("prefix_sum:\n");
    test_impl(LEN, in, exp, prefix_sum);

#if 0
    for (LEN = 256; LEN <= MAXLEN; LEN *= 2) {
#if __linux__
        struct timespec ts1, ts2;
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        for (int i = 0; i < ITERS; ++i) {
            prefix_sum_cpu(LEN, in, exp);
        }
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        double t1 = ts1.tv_sec * 1e3 + ts1.tv_nsec * 1e-6;
        double t2 = ts2.tv_sec * 1e3 + ts2.tv_nsec * 1e-6;
        printf("cpu,%d,%d,%e\n", 0, LEN, (t2 - t1) / ITERS);
#endif

        timing = 0;
        for (int i = 0; i < ITERS; ++i) {
            prefix_sum_naive(LEN, in);
        }
        printf("naive,%d,%d,%e\n", BLOCK_SIZE, LEN, timing / ITERS);

        timing = 0;
        for (int i = 0; i < ITERS; ++i) {
            prefix_sum(LEN, in);
        }
        printf("shared,%d,%d,%e\n", BLOCK_SIZE, LEN, timing / ITERS);
    }
#endif

    delete[] in;
    delete[] exp;
}
