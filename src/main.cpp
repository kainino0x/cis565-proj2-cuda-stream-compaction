#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "prefix_sum.h"
#include "compaction.h"

#define MAXVAL 100
#define ITERS 5
static int MAXLEN = 1024 * 1024 * 64;
static int LEN = 100000;


bool test_equality(const int len, const int *a, const int *b)
{
    for (int i = 0; i < len; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

void test_impl(const int len, const int *in, const int *exp,
        int *(*f)(const int, const int*))
{
    int *out = f(len, in);
    if (test_equality(len, exp, out)) {
        printf("pass\n");
    } else {
        printf("FAIL\n");
    }
}

int main()
{
    srand(0);

    int *in = new int[MAXLEN];
    for (int i = 0; i < MAXLEN; ++i) { in[i] = rand() % MAXVAL; }
    int *exp = new int[MAXLEN];

    prefix_sum_cpu(LEN, in, exp);

    printf("prefix_sum_naive: ");
    test_impl(LEN, in, exp, prefix_sum_naive);

    printf("prefix_sum: ");
    test_impl(LEN, in, exp, prefix_sum);

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

    delete[] in;
    delete[] exp;
}
