#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "prefix_sum.h"
#include "compaction.h"

#define MAXVAL 100
#define ITERS 8
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

    int *exp = prefix_sum_cpu(LEN, in);

    printf("prefix_sum_naive: ");
    test_impl(LEN, in, exp, prefix_sum_naive);

    printf("prefix_sum: ");
    test_impl(LEN, in, exp, prefix_sum);

    for (LEN = 1024; LEN <= MAXLEN; LEN *= 2) {
#if __linux__
        struct timespec ts1, ts2;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts1);
        for (int i = 0; i < ITERS; ++i) {
            prefix_sum_cpu(LEN, in);
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts2);
        float t1 = ts1.tv_sec * 1e3f + ts1.tv_nsec / 1e6f;
        float t2 = ts2.tv_sec * 1e3f + ts2.tv_nsec / 1e6f;
        printf("cpu,%d,%d,%f\n", 0, LEN, t2 - t1);
#endif

        timing = 0;
        for (int i = 0; i < ITERS; ++i) {
            prefix_sum_naive(LEN, in);
        }
        printf("naive,%d,%d,%f\n", BLOCK_SIZE, LEN, timing / ITERS);

        timing = 0;
        for (int i = 0; i < ITERS; ++i) {
            prefix_sum(LEN, in);
        }
        printf("shared,%d,%d,%f\n", BLOCK_SIZE, LEN, timing / ITERS);
    }

    delete[] in;
    delete[] exp;
}
