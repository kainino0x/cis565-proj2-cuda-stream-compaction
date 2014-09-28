#include <cstdlib>
#include <cstdio>

#include "prefix_sum.h"
#include "compaction.h"

#define LEN 100000000
#define MAXVAL 100


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

    int *in = new int[LEN];
    for (int i = 0; i < LEN; ++i) { in[i] = rand() % MAXVAL; }

    int *exp = prefix_sum_cpu(LEN, in);

    printf("prefix_sum_naive: ");
    test_impl(LEN, in, exp, prefix_sum_naive);

    printf("prefix_sum: ");
    test_impl(LEN, in, exp, prefix_sum);

    delete[] in;
    delete[] exp;
}
