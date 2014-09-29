#include "prefix_sum.h"


void prefix_sum_cpu(const int len, const T *in, T *out)
{
    T sum = 0;
    for (int i = 0; i < len; ++i) {
        out[i] = sum;
        sum += in[i];
    }
}

void scatter_cpu(const int len, const T *in, int *out)
{
}
