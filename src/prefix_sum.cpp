#include "prefix_sum.h"


void prefix_sum_cpu(const int len, const T *in, T *out)
{
    T sum = 0;
    for (int i = 0; i < len; ++i) {
        T t = in[i];
        out[i] = sum;
        sum += t;
    }
}

void scatter_cpu(const int len, const T *in, int *out)
{
    for (int i = 0; i < len; ++i) {
        out[i] = in[i] == 0 ? 0 : 1;
    }
    prefix_sum_cpu(len, out, out);
}
