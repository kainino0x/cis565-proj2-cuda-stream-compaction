#include "prefix_sum.h"


int *prefix_sum_cpu(int *in, int len)
{
    int *out = new int[len];
    int sum = 0;
    for (int i = 0; i < len; ++i) {
        out[i] = sum;
        sum += in[i];
    }
    return out;
}
