#include "prefix_sum.h"


void prefix_sum_cpu(const int len, const float *in, float *out)
{
    float sum = 0;
    for (int i = 0; i < len; ++i) {
        out[i] = sum;
        sum += in[i];
    }
}
