#pragma once
#define T int  // hack so I don't have to change types again


void prefix_sum_cpu(const int len, const T *in, T *out);

void scatter_cpu(const int len, const T *in, int *out);
