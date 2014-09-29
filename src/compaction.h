#pragma once
#define T int  // hack so I don't have to change types again

static const int BLOCK_SIZE = 1024;

extern float timing;


T *prefix_sum_naive(const int len, const T *in);

T *prefix_sum(const int len, const T *in);

T *prefix_sum_eff(const int len, const T *in);
