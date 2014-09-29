#pragma once
#define T int  // hack so I don't have to change types again

static const int BLOCK_SIZE = 1024;

extern float timing;


void prefix_sum_naive(const int len, T *dev_in, T *dev_out);

void prefix_sum      (const int len, T *dev_in, T *dev_out);

void prefix_sum_eff  (const int len, T *dev_in, T *dev_out);

void scatter(const int len, const T *dev_in, int *dev_out);

int compact(const int len, const T *dev_in, int *dev_out);

int compact_thrust(const int len, const T *dev_in, int *dev_out);
