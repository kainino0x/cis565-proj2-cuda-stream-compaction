#pragma once

static const int BLOCK_SIZE = 256;

extern float timing;


float *prefix_sum_naive(const int len, const float *in);

float *prefix_sum_eff(const int len, const float *in);
