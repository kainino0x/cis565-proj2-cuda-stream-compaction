#pragma once

static const int BLOCK_SIZE = 1024;

extern float timing;


int *prefix_sum_naive(const int len, const int *in);

int *prefix_sum(const int len, const int *in);
