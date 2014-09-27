#include <cstdlib>
#include <cstdio>

#include "prefix_sum.h"
#include "compaction.h"

#define LEN 20
#define MAXVAL 8

int main()
{
    srand(0);

    printf("** INPUT **\n");
    int in[LEN];
    int *out;
    for (int i = 0; i < LEN; ++i) { in[i] = rand() % MAXVAL; }
    for (int i = 0; i < LEN; ++i) { printf(" %2d", in[i]); } printf("\n");

    printf("** OUTPUT prefix_sum_cpu **\n");
    out = prefix_sum_cpu(in, LEN);
    for (int i = 0; i < LEN; ++i) { printf(" %2d", out[i]); } printf("\n");
    delete[] out;

    printf("** OUTPUT prefix_sum_naive **\n");
    out = prefix_sum_naive(in, LEN);
    for (int i = 0; i < LEN; ++i) { printf(" %2d", out[i]); } printf("\n");
    delete[] out;

    printf("** OUTPUT prefix_sum **\n");
    out = prefix_sum(in, LEN);
    for (int i = 0; i < LEN; ++i) { printf(" %2d", out[i]); } printf("\n");
    delete[] out;
}
