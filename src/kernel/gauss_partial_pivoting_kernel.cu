#include "gauss_partial_pivoting.h"

__global__
void gauss_partial_pivoting_kernel(size_t n, numeric_t* As, numeric_t* bs, numeric_t* xs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    numeric_t* A = As+idx*28*28;
    numeric_t* b = bs+idx*28;
    numeric_t* x = xs+idx*28;
    gauss<numeric_t, 28>(A, b, x);
}