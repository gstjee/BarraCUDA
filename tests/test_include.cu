#include "myheader.h"
#include "myheader.h"  /* second include should be guarded */

__device__ int helper_func(int x)
{
    return MY_ADD(x, MY_CONSTANT);
}

__global__ void kernel(int *data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = helper_func(data[i]);
}
