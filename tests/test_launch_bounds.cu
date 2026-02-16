/* test_launch_bounds.cu — Verify __launch_bounds__ parsing and propagation. */

__launch_bounds__(256)
__global__ void kern_256(float *out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        out[idx] = (float)idx;
}

__launch_bounds__(128, 4)
__global__ void kern_128_4(float *out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        out[idx] = (float)(idx * 2);
}

/* No launch bounds — should show 0/0 */
__global__ void kern_default(float *out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        out[idx] = 1.0f;
}
