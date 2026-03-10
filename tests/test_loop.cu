/* test_loop.cu — loop accumulation: out[tid] = sum(0..n-1) */
extern "C" __device__ int __ockl_get_local_id(int);

extern "C" __global__ void test_loop(float *out, int n) {
    int lid = __ockl_get_local_id(0);
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum = sum + (float)i;
    out[lid] = sum;
}
