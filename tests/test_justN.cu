/* test_justN.cu — 3 args, store just n (no dispatch needed) */
extern "C" __device__ int __ockl_get_local_id(int);

extern "C" __global__ void test_justN(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    out[lid] = (float)n;
}
