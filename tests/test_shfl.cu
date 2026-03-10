/* test_shfl.cu — shuffle-down: each lane reads value from lane+1 */
extern "C" __device__ int __ockl_get_local_id(int);

extern "C" __global__ void test_shfl(float *out, float *in) {
    int lid = __ockl_get_local_id(0);
    float val = in[lid];
    float got = __shfl_down_sync(0xFFFFFFFFu, val, 1, 64);
    out[lid] = got;
}
