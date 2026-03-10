/* test_branch.cu — minimal divergent branch test */
extern "C" __device__ int __ockl_get_local_id(int);

extern "C" __global__ void test_branch(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    float val = 0.0f;
    if (lid < n) {
        val = 1.0f;
    }
    out[lid] = val;
}
