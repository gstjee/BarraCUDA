/* test_11sgpr.cu — use enough SGPRs to cross the threshold, no branch */
extern "C" __device__ int __ockl_get_local_id(int);
extern "C" __device__ int __ockl_get_group_id(int);
extern "C" __device__ int __ockl_get_local_size(int);

extern "C" __global__ void test_11sgpr(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    int gid = __ockl_get_group_id(0);
    int lsz = __ockl_get_local_size(0);
    int idx = gid * lsz + lid;
    out[idx] = (float)(n + lsz);
}
