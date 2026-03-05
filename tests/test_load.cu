/* test_load.cu — test load from in[], store to out[] with 3 args */
extern "C" __device__ int __ockl_get_local_id(int);
extern "C" __device__ int __ockl_get_group_id(int);
extern "C" __device__ int __ockl_get_local_size(int);

extern "C" __global__ void test_load(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    int gid = __ockl_get_group_id(0);
    int lsz = __ockl_get_local_size(0);
    int idx = gid * lsz + lid;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x + 1.0f;
    }
}
