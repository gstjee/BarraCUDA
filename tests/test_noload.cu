/* test_noload.cu — 3 args + dispatch, NO load from in */
extern "C" __device__ int __ockl_get_local_id(int);
extern "C" __device__ int __ockl_get_local_size(int);

extern "C" __global__ void test_noload(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    int lsz = __ockl_get_local_size(0);
    out[lid] = (float)(n + lsz);
}
