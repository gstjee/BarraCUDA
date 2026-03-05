/* test_n_with_dispatch.cu — store just n, but WITH dispatch/hidden kernarg.
 * Isolates: does kernarg_size=48 break reading n at offset 16? */
extern "C" __device__ int __ockl_get_local_id(int);
extern "C" __device__ int __ockl_get_local_size(int);

extern "C" __global__ void test_n_with_dispatch(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    int lsz = __ockl_get_local_size(0);  /* force needs_dispatch */
    out[lid] = (float)(n + lsz * 0);  /* lsz*0=0, so result is n, but lsz isn't DCE'd */
}
