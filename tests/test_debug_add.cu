/* test_debug_add.cu — diagnostic: stores n, lsz, n+lsz to separate offsets.
 * out[0]=n, out[1]=lsz, out[2]=n+lsz */
extern "C" __device__ int __ockl_get_local_id(int);
extern "C" __device__ int __ockl_get_local_size(int);

extern "C" __global__ void test_debug_add(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    int lsz = __ockl_get_local_size(0);
    if (lid == 0) {
        out[0] = (float)n;
        out[1] = (float)lsz;
        out[2] = (float)(n + lsz);
    }
}
