/* test_vadd.cu — same as test_noload but force VALU add via lid trickery */
extern "C" __device__ int __ockl_get_local_id(int);
extern "C" __device__ int __ockl_get_local_size(int);

extern "C" __global__ void test_vadd(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    int lsz = __ockl_get_local_size(0);
    /* n + lid - lid forces divergent computation, cancels out */
    int val = (n + lid) - lid + lsz;
    out[lid] = (float)val;
}
