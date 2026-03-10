/* test_3arg_dispatch.cu — identical logic to test_dispatch but 3 args */
extern "C" __device__ int __ockl_get_local_id(int);
extern "C" __device__ int __ockl_get_local_size(int);

extern "C" __global__ void test_3arg_dispatch(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    int lsz = __ockl_get_local_size(0);
    /* Only use out and lsz — ignore in and n completely */
    out[lid] = (float)lsz;
}
