/* test_dispatch.cu — minimal test for dispatch_ptr access */
extern "C" __device__ int __ockl_get_local_id(int);
extern "C" __device__ int __ockl_get_local_size(int);

extern "C" __global__ void test_dispatch(float *out) {
    int lsz = __ockl_get_local_size(0);
    int lid = __ockl_get_local_id(0);
    out[lid] = (float)lsz;
}
