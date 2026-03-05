/* test_lds.cu — shared memory write + barrier + reversed read */
extern "C" __device__ int __ockl_get_local_id(int);

extern "C" __global__ void test_lds(float *out) {
    __shared__ float smem[64];
    int lid = __ockl_get_local_id(0);
    smem[lid] = (float)(lid * 2);
    __syncthreads();
    out[lid] = smem[63 - lid];
}
