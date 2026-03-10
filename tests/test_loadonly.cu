/* test_loadonly.cu — load from in[], add 1, store to out[], no branch */
extern "C" __device__ int __ockl_get_local_id(int);

extern "C" __global__ void test_loadonly(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    float x = in[lid];
    out[lid] = x + 1.0f;
}
