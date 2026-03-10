/* test_sgpr_dump.cu — dump system SGPRs to verify dispatch+kernarg layout */
extern "C" __device__ int __ockl_get_local_id(int);
extern "C" __device__ int __ockl_get_local_size(int);

extern "C" __global__ void test_sgpr_dump(float *out, float *in, int n) {
    /* Thread 0 writes a diagnostic pattern:
       out[0] = (float)n          → should be 64
       out[1] = (float)local_size → should be 64
       out[2] = in[0]            → should be 0.5  (our test input)
       out[3] = in[1]            → should be 0.6  */
    int lid = __ockl_get_local_id(0);
    if (lid == 0) {
        out[0] = (float)n;
        out[1] = (float)__ockl_get_local_size(0);
        out[2] = in[0];
        out[3] = in[1];
    }
}
