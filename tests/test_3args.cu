/* test_3args.cu — test 3 args with plain threadIdx.x (no dispatch_ptr) */
extern "C" __global__ void test_3args(float *out, float *in, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + 1.0f;
    }
}
