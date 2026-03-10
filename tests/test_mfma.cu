/* test_mfma.cu — prove v_mfma_f32_4x4x1f32 encodes and runs.
 * We pass scalar a, b and a single-float accumulator.  The HW
 * writes 4 VGPRs but we only read the first element back — good
 * enough to confirm the opcode doesn't trap. */
extern "C" __device__ int __ockl_get_local_id(int);

extern "C" __global__ void test_mfma(float *out, float a, float b) {
    float acc = 0.0f;
    float res = __builtin_amdgcn_mfma_f32_4x4x1_f32(a, b, acc);
    int lid = __ockl_get_local_id(0);
    out[lid] = res;
}
