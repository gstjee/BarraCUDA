/* test_dce.cu — Dead code elimination test cases.
 *
 * Each kernel targets a specific DCE edge case.
 * The test harness compiles with --ir and checks which
 * instructions survive. */

/* Dead chain: dead1 unused, dead2 depends only on dead1 */
__global__ void dce_chain(int *out, int a, int b) {
    int live = a + b;
    int dead1 = a * b;
    int dead2 = dead1 + 1;
    out[0] = live;
}

/* Non-volatile load dies when unused */
__global__ void dce_load(int *src) {
    int unused_load = src[0];
}

/* Params always survive */
__global__ void dce_params(int *out, int a, int b, int c) {
    out[0] = a;
}

/* Side effects survive: store, barrier */
__global__ void dce_side_effects(int *out, int x) {
    out[0] = x;
    __syncthreads();
}

/* Empty function body */
__global__ void dce_empty(void) {
}

/* No dead code — everything is live */
__global__ void dce_all_live(float *out, const float *a, const float *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = a[i] + b[i];
}
