/* Struct copy from global — the operation that stumped us.
 * Buffer starts zeroed (memset by harness).  We bitcast to
 * struct Quad*, copy struct to local, then add known offsets
 * to each field to prove the fields read correctly.
 * Expected: 10.0, 20.0, 30.0, 40.0
 * Run: test_gpu_run test_scopy.hsaco test_scopy 1 */
struct Quad { float a; float b; float c; float d; };

__global__ void test_scopy(float *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    /* Bitcast + struct copy from zeroed global memory.
     * All fields should be 0.0, so we add known constants
     * to prove each field is independently addressable
     * and the struct didn't arrive as a pointer-shaped
     * hallucination. */
    struct Quad *arr = (struct Quad *)out;
    struct Quad q;
    q = arr[0];

    out[0] = q.a + 10.0f;
    out[1] = q.b + 20.0f;
    out[2] = q.c + 30.0f;
    out[3] = q.d + 40.0f;
}
