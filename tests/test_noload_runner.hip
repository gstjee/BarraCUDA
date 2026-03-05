/* test_noload_runner.hip — expects out[i] = (float)(n + lsz) = 128.0 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 64
#define BLOCK 64
#define HIP_CHECK(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d: %s at %s:%d\n", \
                e, hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file.hsaco>\n", argv[0]); return 1; }
    hipModule_t mod;
    HIP_CHECK(hipModuleLoad(&mod, argv[1]));
    hipFunction_t kern;
    HIP_CHECK(hipModuleGetFunction(&kern, mod, "test_noload"));
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) { h_in[i] = (float)i; h_out[i] = -1.0f; }
    float *d_out, *d_in;
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_in,  N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice));
    int n = N;
    struct { float *out; float *in; int n; } args = { d_out, d_in, n };
    size_t arg_size = sizeof(args);
    void *config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
        HIP_LAUNCH_PARAM_END
    };
    HIP_CHECK(hipModuleLaunchKernel(kern, 1,1,1, BLOCK,1,1, 0,0, NULL, config));
    HIP_CHECK(hipDeviceSynchronize());
    printf("[ok] kernel launched\n");
    HIP_CHECK(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));
    int pass = 0, fail = 0;
    for (int i = 0; i < N; i++) {
        float expected = 128.0f; /* (float)(n + lsz) where n=64, lsz=64 */
        if (fabsf(h_out[i] - expected) < 0.001f) pass++;
        else { if (fail < 5) printf("[FAIL] i=%d: got %.6f expected %.6f\n", i, h_out[i], expected); fail++; }
    }
    printf("=== %d/%d passed", pass, N);
    if (fail) printf(", %d FAILED", fail);
    printf(" ===\n");
    hipFree(d_out); hipFree(d_in); hipModuleUnload(mod);
    return fail ? 1 : 0;
}
