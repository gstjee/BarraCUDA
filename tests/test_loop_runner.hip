/* test_loop_runner.hip — expects out[i] = n*(n-1)/2 = 2016.0 for n=64 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 64
#define BLOCK 64
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP error %d: %s at %s:%d\n", e, hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file.hsaco>\n", argv[0]); return 1; }
    hipModule_t mod;
    HIP_CHECK(hipModuleLoad(&mod, argv[1]));
    hipFunction_t kern;
    HIP_CHECK(hipModuleGetFunction(&kern, mod, "test_loop"));
    float h_out[N];
    for (int i = 0; i < N; i++) h_out[i] = -1.0f;
    float *d_out;
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));
    int n = N;
    struct { float *out; int n; } args = { d_out, n };
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
    float expected = (float)(N * (N - 1) / 2); /* 2016.0 */
    for (int i = 0; i < N; i++) {
        if (fabsf(h_out[i] - expected) < 0.5f) pass++;
        else { if (fail < 5) printf("[FAIL] i=%d: got %.1f expected %.1f\n", i, h_out[i], expected); fail++; }
    }
    printf("=== %d/%d passed", pass, N);
    if (fail) printf(", %d FAILED", fail);
    printf(" ===\n");
    hipFree(d_out); hipModuleUnload(mod);
    return fail ? 1 : 0;
}
