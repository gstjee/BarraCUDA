/* test_mfma_runner.hip — run 4x4x1 f32 MFMA, check lane 0 output.
 * With a=2.0, b=3.0, acc=0: first element of result should be a*b = 6.0 */
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
    HIP_CHECK(hipModuleGetFunction(&kern, mod, "test_mfma"));
    float h_out[N];
    for (int i = 0; i < N; i++) h_out[i] = -1.0f;
    float *d_out;
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));
    float a = 2.0f, b = 3.0f;
    struct { float *out; float a; float b; } args = { d_out, a, b };
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
    /* Lane 0 should have a*b = 6.0 in the first MFMA result element */
    int pass = 0, fail = 0;
    float got = h_out[0];
    float expected = a * b;
    if (fabsf(got - expected) < 0.01f) { pass = 1; }
    else { printf("[FAIL] out[0]: got %.4f expected %.4f\n", got, expected); fail = 1; }
    printf("=== %d/%d passed", pass, pass + fail);
    if (fail) printf(", %d FAILED", fail);
    printf(" ===\n");
    hipFree(d_out); hipModuleUnload(mod);
    return fail ? 1 : 0;
}
