/* test_debug_add_runner.hip — reads out[0]=n, out[1]=lsz, out[2]=n+lsz */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
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
    HIP_CHECK(hipModuleGetFunction(&kern, mod, "test_debug_add"));
    float h_out[N];
    for (int i = 0; i < N; i++) h_out[i] = -1.0f;
    float h_in[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)i;
    float *d_out, *d_in;
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_in,  N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out, h_out, N * sizeof(float), hipMemcpyHostToDevice));
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
    HIP_CHECK(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));
    printf("out[0] = %.6f  (expected 64.0 = n)\n", h_out[0]);
    printf("out[1] = %.6f  (expected 64.0 = lsz)\n", h_out[1]);
    printf("out[2] = %.6f  (expected 128.0 = n+lsz)\n", h_out[2]);
    hipFree(d_out); hipFree(d_in); hipModuleUnload(mod);
    return 0;
}
