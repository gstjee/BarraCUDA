/* test_dispatch_runner.hip — test dispatch_ptr access */
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
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.hsaco>\n", argv[0]);
        return 1;
    }

    hipModule_t mod;
    HIP_CHECK(hipModuleLoad(&mod, argv[1]));
    printf("[ok] loaded %s\n", argv[1]);

    hipFunction_t kern;
    HIP_CHECK(hipModuleGetFunction(&kern, mod, "test_dispatch"));
    printf("[ok] found test_dispatch\n");

    float h_out[N];
    for (int i = 0; i < N; i++) h_out[i] = -1.0f;

    float *d_out;
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));

    struct { float *out; } args = { d_out };
    size_t arg_size = sizeof(args);
    void *config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    HIP_CHECK(hipModuleLaunchKernel(kern,
        1, 1, 1,       /* grid */
        BLOCK, 1, 1,   /* block */
        0, 0,          /* shared, stream */
        NULL, config));
    HIP_CHECK(hipDeviceSynchronize());
    printf("[ok] kernel launched and completed\n");

    HIP_CHECK(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));

    int pass = 0, fail = 0;
    for (int i = 0; i < N; i++) {
        if (h_out[i] == 64.0f) {
            pass++;
        } else {
            if (fail < 5)
                printf("[FAIL] i=%d: got %.6f, expected 64.000000\n", i, h_out[i]);
            fail++;
        }
    }

    printf("\n=== %d/%d passed", pass, N);
    if (fail) printf(", %d FAILED", fail);
    printf(" ===\n");

    printf("\nFirst 8 results:\n");
    for (int i = 0; i < 8 && i < N; i++)
        printf("  [%d] out=%.6f\n", i, h_out[i]);

    hipFree(d_out);
    hipModuleUnload(mod);
    return fail ? 1 : 0;
}
