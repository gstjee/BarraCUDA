/* test_preproc.cu — exercises preprocessor features */

/* --- Object-like macros --- */
#define BLOCK_SIZE 256
#define PI 3.14159f
#define EMPTY

/* --- Function-like macros --- */
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(v, lo, hi) MIN(MAX(v, lo), hi)
#define SQUARE(x) ((x) * (x))
#define KERNEL_LAUNCH(fn, n) fn<<<((n)+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>

/* --- Stringification --- */
#define STR(x) #x
#define XSTR(x) STR(x)

/* --- Token pasting --- */
#define CONCAT(a, b) a ## b

/* --- Conditional compilation: #ifdef/#ifndef --- */
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifndef CUSTOM_BLOCK_SIZE
#define THREAD_COUNT BLOCK_SIZE
#else
#define THREAD_COUNT CUSTOM_BLOCK_SIZE
#endif

/* --- Conditional compilation: #if/#elif/#else --- */
#if __CUDA_ARCH__ >= 1100
#define HAS_RDNA3 1
#elif __CUDA_ARCH__ >= 1000
#define HAS_RDNA2 1
#else
#define HAS_RDNA3 0
#endif

/* --- Nested conditionals --- */
#if defined(__BARRACUDA__)
#if HAS_RDNA3
#define TARGET_NAME "RDNA3 (gfx1100)"
#else
#define TARGET_NAME "Unknown"
#endif
#endif

/* --- #undef --- */
#define TEMP_VAL 42
#undef TEMP_VAL
#define TEMP_VAL 99

/* --- Test: use the macros --- */
__shared__ float smem[BLOCK_SIZE];

DEVICE float compute(float x)
{
    float val = SQUARE(x);
    float clamped = CLAMP(val, 0.0f, PI);
    return MAX(clamped, 0.0f);
}

__global__ void kernel(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = compute(data[i]);
        smem[threadIdx.x] = data[i];
    }
}

/* --- #if 0 block (should be stripped) --- */
#if 0
THIS SHOULD NOT APPEAR IN OUTPUT
int syntax_error@@@@;
#endif

/* --- Test TEMP_VAL was redefined --- */
int test_val = TEMP_VAL;

/* --- Test THREAD_COUNT defaults to BLOCK_SIZE --- */
int threads = THREAD_COUNT;

/* --- Test nested macro expansion --- */
int max_val = MAX(BLOCK_SIZE, THREAD_COUNT);

int main(void)
{
    float *d;
    cudaMalloc(&d, 1024 * sizeof(float));
    kernel<<<4, BLOCK_SIZE>>>(d, 1024);
    cudaFree(d);
    return 0;
}
