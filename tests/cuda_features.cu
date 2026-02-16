/*
 * cuda_features.cu — exercises all 8 frontend gaps
 *
 * Gap 1: CUDA builtin functions (barriers, atomics, warp ops)
 * Gap 2: Function calls with >5 args (overflow mode)
 * Gap 3: Goto/label lowering
 * Gap 4: Init list / compound initializers
 * Gap 5: Global variable initializers
 * Gap 6: Switch → BIR_SWITCH
 * Gap 7: Short-circuit && and ||
 * Gap 8: Array size from enum / template int param
 */

/* --- Gap 5: Global with initializer (float lit is unambiguous) --- */
__device__ float g_scale = 2.5f;
__device__ float g_bias = 0.5f;

/* --- Gap 8: Enum-sized array --- */
enum { BLOCK_SIZE = 256 };
__shared__ float shared_buf[BLOCK_SIZE];

/* --- Gap 4: Struct for init-list test --- */
struct Vec3 {
    float x, y, z;
};

/* --- Gap 1: CUDA builtins --- */
__device__ void test_builtins(int *data, int n)
{
    /* Barriers */
    __syncthreads();
    __threadfence();

    /* Atomics */
    atomicAdd(&data[0], 1);
    atomicSub(&data[1], 1);
    atomicMin(&data[2], n);
    atomicMax(&data[3], n);
    atomicExch(&data[4], 99);
    atomicAnd(&data[5], 0xFF);
    atomicOr(&data[6], 0x01);
    atomicXor(&data[7], 0xAA);
    atomicCAS(&data[8], 0, 1);

    /* Warp vote */
    unsigned int mask = 0xFFFFFFFF;
    int pred = (threadIdx.x < 16);
    int ballot = __ballot_sync(mask, pred);
    int any    = __any_sync(mask, pred);
    int all    = __all_sync(mask, pred);

    /* Warp shuffle */
    int val = data[threadIdx.x];
    int s0 = __shfl_sync(mask, val, 0, 32);
    int s1 = __shfl_up_sync(mask, val, 1, 32);
    int s2 = __shfl_down_sync(mask, val, 1, 32);
    int s3 = __shfl_xor_sync(mask, val, 1, 32);
    data[threadIdx.x] = s0 + s1 + s2 + s3;
}

/* --- Gap 2: Function with many args (overflow call) --- */
__device__ int many_args(int a, int b, int c, int d, int e, int f, int g)
{
    return a + b + c + d + e + f + g;
}

__device__ void test_many_args(int *out)
{
    out[0] = many_args(1, 2, 3, 4, 5, 6, 7);
}

/* --- Gap 3: Goto/label --- */
__device__ int test_goto(int *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        goto done;
    data[i] = data[i] + 1;
done:
    return i;
}

/* --- Gap 4: Init lists --- */
__device__ void test_init_list(float *out)
{
    int arr[4] = {10, 20, 30, 40};
    out[0] = (float)arr[0];
    out[1] = (float)arr[1];

    struct Vec3 v = {1.0f, 2.0f, 3.0f};
    out[2] = v.x;
    out[3] = v.z;
}

/* --- Gap 6: Switch --- */
__device__ int test_switch(int sel)
{
    int result;
    switch (sel) {
    case 0:
        result = 100;
        break;
    case 1:
        result = 200;
        break;
    case 5:
        result = 500;
        break;
    default:
        result = -1;
        break;
    }
    return result;
}

/* --- Gap 7: Short-circuit && and || --- */
__device__ int test_short_circuit(int *ptr, int n)
{
    /* && : if ptr is null, should NOT dereference */
    int a = (ptr != 0) && (*ptr > 0);

    /* || : if n > 0, should NOT evaluate second */
    int b = (n > 0) || (ptr != 0);

    return a + b;
}

/* --- Gap 8: Template with enum-based array --- */
template<typename T>
__device__ T test_template_sum(const T *input, int n)
{
    T sum = (T)0;
    for (int i = 0; i < n; i++)
        sum += input[i];
    return sum;
}

/* Host main: triggers template instantiation */
int main(void)
{
    float *d;
    cudaMalloc(&d, 1024 * sizeof(float));
    test_template_sum<<<1, 1>>>(d, 2.0f, 1024);
    cudaFree(d);
    return 0;
}
