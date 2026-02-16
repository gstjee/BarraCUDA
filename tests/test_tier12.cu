/* Test Tier 1+2 features: vectors, half, shared, operator overloading */

struct Vec2 {
    float x, y;
};

__device__ Vec2 operator+(Vec2 a, Vec2 b) {
    Vec2 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

__device__ void test_vectors(float* out) {
    float4 v = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;

    int2 iv;
    iv.x = 10;
    iv.y = 20;
    out[4] = (float)(iv.x + iv.y);
}

__device__ void test_half(float* out) {
    __half h = __float2half(1.0f);
    float f = __half2float(h);
    out[0] = f;
}

__global__ void test_shared(float* out, int n) {
    __shared__ float buf[256];
    int tid = threadIdx.x;
    buf[tid] = (float)tid;
    __syncthreads();
    if (tid < n)
        out[tid] = buf[tid];
}

__device__ void test_operator(float* out) {
    Vec2 a;
    a.x = 1.0f;
    a.y = 2.0f;
    Vec2 b;
    b.x = 3.0f;
    b.y = 4.0f;
    Vec2 c = a + b;
    out[0] = c.x;
    out[1] = c.y;
}
