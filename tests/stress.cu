/* stress.cu — Real-world patterns that break compilers for breakfast.
 * Sourced from the dark corners of GitHub where CUDA goes to suffer. */

/* ---- Deeply Nested Control Flow ---- */
/* The "enterprise developer discovered CUDA" pattern. */
__global__ void nested_hell(int *out, int *in, int n, int mode)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int val = in[idx];
        if (mode == 0) {
            if (val > 0) {
                if (val < 100) {
                    out[idx] = val * 2;
                } else {
                    out[idx] = val / 2;
                }
            } else {
                if (val > -100) {
                    out[idx] = val * -1;
                } else {
                    out[idx] = 0;
                }
            }
        } else if (mode == 1) {
            for (int i = 0; i < val; i = i + 1) {
                out[idx] = out[idx] + 1;
            }
        } else {
            while (val > 0) {
                val = val - 1;
                out[idx] = val;
            }
        }
    }
}


/* ---- Multiple Return Paths ---- */
__device__ int multi_return(int x, int y)
{
    if (x == 0) return y;
    if (y == 0) return x;
    if (x < 0) return -x;
    if (y < 0) return -y;
    return x + y;
}


/* ---- Pointer Arithmetic Gymnastics ---- */
__global__ void pointer_fun(float *out, float *a, float *b, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    float *pa = a + idx;
    float *pb = b + idx;
    float *po = out + idx;

    *po = *pa + *pb;

    /* Offset access */
    if (idx + 1 < n) {
        *(po + 1) = *(pa + 1) * 0.5f;
    }
}


/* ---- Struct of Arrays vs Array of Structs ---- */
struct Particle {
    float x;
    float y;
    float vx;
    float vy;
    float mass;
};

__global__ void nbody_step(Particle *particles, int n, float dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float fx = 0.0f;
    float fy = 0.0f;

    /* N-body: O(n^2) because we're not here for efficiency,
     * we're here to stress-test the compiler's loop handling. */
    for (int j = 0; j < n; j = j + 1) {
        if (j == i) continue;

        float dx = particles[j].x - particles[i].x;
        float dy = particles[j].y - particles[i].y;
        float dist2 = dx * dx + dy * dy + 0.0001f;

        /* 1/sqrt approximation: good enough for government work. */
        float inv_dist = 1.0f / dist2;
        float force = particles[j].mass * inv_dist;

        fx = fx + force * dx;
        fy = fy + force * dy;
    }

    particles[i].vx = particles[i].vx + fx * dt;
    particles[i].vy = particles[i].vy + fy * dt;
    particles[i].x = particles[i].x + particles[i].vx * dt;
    particles[i].y = particles[i].y + particles[i].vy * dt;
}


/* ---- Chained Function Calls ---- */
__device__ float clamp(float val, float lo, float hi)
{
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

__device__ float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

__device__ float smoothstep(float edge0, float edge1, float x)
{
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__global__ void apply_smoothstep(float *out, float *in, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    float val = in[idx];
    out[idx] = smoothstep(0.0f, 1.0f, val);
}


/* ---- Bit Manipulation ---- */
__global__ void bitwise_kernel(int *out, int *in, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    int val = in[idx];
    int popcount = 0;
    int tmp = val;

    /* Manual popcount because we enjoy pain. */
    for (int bit = 0; bit < 32; bit = bit + 1) {
        popcount = popcount + (tmp & 1);
        tmp = tmp >> 1;
    }

    /* Bit reversal because why not. */
    int reversed = 0;
    tmp = val;
    for (int bit = 0; bit < 32; bit = bit + 1) {
        reversed = (reversed << 1) | (tmp & 1);
        tmp = tmp >> 1;
    }

    out[idx] = popcount + reversed;
}


/* ---- Do-While (rare but legal) ---- */
__global__ void do_while_test(int *out, int *in, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    int val = in[idx];
    int count = 0;
    do {
        val = val / 2;
        count = count + 1;
    } while (val > 0);

    out[idx] = count;
}


/* ---- Switch With Fallthrough ---- */
__device__ int classify(int x)
{
    int result = 0;
    switch (x % 4) {
        case 0: result = 100; break;
        case 1: result = 200; break;
        case 2: result = 300; break;
        default: result = -1; break;
    }
    return result;
}

__global__ void switch_test(int *out, int *in, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    out[idx] = classify(in[idx]);
}


/* ---- Large Struct Passing ---- */
struct Matrix3x3 {
    float m[9];
};

__device__ Matrix3x3 mat_mul(Matrix3x3 a, Matrix3x3 b)
{
    Matrix3x3 c;
    for (int i = 0; i < 3; i = i + 1)
        for (int j = 0; j < 3; j = j + 1) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k = k + 1)
                sum = sum + a.m[i * 3 + k] * b.m[k * 3 + j];
            c.m[i * 3 + j] = sum;
        }
    return c;
}

__global__ void transform_kernel(float *out, float *in, float *mat_data, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    Matrix3x3 m;
    for (int i = 0; i < 9; i = i + 1)
        m.m[i] = mat_data[i];

    /* Self-multiply: m = m * m. Exercising struct pass-by-value. */
    Matrix3x3 m2 = mat_mul(m, m);
    out[idx] = m2.m[0] * in[idx];
}
