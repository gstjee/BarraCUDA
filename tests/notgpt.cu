/* notgpt.cu — A "large language model" written by a large language model.
 *
 * This is what happens when you ask an AI to write CUDA. Every kernel
 * here was conceived by a neural network that has never once felt the
 * warmth of a GPU dying under its code. Caveat emptor.
 *
 * Features exercised: shared memory, atomics, warp shuffles, barriers,
 * vector types, launch bounds, cooperative groups, half precision,
 * operator overloading, and an alarming amount of hubris.
 */

/* ---- The Obligatory Matrix Multiply ---- */
/* Because no CUDA demo is complete without one. This is the "I read
 * the NVIDIA programming guide once on a train" version with shared
 * memory tiling. It will produce correct results for square matrices
 * whose dimensions are multiples of 16, which is to say, it will
 * produce correct results almost never in production. */

#define TILE 16

__launch_bounds__(256)
__global__ void sgemm_tiled(float *C, float *A, float *B, int N)
{
    __shared__ float As[TILE * TILE];
    __shared__ float Bs[TILE * TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = N / TILE;
    for (int t = 0; t < numTiles; t++) {
        /* Load tile into shared memory. The coalescing here is
         * acceptable. The variable naming is not. */
        As[threadIdx.y * TILE + threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y * TILE + threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        /* The inner product. Sixteen multiply-adds, each one a tiny
         * prayer to the silicon gods of throughput. */
        for (int k = 0; k < TILE; k++)
            sum = sum + As[threadIdx.y * TILE + k] * Bs[k * TILE + threadIdx.x];

        __syncthreads();
    }

    C[row * N + col] = sum;
}


/* ---- Warp Reduction ---- */
/* The "my thesis advisor said use shuffle instructions" approach.
 * Reduces an array by having each warp argue amongst itself about
 * who holds the largest value, then the block argues, then we
 * write one number and call it a day. Democracy in silicon. */

__launch_bounds__(256)
__global__ void warp_reduce_sum(float *out, float *in, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.0f;
    if (tid < n)
        val = in[tid];

    /* Warp-level reduction. Each iteration halves the participants,
     * like a particularly ruthless game show. */
    int mask = 0xFFFFFFFF;
    val = val + __shfl_down_sync(mask, val, 16);
    val = val + __shfl_down_sync(mask, val, 8);
    val = val + __shfl_down_sync(mask, val, 4);
    val = val + __shfl_down_sync(mask, val, 2);
    val = val + __shfl_down_sync(mask, val, 1);

    /* Lane 0 of each warp deposits its hard-won sum into shared memory,
     * where it will be immediately ignored by all but one thread. */
    __shared__ float warp_sums[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0)
        warp_sums[warp_id] = val;
    __syncthreads();

    /* First warp reduces the warp sums. Everyone else just watches. */
    if (warp_id == 0) {
        val = 0.0f;
        if (lane < 8)
            val = warp_sums[lane];
        val = val + __shfl_down_sync(mask, val, 4);
        val = val + __shfl_down_sync(mask, val, 2);
        val = val + __shfl_down_sync(mask, val, 1);
    }

    if (threadIdx.x == 0)
        atomicAdd(out, val);
}


/* ---- Histogram ---- */
/* The "just atomicAdd and pray" approach to computing histograms.
 * This is the moral equivalent of solving a parking problem by
 * having everyone drive at the same spot simultaneously. Yet it
 * works, because atomics are patient and GPUs are fast. */

__launch_bounds__(256)
__global__ void histogram_256(int *hist, unsigned char *data, int n)
{
    __shared__ int local_hist[256];

    /* Clear shared histogram. Each thread clears one bin.
     * If blockDim.x != 256, we have a problem. We do not
     * check for this because living dangerously is the only
     * way to truly live. */
    local_hist[threadIdx.x] = 0;
    __syncthreads();

    /* Each thread processes multiple elements because we believe
     * in giving GPUs a proper workload, not this namby-pamby
     * one-element-per-thread business. */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i = i + stride)
        atomicAdd(&local_hist[data[i]], 1);

    __syncthreads();

    /* Flush to global. The atomic here is necessary because other
     * blocks are also doing this. It's atomics all the way down. */
    atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
}


/* ---- Prefix Sum (Blelloch) ---- */
/* The up-sweep / down-sweep scan, named after Guy Blelloch who
 * presumably had better things to do than debug off-by-one errors
 * in shared memory indexing. We do not. */

__launch_bounds__(128)
__global__ void blelloch_scan(int *out, int *in, int n)
{
    __shared__ int temp[256];
    int tid = threadIdx.x;

    /* Load two elements per thread because we're greedy like that. */
    int ai = tid;
    int bi = tid + 128;
    temp[ai] = in[ai + blockIdx.x * 256];
    temp[bi] = in[bi + blockIdx.x * 256];

    /* Up-sweep: the "reduce but remember everything" phase.
     * Each iteration doubles the stride while halving the
     * number of threads doing useful work. Efficiency! */
    int offset = 1;
    for (int d = 128; d > 0; d = d / 2) {
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            temp[bi2] = temp[bi2] + temp[ai2];
        }
        offset = offset * 2;
    }

    /* Plant the identity at the root. This is the most important
     * zero assignment in the entire kernel. */
    if (tid == 0)
        temp[255] = 0;

    /* Down-sweep: the "undo everything you just did but sideways" phase. */
    for (int d = 1; d < 256; d = d * 2) {
        offset = offset / 2;
        __syncthreads();
        if (tid < d) {
            int ai2 = offset * (2 * tid + 1) - 1;
            int bi2 = offset * (2 * tid + 2) - 1;
            int t = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] = temp[bi2] + t;
        }
    }

    __syncthreads();
    out[ai + blockIdx.x * 256] = temp[ai];
    out[bi + blockIdx.x * 256] = temp[bi];
}


/* ---- 1D Stencil ---- */
/* A 3-point stencil that computes the average of neighbours.
 * This is technically a Jacobi iteration, but calling it that
 * makes it sound like we know what we're doing. The halo cells
 * are handled with all the grace of a hippo on roller skates. */

__launch_bounds__(256)
__global__ void stencil_1d(float *out, float *in, int n)
{
    __shared__ float smem[258];

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    /* Load interior + halo. Thread 0 loads left halo, thread 255
     * loads right halo. Everyone else just loads their own element
     * and tries not to make eye contact. */
    smem[threadIdx.x + 1] = (gid < n) ? in[gid] : 0.0f;
    if (threadIdx.x == 0)
        smem[0] = (gid > 0) ? in[gid - 1] : 0.0f;
    if (threadIdx.x == 255)
        smem[257] = (gid + 1 < n) ? in[gid + 1] : 0.0f;

    __syncthreads();

    if (gid < n) {
        float left  = smem[threadIdx.x];
        float mid   = smem[threadIdx.x + 1];
        float right = smem[threadIdx.x + 2];
        out[gid] = 0.25f * left + 0.5f * mid + 0.25f * right;
    }
}


/* ---- Vector Maths ---- */
/* Because sometimes you need to remind yourself that float4
 * exists and is not just a fever dream. This kernel does
 * absolutely nothing useful but exercises the vector type
 * system with the enthusiasm of a golden retriever. */

struct vec2 {
    float x;
    float y;
};

__device__ vec2 operator+(vec2 a, vec2 b)
{
    vec2 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

__device__ vec2 operator*(vec2 a, float s)
{
    vec2 r;
    r.x = a.x * s;
    r.y = a.y * s;
    return r;
}

__global__ void vector_party(float *out, float *in, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    vec2 a;
    a.x = in[idx * 2];
    a.y = in[idx * 2 + 1];

    vec2 b;
    b.x = 1.0f;
    b.y = 2.0f;

    vec2 c = a + b;
    vec2 d = c * 3.0f;

    out[idx * 2]     = d.x;
    out[idx * 2 + 1] = d.y;
}


/* ---- Half Precision: For When 32 Bits Is Just Too Many ---- */
/* This kernel converts floats to halfs, does some arithmetic that
 * would make a numerical analyst weep, then converts back. The
 * precision loss is a feature, not a bug. At least that's what
 * we tell the machine learning people. */

__launch_bounds__(256)
__global__ void half_precision_yolo(float *out, float *in, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    __half h = __float2half(in[idx]);
    float back = __half2float(h);

    /* The difference between in[idx] and back is what we call
     * "acceptable quantisation error" in polite company and
     * "oh god my gradients" in practice. */
    out[idx] = back;
}


/* ---- Cooperative Groups: Bureaucracy for Threads ---- */
/* Cooperative groups let you name the thing you were already doing.
 * Instead of __syncthreads(), you get tb.sync(), which is the same
 * thing but with an object-oriented veneer that makes the C++
 * committee feel warm inside. */

namespace cooperative_groups {
    struct thread_block {};
    thread_block this_thread_block();
}

__global__ void coop_reduction(float *out, float *in, int n)
{
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();

    __shared__ float smem[256];
    int tid = block.thread_rank();
    int gid = tid + blockIdx.x * block.size();

    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    block.sync();

    /* Tree reduction in shared memory. Each iteration murders half
     * the active threads. Natural selection, GPU edition. */
    for (int s = block.size() / 2; s > 0; s = s / 2) {
        if (tid < s)
            smem[tid] = smem[tid] + smem[tid + s];
        block.sync();
    }

    if (tid == 0)
        atomicAdd(out, smem[0]);
}


/* ---- The Grand Finale: All Features At Once ---- */
/* This kernel exists solely to make the compiler sweat. It uses
 * shared memory, atomics, barriers, launch bounds, vectors,
 * half precision, and questionable life choices — all in one
 * function. If this compiles, we deserve a biscuit. */

__launch_bounds__(128, 4)
__global__ void kitchen_sink(float *out, float *in, int *counters, int n)
{
    __shared__ float tile[128];

    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x;

    /* Load and convert to half because we enjoy suffering. */
    __half h = __float2half((gid < n) ? in[gid] : 0.0f);
    float val = __half2float(h);
    tile[tid] = val;
    __syncthreads();

    /* Warp reduction of the tile segment. */
    int mask = 0xFFFFFFFF;
    val = val + __shfl_down_sync(mask, val, 16);
    val = val + __shfl_down_sync(mask, val, 8);
    val = val + __shfl_down_sync(mask, val, 4);
    val = val + __shfl_down_sync(mask, val, 2);
    val = val + __shfl_down_sync(mask, val, 1);

    /* Lane 0 contributes to global sum and bumps the counter.
     * The counter serves no purpose except to exercise atomicAdd
     * on integers, because we are thorough like that. */
    int lane = tid % 32;
    if (lane == 0) {
        atomicAdd(out, val);
        atomicAdd(counters, 1);
    }

    /* Struct operations, because why not at this point. */
    vec2 v;
    v.x = tile[tid];
    v.y = tile[127 - tid];
    vec2 w;
    w.x = 0.5f;
    w.y = 0.5f;
    vec2 avg = v * 0.5f + w;

    if (gid < n) {
        out[gid] = avg.x + avg.y;
    }
}
