#include <stdio.h>

enum Color { RED, GREEN = 5, BLUE };

struct Vec3 {
    float x, y, z;
};

typedef unsigned int uint;

template<typename T>
__global__ void scale(T *data, T factor, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = data[i] * factor;
}

template<typename T, int N>
__device__ T dotProduct(const T *a, const T *b)
{
    T sum = (T)0;
    for (int i = 0; i < N; i++)
        sum += a[i] * b[i];
    return sum;
}

__shared__ float smem[256];

__device__ __constant__ float coeffs[16];

static inline __host__ __device__ float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

int main(void)
{
    struct Vec3 v;
    v.x = 1.0f;
    v.y = 2.0f;
    v.z = 3.0f;

    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));

    scale<<<4, 256>>>(d_data, 2.0f, 1024);

    enum Color c = RED;
    switch (c) {
    case RED:
        printf("red\n");
        break;
    case GREEN:
        printf("green\n");
        break;
    default:
        printf("other\n");
        break;
    }

    cudaFree(d_data);
    return 0;
}
