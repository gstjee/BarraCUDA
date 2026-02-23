__global__ void matmul_tile(float *C, float *A, float *B, int N)
{
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;

    float sum = 0.0f;
    As[ty][tx] = A[row * N + col];
    Bs[ty][tx] = B[row * N + col];
    __syncthreads();

    for (int k = 0; k < 16; k++)
        sum += As[ty][k] * Bs[k][tx];

    C[row * N + col] = sum;
}
