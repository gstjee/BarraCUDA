/* test_scopy — 6 pointer params + scratch, close to Moa pattern.
 * out[tid] = a[tid] + b[tid] + c[tid] + d[tid]
 * Expected: each array has value (tid + offset), sum = 4*tid + 10 */
__global__ void test_scopy(float *out, float *a, float *b,
                           float *c, float *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
    float tmp[4];
    tmp[0] = a[tid];
    tmp[1] = b[tid];
    tmp[2] = c[tid];
    tmp[3] = d[tid];

    out[tid] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
}
