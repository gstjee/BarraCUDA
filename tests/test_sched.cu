__global__ void sched_loadpair(float *out, const float *a, const float *b) {
    int i = threadIdx.x;
    out[i] = a[i] + b[i];
}
