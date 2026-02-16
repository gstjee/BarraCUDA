/* Test unsigned operations — verify sema wires through to BIR */

__device__ unsigned int div_rem(unsigned int a, unsigned int b,
                                unsigned int *out_rem) {
    *out_rem = a % b;
    return a / b;
}

__device__ unsigned int shift_right(unsigned int x) {
    return x >> 1;
}

__device__ int unsigned_cmp(unsigned int a, unsigned int b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

__device__ float uint_to_float(unsigned int x) {
    return (float)x;
}

__device__ unsigned int float_to_uint(float f) {
    return (unsigned int)f;
}

__device__ int widen_uchar(unsigned char c) {
    return (int)c;
}
