/* Check that suffixed literals keep their width through the IR,
   not just through the constant folder. Separate stores defeat
   folding so we can read the literal type directly. */

__global__ void chk_t(unsigned long long *a,
                      unsigned long long *b,
                      long long *c,
                      unsigned int *d)
{
    a[threadIdx.x] = 0xFFFFFFFFULL;
    b[threadIdx.x] = 1ULL << 40;
    c[threadIdx.x] = 1000000000000LL;
    d[threadIdx.x] = 0xDEADBEEFU;
}
