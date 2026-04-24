/* Int literal suffixes: ull / ll / ul / l / u.
   Previously every literal lowered to i32 regardless of suffix,
   so 0xFFFFFFFFull, 1ull<<40, etc. silently truncated. */

__global__ void sfx_t(unsigned long long *out)
{
    unsigned long long a = 0xFFFFFFFFULL;
    unsigned long long b = 1ULL << 40;
    long long          c = 1000000000000LL;
    unsigned int       d = 0xDEADBEEFU;
    unsigned long      e = 0xCAFEBABEUL;

    out[threadIdx.x] = a + b + (unsigned long long)c
                     + (unsigned long long)d
                     + (unsigned long long)e;
}
