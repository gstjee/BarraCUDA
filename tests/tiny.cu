/* tiny.cu — absolute minimum kernel for MI300X testing.
 * If THIS fails, the issue is ELF structure. If it passes, it's resource counts. */
extern "C" __global__ void tiny(float *out) {
    out[threadIdx.x] = 1.0f;
}
