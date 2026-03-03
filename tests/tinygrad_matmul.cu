/* tinygrad_matmul.cu — tinygrad-style with __int_as_float / __float_as_int */

extern "C" __global__ void __launch_bounds__(256) r_8(const float* data0, float* data1, int n) {
    int gidx0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx0 < n) {
        float acc = __int_as_float(0x7f800000);  /* INFINITY */
        float nan_val = __int_as_float(0x7fc00000);  /* NAN */
        int bits = __float_as_int(data0[gidx0]);
        float val = (bits != 0) ? acc : nan_val;
        data1[gidx0] = val + data0[gidx0];
    }
}
