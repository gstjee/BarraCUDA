/* tinygrad_add.cu — tinygrad-style kernel: extern "C", suffix __launch_bounds__ */

extern "C" __global__ void __launch_bounds__(256) r_4(const float* data0, float* data1) {
    int gidx0 = blockIdx.x;
    float val = data0[gidx0] + 1.0f;
    data1[gidx0] = val;
}
