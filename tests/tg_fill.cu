/* tg_fill.cu — real tinygrad: constant fill (vectorized) */
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void __launch_bounds__(1) E_25_4n1(float* data0) {
  int gidx0 = blockIdx.x; /* 25 */
  *((float4*)((data0+(gidx0<<2)))) = make_float4(3.0f,3.0f,3.0f,3.0f);
}
