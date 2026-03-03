/* tg_vec4.cu — real tinygrad: vectorized elementwise add (float4) */
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void __launch_bounds__(4) E_4_4(float* data0, float* data1, float* data2) {
  int lidx0 = threadIdx.x; /* 4 */
  int alu0 = (lidx0<<2);
  float4 val0 = *((float4*)((data1+alu0)));
  float4 val1 = *((float4*)((data2+alu0)));
  *((float4*)((data0+alu0))) = make_float4((val0.x+val1.x),(val0.y+val1.y),(val0.z+val1.z),(val0.w+val1.w));
}
