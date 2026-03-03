/* tg_elem.cu — real tinygrad: scalar elementwise add (unoptimized) */
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void __launch_bounds__(1) E_16(float* data0, float* data1, float* data2) {
  int gidx0 = blockIdx.x; /* 16 */
  float val0 = *(data1+gidx0);
  float val1 = *(data2+gidx0);
  *(data0+gidx0) = (val0+val1);
}
