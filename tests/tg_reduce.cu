/* tg_reduce.cu — real tinygrad: GPT-2 attention reduction with shared mem */
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void __launch_bounds__(16) r_2_1024_16_64(float* data0, int* data1, int* data2, float* data3, const int start_pos) {
  __shared__ float temp0[16];
  int gidx0 = blockIdx.x; /* 1024 */
  int gidx1 = blockIdx.y; /* 2 */
  int lidx0 = threadIdx.x; /* 16 */
  int val0 = *(data2+(start_pos+gidx1));
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 64; ridx0++) {
    int val1 = *(data1+((lidx0<<6)+ridx0));
    float val2 = *(data3+((ridx0<<10)+gidx0+(lidx0<<16)));
    acc0 = (acc0+(((float)(((val1!=val0)!=1)))*val2));
  }
  *(temp0+lidx0) = acc0;
  __syncthreads();
  if ((((bool)(lidx0))!=1)) {
    float acc1 = 0.0f;
    for (int ridx1 = 0; ridx1 < 16; ridx1++) {
      float val3 = *(temp0+ridx1);
      acc1 = (acc1+val3);
    }
    *(data0+(gidx0+(gidx1<<10))) = acc1;
  }
}
