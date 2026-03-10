/* tg_compat.cu — tinygrad compatibility smoke test
 *
 * Exercises the AMD OCML/OCKL naming conventions that tinygrad
 * emits when generating HIP kernels. If this compiles and
 * llvm-objdump is happy, we can eat tinygrad's output for lunch.
 */

/* ---- Forward declarations (tinygrad style) ---- */
extern "C" __device__ int __ockl_get_local_id(int);
extern "C" __device__ int __ockl_get_group_id(int);
extern "C" __device__ int __ockl_get_local_size(int);
extern "C" __device__ int __ockl_get_num_groups(int);

extern "C" __device__ float __ocml_exp2_f(float);
extern "C" __device__ float __ocml_log2_f(float);
extern "C" __device__ float __ocml_sqrt_f(float);
extern "C" __device__ float __ocml_sin_f(float);
extern "C" __device__ float __ocml_cos_f(float);
extern "C" __device__ float __ocml_fabs_f(float);
extern "C" __device__ float __ocml_floor_f(float);
extern "C" __device__ float __ocml_ceil_f(float);
extern "C" __device__ float __ocml_fmax_f(float, float);
extern "C" __device__ float __ocml_fmin_f(float, float);

extern "C" __device__ _Float16 __ocml_exp2_f16(_Float16);
extern "C" __device__ _Float16 __ocml_sqrt_f16(_Float16);

/* ---- Test kernel: thread model + math ---- */
extern "C" __global__ void tg_smoke(float *out, float *in, int n) {
    int lid = __ockl_get_local_id(0);
    int gid = __ockl_get_group_id(0);
    int lsz = __ockl_get_local_size(0);
    int idx = gid * lsz + lid;

    if (idx < n) {
        float x = in[idx];
        /* exercise every __ocml_ builtin we claim to support */
        float a = __ocml_exp2_f(x);
        float b = __ocml_log2_f(a);
        float c = __ocml_sqrt_f(__ocml_fabs_f(b));
        float d = __ocml_sin_f(c) + __ocml_cos_f(c);
        float e = __ocml_floor_f(d);
        float f = __ocml_ceil_f(d);
        out[idx] = __ocml_fmax_f(e, __ocml_fmin_f(f, x));
    }
}

/* ---- _Float16 kernel ---- */
extern "C" __global__ void tg_half(_Float16 *out, _Float16 *in, int n) {
    int idx = __ockl_get_group_id(0) * __ockl_get_local_size(0)
            + __ockl_get_local_id(0);
    if (idx < n) {
        _Float16 x = in[idx];
        out[idx] = __ocml_sqrt_f16(__ocml_exp2_f16(x));
    }
}
