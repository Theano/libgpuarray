#ifndef CLUDA_H
#define CLUDA_H
#define local_barrier() __syncthreads()
#define WITHIN_KERNEL extern "C" __device__
#define KERNEL extern "C" __global__
#define GLOBAL_MEM /* empty */
#define LOCAL_MEM __shared__
#define LOCAL_MEM_ARG /* empty */
#define MAXFLOAT        3.402823466E+38F
#ifdef NAN
#undef NAN
#endif
#define NAN __int_as_float(0x7fffffff)
/* NULL */
#ifdef INFINITY
#undef INFINITY
#endif
#define INFINITY __int_as_float(0x7f800000)
#define HUGE_VALF INFINITY
#define HUGE_VAL __longlong_as_double(0x7ff0000000000000)

#define M_E            2.7182818284590452354
#define M_LOG2E        1.4426950408889634074
#define M_LOG10E       0.43429448190325182765
#define M_LN2          0.69314718055994530942
#define M_LN10         2.30258509299404568402
#define M_PI           3.14159265358979323846
#define M_PI_2         1.57079632679489661923
#define M_PI_4         0.78539816339744830962
#define M_1_PI         0.31830988618379067154
#define M_2_PI         0.63661977236758134308
#define M_2_SQRTPI     1.12837916709551257390
#define M_SQRT2        1.41421356237309504880
#define M_SQRT1_2      0.70710678118654752440
#define LID_0 threadIdx.x
#define LID_1 threadIdx.y
#define LID_2 threadIdx.z
#define LDIM_0 blockDim.x
#define LDIM_1 blockDim.y
#define LDIM_2 blockDim.z
#define GID_0 blockIdx.x
#define GID_1 blockIdx.y
#define GID_2 blockIdx.z
#define GDIM_0 gridDim.x
#define GDIM_1 gridDim.y
#define GDIM_2 gridDim.z
#define ga_bool unsigned char
#define ga_byte signed char
#define ga_ubyte unsigned char
#define ga_short short
#define ga_ushort unsigned short
#define ga_int int
#define ga_uint unsigned int
#define ga_long long long
#define ga_ulong unsigned long long
#define ga_float float
#define ga_double double
#define ga_size size_t
#define ga_ssize ptrdiff_t
#define GA_DECL_SHARED_PARAM(type, name)
#define GA_DECL_SHARED_BODY(type, name) extern __shared__ type name[];
#define GA_WARP_SIZE warpSize

struct ga_half {
  ga_ushort data;
};

static __device__ inline float ga_half2float(ga_half h) {
  float r;
  asm("{ cvt.f32.f16 %0, %1; }\n" : "=f"(r) : "h"(h.data));
  return r;
}
static __device__ inline ga_half ga_float2half(float f) {
  ga_half r;
  asm("{ cvt.rn.f16.f32 %0, %1; }\n" : "=h"(r.data) : "f"(f));
  return r;
}

/* ga_int */
#define atom_add_ig(a, b) atomicAdd(a, b)
#define atom_add_il(a, b) atomicAdd(a, b)
#define atom_xchg_ig(a, b) atomicExch(a, b)
#define atom_xchg_il(a, b) atomicExch(a, b)
/* ga_uint */
#define atom_add_Ig(a, b) atomicAdd(a, b)
#define atom_add_Il(a, b) atomicAdd(a, b)
#define atom_xchg_Ig(a, b) atomicExch(a, b)
#define atom_xchg_Il(a, b) atomicExch(a, b)
/* ga_long */
__device__ ga_long atom_add_lg(ga_long *addr, ga_long val) {
  unsigned long long *waddr = (unsigned long long *)addr;
  unsigned long long old = *waddr;
  unsigned long long assumed;
  do {
    assumed = old;
    old = atomicCAS(waddr, assumed, (val + (ga_long)(assumed)));
  } while (assumed != old);
  return (ga_long)old;
}
#define atom_add_ll(a, b) atom_add_lg(a, b)
__device__ ga_long atom_xchg_lg(ga_long *addr, ga_long val) {
  unsigned long long res;
  res = atomicExch((unsigned long long *)addr, val);
  return (ga_long)res;
}
#define atom_xchg_ll(a, b) atom_xchg_lg(a, b)
/* ga_ulong */
#define atom_add_Lg(a, b) atomicAdd(a, b)
#define atom_add_Ll(a, b) atomicAdd(a, b)
#define atom_xchg_Lg(a, b) atomicExch(a, b)
#define atom_xchg_Ll(a, b) atomicExch(a, b)
/* ga_float */
#define atom_add_fg(a, b) atomicAdd(a, b)
#define atom_add_fl(a, b) atomicAdd(a, b)
#define atom_xchg_fg(a, b) atomicExch(a, b)
#define atom_xchg_fl(a, b) atomicExch(a, b)
/* ga_double */
#if __CUDA_ARCH__ < 600
__device__ ga_double atom_add_dg(ga_double *addr, ga_double val) {
  unsigned long long *waddr = (unsigned long long *)addr;
  unsigned long long old = *waddr;
  unsigned long long assumed;
  do {
    assumed = old;
    old = atomicCAS(waddr, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#define atom_add_dl(a, b) atom_add_dg(a, b)
#else
#define atom_add_dg(a, b) atomicAdd(a, b)
#define atom_add_dl(a, b) atomicAdd(a, b)
#endif
__device__ ga_double atom_xchg_dg(ga_double *addr, ga_double val) {
  unsigned long long res;
  res = atomicExch((unsigned long long *)addr, __double_as_longlong(val));
  return __longlong_as_double(res);
}
#define atom_xchg_dl(a, b) atom_xchg_dg(a, b)
/* ga_half */
__device__ ga_half atom_add_eg(ga_half *addr, ga_half val) {
  ga_uint *base = (ga_uint *)((ga_size)addr & ~2);
  ga_uint old, assumed, sum, new_;
  ga_half tmp;
  old = *base;
  do {
    assumed = old;
    tmp.data = __byte_perm(old, 0, ((ga_size)addr & 2) ? 0x4432 : 0x4410);
    sum = ga_float2half(ga_half2float(val) + ga_half2float(tmp)).data;
    new_ = __byte_perm(old, sum, ((ga_size)addr & 2) ? 0x5410 : 0x3254);
    old = atomicCAS(base, assumed, new_);
  } while (assumed != old);
  tmp.data = __byte_perm(old, 0, ((ga_size)addr & 2) ? 0x4432 : 0x4410);
  return tmp;
}
#define atom_add_el(a, b) atom_add_eg(a, b)

__device__ ga_half atom_xchg_eg(ga_half *addr, ga_half val) {
  ga_uint *base = (ga_uint *)((ga_size)addr & ~2);
  ga_uint old, assumed, new_;
  ga_half tmp;
  old = *base;
  do {
    assumed = old;
    new_ = __byte_perm(old, val.data, ((ga_size)addr & 2) ? 0x5410 : 0x3254);
    old = atomicCAS(base, assumed, new_);
  } while (assumed != old);
  tmp.data = __byte_perm(old, 0, ((ga_size)addr & 2) ? 0x4432 : 0x4410);
  return tmp;
}
#define atom_xchg_el(a, b) atom_xchg_eg(a, b)
#endif
