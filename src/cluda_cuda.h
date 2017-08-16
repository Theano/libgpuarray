#define local_barrier() __syncthreads()
#define WITHIN_KERNEL extern "C" __device__
#define KERNEL extern "C" __global__
#define GLOBAL_MEM /* empty */
#define LOCAL_MEM __shared__
#define LOCAL_MEM_ARG /* empty */
#ifdef NAN
#undef NAN
#endif
#define NAN __int_as_float(0x7fffffff)
/* NULL */
#ifdef INFINITY
#undef INFINITY
#endif
#define INFINITY __int_as_float(0x7f800000)
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

#define ga_half2float(p) __half2float((p).data)
__device__ static inline ga_half ga_float2half(float f) {
  ga_half r;
  r.data = __float2half_rn(f);
  return r;
}

#define gen_atom_add(name, argtype, wtype)              \
  __device__ argtype name(argtype *addr, argtype val) { \
    union {                                             \
      argtype a;                                        \
      wtype w;                                          \
    } p, n;                                             \
    p.a = *addr;                                        \
    do {                                                \
      n.a = p.a + val;                                  \
      p.w = atomicCAS((wtype *)addr, p.w, n.w);         \
    } while (p.w != n.w);                               \
    return n.a;                                         \
  }

#define gen_atom32_add(name, argtype) gen_atom_add(name, argtype, unsigned int)
#define gen_atom64_add(name, argtype) gen_atom_add(name, argtype, unsigned long long)

#define gen_atom_xchg(name, argtype, wtype)              \
  __device__ argtype name(argtype *addr, argtype val) { \
    union {                                             \
      argtype a;                                        \
      wtype w;                                          \
    } n, p;                                             \
    n.a = val;                                          \
    p.w = atomicExch((wtype *)addr, n.w);               \
    return p.a;                                         \
  }

#define gen_atom32_xchg(name, argtype) gen_atom_xchg(name, argtype, unsigned int)
#define gen_atom64_xchg(name, argtype) gen_atom_xchg(name, argtype, unsigned long long)

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
gen_atom64_add(atom_add_lg, ga_long)
#define atom_add_ll(a, b) atom_add_lg(a, b)
gen_atom64_xchg(atom_xchg_lg, ga_long)
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
gen_atom64_add(atom_add_dg, ga_double)
#define atom_add_dl(a, b) atom_add_dg(a, b)
#else
#define atom_add_dg(a, b, c) atomicAdd(a, b, c)
#define atom_add_dl(a, b, c) atomicAdd(a, b, c)
#endif
gen_atom64_xchg(atom_xchg_dg, ga_double)
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
    sum = __float2half_rn(__half2float(val.data) + __half2float(tmp.data));
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
