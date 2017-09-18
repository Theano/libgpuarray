#ifndef CLUDA_H
#define CLUDA_H
#define local_barrier() barrier(CLK_LOCAL_MEM_FENCE)
#define WITHIN_KERNEL /* empty */
#define KERNEL __kernel
#define GLOBAL_MEM __global
#define LOCAL_MEM __local
#define LOCAL_MEM_ARG __local
/* NAN */
#ifndef NULL
  #define NULL ((void*)0)
#endif
/* INFINITY */
#define LID_0 get_local_id(0)
#define LID_1 get_local_id(1)
#define LID_2 get_local_id(2)
#define LDIM_0 get_local_size(0)
#define LDIM_1 get_local_size(1)
#define LDIM_2 get_local_size(2)
#define GID_0 get_group_id(0)
#define GID_1 get_group_id(1)
#define GID_2 get_group_id(2)
#define GDIM_0 get_num_groups(0)
#define GDIM_1 get_num_groups(1)
#define GDIM_2 get_num_groups(2)
#define ga_bool uchar
#define ga_byte char
#define ga_ubyte uchar
#define ga_short short
#define ga_ushort ushort
#define ga_int int
#define ga_uint uint
#define ga_long long
#define ga_ulong ulong
#define ga_float float
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define ga_double double
#endif
#define ga_size ulong
#define ga_ssize long
#define GA_DECL_SHARED_PARAM(type, name) , __local type *name
#define GA_DECL_SHARED_BODY(type, name)
#define GA_WARP_SIZE __GA_WARP_SIZE

typedef struct _ga_half {
  half data;
} ga_half;

#define ga_half2float(p) vload_half(0, &((p).data))
static inline ga_half ga_float2half(ga_float f) {
  ga_half r;
  vstore_half_rtn(f, 0, &r.data);
  return r;
}

#pragma OPENCL_EXTENSION cl_khr_int64_base_atomics: enable

#define gen_atom32_add(name, argtype, aspace)                     \
  argtype name(volatile aspace argtype *, argtype);               \
  argtype name(volatile aspace argtype *addr, argtype val) {      \
    union {                                                       \
      argtype a;                                                  \
      int w;                                                      \
    } p, n;                                                       \
    int a;                                                        \
    p.a = *addr;                                                  \
    do {                                                          \
      a = p.w;                                                    \
      n.a = p.a + val;                                            \
      p.w = atomic_cmpxchg((volatile aspace int *)addr, a, n.w);  \
    } while (p.w != a);                                           \
    return n.a;                                                   \
  }

#define gen_atom64_add(name, argtype, aspace)                     \
  argtype name(volatile aspace argtype *, argtype);               \
  argtype name(volatile aspace argtype *addr, argtype val) {      \
    union {                                                       \
      argtype a;                                                  \
      long w;                                                     \
    } p, n;                                                       \
    long a;                                                       \
    p.a = *addr;                                                  \
    do {                                                          \
      a = p.w;                                                    \
      n.a = p.a + val;                                            \
      p.w = atom_cmpxchg((volatile aspace long *)addr, a, n.w);   \
    } while (p.w != a);                                           \
    return n.a;                                                   \
  }

#define gen_atom64_xchg(name, argtype, aspace)                  \
  argtype name(volatile aspace argtype *, argtype);             \
  argtype name(volatile aspace argtype *addr, argtype val) {    \
    union {                                                     \
      argtype a;                                                \
      long w;                                                   \
    } p, n;                                                     \
    n.a = val;                                                  \
    p.w = atom_xchg((volatile aspace long *)addr, n.w);         \
    return p.a;                                                 \
  }

/* ga_int */
#define atom_add_ig(a, b) atomic_add(a, b)
#define atom_add_il(a, b) atomic_add(a, b)
#define atom_xchg_ig(a, b) atomic_xchg(a, b)
#define atom_xchg_il(a, b) atomic_xchg(a, b)
/* ga_uint */
#define atom_add_Ig(a, b) atomic_add(a, b)
#define atom_add_Il(a, b) atomic_add(a, b)
#define atom_xchg_Ig(a, b) atomic_xchg(a, b)
#define atom_xchg_Il(a, b) atomic_xchg(a, b)
/* ga_float */
gen_atom32_add(atom_add_fg, ga_float, global)
gen_atom32_add(atom_add_fl, ga_float, local)
#define atom_xchg_fg(a, b) atomic_xchg(a, b)
#define atom_xchg_fl(a, b) atomic_xchg(a, b)

#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
/* ga_long */
#define atom_add_lg(a, b) atom_add(a, b)
#define atom_add_ll(a, b) atom_add(a, b)
#define atom_xchg_lg(a, b) atom_xchg(a, b)
#define atom_xchg_ll(a, b) atom_xchg(a, b)
/* ga_ulong */
#define atom_add_Lg(a, b) atom_add(a, b)
#define atom_add_Ll(a, b) atom_add(a, b)
#define atom_xchg_Lg(a, b) atom_xchg(a, b)
#define atom_xchg_Ll(a, b) atom_xchg(a, b)
/* ga_double */
#ifdef cl_khr_fp64
gen_atom64_add(atom_add_dg, ga_double, global)
gen_atom64_add(atom_add_dl, ga_double, local)
gen_atom64_xchg(atom_xchg_dg, ga_double, global)
gen_atom64_xchg(atom_xchg_dl, ga_double, local)
#endif
#endif
/* ga_half */
#define gen_atomh_add(name, aspace)                                     \
  ga_half name(volatile aspace ga_half *addr, ga_half val);             \
  ga_half name(volatile aspace ga_half *addr, ga_half val) {            \
    ga_uint idx = ((ga_size)addr & 2) >> 1;                             \
    volatile aspace int *base = (volatile aspace int *)((ga_size)addr & ~2); \
    union {                                                             \
      int i;                                                            \
      ga_half h[2];                                                     \
    } o, a, n;                                                          \
    float fo;                                                           \
    float fval;                                                         \
    fval = ga_half2float(val);                                          \
    o.i = *base;                                                        \
    do {                                                                \
      a.i = o.i;                                                        \
      fo = ga_half2float(o.h[idx]);                                     \
      n.i = o.i;                                                        \
      n.h[idx] = ga_float2half(fval + fo);                              \
      o.i = atomic_cmpxchg(base, a.i, n.i);                             \
    } while (o.i != a.i);                                               \
    return n.h[idx];                                                    \
  }

#define gen_atomh_xchg(name, aspace)                                    \
  ga_half name(volatile aspace ga_half *addr, ga_half val);             \
  ga_half name(volatile aspace ga_half *addr, ga_half val) {            \
    ga_uint idx = ((ga_size)addr & 2) >> 1;                             \
    volatile aspace int *base = (volatile aspace int *)((ga_size)addr & ~2); \
    union {                                                             \
      int i;                                                            \
      ga_half h[2];                                                     \
    } o, a, n;                                                          \
    o.i = *base;                                                        \
    do {                                                                \
      a.i = o.i;                                                        \
      n.i = o.i;                                                        \
      n.h[idx] = val;                                                   \
      o.i = atomic_cmpxchg(base, a.i, n.i);                             \
    } while (o.i != a.i);                                               \
    return o.h[idx];                                                    \
  }

gen_atomh_add(atom_add_eg, global)
gen_atomh_add(atom_add_el, local)
gen_atomh_xchg(atom_xchg_eg, global)
gen_atomh_xchg(atom_xchg_el, local)

#endif
