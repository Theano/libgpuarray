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
#define ga_double double
#define ga_size ulong
#define ga_ssize long
#define load_half(p) vload_half(0, &(p)->data)
#define store_half(p, v) vstore_half_rtn(v, 0, &(p)->data)
#define GA_DECL_SHARED_PARAM(type, name) , __local type *name
#define GA_DECL_SHARED_BODY(type, name)
#define GA_WARP_SIZE __GA_WARP_SIZE

struct ga_half {
  half data;
};

#pragma OPENCL_EXTENSION cl_khr_int64_base_atomics: enable

#define gen_atom32_add(name, argtype, aspace)                     \
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
      p.w = atomic_cmpxhg((volatile aspace int *)addr, a, n.w);   \
    } while (p.w != a);                                           \
    return n.a;                                                   \
  }

#define gen_atom64_add(name, argtype, aspace)                     \
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
      p.w = atom_cmpxhg((volatile aspace long *)addr, a, n.w);    \
    } while (p.w != a);                                           \
    return n.a;                                                   \
  }

#define gen_atom64_xchg(name, argtype, aspace)                  \
  argtype name(volatile aspace argtype *addr, argtype val) {    \
    union {                                                     \
      argtype a;                                                \
      long w;                                                   \
    } p, n;                                                     \
    n.a = val;                                                  \
    p.w = atom_xchg((volatile aspace wtype *)addr, n.w);        \
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
/* ga_float */
gen_atom32_add(atom_add_fg, ga_float, global)
gen_atom32_add(atom_add_fl, ga_float, local)
#define atom_xchg_fg(a, b) atomic_xchg(a, b)
#define atom_xchg_fl(a, b) atomic_xchg(a, b)
/* ga_double */
gen_atom64_add(atom_add_dg, ga_double, global)
gen_atom64_add(atom_add_dl, ga_double, local)
gen_atom64_xchg(atom_xchg_dg, ga_double, global)
gen_atom64_xchg(atom_xchg_dl, ga_double, local)
/* ga_half */
#define gen_atomh_add(name, aspace) \
  ga_half name(volatile aspace ga_half *addr, ga_half val) {            \
    ga_size off = (ga_size)addr & 2;                                    \
    volatile aspace int *base = (volatile aspace int *)((ga_size)addr - off); \
    int o, a, n;                                                        \
    float fo;                                                           \
    float fval;                                                         \
    ga_half hn;                                                         \
    fval = vload_half(0, &val->data);                                   \
    o = *base;                                                          \
    do {                                                                \
      a = o;                                                            \
      /* This loads the half of `o` that we want to update */           \
      fo = vload_half(off, (__private half *)&o);                       \
      /* We compute the half addition in float 32 */                    \
      store_half(fval + fo, &hn);                                       \
      /* Now we reassemble the the parts to form a 32-bits n */         \
      if (off == 2)                                                     \
        n = (int)hn->data << 16 & (o & 0xffff);                         \
      else                                                              \
        n = (int)hn->data & (o & 0xffff0000);                           \
      o = atomic_cmpxchg(base, a, n);                                   \
    } while (o != a);                                                   \
    if (off == 2)                                                       \
      hn->data = (ushort)(o >> 16);                                     \
    else                                                                \
      hn->data = (ushort)(o & 0xffff);                                  \
    return hn;                                                          \
  }

#define gen_atomh_xchg(name, aspace) \
  ga_half name(volatile aspace ga_half *addr, ga_half *val) { \
    ga_size off = (ga_size)addr & 2;                                    \
    volatile aspace int *base = (volatile aspace int *)((ga_size)addr - off); \
    int o, a, n;                                                        \
    ga_half hr;                                                         \
    o = *base;                                                          \
    do {                                                                \
      a = o;                                                            \
      /* we have to combine our half value with the right part of `o` */ \
      if (off == 2)                                                     \
        n = (int)val->data << 16 & (o & 0xffff);                        \
      else                                                              \
        n = (int)val->data & (o & 0xffff0000);                          \
      o = atomic_cmpxchg(base, a, n);                                   \
    } while (o != a);                                                   \
    if (off == 2)                                                       \
      hr->data = (ushort)o << 16;                                       \
    else                                                                \
      hr->data = (ushort)o & 0xffff;                                    \
    return hr;                                                          \
  }

gen_atomh_add(atom_add_hg, global)
gen_atomh_add(atom_add_hl, local)
gen_atomh_xchg(atom_xchg_hg, global)
gen_atomh_xchg(atom_xchg_hl, local)
