#define local_barrier() barrier(CLK_LOCAL_MEM_FENCE)
#define WITHIN_KERNEL /* empty */
#define KERNEL __kernel
#define GLOBAL_MEM __global
#define LOCAL_MEM __local
#define LOCAL_MEM_ARG __local
#ifndef NULL
  #define NULL ((void*)0)
#endif
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
