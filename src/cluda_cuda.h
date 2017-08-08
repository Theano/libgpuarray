#define local_barrier() __syncthreads()
#define WITHIN_KERNEL extern \"C\" __device__
#define KERNEL extern \"C\" __global__
#define GLOBAL_MEM /* empty */
#define LOCAL_MEM __shared__
#define LOCAL_MEM_ARG /* empty */
#ifdef NAN
#undef NAN
#endif
#define NAN __int_as_float(0x7fffffff)
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
#define ga_half ga_ushort
#define ga_size size_t
#define ga_ssize ptrdiff_t
#define load_half(p) __half2float(*(p))
#define store_half(p, v) (*(p) = __float2half_rn(v))
#define GA_DECL_SHARED_PARAM(type, name)
#define GA_DECL_SHARED_BODY(type, name) extern __shared__ type name[];
#define GA_WARP_SIZE warpSize
