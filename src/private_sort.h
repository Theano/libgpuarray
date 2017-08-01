#ifndef PRIVATE_SORT_H
#define PRIVATE_SORT_H

#define SHARED_SIZE_LIMIT 	1024U
#define SAMPLE_STRIDE 		128

typedef struct _GpuSortData {

  GpuArray       d_RanksA;
  GpuArray       d_RanksB;
  GpuArray       d_LimitsA;
  GpuArray       d_LimitsB;

} GpuSortData;

typedef struct _GpuSortConfig {
  
  unsigned int   dims;
  unsigned int   Nfloor;
  int            Nleft;
  unsigned int   sortDirFlg;
  unsigned int   argSortFlg;
  int            typecodeKey;
  size_t         typesizeKey;
  int            typecodeArg;
  size_t         typesizeArg;

} GpuSortConfig;

typedef struct _GpuSortBuffers {
  
  GpuArray       BufKey;
  GpuArray       BufArg;

} GpuSortBuff;

typedef struct _GpuSortKernels {
  
  GpuKernel      k_bitonic;
  GpuKernel      k_ranks;
  GpuKernel      k_ranks_idxs;
  GpuKernel      k_merge;
  GpuKernel      k_merge_global;

} GpuSortKernels;

#endif
