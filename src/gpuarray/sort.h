#ifndef GPUARRAY_SORT_H
#define GPUARRAY_SORT_H
/** \file sort.h
 *  \brief Sort operations generator.
 */

#include <gpuarray/buffer.h>
#include <gpuarray/array.h>
#include <gpuarray/kernel.h>


#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

#define SHARED_SIZE_LIMIT 	1024U
#define SAMPLE_STRIDE 		128

typedef struct _GpuSortData {
  GpuArray       BufKey;
  GpuArray       BufArg;
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


int GpuArray_sort(GpuArray *r, GpuArray *a, unsigned int sortDir, GpuArray *dstArg);


#ifdef __cplusplus
}
#endif

#endif