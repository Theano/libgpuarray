#ifndef GPUARRAY_SORT_H
#define GPUARRAY_SORT_H
/** \file sort.h
 *  \brief Sort operations generator.
 */

#include <gpuarray/buffer.h>
#include <gpuarray/array.h>


#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

#define SHARED_SIZE_LIMIT 	1024U
#define SAMPLE_STRIDE 		128


int GpuArray_sort(GpuArray *r, GpuArray *a, unsigned int sortDir, GpuArray *dstArg, GpuArray *srcArg);


#ifdef __cplusplus
}
#endif

#endif