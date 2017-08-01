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


int GpuArray_sort(GpuArray *r, GpuArray *a, unsigned int sortDir, GpuArray *dstArg);


#ifdef __cplusplus
}
#endif

#endif
