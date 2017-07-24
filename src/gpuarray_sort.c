#include <assert.h>
#include <limits.h>
#include <float.h>

#include <gpuarray/sort.h>
#include <gpuarray/array.h>
#include <gpuarray/kernel.h>

#include "util/strb.h"
#include "private.h"

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * This software contains source code provided by NVIDIA Corporation.
 *
 * Read more at: http://docs.nvidia.com/cuda/eula/index.html#ixzz4lUbgXjsr
 * Follow us: @GPUComputing on Twitter | NVIDIA on Facebook
 *
 *
 */

const int flags = GA_USE_CLUDA;

static const char *code_helper_funcs =                                                                               \
"\n#define SAMPLE_STRIDE 128 \n"                                                                                     \
"\n#define SHARED_SIZE_LIMIT  1024U \n"                                                                              \
"__device__ unsigned int iDivUp(unsigned int a, unsigned int b)"                                                     \
"{"                                                                                                                  \
"    return ((a % b) == 0) ? (a / b) : (a / b + 1); "                                                                \
"} "                                                                                                                 \
"__device__ unsigned int getSampleCount(unsigned int dividend) "                                                     \
"{ "                                                                                                                 \
"    return iDivUp(dividend, SAMPLE_STRIDE); "                                                                       \
"}"                                                                                                                  \
"\n #define W (sizeof(unsigned int) * 8) \n"                                                                         \
"__device__ unsigned int nextPowerOfTwo(unsigned int x) "                                                            \
"{"                                                                                                                  \
"    return 1U << (W - __clz(x - 1));"                                                                               \
"} "                                                                                                                 \
"template<typename T> __device__ T readArray(T *a, unsigned int pos, unsigned int length, unsigned int sortDir){"    \
"      if (pos >= length) { "                                                                                        \
"          if (sortDir) { "                                                                                          \
"             return MAX_NUM; "                                                                                      \
"          } "                                                                                                       \
"          else { "                                                                                                  \
"             return MIN_NUM; "                                                                                      \
"          } "                                                                                                       \
"      } "                                                                                                           \
"      else { "                                                                                                      \
"          return a[pos]; "                                                                                          \
"      } "                                                                                                           \
"  } "                                                                                                               \
"template<typename T> __device__ void writeArray(T *a, unsigned int pos, T value, unsigned int length) "             \
" { "                                                                                                                \
"     if (pos >= length) "                                                                                           \
"     { "                                                                                                            \
"          return; "                                                                                                 \
"     } "                                                                                                            \
"     else { "                                                                                                       \
"         a[pos] = value; "                                                                                          \
"     } "                                                                                                            \
" }\n";

static unsigned int iDivUp(unsigned int a, unsigned int b) {
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static unsigned int getSampleCount(unsigned int dividend) {
    return iDivUp(dividend, SAMPLE_STRIDE);
}

static unsigned int roundDown(unsigned int numToRound, unsigned int multiple) {
  if (numToRound <= multiple)
    return numToRound;
  else
    return (numToRound / multiple) * multiple;    
}

static inline const char *ctype(int typecode) {
  return gpuarray_get_type(typecode)->cluda_name;
}

static inline size_t typesize(int typecode) {
  return gpuarray_get_type(typecode)->size;
}

static const char *code_bin_search =                                                                              \
"template<typename T> __device__ unsigned int binarySearchInclusive(T val, T *data, unsigned int L, "             \
"                                              unsigned int stride, unsigned int sortDir){"                       \
"    if (L == 0) "                                                                                                \
"        return 0; "                                                                                              \
"    unsigned int pos = 0; "                                                                                      \
"    for (; stride > 0; stride >>= 1){ "                                                                          \
"      unsigned int newPos = min(pos + stride, L); "                                                              \
"      if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val))){ "                  \
"          pos = newPos; "                                                                                        \
"      } "                                                                                                        \
"    } "                                                                                                          \
"    return pos; "                                                                                                \
"} "                                                                                                              \
" template<typename T> __device__ unsigned int binarySearchExclusive(T val, T *data, unsigned int L, "            \
"                                              unsigned int stride, unsigned int sortDir) "                       \
"{ "                                                                                                              \
"    if (L == 0) "                                                                                                \
"        return 0; "                                                                                              \
"    unsigned int pos = 0; "                                                                                      \
"    for (; stride > 0; stride >>= 1){ "                                                                          \
"      unsigned int newPos = min(pos + stride, L); "                                                              \
"      if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val))){ "                    \
"          pos = newPos; "                                                                                        \
"      } "                                                                                                        \
"    } "                                                                                                          \
"    return pos; "                                                                                                \
"}"                                                                                                               \
"template<typename T> __device__ unsigned int binarySearchLowerBoundExclusive(T val, T *ptr, unsigned int first," \
"                                                                             unsigned int last, unsigned int sortDir) "  \
"{ "                                                                                                                      \
"    unsigned int len = last - first; "                                                                                   \
"    unsigned int half; "                                                                                                 \
"    unsigned int middle; "                                                                                               \
"    while (len > 0) { "                                                                                                  \
"        half = len >> 1; "                                                                                               \
"        middle = first; "                                                                                                \
"        middle += half; "                                                                                                \
"        if ( (sortDir && ptr[middle] < val) || (!sortDir && ptr[middle] > val) ) { "                                     \
"            first = middle; "                                                                                            \
"            ++first; "                                                                                                   \
"            len = len - half - 1; "                                                                                      \
"        } "                                                                                                              \
"        else "                                                                                                           \
"            len = half; "                                                                                                \
"    } "                                                                                                                  \
"    return first; "                                                                                                      \
"} "                                                                                                                      \
"template<typename T> __device__ unsigned int binarySearchLowerBoundInclusive(T val, T *ptr, unsigned int first,  "       \
"                                                                             unsigned int last, unsigned int sortDir) "  \
"{ "                                                                                                                      \
"    unsigned int len = last - first; "                                                                                   \
"    unsigned int half; "                                                                                                 \
"    unsigned int middle; "                                                                                               \
"    while (len > 0) { "                                                                                                  \
"        half = len >> 1; "                                                                                               \
"        middle = first; "                                                                                                \
"        middle += half; "                                                                                                \
"        if ( (sortDir && ptr[middle] <= val) || (!sortDir && ptr[middle] >= val) ) { "                                   \
"            first = middle; "                                                                                            \
"            ++first; "                                                                                                   \
"            len = len - half - 1; "                                                                                      \
"        } "                                                                                                              \
"        else "                                                                                                           \
"            len = half; "                                                                                                \
"    } "                                                                                                                  \
"    return first; "                                                                                                      \
"}\n";

#define NUMARGS_BITONIC_KERNEL 8
const int type_args_bitonic[NUMARGS_BITONIC_KERNEL] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT, GA_UINT};
static const char *code_bitonic_smem =                                                                                                          \
" extern \"C\" __global__ void bitonicSortSharedKernel( "                                                                                       \
"      t_key *d_DstKey, "                                                                                                                       \
"      size_t dstOff,"                                                                                                                          \
"      t_key *d_SrcKey, "                                                                                                                       \
"      size_t srcOff,"                                                                                                                          \
"      unsigned int batchSize, "                                                                                                                \
"      unsigned int arrayLength, "                                                                                                              \
"      unsigned int elemsOff, "                                                                                                                 \
"      unsigned int sortDir "                                                                                                                   \
"  ) "                                                                                                                                          \
"  { "                                                                                                                                          \
"      d_DstKey = (t_key*) (((char*)d_DstKey)+ dstOff);"                                                                                        \
"      d_SrcKey = (t_key*) (((char*)d_SrcKey)+ srcOff);"                                                                                        \
"      d_DstKey += elemsOff;"                                                                                                                   \
"      d_SrcKey += elemsOff;"                                                                                                                   \
"      __shared__ t_key s_key[SHARED_SIZE_LIMIT]; "                                                                                             \
"      s_key[threadIdx.x] = readArray<t_key>( d_SrcKey, "                                                                                       \
"                                             blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x, "                                                   \
"                                             arrayLength * batchSize, "                                                                        \
"                                             sortDir "                                                                                         \
"                                           ); "                                                                                                \ 
"      s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = readArray<t_key>( d_SrcKey, "                                                             \
"                                                                       blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT / 2),"\
"                                                                       arrayLength * batchSize, "                                              \
"                                                                       sortDir "                                                               \
"                                                                     ); "                                                                      \
"      for (unsigned int size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) { "                                                                    \
"          unsigned int ddd = sortDir ^ ((threadIdx.x & (size / 2)) != 0); "                                                                    \
"          for (unsigned int stride = size / 2; stride > 0; stride >>= 1) "                                                                     \
"          { "                                                                                                                                  \
"              __syncthreads(); "                                                                                                               \
"              unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1)); "                                                             \
"              t_key t; "                                                                                                                       \
"              if ((s_key[pos] > s_key[pos + stride]) == ddd) { "                                                                               \
"                  t = s_key[pos]; "                                                                                                            \
"                  s_key[pos] = s_key[pos + stride]; "                                                                                          \
"                  s_key[pos + stride] = t; "                                                                                                   \
"              } "                                                                                                                              \
"          } "                                                                                                                                  \
"      } "                                                                                                                                      \
"      { "                                                                                                                                      \
"          for (unsigned int stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {"                                                       \
"              __syncthreads(); "                                                                                                               \
"              unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1)); "                                                             \
"              t_key t; "                                                                                                                       \
"              if ((s_key[pos] > s_key[pos + stride]) == sortDir) {"                                                                            \
"                  t = s_key[pos]; "                                                                                                            \
"                  s_key[pos] = s_key[pos + stride]; "                                                                                          \
"                  s_key[pos + stride] = t; "                                                                                                   \
"              } "                                                                                                                              \
"          } "                                                                                                                                  \
"      } "                                                                                                                                      \
"      __syncthreads(); "                                                                                                                       \
"      writeArray<t_key>( d_DstKey, "                                                                                                           \
"                  blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x,  "                                                                             \
"                  s_key[threadIdx.x], "                                                                                                        \
"                  arrayLength * batchSize "                                                                                                    \
"                ); "                                                                                                                           \
"      writeArray<t_key>( d_DstKey, "                                                                                                           \
"                  blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT / 2), "                                                    \
"                  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)], "                                                                              \
"                  arrayLength * batchSize "                                                                                                    \
"                ); "                                                                                                                           \
"  }\n";
static int bitonicSortShared(
    GpuArray *d_DstKey,
    GpuArray *d_SrcKey,
    unsigned int batchSize,
    unsigned int arrayLength,
    unsigned int sortDir,
    unsigned int elemsOff,
    GpuKernel *k_bitonic,
    gpucontext *ctx
)
{
  size_t ls, gs;
  unsigned int p = 0;
  int err = GA_NO_ERROR;

  ls = SHARED_SIZE_LIMIT / 2;
  gs = batchSize;

  err = GpuKernel_setarg(k_bitonic, p++, d_DstKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_bitonic, p++, &d_DstKey->offset);
  if (err != GA_NO_ERROR) return err;
  
  err = GpuKernel_setarg(k_bitonic, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_bitonic, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) return err;
  
  err = GpuKernel_setarg(k_bitonic, p++, &batchSize);
  if (err != GA_NO_ERROR) return err;
  
  err = GpuKernel_setarg(k_bitonic, p++, &arrayLength);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_bitonic, p++, &elemsOff);
  if (err != GA_NO_ERROR) return err;
  
  err = GpuKernel_setarg(k_bitonic, p++, &sortDir);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_bitonic, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;

  return err;  
/*
  float *h_dst2 = (float *) malloc ( 16 * sizeof(float));
  err = GpuArray_read(h_dst2, 16 * sizeof(float), d_DstKey);
  if (err != GA_NO_ERROR) printf("error reading \n");

  
  int i;
  for (i = 0; i < 16; i++)
  {
      printf("%d afterbitonic %f \n", i, h_dst2[i]);
  }
  */
}

#define NUMARGS_SAMPLE_RANKS 10
const int type_args_ranks[NUMARGS_SAMPLE_RANKS] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT, GA_UINT};
static const char *code_sample_ranks =                                                                              \
"extern \"C\" __global__ void generateSampleRanksKernel("                                                           \
"    unsigned int *d_RanksA,"                                                                                       \
"    size_t rankAOff,"                                                                                              \
"    unsigned int *d_RanksB,"                                                                                       \
"    size_t rankBOff,"                                                                                              \
"    t_key *d_SrcKey,"                                                                                              \
"    size_t srcOff,"                                                                                                \
"    unsigned int stride,"                                                                                          \
"    unsigned int N,"                                                                                               \
"    unsigned int threadCount,"                                                                                     \
"    unsigned int sortDir"                                                                                          \
")"                                                                                                                 \
"{"                                                                                                                 \
"    d_RanksA = (unsigned int*) (((char*)d_RanksA)+ rankAOff);"                                                     \
"    d_RanksB = (unsigned int*) (((char*)d_RanksB)+ rankBOff);"                                                     \
"    d_SrcKey = (t_key*) (((char*)d_SrcKey)+ srcOff);"                                                              \
"    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;"                                                     \
"    if (pos >= threadCount) {"                                                                                     \
"        return;"                                                                                                   \
"    }"                                                                                                             \
"    const unsigned int           i = pos & ((stride / SAMPLE_STRIDE) - 1);"                                        \
"    const unsigned int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);"                                             \
"    d_SrcKey += segmentBase;"                                                                                      \
"    d_RanksA += segmentBase / SAMPLE_STRIDE;"                                                                      \
"    d_RanksB += segmentBase / SAMPLE_STRIDE;"                                                                      \
"    const unsigned int segmentElementsA = stride;"                                                                 \
"    const unsigned int segmentElementsB = min(stride, N - segmentBase - stride);"                                  \
"    const unsigned int  segmentSamplesA = getSampleCount(segmentElementsA);"                                       \
"    const unsigned int  segmentSamplesB = getSampleCount(segmentElementsB);"                                       \
"    if (i < segmentSamplesA) {"                                                                                    \
"        d_RanksA[i] = i * SAMPLE_STRIDE;"                                                                          \
"        d_RanksB[i] = binarySearchExclusive<t_key>("                                                               \
"                          d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride,"                                         \
"                          segmentElementsB, nextPowerOfTwo(segmentElementsB), sortDir"                             \
"                      );"                                                                                          \
"    }"                                                                                                             \
"    if (i < segmentSamplesB) {"                                                                                    \
"        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;"                                               \
"        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive<t_key>("                                    \
"                                                     d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0,"          \
"                                                     segmentElementsA, nextPowerOfTwo(segmentElementsA), sortDir"  \
"                                                 );"                                                               \
"    }"                                                                                                             \
"}\n";
static int generateSampleRanks(
  GpuArray  *d_RanksA,
  GpuArray  *d_RanksB,
  GpuArray *d_SrcKey,
  unsigned int stride,
  unsigned int N,
  unsigned int sortDir,
  GpuKernel *k_ranks,
  gpucontext *ctx
)
{
  unsigned int lastSegmentElements = N % (2 * stride);
  unsigned int threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

  size_t ls, gs;
  unsigned int p = 0;
  int err = GA_NO_ERROR;

  ls = 256;
  gs = iDivUp(threadCount, 256);

  err = GpuKernel_setarg(k_ranks, p++, d_RanksA->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &d_RanksA->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, d_RanksB->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &d_RanksB->offset);
  if (err != GA_NO_ERROR) return err;
  
  err = GpuKernel_setarg(k_ranks, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &stride);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &N);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &threadCount);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks, p++, &sortDir);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_ranks, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;
 
  return err;
}

#define NUMARGS_RANKS_IDXS 7
const int type_args_ranks_idxs[NUMARGS_RANKS_IDXS] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT};
static const char *code_ranks_idxs =                                                                                           \
"extern \"C\" __global__ void mergeRanksAndIndicesKernel( "                                                                    \
"    unsigned int *d_Limits, "                                                                                                 \
"    size_t limOff,"                                                                                                           \
"    unsigned int *d_Ranks, "                                                                                                  \
"    size_t rankOff,"                                                                                                          \
"    unsigned int stride, "                                                                                                    \
"    unsigned int N, "                                                                                                         \
"    unsigned int threadCount "                                                                                                \
") "                                                                                                                           \
"{ "                                                                                                                           \
"    d_Limits = (unsigned int*) (((char*)d_Limits)+ limOff);"                                                                  \
"    d_Ranks = (unsigned int*) (((char*)d_Ranks)+ rankOff);"                                                                   \
"    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x; "                                                               \
"    if (pos >= threadCount) "                                                                                                 \
"        return; "                                                                                                             \
"    const unsigned int           i = pos & ((stride / SAMPLE_STRIDE) - 1); "                                                  \
"    const unsigned int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE); "                                                       \
"    d_Ranks  += (pos - i) * 2; "                                                                                              \
"    d_Limits += (pos - i) * 2; "                                                                                              \
"    const unsigned int segmentElementsA = stride; "                                                                           \
"    const unsigned int segmentElementsB = min(stride, N - segmentBase - stride); "                                            \
"    const unsigned int  segmentSamplesA = getSampleCount(segmentElementsA); "                                                 \
"    const unsigned int  segmentSamplesB = getSampleCount(segmentElementsB); "                                                 \
"    if (i < segmentSamplesA) { "                                                                                              \
"        unsigned int dstPos = binarySearchExclusive<unsigned int>(d_Ranks[i], d_Ranks + segmentSamplesA,"                     \
"                                                                  segmentSamplesB, nextPowerOfTwo(segmentSamplesB), 1U) + i;" \
"        d_Limits[dstPos] = d_Ranks[i]; "                                                                                      \
"    } "                                                                                                                       \
"    if (i < segmentSamplesB) { "                                                                                              \
"        unsigned int dstPos = binarySearchInclusive<unsigned int>(d_Ranks[segmentSamplesA + i], d_Ranks,"                     \
"                                                                   segmentSamplesA, nextPowerOfTwo(segmentSamplesA), 1U) + i;"\
"        d_Limits[dstPos] = d_Ranks[segmentSamplesA + i]; "                                                                    \
"    } "                                                                                                                       \
"}\n";
static int mergeRanksAndIndices(
  GpuArray *d_LimitsA,
  GpuArray *d_LimitsB,
  GpuArray *d_RanksA,
  GpuArray *d_RanksB,
  unsigned int stride,
  unsigned int N,
  unsigned int sortDir,
  GpuKernel *k_ranks_idxs,
  gpucontext *ctx
)
{
  unsigned int lastSegmentElements = N % (2 * stride);
  unsigned int threadCount = (lastSegmentElements > stride) ? 
                             (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);
  size_t ls, gs;
  unsigned int p = 0;
  int err = GA_NO_ERROR;

  ls = 256U;
  gs = iDivUp(threadCount, 256U);

  err = GpuKernel_setarg(k_ranks_idxs, p++, d_LimitsA->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &d_LimitsA->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, d_RanksA->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &d_RanksA->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &stride);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &N);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &threadCount);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_ranks_idxs, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;

  p = 0;

  err = GpuKernel_setarg(k_ranks_idxs, p++, d_LimitsB->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &d_LimitsB->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, d_RanksB->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_ranks_idxs, p++, &d_RanksB->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_ranks_idxs, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;

  return err;
}

#define NUMARGS_MERGE 11
const int type_args_merge[NUMARGS_MERGE] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT};
static const char *code_merge =                                                                                     \
" template<typename T> __device__ void merge( "                                                                     \
"    T *dstKey, "                                                                                                   \
"    T *srcAKey, "                                                                                                  \
"    T *srcBKey, "                                                                                                  \
"    unsigned int lenA, "                                                                                           \
"    unsigned int nPowTwoLenA, "                                                                                    \
"    unsigned int lenB, "                                                                                           \
"    unsigned int nPowTwoLenB, "                                                                                    \
"    unsigned int sortDir "                                                                                         \
") "                                                                                                                \
"{ "                                                                                                                \
"    T keyA, keyB; "                                                                                                \
"    unsigned int dstPosA , dstPosB;"                                                                               \
"    if (threadIdx.x < lenA) { "                                                                                    \
"        keyA = srcAKey[threadIdx.x]; "                                                                             \
"        dstPosA = binarySearchExclusive<T>(keyA, srcBKey, lenB, nPowTwoLenB, sortDir) + threadIdx.x; "             \
"    } "                                                                                                            \
"    if (threadIdx.x < lenB) { "                                                                                    \
"        keyB = srcBKey[threadIdx.x]; "                                                                             \
"        dstPosB = binarySearchInclusive<T>(keyB, srcAKey, lenA, nPowTwoLenA, sortDir) + threadIdx.x; "             \
"    } "                                                                                                            \
"    __syncthreads(); "                                                                                             \
"    if (threadIdx.x < lenA) { "                                                                                    \
"        dstKey[dstPosA] = keyA; "                                                                                  \
"    } "                                                                                                            \
"    if (threadIdx.x < lenB) { "                                                                                    \
"        dstKey[dstPosB] = keyB; "                                                                                  \
"    } "                                                                                                            \
"} "                                                                                                                \
"extern \"C\" __global__ void mergeElementaryIntervalsKernel( "                                                     \
"    t_key *d_DstKey, "                                                                                             \
"    size_t dstOff,"                                                                                                \
"    t_key *d_SrcKey, "                                                                                             \
"    size_t srcOff,"                                                                                                \
"    unsigned int *d_LimitsA, "                                                                                     \
"    size_t limAOff,"                                                                                               \
"    unsigned int *d_LimitsB, "                                                                                     \
"    size_t limBOff,"                                                                                               \
"    unsigned int stride, "                                                                                         \
"    unsigned int N, "                                                                                              \
"    unsigned int sortDir"                                                                                          \
") "                                                                                                                \
"{ "                                                                                                                \
"    d_DstKey = (t_key*) (((char*)d_DstKey)+ dstOff);"                                                              \
"    d_SrcKey = (t_key*) (((char*)d_SrcKey)+ srcOff);"                                                              \
"    d_LimitsA = (unsigned int*) (((char*)d_LimitsA)+ limAOff);"                                                    \
"    d_LimitsB = (unsigned int*) (((char*)d_LimitsB)+ limBOff);"                                                    \
"    __shared__ t_key s_key[2 * SAMPLE_STRIDE]; "                                                                   \
"    const unsigned int   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1); "                            \
"    const unsigned int segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE; "                                   \
"    d_SrcKey += segmentBase; "                                                                                     \
"    d_DstKey += segmentBase; "                                                                                     \
"    __shared__ unsigned int startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB; "                        \
"    if (threadIdx.x == 0) { "                                                                                      \
"        unsigned int segmentElementsA = stride; "                                                                  \
"        unsigned int segmentElementsB = min(stride, N - segmentBase - stride); "                                   \
"        unsigned int  segmentSamplesA = getSampleCount(segmentElementsA); "                                        \
"        unsigned int  segmentSamplesB = getSampleCount(segmentElementsB); "                                        \
"        unsigned int   segmentSamples = segmentSamplesA + segmentSamplesB; "                                       \
"        startSrcA    = d_LimitsA[blockIdx.x]; "                                                                    \
"        startSrcB    = d_LimitsB[blockIdx.x]; "                                                                    \
"        unsigned int endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA; "  \
"        unsigned int endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB; "  \
"        lenSrcA      = endSrcA - startSrcA; "                                                                      \
"        lenSrcB      = endSrcB - startSrcB; "                                                                      \
"        startDstA    = startSrcA + startSrcB; "                                                                    \
"        startDstB    = startDstA + lenSrcA; "                                                                      \
"    } "                                                                                                            \
"    __syncthreads(); "                                                                                             \
"    if (threadIdx.x < lenSrcA) { "                                                                                 \
"        s_key[threadIdx.x +             0] = d_SrcKey[0 + startSrcA + threadIdx.x]; "                              \
"    } "                                                                                                            \
"    if (threadIdx.x < lenSrcB) { "                                                                                 \
"        s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x]; "                         \
"    } "                                                                                                            \
"    __syncthreads(); "                                                                                             \
"    merge<t_key>( "                                                                                                \
"        s_key, "                                                                                                   \
"        s_key + 0, "                                                                                               \
"        s_key + SAMPLE_STRIDE, "                                                                                   \
"        lenSrcA, SAMPLE_STRIDE, "                                                                                  \
"        lenSrcB, SAMPLE_STRIDE, "                                                                                  \
"        sortDir "                                                                                                  \
"    ); "                                                                                                           \
"    __syncthreads(); "                                                                                             \
"    if (threadIdx.x < lenSrcA) { "                                                                                 \
"        d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x]; "                                                  \
"    } "                                                                                                            \
"    if (threadIdx.x < lenSrcB) { "                                                                                 \
"        d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x]; "                                        \
"    } "                                                                                                            \
"}\n";
static int mergeElementaryIntervals(
    GpuArray *d_DstKey,
    GpuArray *d_SrcKey,
    GpuArray  *d_LimitsA,
    GpuArray  *d_LimitsB,
    unsigned int stride,
    unsigned int N,
    unsigned int sortDir,
    GpuKernel *k_merge,
    gpucontext *ctx
)
{
  unsigned int lastSegmentElements = N % (2 * stride);
  unsigned int mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) : (N - lastSegmentElements) / SAMPLE_STRIDE;

  size_t ls, gs;
  unsigned int p = 0;
  int err = GA_NO_ERROR;

  ls = SAMPLE_STRIDE;
  gs = mergePairs;

  err = GpuKernel_setarg(k_merge, p++, d_DstKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &d_DstKey->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, d_LimitsA->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &d_LimitsA->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, d_LimitsB->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &d_LimitsB->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &stride);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &N);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge, p++, &sortDir);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_merge, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;
 
  return err;
}

#define NUMARGS_MERGE_GLB 8
const int type_args_merge_glb[NUMARGS_MERGE_GLB] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT, GA_UINT};
static const char *code_merge_glb =                                                                                     \
"extern \"C\" __global__ void mergeGlobalMemKernel( "                                                                   \
"    t_key *d_DstKey, "                                                                                                 \
"    size_t dstOff, "                                                                                                   \
"    t_key *d_SrcKey, "                                                                                                 \
"    size_t srcOff, "                                                                                                   \
"    unsigned int segmentSizeA, "                                                                                       \
"    unsigned int segmentSizeB, "                                                                                       \
"    unsigned int N, "                                                                                                  \
"    unsigned int sortDir "                                                                                             \
") "                                                                                                                    \
"{ "                                                                                                                    \
"    d_DstKey = (t_key*) (((char*)d_DstKey)+ dstOff);"                                                                  \
"    d_SrcKey = (t_key*) (((char*)d_SrcKey)+ srcOff);"                                                                  \
"    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; "                                                        \
"    t_key *segmentPtrA = d_SrcKey; "                                                                                   \
"    t_key *segmentPtrB = d_SrcKey + segmentSizeA; "                                                                    \
"    unsigned int idxSegmentA = idx % segmentSizeA; "                                                                   \
"    unsigned int idxSegmentB = idx - segmentSizeA; "                                                                   \
"    if (idx >= N) "                                                                                                    \
"        return; "                                                                                                      \
"    t_key value = d_SrcKey[idx]; "                                                                                     \
"    unsigned int dstPos; "                                                                                             \
"    if (idx < segmentSizeA) { "                                                                                        \
"        dstPos = binarySearchLowerBoundExclusive<t_key>(value, segmentPtrB, 0, segmentSizeB, sortDir) + idxSegmentA;"  \
"    } "                                                                                                                \
"    else { "                                                                                                           \
"        dstPos = binarySearchLowerBoundInclusive<t_key>(value, segmentPtrA, 0, segmentSizeA, sortDir) + idxSegmentB;"  \
"    } "                                                                                                                \
"    d_DstKey[dstPos] = value; "                                                                                        \
"}\n";

static int mergeGlobalMem(
    GpuArray *d_DstKey,
    GpuArray *d_SrcKey,
    unsigned int segmentSizeA,
    unsigned int segmentSizeB,
    unsigned int N,
    unsigned int sortDir,
    GpuKernel *k_merge_global,
    gpucontext *ctx
)
{
  size_t ls, gs;
  unsigned int p = 0;
  int err = GA_NO_ERROR;

  ls = 256;
  gs = iDivUp(N, ls);

  err = GpuKernel_setarg(k_merge_global, p++, d_DstKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &d_DstKey->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &segmentSizeA);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &segmentSizeB);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &N);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_setarg(k_merge_global, p++, &sortDir);
  if (err != GA_NO_ERROR) return err;

  err = GpuKernel_call(k_merge_global, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) return err;

  return err;
}

// Generate type specific GPU code
static int genMergeSortTypeCode(strb *str, int typecode)
{
  int err = GA_NO_ERROR;
  // Generate typedef for the data type to be sorted    
  strb_appendf(str, "typedef %s t_key;\n", ctype(typecode));

  // Generate macro for MIN and MAX value of a given data type
  switch (typecode){
  case GA_UINT:
    strb_appendf(str, "#define MAX_NUM %u \n#define MIN_NUM %u \n", UINT_MAX, 0);
    break;
  case GA_INT:
    strb_appendf(str, "#define MAX_NUM %d \n#define MIN_NUM %d \n", INT_MAX, INT_MIN);
    break;
  case GA_FLOAT:
    strb_appendf(str, "#define MAX_NUM %f \n#define MIN_NUM %f \n", FLT_MAX, -FLT_MAX);
    break;
  case GA_DOUBLE:
    strb_appendf(str, "#define MAX_NUM %g \n#define MIN_NUM %g \n", DBL_MAX, -DBL_MAX);
    break;
  case GA_UBYTE:
    strb_appendf(str, "#define MAX_NUM %u \n#define MIN_NUM %u \n", UCHAR_MAX, 0);
    break;
  case GA_BYTE:
    strb_appendf(str, "#define MAX_NUM %d \n#define MIN_NUM %d \n", SCHAR_MAX, SCHAR_MIN);
    break;
  case GA_USHORT:
    strb_appendf(str, "#define MAX_NUM %u \n#define MIN_NUM %u \n", USHRT_MAX, 0);
    break;
  case GA_SHORT:
    strb_appendf(str, "#define MAX_NUM %d \n#define MIN_NUM %d \n", SHRT_MAX, SHRT_MIN);
    break;
  default:
    return GA_IMPL_ERROR;
    break;
  }
  return strb_error(&str);
}

#define NSTR_BITONIC 3
#define NSTR_RANKS 4
#define NSTRINGS_RKS_IDX 4
#define NSTRINGS_MERGE 4
#define NSTRINGS_MERGE_GLB 4
static int compileKernels(GpuKernel *k_bitonic, GpuKernel *k_ranks, GpuKernel *k_ranks_idxs, GpuKernel *k_merge,
                           GpuKernel *k_merge_global, gpucontext *ctx, int typecode)
{
  char *err_str = NULL;
  int err = GA_NO_ERROR;

  size_t lens_bitonic[NSTR_BITONIC]         = {0, strlen(code_helper_funcs), strlen(code_bitonic_smem)};
  size_t lens_ranks[NSTR_RANKS]             = {0, strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_sample_ranks)};
  size_t lens_rks_idx[NSTRINGS_RKS_IDX]     = {0, strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_ranks_idxs)};
  size_t lens_merge[NSTRINGS_MERGE]         = {0, strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_merge)};
  size_t lens_merge_glb[NSTRINGS_MERGE_GLB] = {0, strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_merge_glb)};

  const char *codes_bitonic[NSTR_BITONIC]         = {NULL, code_helper_funcs, code_bitonic_smem};
  const char *codes_ranks[NSTR_RANKS]             = {NULL, code_helper_funcs, code_bin_search, code_sample_ranks};
  const char *codes_rks_idx[NSTRINGS_RKS_IDX]     = {NULL, code_helper_funcs, code_bin_search, code_ranks_idxs};
  const char *codes_merge[NSTRINGS_MERGE]         = {NULL, code_helper_funcs, code_bin_search, code_merge};
  const char *codes_merge_glb[NSTRINGS_MERGE_GLB] = {NULL, code_helper_funcs, code_bin_search, code_merge_glb};

  strb sb = STRB_STATIC_INIT;
  err = genMergeSortTypeCode(&sb, typecode);
  if (err != GA_NO_ERROR) return err;

  // Compile Bitonic sort Kernel  
  lens_bitonic[0] = sb.l;
  codes_bitonic[0] = sb.s;
  err = GpuKernel_init( k_bitonic,
                        ctx,
                        NSTR_BITONIC,
                        codes_bitonic,
                        lens_bitonic,
                        "bitonicSortSharedKernel",
                        NUMARGS_BITONIC_KERNEL,
                        type_args_bitonic,
                        flags,
                        &err_str
                      );
  if (err != GA_NO_ERROR) {
   printf("error kernel init: %s \n", gpuarray_error_str(err));
   printf("error backend: %s \n", err_str);
   return err;
  }

  // Compile ranks kernel
  lens_ranks[0]  = sb.l;
  codes_ranks[0] = sb.s;
  err = GpuKernel_init( k_ranks,
                        ctx,
                        NSTR_RANKS, 
                        codes_ranks,
                        lens_ranks,
                        "generateSampleRanksKernel",
                        NUMARGS_SAMPLE_RANKS,
                        type_args_ranks,
                        flags,
                        &err_str
                      );
  if (err != GA_NO_ERROR) {
    printf("error kernel init: %s \n", gpuarray_error_str(err));
    printf("error backend: %s \n", err_str);
    return err;
  }

  // Compile ranks and idxs kernel
  lens_rks_idx[0]  = sb.l;
  codes_rks_idx[0] = sb.s;
  err = GpuKernel_init( k_ranks_idxs,
                        ctx, 
                        NSTRINGS_RKS_IDX, 
                        codes_rks_idx,
                        lens_rks_idx,
                        "mergeRanksAndIndicesKernel",
                        NUMARGS_RANKS_IDXS,
                        type_args_ranks_idxs,
                        flags,
                        &err_str
                      );
  if (err != GA_NO_ERROR) {
    printf("error kernel init: %s \n", gpuarray_error_str(err)); 
    printf("error backend: %s \n", err_str);
    return err;
  }

  // Compile merge kernel
  lens_merge[0]  = sb.l;
  codes_merge[0] = sb.s;
  err = GpuKernel_init( k_merge,
                        ctx,
                        NSTRINGS_MERGE, 
                        codes_merge,
                        lens_merge,
                        "mergeElementaryIntervalsKernel",
                        NUMARGS_MERGE,
                        type_args_merge,
                        flags,
                        &err_str
                      );
  if (err != GA_NO_ERROR) {
    printf("error kernel init: %s \n", gpuarray_error_str(err));
    printf("error backend: %s \n", err_str);
    return err;
  }

  // Compile merge global kernel
  lens_merge_glb[0]  = sb.l;
  codes_merge_glb[0] = sb.s;
  err = GpuKernel_init( k_merge_global,
                        ctx, 
                        NSTRINGS_MERGE_GLB, 
                        codes_merge_glb,
                        lens_merge_glb,
                        "mergeGlobalMemKernel",
                        NUMARGS_MERGE_GLB,
                        type_args_merge_glb,
                        flags,
                        &err_str
                      );
  if (err != GA_NO_ERROR) {
    printf("error kernel init: %s \n", gpuarray_error_str(err));
    printf("error backend: %s \n", err_str);
    return err;
  }
  return err;
}

static int sort(
  GpuArray *d_DstKey,
  GpuArray *d_BufKey,
  GpuArray *d_SrcKey,
  GpuArray  *d_RanksA,
  GpuArray  *d_RanksB,
  GpuArray  *d_LimitsA,
  GpuArray  *d_LimitsB,
  unsigned int N,
  unsigned int Nfloor,
  int Nleft,
  unsigned int sortDir,
  gpucontext *ctx
)
{
  int    typecode = d_SrcKey->typecode;
  size_t typeSize = typesize(typecode);
  size_t lstCopyOff;
  int err = GA_NO_ERROR;

  unsigned int stageCount = 0;
  unsigned int stride;

  GpuArray *ikey, *okey, *t;
  GpuKernel k_bitonic, k_ranks, k_ranks_idxs, k_merge, k_merge_global;
  err = compileKernels(&k_bitonic, &k_ranks, &k_ranks_idxs, &k_merge, &k_merge_global, ctx, typecode);
  if (err != GA_NO_ERROR) return err;

  for (stride = SHARED_SIZE_LIMIT; stride < Nfloor; stride <<= 1, stageCount++);

  if (stageCount & 1) {
    ikey = d_BufKey;
    okey = d_DstKey;
  }
  else {
    ikey = d_DstKey;
    okey = d_BufKey;
  }

  // Bitonic sort for short arrays
  if (N <= SHARED_SIZE_LIMIT) {  
    err = bitonicSortShared(d_DstKey, d_SrcKey, 1, N, sortDir, 0, &k_bitonic, ctx);
    if (err != GA_NO_ERROR) return err;
  }
  // Merge - Bitonic sort for bigger arrays
  else {
    unsigned int batchSize = Nfloor / SHARED_SIZE_LIMIT;
    unsigned int arrayLength = SHARED_SIZE_LIMIT;
    err = bitonicSortShared(ikey, d_SrcKey, batchSize, arrayLength, sortDir, 0, &k_bitonic, ctx);
    if (err != GA_NO_ERROR) return err;

    for (stride = SHARED_SIZE_LIMIT; stride < Nfloor; stride <<= 1) {
      unsigned int lastSegmentElements = Nfloor % (2 * stride);

      //Find sample ranks and prepare for limiters merge
      err = generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, Nfloor, sortDir, &k_ranks, ctx);
      if (err != GA_NO_ERROR) return err;

      //Merge ranks and indices
      err = mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, Nfloor, sortDir, &k_ranks_idxs, ctx);
      if (err != GA_NO_ERROR) return err;

      //Merge elementary intervals
      err = mergeElementaryIntervals(okey, ikey, d_LimitsA, d_LimitsB, stride, Nfloor, sortDir, &k_merge, ctx);
      if (err != GA_NO_ERROR) return err;

      if (lastSegmentElements <= stride) {
        //Last merge segment consists of a single array which just needs to be passed through          
        lstCopyOff = okey->offset + ((Nfloor - lastSegmentElements) * typeSize);
        err = gpudata_move(okey->data, lstCopyOff, ikey->data, lstCopyOff, lastSegmentElements * typeSize);
        if (err != GA_NO_ERROR) return err;
      }
      // Swap pointers
      t = ikey;
      ikey = okey;
      okey = t;
    }
    // If the array is not multiple of 1024, sort the remaining and merge
    if (Nleft > 0) {
      err = bitonicSortShared(d_SrcKey, d_DstKey, 1, Nleft, sortDir, Nfloor, &k_bitonic, ctx);
      if (err != GA_NO_ERROR) return err;

      // Copy the leftMost segment to the output array of which contains the first sorted sequence
      lstCopyOff = okey->offset + Nfloor * typeSize;
      err = gpudata_move(d_DstKey->data, lstCopyOff, d_SrcKey->data, lstCopyOff, Nleft * typeSize);
      if (err != GA_NO_ERROR) return err;

      err = mergeGlobalMem(d_SrcKey, d_DstKey, Nfloor, (unsigned int)Nleft, N, sortDir, &k_merge_global, ctx);
      if (err != GA_NO_ERROR) return err;

      err = GpuArray_copy(d_DstKey, d_SrcKey, GA_C_ORDER);
      if (err != GA_NO_ERROR) return err;
    }
  }
  return err;
}

static int initMergeSort(
  GpuArray *d_RanksA,
  GpuArray *d_RanksB,
  GpuArray *d_LimitsA,
  GpuArray *d_LimitsB,
  unsigned int len,
  unsigned int nd,
  gpucontext *ctx
)
{
  int err = GA_NO_ERROR;
  const size_t dims = len * sizeof(unsigned int);

  err = GpuArray_empty(d_RanksA, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (err != GA_NO_ERROR) printf("error allocating aux structures %d\n", err);

  err = GpuArray_empty(d_RanksB, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (err != GA_NO_ERROR) printf("error allocating aux structures %d\n", err);
  
  err = GpuArray_empty(d_LimitsA, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (err != GA_NO_ERROR) printf("error allocating aux structures %d\n", err);

  err = GpuArray_empty(d_LimitsB, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (err != GA_NO_ERROR) printf("error allocating aux structures %d\n", err);

  return err;
}

static void destroyMergeSort(
  GpuArray *d_RanksA,
  GpuArray *d_RanksB,
  GpuArray *d_LimitsA,
  GpuArray *d_LimitsB,
  GpuArray *BufKey
)
{
  GpuArray_clear(d_RanksA);
  GpuArray_clear(d_RanksB);
  GpuArray_clear(d_LimitsA);
  GpuArray_clear(d_LimitsB);
  GpuArray_clear(BufKey);
}


int GpuArray_sort(
  GpuArray *dstKey,
  GpuArray *srcKey,
  unsigned int sortDir
)
{
  int err = GA_NO_ERROR;

  const size_t dims         = srcKey->dimensions[0];
  const unsigned int Nfloor = roundDown(dims, SHARED_SIZE_LIMIT);
  const int Nleft           = dims - Nfloor;

  // Buffer data structure
  GpuArray BufKey;

  // Device pointers - auxiiary data structure    
  GpuArray d_RanksA, d_RanksB, d_LimitsA, d_LimitsB;
  
  gpucontext *ctx = GpuArray_context(srcKey);

  if (srcKey->nd > 1)                      return GA_IMPL_ERROR;
  // if (dstArg != NULL || srcArg != NULL)    return GA_IMPL_ERROR;

  /*
	if (dstArg != NULL || srcArg != NULL) {
    err = GpuArray_empty(&BufArg, ctx, GA_UINT, srcKey->nd, &dims, GA_C_ORDER);
  }*/

  err = GpuArray_empty(&BufKey, ctx, srcKey->typecode, srcKey->nd, &dims, GA_C_ORDER);

  // Auxiliary data structure for MergeSort 
  err = initMergeSort(&d_RanksA, &d_RanksB, &d_LimitsA, &d_LimitsB, Nfloor / 128, srcKey->nd, ctx);

  // perform regular sort
  err = sort(
          dstKey,
          &BufKey,
          srcKey,
          &d_RanksA,
          &d_RanksB,
          &d_LimitsA,
          &d_LimitsB,
          dims,
          Nfloor,
          Nleft,
          sortDir,
          ctx
        );

  // Destroy auxiliary data structures
  destroyMergeSort(&d_RanksA, &d_RanksB, &d_LimitsA, &d_LimitsB, &BufKey);

  return err;
}