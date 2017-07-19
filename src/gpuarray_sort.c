#include <assert.h>

#include <gpuarray/sort.h>
#include <gpuarray/array.h>
#include <gpuarray/kernel.h>


#include "util/strb.h"
#include "private.h"

const int flags = GA_USE_CUDA;

static const char *code_helper_funcs = \
"\n#define SAMPLE_STRIDE 128 \n" \
"\n#define SHARED_SIZE_LIMIT  1024U \n" \
"__device__ unsigned int iDivUp(unsigned int a, unsigned int b)"\
"{"\
"    return ((a % b) == 0) ? (a / b) : (a / b + 1); "\
"} "\
"__device__ unsigned int getSampleCount(unsigned int dividend) "\
"{ "\
"    return iDivUp(dividend, SAMPLE_STRIDE); "\
"}"\
" \n #define W (sizeof(unsigned int) * 8) \n"\
"__device__ unsigned int nextPowerOfTwo(unsigned int x) "\
"{"\
"    return 1U << (W - __clz(x - 1));"\
"} "\
" __device__ unsigned int readArray(unsigned int *a, unsigned int pos, unsigned int length, unsigned int sortDir){"                       \
"      if (pos >= length) { "                                                                                                                           \
"          if (sortDir) { "                                                                                                                             \
"             return 4294967295; "                                                                                                                        \
"          } "                                                                                                                                          \
"          else { "                                                                                                                                     \
"              return 0; "                                                                                                                              \
"          } "                                                                                                                                          \
"      } "                                                                                                                                              \
"      else { "                                                                                                                                         \
"          return a[pos]; "                                                                                                                             \
"      } "                                                                                                                                              \
"  } "                                                                                                                                                  \
" __device__ void writeArray(unsigned int *a, unsigned int pos, unsigned int value, unsigned int length) "                                             \
" { "                                                                                                                                                  \
"     if (pos >= length) "                                                                                                                             \    
"     { "                                                                                                                                              \
"          return; "                                                                                                                                   \
"     } "\
"     else { "    \
"         a[pos] = value; "                                                                                                                            \
"     } "\
" }\n";

static unsigned int iDivUp(unsigned int a, unsigned int b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static unsigned int getSampleCount(unsigned int dividend)
{
    return iDivUp(dividend, SAMPLE_STRIDE);
}

static unsigned int ceiling(unsigned int n, unsigned int v)
{
    return (!n%v) ? n/v : (n/v) + 1;
}

static const char *code_bin_search =                                                          \
"__device__ unsigned int binarySearchInclusive(unsigned int val, unsigned int *data, unsigned int L, "\
"                                              unsigned int stride, unsigned int sortDir){"\
"    if (L == 0) "\
"        return 0; "\
"    unsigned int pos = 0; "\
"    for (; stride > 0; stride >>= 1){ "\
"      unsigned int newPos = min(pos + stride, L); "\
"      if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val))){ "\
"          pos = newPos; "\
"      } "\
"    } "\
"    return pos; "\
"} "\
"__device__ unsigned int binarySearchExclusive(unsigned int val, unsigned int *data, unsigned int L, " \
"                                              unsigned int stride, unsigned int sortDir) "\
"{ "\
"    if (L == 0) "\
"        return 0; "\
"    unsigned int pos = 0; "\
"    for (; stride > 0; stride >>= 1){ "\
"      unsigned int newPos = min(pos + stride, L); "\
"      if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val))){ "\
"          pos = newPos; "\
"      } "\
"    } "\
"    return pos; "\
"}"\
"__device__ unsigned int binarySearchLowerBoundExclusive(unsigned int val, unsigned int *ptr, unsigned int first, "\
"                                                        unsigned int last, unsigned int sortDir) "\
"{ "\    
"    unsigned int len = last - first; "\
"    unsigned int half; "\
"    unsigned int middle; "\
"    while (len > 0) "\
"    { "\
"        half = len >> 1; "\
"        middle = first; "\
"        middle += half; "\
"        if ( (sortDir && ptr[middle] < val) || (!sortDir && ptr[middle] > val) ) "\
"        { "\
"            first = middle; "\
"            ++first; "\
"            len = len - half - 1; "\
"        } "\
"        else "\
"            len = half; "\
"    } "\
"    return first; "\
"} "\
"__device__ unsigned int binarySearchLowerBoundInclusive(unsigned int val, unsigned int *ptr, unsigned int first,  "\
"                                                        unsigned int last, unsigned int sortDir) "\
"{    "\
"    unsigned int len = last - first; "\
"    unsigned int half; "\
"    unsigned int middle; "\
"    while (len > 0) "\
"    { "\
"        half = len >> 1; "\
"        middle = first; "\
"        middle += half; "\
"        if ( (sortDir && ptr[middle] <= val) || (!sortDir && ptr[middle] >= val) ) "\
"        { "\
"            first = middle; "\
"            ++first; "\ 
"            len = len - half - 1; "\
"        } "\
"        else "\
"            len = half; "\
"    } "\
"    return first; "\
"}\n";

#define NUMARGS_BITONIC_KERNEL 8
const int type_args_bitonic[NUMARGS_BITONIC_KERNEL] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT, GA_UINT};
static const char *code_bitonic_smem =                                                                                                                  \
" extern \"C\" __global__ void bitonicSortSharedKernel( "\
"      unsigned int *d_DstKey, "\
"      size_t dstOff,"
"      unsigned int *d_SrcKey, "\
"      size_t srcOff,"
"      unsigned int batchSize, "\
"      unsigned int arrayLength, "\
"      unsigned int elemsOff, " \
"      unsigned int sortDir "\
"  ) "\
"  { "\
"      d_DstKey = (unsigned int*) (((char*)d_DstKey)+ dstOff);" \
"      d_SrcKey = (unsigned int*) (((char*)d_SrcKey)+ srcOff);" \
"      d_DstKey += elemsOff;" \
"      d_SrcKey += elemsOff;" \
"      __shared__ unsigned int s_key[SHARED_SIZE_LIMIT]; "\
"      s_key[threadIdx.x] = readArray( d_SrcKey, "\
"                                      blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x, "\
"                                      arrayLength * batchSize, "\
"                                      sortDir "\
"                                      ); "\
"      s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = readArray( d_SrcKey, "\
"                                                   blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT / 2), "\
"                                                   arrayLength * batchSize, "\
"                                                   sortDir "\
"                                                  ); "\
"      for (unsigned int size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) "\
"      { "\
"          unsigned int ddd = sortDir ^ ((threadIdx.x & (size / 2)) != 0); "\
"          for (unsigned int stride = size / 2; stride > 0; stride >>= 1) "\
"          { "\
"              __syncthreads(); "\
"              unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1)); "\
"              unsigned int t; "\
"              if ((s_key[pos] > s_key[pos + stride]) == ddd) { "\
"                  t = s_key[pos]; "\
"                  s_key[pos] = s_key[pos + stride]; "\
"                  s_key[pos + stride] = t; "\
"              } "\
"          } "\
"      } "\
"      { "\
"          for (unsigned int stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {" \
"              __syncthreads(); "\
"              unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1)); "\
"              unsigned int t; "\
"              if ((s_key[pos] > s_key[pos + stride]) == sortDir) {" \
"                  t = s_key[pos]; "\
"                  s_key[pos] = s_key[pos + stride]; "\
"                  s_key[pos + stride] = t; "\
"              } "\
"          } "\
"      } "\
"      __syncthreads(); "\
"      writeArray( d_DstKey, "\
"                  blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x,  "\
"                  s_key[threadIdx.x], "\
"                  arrayLength * batchSize "\
"                ); "\
"      writeArray( d_DstKey, "\
"                  blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT / 2), "\
"                  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)], "\
"                  arrayLength * batchSize "\
"                ); "\
"  }\n";
#define NSTR_BITONIC 2
static void bitonicSortShared(
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
  char *err_str = NULL;
  size_t ls, gs;
  unsigned int p = 0;
  int err;
  size_t lens[NSTR_BITONIC] = {strlen(code_helper_funcs), strlen(code_bitonic_smem)};
  const char *codes[NSTR_BITONIC] = {code_helper_funcs, code_bitonic_smem};

  err = GpuKernel_init( k_bitonic, ctx, NSTR_BITONIC, 
                         codes, lens, "bitonicSortSharedKernel",
                         NUMARGS_BITONIC_KERNEL, type_args_bitonic, flags, &err_str);
  if (err != GA_NO_ERROR) printf("error kernel init: %s \n", gpuarray_error_str(err));
  if (err != GA_NO_ERROR)  printf("error backend: %s \n", err_str);

  ls = SHARED_SIZE_LIMIT / 2;
  gs = batchSize;

  err = GpuKernel_setarg(k_bitonic, p++, d_DstKey->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_bitonic, p++, &d_DstKey->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);
  
  err = GpuKernel_setarg(k_bitonic, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_bitonic, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);
  
  err = GpuKernel_setarg(k_bitonic, p++, &batchSize);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);
  
  err = GpuKernel_setarg(k_bitonic, p++, &arrayLength);
  if (err != GA_NO_ERROR) printf("eror setting arg %d \n", p);

  err = GpuKernel_setarg(k_bitonic, p++, &elemsOff);
  if (err != GA_NO_ERROR) printf("eror setting arg %d \n", p);
  
  err = GpuKernel_setarg(k_bitonic, p++, &sortDir);
  if (err != GA_NO_ERROR) printf("eror setting arg %d \n", p);

  err = GpuKernel_call(k_bitonic, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) printf("error calling kernel %d \n", p);

}

#define NUMARGS_SAMPLE_RANKS 10
const int type_args_ranks[NUMARGS_SAMPLE_RANKS] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT, GA_UINT};
static const char *code_sample_ranks =                                            \
"extern \"C\" __global__ void generateSampleRanksKernel("               \
"    unsigned int *d_RanksA,"\
"    size_t rankAOff,"         \
"    unsigned int *d_RanksB,"\
"    size_t rankBOff,"         \
"    unsigned int *d_SrcKey,"\
"    size_t srcOff,"         \
"    unsigned int stride,"   \
"    unsigned int N,"        \
"    unsigned int threadCount,"\
"    unsigned int sortDir"   \
")" \
"{" \
"    d_RanksA = (unsigned int*) (((char*)d_RanksA)+ rankAOff);" \
"    d_RanksB = (unsigned int*) (((char*)d_RanksB)+ rankBOff);" \
"    d_SrcKey = (unsigned int*) (((char*)d_SrcKey)+ srcOff);" \
"    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;"\
"    if (pos >= threadCount)" \
"    {"\
"        return;"\
"    }"\
"    const unsigned int           i = pos & ((stride / SAMPLE_STRIDE) - 1);"\
"    const unsigned int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);"\
"    d_SrcKey += segmentBase;"\
"    d_RanksA += segmentBase / SAMPLE_STRIDE;"\
"    d_RanksB += segmentBase / SAMPLE_STRIDE;"\
"    const unsigned int segmentElementsA = stride;"\
"    const unsigned int segmentElementsB = min(stride, N - segmentBase - stride);"\
"    const unsigned int  segmentSamplesA = getSampleCount(segmentElementsA);"\
"    const unsigned int  segmentSamplesB = getSampleCount(segmentElementsB);"\
"    if (i < segmentSamplesA)"\
"    {"\
"        d_RanksA[i] = i * SAMPLE_STRIDE;"\
"        d_RanksB[i] = binarySearchExclusive("\
"                          d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride,"\
"                          segmentElementsB, nextPowerOfTwo(segmentElementsB), sortDir"\
"                      );"\
"    }"\
"    if (i < segmentSamplesB)"\
"    {"\
"        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;"\
"        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive("\
"                                                     d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0,"\
"                                                     segmentElementsA, nextPowerOfTwo(segmentElementsA), sortDir"\
"                                                 );"\
"    }"\
"}\n";
#define NSTR_RANKS 3
static void generateSampleRanks(
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

  char *err_str = NULL;
  size_t ls, gs;
  unsigned int p = 0;
  int err;
  const char *codes[NSTR_RANKS] = {code_helper_funcs, code_bin_search, code_sample_ranks};
  size_t lens[NSTR_RANKS] = {strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_sample_ranks)};

  err = GpuKernel_init(k_ranks, ctx, NSTR_RANKS, 
                       codes, lens, "generateSampleRanksKernel",
                       NUMARGS_SAMPLE_RANKS, type_args_ranks, flags, &err_str);
  if (err != GA_NO_ERROR) printf("error kernel init: %s \n", gpuarray_error_str(err));
  if (err != GA_NO_ERROR)  printf("error backend: %s \n", err_str);

  ls = 256;
  gs = iDivUp(threadCount, 256);

  err = GpuKernel_setarg(k_ranks, p++, d_RanksA->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks, p++, &d_RanksA->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks, p++, d_RanksB->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks, p++, &d_RanksB->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);
  
  err = GpuKernel_setarg(k_ranks, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks, p++, &stride);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks, p++, &N);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks, p++, &threadCount);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks, p++, &sortDir);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_call(k_ranks, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) printf("error calling Ranks kernel %d \n", p);

  

  /*unsigned int *h_dst = (unsigned int *) malloc ( (2048/128) * sizeof(unsigned int));
  err = GpuArray_read(h_dst, (2048/128) * sizeof(unsigned int), d_RanksA);
  if (err != GA_NO_ERROR) printf("error reading \n");

  unsigned int *h_dst2 = (unsigned int *) malloc ( (2048/128) * sizeof(unsigned int));
  err = GpuArray_read(h_dst2, (2048/128) * sizeof(unsigned int), d_RanksB);
  if (err != GA_NO_ERROR) printf("error reading \n");

  int i;
  for (i = 0; i < 2048/128; i++)
  {
      printf("%d rankA %u rankB %u \n", i, h_dst[i], h_dst2[i]);
  }
  */
}

#define NUMARGS_RANKS_IDXS 7
const int type_args_ranks_idxs[NUMARGS_RANKS_IDXS] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT};
static const char *code_ranks_idxs =                                            \
"extern \"C\" __global__ void mergeRanksAndIndicesKernel( "\
"    unsigned int *d_Limits, "\
"    size_t limOff,"         \
"    unsigned int *d_Ranks, "\
"    size_t rankOff,"         \
"    unsigned int stride, "\
"    unsigned int N, "\
"    unsigned int threadCount "\
") "\ 
"{ "\
"    d_Limits = (unsigned int*) (((char*)d_Limits)+ limOff);" \
"    d_Ranks = (unsigned int*) (((char*)d_Ranks)+ rankOff);" \
"    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x; "\
"    if (pos >= threadCount) "\
"        return; "\
"    const unsigned int           i = pos & ((stride / SAMPLE_STRIDE) - 1); "\
"    const unsigned int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE); "\
"    d_Ranks  += (pos - i) * 2; "\
"    d_Limits += (pos - i) * 2; "\
"    const unsigned int segmentElementsA = stride; "\
"    const unsigned int segmentElementsB = min(stride, N - segmentBase - stride); "\
"    const unsigned int  segmentSamplesA = getSampleCount(segmentElementsA); "\
"    const unsigned int  segmentSamplesB = getSampleCount(segmentElementsB); "\
"    if (i < segmentSamplesA) "\
"    { "\
"        unsigned int dstPos = binarySearchExclusive(d_Ranks[i], d_Ranks + segmentSamplesA, segmentSamplesB, nextPowerOfTwo(segmentSamplesB), 1U) + i; "\
"        d_Limits[dstPos] = d_Ranks[i]; "\
"    } "\
"    if (i < segmentSamplesB) "\
"    { "\
"        unsigned int dstPos = binarySearchInclusive(d_Ranks[segmentSamplesA + i], d_Ranks, segmentSamplesA, nextPowerOfTwo(segmentSamplesA), 1U) + i; "\
"        d_Limits[dstPos] = d_Ranks[segmentSamplesA + i]; "\
"    } "\
"}\n";
#define NSTRINGS_RKS_IDX 3
static void mergeRanksAndIndices(
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
  unsigned int threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);


  char *err_str = NULL;
  size_t ls, gs;
  unsigned int p = 0;
  int err;
  const char *codes[NSTRINGS_RKS_IDX] = {code_helper_funcs, code_bin_search, code_ranks_idxs};
  size_t lens[NSTRINGS_RKS_IDX] = {strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_ranks_idxs)};

  err = GpuKernel_init(k_ranks_idxs, ctx, NSTRINGS_RKS_IDX, 
                       codes, lens, "mergeRanksAndIndicesKernel",
                       NUMARGS_RANKS_IDXS, type_args_ranks_idxs, flags, &err_str);
  if (err != GA_NO_ERROR) printf("error kernel init: %s \n", gpuarray_error_str(err));
  if (err != GA_NO_ERROR)  printf("error backend: %s \n", err_str);

  ls = 256U;
  gs = iDivUp(threadCount, 256U);

  err = GpuKernel_setarg(k_ranks_idxs, p++, d_LimitsA->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks_idxs, p++, &d_LimitsA->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks_idxs, p++, d_RanksA->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks_idxs, p++, &d_RanksA->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks_idxs, p++, &stride);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks_idxs, p++, &N);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks_idxs, p++, &threadCount);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_call(k_ranks_idxs, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) printf("error calling Ranks kernel %d \n", p);

  p = 0;

  err = GpuKernel_setarg(k_ranks_idxs, p++, d_LimitsB->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks_idxs, p++, &d_LimitsB->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks_idxs, p++, d_RanksB->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_ranks_idxs, p++, &d_RanksB->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_call(k_ranks_idxs, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) printf("error calling Ranks kernel %d \n", p);


  unsigned int *h_dst = (unsigned int *) malloc ( (2048/128) * sizeof(unsigned int));
  err = GpuArray_read(h_dst, (2048/128) * sizeof(unsigned int), d_LimitsB);
  if (err != GA_NO_ERROR) printf("error reading \n");

  unsigned int *h_dst2 = (unsigned int *) malloc ( (2048/128) * sizeof(unsigned int));
  err = GpuArray_read(h_dst2, (2048/128) * sizeof(unsigned int), d_RanksB);
  if (err != GA_NO_ERROR) printf("error reading \n");

  /*
  int i;
  for (i = 0; i < 2048/128; i++)
  {
      printf("%d Limit %u Rank %u \n", i, h_dst[i], h_dst2[i]);
  }
  */

  /*mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
      d_LimitsA,
      d_RanksA,
      stride,
      N,
      threadCount
  );
  printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);

  mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
      d_LimitsB,
      d_RanksB,
      stride,
      N,
      threadCount
  );
  printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);
  */
}

#define NUMARGS_MERGE 11
const int type_args_merge[NUMARGS_MERGE] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT};
static const char *code_merge =                                            \
"__device__ void merge( "\
"    unsigned int *dstKey, "\
"    unsigned int *srcAKey, "\
"    unsigned int *srcBKey, "\
"    unsigned int lenA, "\
"    unsigned int nPowTwoLenA, "\
"    unsigned int lenB, "\
"    unsigned int nPowTwoLenB, "\
"    unsigned int sortDir "\
") "\
"{ "\
"    unsigned int keyA, keyB; "\
"    unsigned int dstPosA , dstPosB;"\
"    if (threadIdx.x < lenA) "\
"    { "\
"        keyA = srcAKey[threadIdx.x]; "\
"        dstPosA = binarySearchExclusive(keyA, srcBKey, lenB, nPowTwoLenB, sortDir) + threadIdx.x; "\
"    } "\
"    if (threadIdx.x < lenB) "\
"    { "\
"        keyB = srcBKey[threadIdx.x]; "\
"        dstPosB = binarySearchInclusive(keyB, srcAKey, lenA, nPowTwoLenA, sortDir) + threadIdx.x; "\
"    } "\
"    __syncthreads(); "\
"    if (threadIdx.x < lenA) "\
"    { "\
"        dstKey[dstPosA] = keyA; "\
"    } "\
"    if (threadIdx.x < lenB) "\
"    { "\
"        dstKey[dstPosB] = keyB; "\
"    } "\
"} "\
"extern \"C\" __global__ void mergeElementaryIntervalsKernel( "\
"    unsigned int *d_DstKey, "\
"    size_t dstOff,"         \
"    unsigned int *d_SrcKey, "\
"    size_t srcOff,"         \
"    unsigned int *d_LimitsA, "\
"    size_t limAOff,"         \
"    unsigned int *d_LimitsB, "\
"    size_t limBOff,"         \
"    unsigned int stride, "\
"    unsigned int N, "\
"    unsigned int sortDir"
") "\
"{ "\
"    d_DstKey = (unsigned int*) (((char*)d_DstKey)+ dstOff);" \
"    d_SrcKey = (unsigned int*) (((char*)d_SrcKey)+ srcOff);" \
"    d_LimitsA = (unsigned int*) (((char*)d_LimitsA)+ limAOff);" \
"    d_LimitsB = (unsigned int*) (((char*)d_LimitsB)+ limBOff);" \
"    __shared__ unsigned int s_key[2 * SAMPLE_STRIDE]; "\
"    const unsigned int   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1); "\
"    const unsigned int segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE; "\
"    d_SrcKey += segmentBase; "\
"    d_DstKey += segmentBase; "\
"    __shared__ unsigned int startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB; "\
"    if (threadIdx.x == 0) "\
"    { "\
"        unsigned int segmentElementsA = stride; "\
"        unsigned int segmentElementsB = min(stride, N - segmentBase - stride); "\
"        unsigned int  segmentSamplesA = getSampleCount(segmentElementsA); "\
"        unsigned int  segmentSamplesB = getSampleCount(segmentElementsB); "\
"        unsigned int   segmentSamples = segmentSamplesA + segmentSamplesB; "\
"        startSrcA    = d_LimitsA[blockIdx.x]; "\
"        startSrcB    = d_LimitsB[blockIdx.x]; "\
"        unsigned int endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA; "\
"        unsigned int endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB; "\
"        lenSrcA      = endSrcA - startSrcA; "\ 
"        lenSrcB      = endSrcB - startSrcB; "\
"        startDstA    = startSrcA + startSrcB; "\
"        startDstB    = startDstA + lenSrcA; "\
"    } "\
"    __syncthreads(); "\
"    if (threadIdx.x < lenSrcA) "\
"    { "\
"        s_key[threadIdx.x +             0] = d_SrcKey[0 + startSrcA + threadIdx.x]; "\
"    } "\
"   if (threadIdx.x < lenSrcB) "\
"    { "\
"        s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x]; "\
"    } "\
"    __syncthreads(); "\
"    merge( "\
"        s_key, "\
"        s_key + 0, "\
"        s_key + SAMPLE_STRIDE, "\
"        lenSrcA, SAMPLE_STRIDE, "\
"        lenSrcB, SAMPLE_STRIDE, "\
"        sortDir "\
"    ); "\
"    __syncthreads(); "\
"    if (threadIdx.x < lenSrcA) "\
"    { "\
"        d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x]; "\
"    } "\
"    if (threadIdx.x < lenSrcB) "\
"    { "\
"        d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x]; "\
"    } "\
"}\n";
#define NSTRINGS_MERGE 3
static void mergeElementaryIntervals(
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

  char *err_str = NULL;
  size_t ls, gs;
  unsigned int p = 0;
  int err;
  const char *codes[NSTRINGS_MERGE] = {code_helper_funcs, code_bin_search, code_merge};
  size_t lens[NSTRINGS_MERGE] = {strlen(code_helper_funcs), strlen(code_bin_search), strlen(code_merge)};

  err = GpuKernel_init(k_merge, ctx, NSTRINGS_MERGE, 
                       codes, lens, "mergeElementaryIntervalsKernel",
                       NUMARGS_MERGE, type_args_merge, flags, &err_str);
  if (err != GA_NO_ERROR) printf("error kernel init: %s \n", gpuarray_error_str(err));
  if (err != GA_NO_ERROR)  printf("error backend: %s \n", err_str);  


  ls = SAMPLE_STRIDE;
  gs = mergePairs;

  err = GpuKernel_setarg(k_merge, p++, d_DstKey->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge, p++, &d_DstKey->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p); 

  err = GpuKernel_setarg(k_merge, p++, d_LimitsA->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge, p++, &d_LimitsA->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p); 

  err = GpuKernel_setarg(k_merge, p++, d_LimitsB->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge, p++, &d_LimitsB->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p); 

  err = GpuKernel_setarg(k_merge, p++, &stride);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p); 

  err = GpuKernel_setarg(k_merge, p++, &N);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p); 

  err = GpuKernel_setarg(k_merge, p++, &sortDir);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p); 

  err = GpuKernel_call(k_merge, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) printf("error calling Ranks kernel %d \n", p);  

/*
    if (sortDir)
    {
       mergeElementaryIntervalsKernel<1U><<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_SrcKey,
            d_LimitsA,
            d_LimitsB,
            stride,
            N
        );
        printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);
    }
    else
    {
      
        mergeElementaryIntervalsKernel<0U><<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_SrcKey,
            d_LimitsA,
            d_LimitsB,
            stride,
            N
        );
        printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);
      
    }
  */
}

#define NUMARGS_MERGE_GLB 8
const int type_args_merge_glb[NUMARGS_MERGE_GLB] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT, GA_UINT, GA_UINT, GA_UINT};
static const char *code_merge_glb =                                            \
"extern \"C\" __global__ void mergeGlobalMemKernel( "\
"    unsigned int *d_DstKey, "\
"    size_t dstOff, "\
"    unsigned int *d_SrcKey, "\
"    size_t srcOff, "\
"    unsigned int segmentSizeA, "\
"    unsigned int segmentSizeB, "\
"    unsigned int N, "\
"    unsigned int sortDir "\
") "\ 
"{ "\
"    d_DstKey = (unsigned int*) (((char*)d_DstKey)+ dstOff);" \
"    d_SrcKey = (unsigned int*) (((char*)d_SrcKey)+ srcOff);" \
"    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; "\
"    unsigned int *segmentPtrA = d_SrcKey; "\
"    unsigned int *segmentPtrB = d_SrcKey + segmentSizeA; "\
"    unsigned int idxSegmentA = idx % segmentSizeA; "\
"    unsigned int idxSegmentB = idx - segmentSizeA; "\
"    if (idx >= N) "\
"        return; "\
"    unsigned int value = d_SrcKey[idx]; "\
"    unsigned int dstPos; "\
"    if (idx < segmentSizeA) "\
"    { "\
"        dstPos = binarySearchLowerBoundExclusive(value, segmentPtrB, 0, segmentSizeB, sortDir) + idxSegmentA; "\
"    } "\
"    else "\
"    { "\
"        dstPos = binarySearchLowerBoundInclusive(value, segmentPtrA, 0, segmentSizeA, sortDir) + idxSegmentB; "\
"    } "\
"    d_DstKey[dstPos] = value; "\
"}\n";

#define NSTRINGS_MERGE_GLB 2
static void mergeGlobalMem(
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
  char *err_str = NULL;
  size_t ls, gs;
  unsigned int p = 0;
  int err;
  const char *codes[NSTRINGS_MERGE_GLB] = {code_bin_search, code_merge_glb};
  size_t lens[NSTRINGS_MERGE_GLB] = {strlen(code_bin_search), strlen(code_merge_glb)};

  err = GpuKernel_init(k_merge_global, ctx, NSTRINGS_MERGE_GLB, 
                       codes, lens, "mergeGlobalMemKernel",
                       NUMARGS_MERGE_GLB, type_args_merge_glb, flags, &err_str);
  if (err != GA_NO_ERROR) printf("error kernel init: %s \n", gpuarray_error_str(err));
  if (err != GA_NO_ERROR)  printf("error backend: %s \n", err_str);  

  ls = 256;
  gs = ceiling(N, ls);


  err = GpuKernel_setarg(k_merge_global, p++, d_DstKey->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge_global, p++, &d_DstKey->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge_global, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge_global, p++, &d_SrcKey->offset);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge_global, p++, &segmentSizeA);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge_global, p++, &segmentSizeB);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge_global, p++, &N);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_merge_global, p++, &sortDir);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_call(k_merge_global, 1, &gs, &ls, 0, NULL);
  if (err != GA_NO_ERROR) printf("error calling Ranks kernel %d \n", p);
/*
    if (sortDir)
    {
        //mergeLeftMostSegmentKernel<1U><<<blockCount, blockDim>>>(d_DstKey, d_SrcKey, segmentSizeA, segmentSizeB, N);
        //printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);
    }
    else
    {
        //mergeLeftMostSegmentKernel<0U><<<blockCount, blockDim>>>(d_DstKey, d_SrcKey, segmentSizeA, segmentSizeB, N);
        //printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);
    }
*/
}

static void sort(
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
    GpuArray *ikey, *okey;
    GpuArray *t; // Aux pointer

    GpuKernel k_bitonic;
    GpuKernel k_ranks;
    GpuKernel k_ranks_idxs;
    GpuKernel k_merge;
    GpuKernel k_merge_global;

    size_t lstCopyOff;
    int err;

    unsigned int stageCount = 0;
    unsigned int stride;
    for (stride = SHARED_SIZE_LIMIT; stride < Nfloor; stride <<= 1, stageCount++);

    if (stageCount & 1)
    {
      printf("bffkey\n");
      ikey = d_BufKey;
      okey = d_DstKey;
    }
    else
    {
      printf("d_DstKey\n");
      ikey = d_DstKey;
      okey = d_BufKey;
    }

    /////////////////////////////////////////////////////////////////////////
    // Sort the array with bitonic sort for arrays shorter than 1024 elements
    // Bitonic sort gives better performance than merge sort for short arrays
    /////////////////////////////////////////////////////////////////////////
  
    if (N <= SHARED_SIZE_LIMIT)
    {  
      bitonicSortShared(d_DstKey, d_SrcKey, 1, N, sortDir, 0, &k_bitonic, ctx); 
    }
    ///////////////////////////////////////////////////////////////////////////////
    // Sort the array with merge sort for arrays equal or bigger than 1024 elements
    ///////////////////////////////////////////////////////////////////////////////
    else
    {
      unsigned int batchSize = Nfloor / SHARED_SIZE_LIMIT;
      unsigned int arrayLength = SHARED_SIZE_LIMIT;
      bitonicSortShared(ikey, d_SrcKey, batchSize, arrayLength, sortDir, 0, &k_bitonic, ctx);

      for (stride = SHARED_SIZE_LIMIT; stride < Nfloor; stride <<= 1)
      {
        unsigned int lastSegmentElements = Nfloor % (2 * stride);

        //Find sample ranks and prepare for limiters merge
        generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, Nfloor, sortDir, &k_ranks, ctx);   

        //Merge ranks and indices
        mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, Nfloor, sortDir, &k_ranks_idxs, ctx);

        //Merge elementary intervals
        mergeElementaryIntervals(okey, ikey, d_LimitsA, d_LimitsB, stride, Nfloor, sortDir, &k_merge, ctx);

        if (lastSegmentElements <= stride)
        {
          //Last merge segment consists of a single array which just needs to be passed through
          printf("inside last segment\n");
           // TODO: uncomment and fix sizeof
          //////////////////////////////////
          //cudaMemcpy( okey + (Nfloor - lastSegmentElements), 
          //              ikey + (Nfloor - lastSegmentElements), 
          //              lastSegmentElements * sizeof(t_key), 
          //              cudaMemcpyDeviceToDevice
          //         );
          
          lstCopyOff = okey->offset + ((Nfloor - lastSegmentElements) * sizeof(unsigned int));
          err = gpudata_move(okey->data, lstCopyOff, ikey->data, lstCopyOff, lastSegmentElements * sizeof(unsigned int));
          if (err != GA_NO_ERROR) printf("error move data\n");
          //err = GpuArray_copy(okey, ikey, GA_C_ORDER);
          //if (err != GA_NO_ERROR) printf("error move data\n");
        }
        // Swap pointers
        t = ikey;
        ikey = okey;
        okey = t;
      }
      
      // If the array is not multiple of 1024, sort the leftmost part
      // and perform merge sort of the two last segments
      if (Nleft > 0)
      {
        printf("Sorting Remaining part %d \n", Nleft);
        bitonicSortShared(d_SrcKey, d_DstKey, 1, Nleft, sortDir, Nfloor, &k_bitonic, ctx);

        unsigned int *h_dst = (unsigned int *) malloc ( N * sizeof(unsigned int));
        err = GpuArray_read(h_dst, N * sizeof(unsigned int), d_SrcKey);
        if (err != GA_NO_ERROR) printf("error reading \n");

        int i;
        for (i = 0; i < N; i++)
        {
            printf("%d value %u \n", i, h_dst[i]);
        }


        // Copy the leftMost segment to the output array of which contains the first sorted sequence
        // TODO: uncomment and fix sizeof
        //////////////////////////////////
        //checkCudaErrors(cudaMemcpy(d_DstKey + Nfloor, d_SrcKey + Nfloor, Nleft * sizeof(t_key), cudaMemcpyDeviceToDevice));
        lstCopyOff = okey->offset + Nfloor;
        err = gpudata_move(d_DstKey->data, lstCopyOff, d_SrcKey->data, lstCopyOff, Nleft * sizeof(unsigned int));
        //GpuArray_copy(d_DstKey, d_SrcKey, GA_C_ORDER); // TODO: copy just the needed part of the buffer

        mergeGlobalMem(d_SrcKey, d_DstKey, Nfloor, (unsigned int)Nleft, N, sortDir, &k_merge_global, ctx);

        GpuArray_copy(d_DstKey, d_SrcKey, GA_C_ORDER);
      }
    }
    //GpuArray_copy(d_DstKey, d_BufKey, GA_C_ORDER);
    //cudaDeviceSynchronize();
}

unsigned int roundDown(unsigned int numToRound, unsigned int multiple)
{
  if (numToRound <= multiple)
  {
      return numToRound;
  }
  else
  {
      return (numToRound / multiple) * multiple;    
  }
}

void initMergeSort(
  GpuArray *d_RanksA,
  GpuArray *d_RanksB,
  GpuArray *d_LimitsA,
  GpuArray *d_LimitsB,
  unsigned int MAX_SAMPLE_COUNT,
  gpucontext *ctx
)
{
  int res = GA_NO_ERROR;
  const unsigned int nd = 1;
  const size_t dims =  MAX_SAMPLE_COUNT * sizeof(unsigned int);

  //d_RanksA = gpudata_alloc(ctx, MAX_SAMPLE_COUNT * sizeof(unsigned int), NULL, GA_BUFFER_READ_WRITE, &res);
  res = GpuArray_empty(d_RanksA, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (res != GA_NO_ERROR) printf("error allocating aux structures %d\n", res);

  //d_RanksB = gpudata_alloc(ctx, MAX_SAMPLE_COUNT * sizeof(unsigned int), NULL, GA_BUFFER_READ_WRITE, &res);
  res = GpuArray_empty(d_RanksB, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (res != GA_NO_ERROR) printf("error allocating aux structures %d\n", res);
  
  //d_LimitsA = gpudata_alloc(ctx, MAX_SAMPLE_COUNT * sizeof(unsigned int), NULL, GA_BUFFER_READ_WRITE, &res);
  res = GpuArray_empty(d_LimitsA, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (res != GA_NO_ERROR) printf("error allocating aux structures %d\n", res);

  //d_LimitsB = gpudata_alloc(ctx, MAX_SAMPLE_COUNT * sizeof(unsigned int), NULL, GA_BUFFER_READ_WRITE, &res);
  res = GpuArray_empty(d_LimitsB, ctx, GA_UINT, nd, &dims, GA_C_ORDER);
  if (res != GA_NO_ERROR) printf("error allocating aux structures %d\n", res);
}


int GpuArray_sort(GpuArray *dst, GpuArray *src, unsigned int sortDir, GpuArray *arg)
{

  int type = src->typecode;
  gpucontext *ctx = GpuArray_context(src);

  // Device pointers - auxiiary data structure    
  //gpudata *d_RanksA = NULL, *d_RanksB = NULL, *d_LimitsA = NULL, *d_LimitsB = NULL;
  GpuArray d_RanksA, d_RanksB, d_LimitsA, d_LimitsB;

	if (arg != NULL)
  {
    // perform argsort
    assert(arg != NULL);
  }
  else
  {
    const unsigned int nd = 1;
    const size_t dims = src->dimensions[0];

    const unsigned int Nfloor = roundDown(dims, SHARED_SIZE_LIMIT);
    const int Nleft = dims - Nfloor;

    // Device pointers - buffer data strucute
    GpuArray BufKey;
    GpuArray_empty(&BufKey, ctx, type, nd, &dims, GA_C_ORDER);

    // Initialize device  auxiliary data structure  
    initMergeSort(&d_RanksA, &d_RanksB, &d_LimitsA, &d_LimitsB, Nfloor / 128, ctx);

    // perform regular sort
    sort(
      dst,
      &BufKey,
      src,
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


    // type -> get typecode of the array

    // vectorType -> "type"

    // stbr_append all the kernels....

    // Set arguments
  }

  return 0;

}
