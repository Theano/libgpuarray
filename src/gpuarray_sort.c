#include <assert.h>

#include <gpuarray/sort.h>
#include <gpuarray/array.h>
#include <gpuarray/kernel.h>


#include "util/strb.h"
#include "private.h"

const int flags = GA_USE_CUDA;

#define NUMARGS_BITONIC_KERNEL 5
const int type_args_bitonic[NUMARGS_BITONIC_KERNEL] = {GA_BUFFER, GA_BUFFER, GA_UINT, GA_UINT, GA_UINT};
static const char *code_bitonic_smem =                                                                                                                  \
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
"  __device__ void writeArray(unsigned int *a, unsigned int pos, unsigned int value, unsigned int length) "                                             \
"  { "                                                                                                                                                  \
"      if (pos >= length) "                                                                                                                             \    
"      { "                                                                                                                                              \
"           return; "                                                                                                                                   \
"      } "\
"      else { "    \
"          a[pos] = value; "                                                                                                                            \
"      } "\
"  } "  \
" extern \"C\" __global__ void bitonicSortSharedKernel( "\
"      unsigned int *d_DstKey, "\
"      unsigned int *d_SrcKey, "\
"      unsigned int batchSize, "\
"      unsigned int arrayLength, "\
"      unsigned int sortDir "\
"  ) "\
"  { "\
"      __shared__ unsigned int s_key[1024]; "\
"      s_key[threadIdx.x] = readArray( d_SrcKey, "\
"                                      blockIdx.x * 1024 + threadIdx.x, "\
"                                      arrayLength * batchSize, "\
"                                      sortDir "\
"                                      ); "\
"      s_key[threadIdx.x + (1024 / 2)] = readArray( d_SrcKey, "\
"                                                                blockIdx.x * 1024 + threadIdx.x + (1024 / 2), "\
"                                                                arrayLength * batchSize, "\
"                                                                sortDir "\
"                                                              ); "\
"      for (unsigned int size = 2; size < 1024; size <<= 1) "\
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
"          for (unsigned int stride = 1024 / 2; stride > 0; stride >>= 1) {" \
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
"                  blockIdx.x * 1024 + threadIdx.x,  "\
"                  s_key[threadIdx.x], "\
"                  arrayLength * batchSize "\
"                ); "\
"      writeArray( d_DstKey, "\
"                  blockIdx.x * 1024 + threadIdx.x + (1024 / 2), "\
"                  s_key[threadIdx.x + (1024 / 2)], "\
"                  arrayLength * batchSize "\
"                ); "\
"  }\n";

#define NUMARGS_CODE_K 5
const int type_args_code_k[NUMARGS_CODE_K] = {GA_BUFFER, GA_SIZE, GA_BUFFER, GA_SIZE, GA_UINT};
static const char *code_k =                                                                                                                  \
" extern \"C\" __global__ void add( "\
"      unsigned int *d_DstKey, size_t dstOff, unsigned int *d_SrcKey, size_t srcOff, unsigned int N "\
"  ) "\
"  { "\
"    d_DstKey = (unsigned int*) (((char*)d_DstKey)+ dstOff);" \
"    d_SrcKey = (unsigned int*) (((char*)d_SrcKey)+ srcOff);" \
"    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;" \
"    if (i < N) d_DstKey[i] = d_SrcKey[i] + 1;" \
"  }\n";
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

static void bitonicSortShared(
    GpuArray *d_DstKey,
    GpuArray *d_SrcKey,
    unsigned int batchSize,
    unsigned int arrayLength,
    unsigned int sortDir,
    GpuKernel *k_bitonic,
    gpucontext *ctx
)
{

  int errI;
  size_t lens[1] = {strlen(code_k)};
  char *err_str = NULL;
  size_t ls, gs;
  unsigned int p = 0;
  int err;

  //void *arguments[NUMARGS_BITONIC_KERNEL]; // = (void**) malloc(sizeof(void *) * NUM_ARGS_KERNEL_1);

  err = GpuKernel_init( k_bitonic, ctx, 1, 
                         &code_k, lens, "add",
                         NUMARGS_CODE_K, type_args_code_k, flags, &err_str);

  if (err != GA_NO_ERROR) printf("error kernel init: %s \n", gpuarray_error_str(errI));
  if (err != GA_NO_ERROR)  printf("error backend: %s \n", err_str);

  //unsigned int blockCount = batchSize;
  //unsigned int blockDim = SHARED_SIZE_LIMIT / 2;
  ls = SHARED_SIZE_LIMIT / 2;
  gs = 1;
  GpuKernel_sched(k_bitonic, (size_t)arrayLength * batchSize, &gs, &ls);

  /*arguments[0] = (void*)d_DstKey->data;
  arguments[1] = (void*)d_SrcKey->data;
  arguments[2] = (void*)&batchSize;
  arguments[3] = (void*)&arrayLength;
  arguments[4] = (void*)&sortDir;
*/
  err = GpuKernel_setarg(k_bitonic, p++, d_DstKey->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_bitonic, p++, &d_DstKey->offset);
  
  err = GpuKernel_setarg(k_bitonic, p++, d_SrcKey->data);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);

  err = GpuKernel_setarg(k_bitonic, p++, &d_SrcKey->offset);
  
  unsigned int sz = 16;
  err = GpuKernel_setarg(k_bitonic, p++, &sz);
  if (err != GA_NO_ERROR) printf("error setting arg %d \n", p);
  
  /*err = GpuKernel_setarg(k_bitonic, p++, &arrayLength);
  if (err != GA_NO_ERROR) printf("eror setting arg %d \n", p);
  
  err = GpuKernel_setarg(k_bitonic, p++, &sortDir);
  if (err != GA_NO_ERROR) printf("eror setting arg %d \n", p);
*/
  err = GpuKernel_call(k_bitonic, 1, &gs, &ls, 0, NULL /*arguments*/);
  if (err != GA_NO_ERROR) printf("error calling kernel %d \n", p);

  /*if (sortDir)
  {
      //bitonicSortSharedKernel<1U><<<blockCount, blockDim>>>(d_DstKey, d_SrcKey, batchSize, arrayLength);
      //printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);
  }
  else
  {
      //bitonicSortSharedKernel<0U><<<blockCount, blockDim>>>(d_DstKey, d_SrcKey, batchSize, arrayLength);
      //printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);
  }*/
}

static void generateSampleRanks(
    gpudata  *d_RanksA,
    gpudata  *d_RanksB,
    GpuArray *d_SrcKey,
    unsigned int stride,
    unsigned int N,
    unsigned int sortDir
)
{
    unsigned int lastSegmentElements = N % (2 * stride);
    unsigned int threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

    if (sortDir)
    {
        //generateSampleRanksKernel<1U><<<iDivUp(threadCount, 256), 256>>>(d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
        //printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);
    }
    else
    {
        //generateSampleRanksKernel<0U><<<iDivUp(threadCount, 256), 256>>>(d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
        //printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);
    }
}

static void mergeRanksAndIndices(
  gpudata *d_LimitsA,
  gpudata *d_LimitsB,
  gpudata *d_RanksA,
  gpudata *d_RanksB,
  unsigned int stride,
  unsigned int N
)
{
  unsigned int lastSegmentElements = N % (2 * stride);
  unsigned int threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

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

static void mergeElementaryIntervals(
    GpuArray *d_DstKey,
    GpuArray *d_SrcKey,
    gpudata  *d_LimitsA,
    gpudata  *d_LimitsB,
    unsigned int stride,
    unsigned int N,
    unsigned int sortDir
)
{
    unsigned int lastSegmentElements = N % (2 * stride);
    unsigned int mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) : (N - lastSegmentElements) / SAMPLE_STRIDE;

    if (sortDir)
    {
       /* mergeElementaryIntervalsKernel<1U><<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_SrcKey,
            d_LimitsA,
            d_LimitsB,
            stride,
            N
        );
        printLastCudaError(cudaGetLastError(), __LINE__, __FILE__); */
    }
    else
    {
      /*
        mergeElementaryIntervalsKernel<0U><<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_SrcKey,
            d_LimitsA,
            d_LimitsB,
            stride,
            N
        );
        printLastCudaError(cudaGetLastError(), __LINE__, __FILE__);
      */
    }
}

static void mergeLeftMostSegment(
    GpuArray *d_DstKey,
    GpuArray *d_SrcKey,
    unsigned int segmentSizeA,
    unsigned int segmentSizeB,
    unsigned int N,
    unsigned int sortDir
)
{
    unsigned int blockDim = 256;
    unsigned int blockCount = ceiling(N, blockDim);

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
}

static void sort(
  GpuArray *d_DstKey,
  GpuArray *d_BufKey,
  GpuArray *d_SrcKey,
  gpudata  *d_RanksA,
  gpudata  *d_RanksB,
  gpudata  *d_LimitsA,
  gpudata  *d_LimitsB,
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

    unsigned int stageCount = 0;
    unsigned int stride;
    for (stride = SHARED_SIZE_LIMIT; stride < Nfloor; stride <<= 1, stageCount++);

    if (stageCount & 1)
    {
      ikey = d_BufKey;
      okey = d_DstKey;
    }
    else
    {
      ikey = d_DstKey;
      okey = d_BufKey;
    }

    /////////////////////////////////////////////////////////////////////////
    // Sort the array with bitonic sort for arrays shorter than 1024 elements
    // Bitonic sort gives better performance than merge sort for short arrays
    /////////////////////////////////////////////////////////////////////////
  
    if (N <= SHARED_SIZE_LIMIT)
    {  
      bitonicSortShared(d_DstKey, d_SrcKey, 1, N, sortDir, &k_bitonic, ctx); 
    }
    ///////////////////////////////////////////////////////////////////////////////
    // Sort the array with merge sort for arrays equal or bigger than 1024 elements
    ///////////////////////////////////////////////////////////////////////////////
    else
    {
      unsigned int batchSize = Nfloor / SHARED_SIZE_LIMIT;
      unsigned int arrayLength = SHARED_SIZE_LIMIT;
      //mergeSortShared(ikey, d_SrcKey, batchSize, arrayLength, sortDir, 0U, Nfloor); 
      bitonicSortShared(ikey, d_SrcKey, batchSize, arrayLength, sortDir, &k_bitonic, ctx);

      for (stride = SHARED_SIZE_LIMIT; stride < Nfloor; stride <<= 1)
      {
        unsigned int lastSegmentElements = Nfloor % (2 * stride);

        //Find sample ranks and prepare for limiters merge
        generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, Nfloor, sortDir);   

        //Merge ranks and indices
        mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, Nfloor);

        //Merge elementary intervals
        mergeElementaryIntervals(okey, ikey, d_LimitsA, d_LimitsB, stride, Nfloor, sortDir);

        if (lastSegmentElements <= stride)
        {
          //Last merge segment consists of a single array which just needs to be passed through
          printf("inside last segment\n");
           // TODO: uncomment and fix sizeof
          //////////////////////////////////
          /*cudaMemcpy( okey + (Nfloor - lastSegmentElements), 
                        ikey + (Nfloor - lastSegmentElements), 
                        lastSegmentElements * sizeof(t_key), 
                        cudaMemcpyDeviceToDevice
                   );
          */
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
        bitonicSortShared(d_SrcKey + Nfloor, d_DstKey + Nfloor, 1, Nleft, sortDir, &k_bitonic, ctx);

        // Copy the leftMost segment to the output array of which contains the first sorted sequence

        // TODO: uncomment and fix sizeof
        //////////////////////////////////
        //checkCudaErrors(cudaMemcpy(d_DstKey + Nfloor, d_SrcKey + Nfloor, Nleft * sizeof(t_key), cudaMemcpyDeviceToDevice));
        GpuArray_copy(d_DstKey, d_SrcKey, GA_C_ORDER); // TODO: copy just the needed part of the buffer

        mergeLeftMostSegment(d_SrcKey, d_DstKey, Nfloor, (unsigned int)Nleft, N, sortDir);

        // TODO: uncomment and fix sizeof
        //////////////////////////////////
        //checkCudaErrors(cudaMemcpy(d_DstKey, d_SrcKey , N * sizeof(t_key), cudaMemcpyDeviceToDevice));
        GpuArray_copy(d_DstKey, d_SrcKey, GA_C_ORDER);
      }
    }
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
  gpudata *d_RanksA,
  gpudata *d_RanksB,
  gpudata *d_LimitsA,
  gpudata *d_LimitsB,
  unsigned int MAX_SAMPLE_COUNT,
  gpucontext *ctx
)
{
    /*cudaMalloc((void **)d_RanksA,  MAX_SAMPLE_COUNT * sizeof(unsigned int));
    cudaMalloc((void **)d_RanksB,  MAX_SAMPLE_COUNT * sizeof(unsigned int));
    cudaMalloc((void **)d_LimitsA, MAX_SAMPLE_COUNT * sizeof(unsigned int));
    cudaMalloc((void **)d_LimitsB, MAX_SAMPLE_COUNT * sizeof(unsigned int));
    */

    int res = GA_NO_ERROR;

    d_RanksA = gpudata_alloc(ctx, MAX_SAMPLE_COUNT * sizeof(unsigned int), NULL, 0, &res);
    if (res != GA_NO_ERROR) printf("error allocating aux structures %d\n", res);

    d_RanksB = gpudata_alloc(ctx, MAX_SAMPLE_COUNT * sizeof(unsigned int), NULL, 0, &res);
    if (res != GA_NO_ERROR) printf("error allocating aux structures %d\n", res);
    
    d_LimitsA = gpudata_alloc(ctx, MAX_SAMPLE_COUNT * sizeof(unsigned int), NULL, 0, &res);
    if (res != GA_NO_ERROR) printf("error allocating aux structures %d\n", res);

    d_LimitsB = gpudata_alloc(ctx, MAX_SAMPLE_COUNT * sizeof(unsigned int), NULL, 0, &res);
    if (res != GA_NO_ERROR) printf("error allocating aux structures %d\n", res);
}


int GpuArray_sort(GpuArray *dst, GpuArray *src, unsigned int numaxes, unsigned int *axes, GpuArray *arg)
{

  int type = src->typecode;
  gpucontext *ctx = GpuArray_context(src);

  // Device pointers - auxiiary data structure    
  gpudata *d_RanksA = NULL, *d_RanksB = NULL, *d_LimitsA = NULL, *d_LimitsB = NULL;

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

    const unsigned int DIR = 0;

    // Device pointers - buffer data strucute
    GpuArray BufKey;
    GpuArray_empty(&BufKey, ctx, type, nd, &dims, GA_C_ORDER);


    //checkCudaErrors(cudaMalloc((void **)&d_BufKey, N * sizeof(t_key)));

    // Initialize device  auxiliary data structure  
    initMergeSort(d_RanksA, d_RanksB, d_LimitsA, d_LimitsB, Nfloor / 128, ctx);

    // perform regular sort
    sort(
      dst,
      &BufKey,
      src,
      d_RanksA,
      d_RanksB,
      d_LimitsA,
      d_LimitsB,
      dims,
      Nfloor,
      Nleft,
      DIR,
      ctx
    );


    // type -> get typecode of the array

    // vectorType -> "type"

    // stbr_append all the kernels....

    // Set arguments 

    


  }

  return 0;

}
