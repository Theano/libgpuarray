/* Includes */
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include "gpuarray/config.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "private.h"
#include "gpuarray/error.h"
#include "gpuarray/kernel.h"
#include "gpuarray/reduction.h"
#include "gpuarray/util.h"

#include "util/strb.h"
#include "util/srcgen.h"
#include "util/integerfactoring.h"


/* Defines */
#define  MAX_HW_DIMS                   3
#define  KERNEL_PRIMARY                0
#define  KERNEL_AUXILIARY              1
#define  AXIS_FREE                     0
#define  AXIS_REDUX                    1



/* Datatypes */

/**
 * @brief Axis Description.
 */

struct axis_desc{
	int      reduxNum;
	unsigned isReduced     : 1;
	unsigned isHW          : 1;
	unsigned isSW          : 1;
	size_t   warpLen;
	size_t   len;
	ssize_t  srcStride,       srcOffset;
	ssize_t  dstStride,       dstOffset;
	ssize_t  dstArgStride,    dstArgOffset;
	ssize_t  tmpDstStride,    tmpDstOffset;
	ssize_t  tmpDstArgStride, tmpDstArgOffset;
};
typedef struct axis_desc axis_desc;

/**
 *                    Reduction Kernel Generator.
 * 
 * INTRO
 * 
 * Generates the source code for a reduction kernel over arbitrarily-dimensioned,
 * -shaped and -typed tensors.
 * 
 * 
 * GOALS
 * 
 * The generator has the following goals:
 * 
 *   1. Maximizing the use of coalesced memory loads within a warp.
 *   2. Maximizing the # of useful threads within a warp.
 *   3. Maximizing the number of warps within a block.
 * 
 *   NOTE: It is possible to guarantee for any tensor problem of at least
 *         2*WARP_SIZE in scale that either
 *         1. All warp blocks in the X dimension have more than 50% threads
 *            active 100% of the time, or
 *         2. The warp blocks in the X dimension have 100% threads active more
 *            than 50% of the time.
 * 
 *   4. Ensuring there are no more blocks than are permitted by the warp
 *      configuration and 2nd-stage workspace size (if required).
 *   5. Ensuring there are no more than 5 blocks per multiprocessor.
 *   6. Minimizing the 2nd-stage workspace (if it is required).
 *   7. Striding the 2nd-stage workspace for maximum convenience (if it is
 *      required). Make it contiguous.
 * 
 * 
 * NOTES
 * 
 * Information elements required to perform reduction.
 * 
 *   1. Ndim, shape and dtype of src tensor
 *   2. Ndim, shape and dtype of dst/dstArg tensors
 *   3. GPU context
 *   4. Number of processors
 *   5. Warp size
 *   6. Maximum size of block
 *   7. Maximum size of block dimension X, Y, Z
 *   8. Maximum size of grid
 *   9. Maximum size of grid  dimension X, Y, Z
 *  10. Dtype and initializer of accumulator
 *  11. Sorted src axes for contiguous memory accesses
 *  12. Ndim, shape and dtype of flattened src tensor
 *  13. Number of stages (1 or 2)
 *  14. Ndim, shape and dtype of workspace tensor
 *  15. Warp     axes
 *  16. Hardware axes
 *  17. Software axes
 *  18. Source code
 * 
 * Rationale for dependencies:
 * 
 *   1) Get the GPU context and its properties immediately, since an invalid
 *      context is a likely error and we want to fail fast.
 *   2) The type and initializer of the accumulator should be determined after
 *      the context's properties have been retrieved since they provide
 *      information about the device's natively-supported types and operations.
 * 
 * REFERENCES
 * 
 * http://lpgpu.org/wp/wp-content/uploads/2013/05/poster_andresch_acaces2014.pdf
 * 
 * 
 * 
 * 
 * 
 *                           Kernel Template:
 *
 * The following kernel code template displays the code generated for the
 * small code model. For the large code model, no pre/postRedux() kernels
 * are generated (since their functionality is incorporated within the main
 * redux() kernel), no atomicRedux() function needs to be generated because
 * writes to global memory are unconditional and not contended.
 *
 *
 *     //Macros
 *     #define FOROVER
 *     #define ESCAPE
 *     #define srcVal       //Indexer
 *     #define dstVal       //Indexer
 *     #define dstArgVal    //Indexer
 *     #define rdxIdx       //Special reduction index computer
 *
 *
 *     //Typedefs:
 *     typedef  float    S  //The type of the source array.
 *     typedef  float    T  //The type of the destination array.
 *     typedef  ssize_t  A  //The type of the destination argument array.
 *     typedef  ssize_t  X  //The type of the indices: signed 32/64-bit.
 *     typedef  float    K  //The type of the accumulator variable.
 *
 *
 *     //Initializer (in case initial value of accumulator cannot be expressed
 *     //as a literal)
 *     static K    getInitValTFn(void){
 *         return ...
 *     }
 *     static K    getInitValKFn(void){
 *         return ...
 *     }
 *
 *
 *     //Reduce into global memory destination a value.
 *     static void writeBackFn(GLOBAL_MEM T* d_, T d,
 *                             GLOBAL_MEM A* a_, A a){
 *         //Large code model:
 *         *dPtr = d;
 *         *aPtr = a;
 *
 *         //Small code model:
 *         // Something complex possibly involving CAS loops
 *     }
 *
 *
 *     //Load data from source and apply pre-operations, coercing the type to
 *     //the accumulator type K.
 *     static K loadValFn(X i0, X i1, ..., X iN,
 *                        const GLOBAL_MEM S* srcPtr,
 *                        const X             srcOff,
 *                        const GLOBAL_MEM X* srcSteps,
 *                        ...?){
 *         return ...
 *     }
 *
 *
 *     //Initialization kernel
 *     KERNEL void initKer(const GLOBAL_MEM X*        srcSize,
 *                         const GLOBAL_MEM X*        chunkSize,
 *                         GLOBAL_MEM T*              dstPtr,
 *                         const X                    dstOff,
 *                         const GLOBAL_MEM X*        dstSteps){
 *         dstVal = getInitValTFn();
 *     }
 *
 *
 *     //Reduction Kernel.
 *     KERNEL void reduxKer(GLOBAL_MEM S*              srcPtr,
 *                          const X                    srcOff,
 *                          const GLOBAL_MEM X*        srcSteps,
 *                          const GLOBAL_MEM X*        srcSize,
 *                          const GLOBAL_MEM X*        chunkSize,
 *                          GLOBAL_MEM T*              dstPtr,
 *                          const X                    dstOff,
 *                          const GLOBAL_MEM X*        dstSteps,
 *                          GLOBAL_MEM A*              dstArgPtr,
 *                          const X                    dstArgOff,
 *                          const GLOBAL_MEM X*        dstArgSteps){
 *         //Declare Indices
 *         //Compute Ranges
 *
 *         //Outer Loops
 *            K rdxK = getInitValKFn();
 *            A rdxA = 0;
 *            //Inner Loops
 *                K k  = loadValFn(indices..., srcPtr, srcOff, srcSteps)
 *                rdxK = k
 *                rdxA = rdxIdx
 *            writeBackFn(&dstVal, d, &dstArgVal, a);
 *     }
 *
 *
 *     //Post-scalar kernel,
 *     KERNEL void postKer(const GLOBAL_MEM X*        srcSize,
 *                         const GLOBAL_MEM X*        chunkSize,
 *                         GLOBAL_MEM T*              dst,
 *                         const X                    dstOff,
 *                         const GLOBAL_MEM X*        dstSteps){
 *         //Default: Nothing.
 *         dstVal = dstVal
 *     }
 *
 *
 *                           Initial Reduction Values
 * +--------------+-----+-----+---------+---------+-----+-----+-----+-----+-----+
 * | Type\Op      |  +  |  *  |   max   |   min   |  &  |  |  |  ^  | &&  | ||  |
 * +--------------+-----+-----+---------+---------+-----+-----+-----+-----+-----+
 * | signed   int |  0  |  1  | INT_MIN | INT_MAX | ~0  |  0  |  0  | ~0  |  0  |
 * | unsigned int |  0  |  1  |    0    |   ~0    | ~0  |  0  |  0  | ~0  |  0  |
 * | floating     | 0.0 | 1.0 |   NAN   |   NAN   |     |     |     |     |     |
 * +--------------+-----+-----+---------+---------+-----+-----+-----+-----+-----+
 */

struct redux_ctx{
	/* Function Arguments. */
	ga_reduce_op    op;
	GpuArray*       dst;
	GpuArray*       dstArg;
	const GpuArray* src;
	int             reduxLen;
	const int*      reduxList;

	/* General. */
	int             nds;          /* # Source              dimensions */
	int             ndr;          /* # Reduced             dimensions */
	int             ndd;          /* # Destination         dimensions */
	int             ndw;          /* # Warp                dimensions */
	int             ndp;          /* # Partial warp        dimensions */
	int             ndf;          /* # Flattened source    dimensions */
	int             ndt;          /* # Temporary workspace dimensions */
	int             zeroAllAxes;  /* # of zero-length                   axes in source tensor */
	int             zeroRdxAxes;  /* # of zero-length         reduction axes in source tensor */
	size_t          prodAllAxes;  /* Product of length of all           axes in source tensor */
	size_t          prodRdxAxes;  /* Product of length of all reduction axes in source tensor */
	size_t          prodFreeAxes; /* Product of length of all free      axes in source tensor */
	size_t          prodWarpAxes; /* Number of active threads per warp. Strictly <= warpSize. */
	int             splitWarpAxis;/* Index of the split warp axis within the source tensor's shape; -1 otherwise. */
	
	gpucontext*     gpuCtx;
	unsigned        numProcs;
	size_t          warpSize;
	size_t          maxLg;
	size_t          maxLs[MAX_HW_DIMS];
	size_t          maxGg;
	size_t          maxGs[MAX_HW_DIMS];
	
	axis_desc*      xdSrc;
	axis_desc*      xdSrcFlat;
	axis_desc*      xdTmp;
	
	axis_desc**     xdSrcPtrs;
	
	int             numStages;
	
	GpuArray*       wsDst;
	GpuArray*       wsDstArg;
	int*            srcAxisList;
	size_t*         dstDims;

	/* Source code Generator. */
	int             srcTypeCode;
	int             dstTypeCode;
	int             dstArgTypeCode;
	int             idxTypeCode;
	int             accTypeCode;
	const char*     srcTypeStr;
	const char*     dstTypeStr;
	const char*     dstArgTypeStr;
	const char*     idxTypeStr;
	const char*     accTypeStr;
	const char*     initValT;
	const char*     initValK;
	int             largeCodeModel;
	strb            s;
	srcb            srcGen;
	char*           sourceCode;
	size_t          sourceCodeLen;
	char*           errorString0;
	char*           errorString1;
	char*           errorString2;
	GpuKernel       preKernel;
	GpuKernel       kernel;
	GpuKernel       postKernel;

	/**
	 * Scheduler
	 *
	 * There are two sets of kernels that may be scheduled:
	 *   1) The reduction kernel. This is the only kernel scheduled in the
	 *      large code model.
	 *   2) The initialization and post-scalar kernels. These are scheduled
	 *      only in the small code model.
	 *
	 * The reduction kernel is the "primary" kernel. The other two, if needed,
	 * are referred to as "auxiliary" kernels.
	 */

	struct{
		int         ndh;
		int         ndhd;
		int         ndhr;
		int         axisList   [MAX_HW_DIMS];
		size_t      bs         [MAX_HW_DIMS];
		size_t      gs         [MAX_HW_DIMS];
		size_t      cs         [MAX_HW_DIMS];
		gpudata*    chunkSizeGD;
	} pri, aux;

	/* Invoker */
	gpudata*        srcStepsGD;
	gpudata*        srcSizeGD;
	gpudata*        chunkSizeGD;
	gpudata*        dstStepsGD;
	gpudata*        dstArgStepsGD;
};
typedef struct redux_ctx redux_ctx;



/* Static Function prototypes */
/* Utilities */
static int        reduxGetSumInit               (int typecode, const char** property);
static int        reduxGetProdInit              (int typecode, const char** property);
static int        reduxGetMinInit               (int typecode, const char** property);
static int        reduxGetMaxInit               (int typecode, const char** property);
static int        reduxGetAndInit               (int typecode, const char** property);
static int        reduxGetOrInit                (int typecode, const char** property);
static int        reduxSortFlatSensitive        (const void* a, const void* b);
static int        reduxSortFlatInsensitive      (const void* a, const void* b);
static int        reduxSortWarp                 (const void* a, const void* b);
static int        axisInSet                     (int         v,
                                                 const int*  set,
                                                 size_t      setLen,
                                                 size_t*     where);
static void       appendIdxes                   (strb*       s,
                                                 const char* prologue,
                                                 const char* prefix,
                                                 int         startIdx,
                                                 int         endIdx,
                                                 const char* suffix,
                                                 const char* epilogue);

/* Axis Description API */
static void       axisInit                      (axis_desc*       axis,
                                                 ssize_t          len,
                                                 ssize_t          srcStride);
static void       axisMarkReduced               (axis_desc*       axis, int    reduxNum);
static void       axisMarkWarp                  (axis_desc*       axis, size_t partialSlice);
static int        axisGetReduxNum               (const axis_desc* axis);
static size_t     axisGetLen                    (const axis_desc* axis);
static ssize_t    axisGetSrcStride              (const axis_desc* axis);
static size_t     axisGetSrcAbsStride           (const axis_desc* axis);
static ssize_t    axisGetSrcOffset              (const axis_desc* axis);
static ssize_t    axisGetDstStride              (const axis_desc* axis);
static size_t     axisGetDstAbsStride           (const axis_desc* axis);
static ssize_t    axisGetDstOffset              (const axis_desc* axis);
static ssize_t    axisGetDstArgStride           (const axis_desc* axis);
static size_t     axisGetDstArgAbsStride        (const axis_desc* axis);
static ssize_t    axisGetDstArgOffset           (const axis_desc* axis);
static int        axisIsReduced                 (const axis_desc* axis);
static int        axisIsWarp                    (const axis_desc* axis);
static int        axisIsPartialWarp             (const axis_desc* axis);

/* Reduction Context API */
/*     Utilities */
static int        reduxRequiresDst              (const redux_ctx*  ctx);
static int        reduxRequiresDstArg           (const redux_ctx*  ctx);
static int        reduxKernelRequiresDst        (const redux_ctx*  ctx);
static int        reduxKernelRequiresDstArg     (const redux_ctx*  ctx);
static int        reduxIsSensitive              (const redux_ctx*  ctx);
static int        reduxIsSmallCodeModel         (const redux_ctx*  ctx);
static int        reduxIsLargeCodeModel         (const redux_ctx*  ctx);
static axis_desc* reduxGetSrcAxis               (const redux_ctx*  ctx, int i);
static axis_desc* reduxGetSrcSortAxis           (const redux_ctx*  ctx, int i);
static axis_desc* reduxGetSrcFlatAxis           (const redux_ctx*  ctx, int i);
static int        reduxTryFlattenInto           (const redux_ctx* ctx,
                                                 axis_desc*       into,
                                                 const axis_desc* from);
static int        reduxCanAppendHwAxis          (redux_ctx*  ctx,
                                                 int         kernelType,
                                                 int         axisType);
static void       reduxAppendLargestAxisToHwList(redux_ctx*  ctx,
                                                 int         kernelType,
                                                 int         axisType);
/*     Control Flow */
static int        reduxInit                     (redux_ctx*  ctx);
static int        reduxInferProperties          (redux_ctx*  ctx);
static int        reduxFlattenSource            (redux_ctx*  ctx);
static int        reduxSelectWarpAxes           (redux_ctx*  ctx);
static int        reduxSelectNumStages          (redux_ctx*  ctx);
static int        reduxSelectHwAxes             (redux_ctx*  ctx);
static int        reduxComputeAxisList          (redux_ctx*  ctx);
static int        reduxGenSource                (redux_ctx*  ctx);
static void       reduxAppendSource             (redux_ctx*  ctx);
static void       reduxAppendIncludes           (redux_ctx*  ctx);
static void       reduxAppendTensorDeclArgs     (redux_ctx*  ctx,
                                                 const char* type,
                                                 const char* baseName);
static void       reduxAppendTensorCallArgs     (redux_ctx*  ctx,
                                                 const char* baseName);
static void       reduxAppendMacroDefs          (redux_ctx*  ctx);
static void       reduxAppendTypedefs           (redux_ctx*  ctx);
static void       reduxAppendGetInitValFns      (redux_ctx*  ctx);
static void       reduxAppendWriteBackFn        (redux_ctx*  ctx);
static void       reduxAppendReduxKernel        (redux_ctx*  ctx);
static void       reduxAppendPrototype          (redux_ctx*  ctx);
static void       reduxAppendIndexDeclarations  (redux_ctx*  ctx);
static void       reduxAppendRangeCalculations  (redux_ctx*  ctx);
static void       reduxAppendLoops              (redux_ctx*  ctx);
static void       reduxAppendInitKernel         (redux_ctx*  ctx);
static void       reduxAppendPostKernel         (redux_ctx*  ctx);
static int        reduxCompile                  (redux_ctx*  ctx);
static int        reduxSchedule                 (redux_ctx*  ctx);
static void       reduxScheduleKernel           (int         ndims,
                                                 uint64_t*   dims,
                                                 uint64_t    warpSize,
                                                 uint64_t    maxLg,
                                                 uint64_t*   maxLs,
                                                 uint64_t    maxGg,
                                                 uint64_t*   maxGs,
                                                 uint64_t*   bs,
                                                 uint64_t*   gs,
                                                 uint64_t*   cs);
static int        reduxInvoke                   (redux_ctx*  ctx);
static int        reduxCleanup                  (redux_ctx*  ctx, int ret);
static int        reduxCleanupMsg               (redux_ctx*  ctx, int ret,
                                                 const char* fmt, ...);


/* Function implementation */
GPUARRAY_PUBLIC int  GpuArray_reduction   (ga_reduce_op    op,
                                           GpuArray*       dst,
                                           GpuArray*       dstArg,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	redux_ctx ctxSTACK, *ctx = &ctxSTACK;
	memset(ctx, 0, sizeof(*ctx));

	ctx->op        = op;
	ctx->dst       = dst;
	ctx->dstArg    = dstArg;
	ctx->src       = src;
	ctx->reduxLen  = reduxLen;
	ctx->reduxList = (const int*)reduxList;

	return reduxInit(ctx);
}

/**
 * @brief Get an expression representing a suitable initialization value for
 *        the given datatype and a sum-reduction operation.
 *
 * @param [in]  typecode  Typecode of the type whose initializer is to be
 *                        requested.
 * @param [out] property  A pointer to a string. On return it will be set to
 *                        the initializer expression.
 * @return Zero if successful; Non-zero if the datatype is not supported.
 */

static int        reduxGetSumInit               (int typecode, const char** property){
	if (typecode == GA_POINTER ||
	    typecode == GA_BUFFER){
		return GA_UNSUPPORTED_ERROR;
	}
	*property = "0";
	return GA_NO_ERROR;
}

/**
 * @brief Get an expression representing a suitable initialization value for
 *        the given datatype and a prod-reduction operation.
 *
 * @param [in]  typecode  Typecode of the type whose initializer is to be
 *                        requested.
 * @param [out] property  A pointer to a string. On return it will be set to
 *                        the initializer expression.
 * @return Zero if successful; Non-zero if the datatype is not supported.
 */

static int        reduxGetProdInit              (int typecode, const char** property){
	if (typecode == GA_POINTER ||
	    typecode == GA_BUFFER){
		return GA_UNSUPPORTED_ERROR;
	}
	*property = "1";
	return GA_NO_ERROR;
}

/**
 * @brief Get an expression representing a suitable initialization value for
 *        the given datatype and a max-reduction operation.
 *
 * @param [in]  typecode  Typecode of the type whose initializer is to be
 *                        requested.
 * @param [out] property  A pointer to a string. On return it will be set to
 *                        the initializer expression.
 * @return Zero if successful; Non-zero if the datatype is not supported.
 */

static int        reduxGetMinInit               (int typecode, const char** property){
	switch (typecode){
		case GA_BYTE2:
		case GA_BYTE3:
		case GA_BYTE4:
		case GA_BYTE8:
		case GA_BYTE16:
		case GA_BYTE:
		  *property = "SCHAR_MIN";
		break;
		case GA_SHORT2:
		case GA_SHORT3:
		case GA_SHORT4:
		case GA_SHORT8:
		case GA_SHORT16:
		case GA_SHORT:
		  *property = "SHRT_MIN";
		break;
		case GA_INT2:
		case GA_INT3:
		case GA_INT4:
		case GA_INT8:
		case GA_INT16:
		case GA_INT:
		  *property = "INT_MIN";
		break;
		case GA_LONG2:
		case GA_LONG3:
		case GA_LONG4:
		case GA_LONG8:
		case GA_LONG16:
		case GA_LONG:
		  *property = "LONG_MIN";
		break;
		case GA_LONGLONG:
		  *property = "LLONG_MIN";
		break;
		case GA_BOOL:
		case GA_UBYTE2:
		case GA_UBYTE3:
		case GA_UBYTE4:
		case GA_UBYTE8:
		case GA_UBYTE16:
		case GA_UBYTE:
		case GA_USHORT2:
		case GA_USHORT3:
		case GA_USHORT4:
		case GA_USHORT8:
		case GA_USHORT16:
		case GA_USHORT:
		case GA_UINT2:
		case GA_UINT3:
		case GA_UINT4:
		case GA_UINT8:
		case GA_UINT16:
		case GA_UINT:
		case GA_ULONG2:
		case GA_ULONG3:
		case GA_ULONG4:
		case GA_ULONG8:
		case GA_ULONG16:
		case GA_ULONG:
		case GA_ULONGLONG:
		case GA_SIZE:
		  *property = "0";
		break;
		case GA_HALF:
		case GA_FLOAT:
		case GA_DOUBLE:
		case GA_QUAD:
		  *property = "NAN";
		break;
		default:
		  return GA_UNSUPPORTED_ERROR;
	}

	return GA_NO_ERROR;
}

/**
 * @brief Get an expression representing a suitable initialization value for
 *        the given datatype and a min-reduction operation.
 *
 * @param [in]  typecode  Typecode of the type whose initializer is to be
 *                        requested.
 * @param [out] property  A pointer to a string. On return it will be set to
 *                        the initializer expression.
 * @return Zero if successful; Non-zero if the datatype is not supported.
 */

static int        reduxGetMaxInit               (int typecode, const char** property){
	switch (typecode){
		case GA_BOOL:
		  *property = "1";
		break;
		case GA_BYTE2:
		case GA_BYTE3:
		case GA_BYTE4:
		case GA_BYTE8:
		case GA_BYTE16:
		case GA_BYTE:
		  *property = "SCHAR_MAX";
		break;
		case GA_UBYTE2:
		case GA_UBYTE3:
		case GA_UBYTE4:
		case GA_UBYTE8:
		case GA_UBYTE16:
		case GA_UBYTE:
		  *property = "UCHAR_MAX";
		break;
		case GA_SHORT2:
		case GA_SHORT3:
		case GA_SHORT4:
		case GA_SHORT8:
		case GA_SHORT16:
		case GA_SHORT:
		  *property = "SHRT_MAX";
		break;
		case GA_USHORT2:
		case GA_USHORT3:
		case GA_USHORT4:
		case GA_USHORT8:
		case GA_USHORT16:
		case GA_USHORT:
		  *property = "USHRT_MAX";
		break;
		case GA_INT2:
		case GA_INT3:
		case GA_INT4:
		case GA_INT8:
		case GA_INT16:
		case GA_INT:
		  *property = "INT_MAX";
		break;
		case GA_UINT2:
		case GA_UINT3:
		case GA_UINT4:
		case GA_UINT8:
		case GA_UINT16:
		case GA_UINT:
		  *property = "UINT_MAX";
		break;
		case GA_LONG2:
		case GA_LONG3:
		case GA_LONG4:
		case GA_LONG8:
		case GA_LONG16:
		case GA_LONG:
		  *property = "LONG_MAX";
		break;
		case GA_ULONG2:
		case GA_ULONG3:
		case GA_ULONG4:
		case GA_ULONG8:
		case GA_ULONG16:
		case GA_ULONG:
		  *property = "ULONG_MAX";
		break;
		case GA_LONGLONG:
		  *property = "LLONG_MAX";
		break;
		case GA_ULONGLONG:
		  *property = "ULLONG_MAX";
		break;
		case GA_HALF:
		case GA_FLOAT:
		case GA_DOUBLE:
		case GA_QUAD:
		  *property = "NAN";
		break;
		default:
		  return GA_UNSUPPORTED_ERROR;
	}

	return GA_NO_ERROR;
}

/**
 * @brief Get an expression representing a suitable initialization value for
 *        the given datatype and a and-reduction operation.
 *
 * @param [in]  typecode  Typecode of the type whose initializer is to be
 *                        requested.
 * @param [out] property  A pointer to a string. On return it will be set to
 *                        the initializer expression.
 * @return Zero if successful; Non-zero if the datatype is not supported.
 */

static int        reduxGetAndInit               (int typecode, const char** property){
	if (typecode == GA_POINTER ||
	    typecode == GA_BUFFER){
		return GA_UNSUPPORTED_ERROR;
	}
	*property = "~0";
	return GA_NO_ERROR;
}

/**
 * @brief Get an expression representing a suitable initialization value for
 *        the given datatype and a or-reduction operation.
 *
 * @param [in]  typecode  Typecode of the type whose initializer is to be
 *                        requested.
 * @param [out] property  A pointer to a string. On return it will be set to
 *                        the initializer expression.
 * @return Zero if successful; Non-zero if the datatype is not supported.
 */

static int        reduxGetOrInit                (int typecode, const char** property){
	if (typecode == GA_POINTER ||
	    typecode == GA_BUFFER){
		return GA_UNSUPPORTED_ERROR;
	}
	*property = "0";
	return GA_NO_ERROR;
}

/**
 * @brief Sort the axes into optimal order for flattening.
 * 
 * Two orderings exist: "Sensitive" and "Insensitive", for reductions that are
 * sensitive (or not) to indexing.
 * 
 * In all cases:
 * 
 *   1. Free axes are sorted before reduction axes.
 *   2. Free axes are sorted by decreasing absolute stride.
 *   3.                 then by increasing source axis number.
 * 
 * In the sensitive case:
 * 
 *   4. Reduction axes are sorted by their position in reduxList.
 * 
 * In the insensitive case:
 * 
 *   4. Reduction axes are sorted by decreasing absolute stride.
 *   5.                      then by increasing source axis number.
 */

static int        reduxSortFlatInsensitive      (const void* a, const void* b){
	const axis_desc* xda  = (const axis_desc*)a;
	const axis_desc* xdb  = (const axis_desc*)b;

	if       ( axisIsReduced(xda)      && !axisIsReduced(xdb)){
		return +1;
	}else if (!axisIsReduced(xda)      &&  axisIsReduced(xdb)){
		return -1;
	}
	
	if       (axisGetSrcAbsStride(xda)  <  axisGetSrcAbsStride(xdb)){
		return +1;
	}else if (axisGetSrcAbsStride(xda)  >  axisGetSrcAbsStride(xdb)){
		return -1;
	}

	return 0;
}
static int        reduxSortFlatSensitive        (const void* a, const void* b){
	const axis_desc* xda  = (const axis_desc*)a;
	const axis_desc* xdb  = (const axis_desc*)b;

	if       ( axisIsReduced(xda)      && !axisIsReduced(xdb)){
		return +1;
	}else if (!axisIsReduced(xda)      &&  axisIsReduced(xdb)){
		return -1;
	}

	if (axisIsReduced(xda)){
		return axisGetReduxNum(xda)<axisGetReduxNum(xdb) ? -1 : +1;
	}else{
		if       (axisGetSrcAbsStride(xda)  <  axisGetSrcAbsStride(xdb)){
			return +1;
		}else if (axisGetSrcAbsStride(xda)  >  axisGetSrcAbsStride(xdb)){
			return -1;
		}
		
		return 0;
	}
}

/**
 * @brief Sort axes in preferred order for integration into warp.
 * 
 * The axes with stride != 0 are sorted by lowest absolute
 * stride. Picking the few axes with the lowest absolute stride (while
 * keeping the product of their dimensions <= warpSize) should maximize
 * memory bandwidth of the warp.
 * 
 * The restriction stride != 0 is intended to avoid waste of memory
 * bandwidth. Once a memory transaction is necessary, it typically operates at
 * far greater granularity than just 32 bits (4 bytes).
 * 
 * Sorting by absolute stride should result, in the case of a packed tensor, in
 * the memory accesses being close to perfectly contiguous.
 */

static int        reduxSortWarp                 (const void* a, const void* b){
	const axis_desc* xda  = *(const axis_desc* const *)a;
	const axis_desc* xdb  = *(const axis_desc* const *)b;

	if       ( axisGetSrcStride(xda)   && !axisGetSrcStride(xdb)){
		return -1;
	}else if (!axisGetSrcStride(xda)   &&  axisGetSrcStride(xdb)){
		return +1;
	}
	
	if       (axisGetSrcAbsStride(xda)    <   axisGetSrcAbsStride(xdb)){
		return -1;
	}else if (axisGetSrcAbsStride(xda)    >   axisGetSrcAbsStride(xdb)){
		return +1;
	}

	if       ( axisIsReduced(xda)      && !axisIsReduced(xdb)){
		return -1;
	}else if (!axisIsReduced(xda)      &&  axisIsReduced(xdb)){
		return +1;
	}

	return 0;
}

/**
 * @brief Check whether axis numbered v is already in the given set of axes.
 *
 * @param [in]  v
 * @param [in]  set
 * @param [in]  setLen
 * @param [out] where
 * @return Non-zero if the set is non-empty and v is in it; Zero otherwise.
 */

static int        axisInSet                     (int         v,
                                                 const int*  set,
                                                 size_t      setLen,
                                                 size_t*     where){
	size_t i;

	for (i=0;i<setLen;i++){
		if (set[i] == v){
			if (where){*where = i;}
			return 1;
		}
	}

	return 0;
}

/**
 * @brief Append a comma-separated list of indices, whose name contains an
 *        incrementing integer, to a string buffer.
 *
 *
 * @param [in]  s         The string buffer to which to append.
 * @param [in]  prologue  Text that is prepended in front and NOT repeated.
 * @param [in]  prefix    Text that is prepended in front of the integer and
 *                        repeated.
 * @param [in]  startIdx  First value of the integer (inclusive)
 * @param [in]  endIdx    Last  value of the integer (exclusive)
 * @param [in]  suffix    Text that is appended after the integer, followed by
 *                        a comma if it isn't the last index, and repeated.
 * @param [in]  epilogue  Text that is appended and NOT repeated.
 */

static void       appendIdxes                   (strb*       s,
                                                 const char* prologue,
                                                 const char* prefix,
                                                 int         startIdx,
                                                 int         endIdx,
                                                 const char* suffix,
                                                 const char* epilogue){
	int i;

	prologue = prologue ? prologue : "";
	prefix   = prefix   ? prefix   : "";
	suffix   = suffix   ? suffix   : "";
	epilogue = epilogue ? epilogue : "";

	strb_appends(s, prologue);
	for (i=startIdx;i<endIdx;i++){
		strb_appendf(s, "%s%d%s%s", prefix, i, suffix, &","[i==endIdx-1]);
	}
	strb_appends(s, epilogue);
}

/* Axis Description API */

/**
 * @brief Initialize Axis Description.
 */

static void       axisInit                      (axis_desc*       axis,
                                                 ssize_t          len,
                                                 ssize_t          srcStride){
	memset(axis, 0, sizeof(*axis));
	
	axis->reduxNum        = -1;
	axis->warpLen         = 0;
	axis->len             = len;
	
	axis->srcStride       = srcStride;
	axis->srcOffset       = 0;
	
	axis->dstStride       = 0;
	axis->dstOffset       = 0;
	
	axis->dstArgStride    = 0;
	axis->dstArgOffset    = 0;
	
	axis->tmpDstStride    = 0;
	axis->tmpDstOffset    = 0;
	
	axis->tmpDstArgStride = 0;
	axis->tmpDstArgOffset = 0;
}

/**
 * @brief Mark axis as reduction axis, with position reduxNum in the axis list.
 */

static void       axisMarkReduced               (axis_desc*       axis, int    reduxNum){
	axis->isReduced = 1;
	axis->reduxNum  = reduxNum;
}

/**
 * @brief Mark axis as warp axis.
 */

static void       axisMarkWarp                  (axis_desc*       axis, size_t warpLen){
	axis->warpLen = warpLen;
}

/**
 * @brief Get properties of an axis.
 */

static int        axisGetReduxNum               (const axis_desc* axis){
	return axis->reduxNum;
}
static size_t     axisGetLen                    (const axis_desc* axis){
	return axis->len;
}
static ssize_t    axisGetSrcStride              (const axis_desc* axis){
	return axisGetLen(axis) > 1 ? axis->srcStride : 0;
}
static size_t     axisGetSrcAbsStride           (const axis_desc* axis){
	return axisGetSrcStride(axis)<0 ? -(size_t)axisGetSrcStride(axis):
	                                  +(size_t)axisGetSrcStride(axis);
}
static ssize_t    axisGetSrcOffset              (const axis_desc* axis){
	return axis->srcOffset;
}
static ssize_t    axisGetDstStride              (const axis_desc* axis){
	return axisGetLen(axis) > 1 ? axis->dstStride : 0;
}
static size_t     axisGetDstAbsStride           (const axis_desc* axis){
	return axisGetDstStride(axis)<0 ? -(size_t)axisGetDstStride(axis):
	                                  +(size_t)axisGetDstStride(axis);
}
static ssize_t    axisGetDstOffset              (const axis_desc* axis){
	return axis->dstOffset;
}
static ssize_t    axisGetDstArgStride           (const axis_desc* axis){
	return axisGetLen(axis) > 1 ? axis->dstArgStride : 0;
}
static size_t     axisGetDstArgAbsStride        (const axis_desc* axis){
	return axisGetDstArgStride(axis)<0 ? -(size_t)axisGetDstArgStride(axis):
	                                     +(size_t)axisGetDstArgStride(axis);
}
static ssize_t    axisGetDstArgOffset           (const axis_desc* axis){
	return axis->dstArgOffset;
}
static int        axisIsReduced                 (const axis_desc* axis){
	return axis->isReduced;
}
static int        axisIsWarp                    (const axis_desc* axis){
	return !!axis->warpLen;
}
static int        axisIsPartialWarp             (const axis_desc* axis){
	return axis->warpLen > 0 && axis->warpLen != axis->len;
}

/**
 * @brief Returns whether the reduction interface requires a dst argument.
 */

static int        reduxRequiresDst              (const redux_ctx*  ctx){
	switch (ctx->op){
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_ARGMAX:
		  return 0;
		default:
		  return 1;
	}
}

/**
 * @brief Returns whether the reduction interface requires a dstArg argument.
 */

static int        reduxRequiresDstArg           (const redux_ctx*  ctx){
	switch (ctx->op){
		case GA_REDUCE_MINANDARGMIN:
		case GA_REDUCE_MAXANDARGMAX:
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_ARGMAX:
		  return 1;
		default:
		  return 0;
	}
}

/**
 * @brief Returns whether the generated kernel internally requires a dst
 *        argument.
 *
 * This is semantically subtly different from reduxHasDst(). The main
 * difference is in the implementation of the GA_REDUCE_ARGMIN/ARGMAX
 * reductions; Either *might* require a dst buffer, which will have to be
 * allocated, even though it will be discared.
 */

static int        reduxKernelRequiresDst        (const redux_ctx*  ctx){
	switch (ctx->op){
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_ARGMAX:
		  return reduxIsSmallCodeModel(ctx);
		default:
		  return 1;
	}
}

/**
 * @brief Returns whether the generated kernel internally requires a dstArg
 *        argument.
 *
 * This is semantically subtly different from reduxHasDstArg(), since it asks
 * whether the reduction, even though it does not accept a dstArg argument,
 * still requires a dstArg internally.
 */

static int        reduxKernelRequiresDstArg     (const redux_ctx*  ctx){
	/**
	 * At present there exists no reduction whose implementation requires
	 * a dstArg but whose interface does not.
	 *
	 * E.g. the max() and min() reductions do NOT currently require a temporary
	 *      buffer for indexes, and will not in the foreseeable future.
	 */

	return reduxRequiresDstArg(ctx);
}

/**
 * @brief Returns whether the reduction is sensitive.
 * 
 * A reduction is sensitive when its output satisfies at least one of the
 * following conditions:
 * 
 *   - It depends on the exact order of axes in the reduxList
 *   - It depends on exact signs of the strides of axes in the reduxList
 * 
 * Such sensitivity may prevent a flattening of contiguous axes even when it
 * would have been otherwise permitted.
 * 
 * For instance, ARGMIN/ARGMAX have this sensitivity, because the dstArg
 * tensor's contents are flattened coordinates into the source tensor, and
 * the flattening order is precisely reduxList. Permuting it would thus produce
 * incorrect output. Moreover, if the strides of a reduction axis were to be
 * reversed for the purpose of flattening the axis into another, the computed
 * coordinate would again be incorrect.
 * 
 * 
 * TL;DR: Reduction is sensitive if
 *   reduce(x, axis=axisList) != reduce(x, axis=axisList[::-1])
 * or
 *   reduce(x) != reduce(x[::-1])
 * .
 */

static int        reduxIsSensitive              (const redux_ctx*  ctx){
	switch (ctx->op){
		case GA_REDUCE_MINANDARGMIN:
		case GA_REDUCE_MAXANDARGMAX:
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_ARGMAX:
		  return 1;
		default:
		  return 0;
	}
}

/**
 * @brief Returns whether we are using the small code model or not.
 */

static int        reduxIsSmallCodeModel         (const redux_ctx*  ctx){
	return !reduxIsLargeCodeModel(ctx);
}

/**
 * @brief Returns whether we are using the large code model or not.
 */

static int        reduxIsLargeCodeModel         (const redux_ctx*  ctx){
	return ctx->largeCodeModel;
}

/**
 * @brief Get description of source axis with given number.
 */

static axis_desc* reduxGetSrcAxis               (const redux_ctx*  ctx, int i){
	return &ctx->xdSrc[i];
}

/**
 * @brief Get description of source axis with given number in sort-order.
 */

static axis_desc* reduxGetSrcSortAxis           (const redux_ctx*  ctx, int i){
	return ctx->xdSrcPtrs[i];
}

/**
 * @brief Get description of flattened source axis with given number.
 */

static axis_desc* reduxGetSrcFlatAxis           (const redux_ctx*  ctx, int i){
	return &ctx->xdSrcFlat[i];
}

/**
 * @brief Attempt to flatten an axis `from` into an axis `into`.
 * 
 * An axis can be considered for flattening into the previous one if ALL of
 * the following conditions hold:
 * 
 *   1. The product of the previous axis' length by its stride exactly
 *      matches the current axis' stride.
 *   2. Both axes are reduced.
 * 
 * For reductions where axis order matters (e.g. those that compute
 * indices, like argmax/argmin), ALL of the following additional conditions
 * must hold:
 * 
 *   3. The sign of the strides must match.
 *   4. The axis numbers must follow consecutively in the reduction list
 *      (this is ensured by the "sensitive" sort order)
 * 
 * @return Non-zero if flattening attempt successful; Zero otherwise.
 */

static int        reduxTryFlattenInto           (const redux_ctx* ctx,
                                                 axis_desc*       into,
                                                 const axis_desc* from){
	int signSrc    = 0, signDst    = 0, signDstArg    = 0,
	    reverseSrc = 0, reverseDst = 0, reverseDstArg = 0;
	
	if (axisIsReduced         (into) != axisIsReduced         (from)                 ||
	    axisGetSrcAbsStride   (into) != axisGetSrcAbsStride   (from)*axisGetLen(from)){
		return 0;
	}
	
	if (reduxRequiresDst(ctx) &&
	    axisGetDstAbsStride   (into) != axisGetDstAbsStride   (from)*axisGetLen(from)){
		return 0;
	}
	
	if (reduxRequiresDstArg(ctx) &&
	    axisGetDstArgAbsStride(into) != axisGetDstArgAbsStride(from)*axisGetLen(from)){
		return 0;
	}
	
	signSrc       = (axisGetSrcStride   (into)^axisGetSrcStride   (from)) < 0;
	signDst       = (axisGetDstStride   (into)^axisGetDstStride   (from)) < 0;
	signDstArg    = (axisGetDstArgStride(into)^axisGetDstArgStride(from)) < 0;
	reverseSrc    = signSrc;
	reverseDst    = signDst    && reduxRequiresDst   (ctx);
	reverseDstArg = signDstArg && reduxRequiresDstArg(ctx);
	
	if (reduxIsSensitive(ctx)){
		if(reverseSrc || reverseDst || reverseDstArg){
			return 0;
		}
	}
	
	if (reduxRequiresDst   (ctx) &&
	    reduxRequiresDstArg(ctx) &&
	    reverseDst != reverseDstArg){
		/* Either both, or neither, of dst and dstArg must require reversal. */
		return 0;
	}
	
	if (reverseSrc){
		into->srcOffset    += (ssize_t)(axisGetLen(from)-1)*axisGetSrcStride(from);
		into->srcStride     = -axisGetSrcStride   (from);
	}else{
		into->srcStride     =  axisGetSrcStride   (from);
	}
	
	if (reverseDst){
		into->dstOffset    += (ssize_t)(axisGetLen(from)-1)*axisGetDstStride(from);
		into->dstStride     = -axisGetDstStride   (from);
	}else{
		into->dstStride     =  axisGetDstStride   (from);
	}
	
	if (reverseDstArg){
		into->dstArgOffset += (ssize_t)(axisGetLen(from)-1)*axisGetDstArgStride(from);
		into->dstArgStride  = -axisGetDstArgStride(from);
	}else{
		into->dstArgStride  =  axisGetDstArgStride(from);
	}
	
	into->srcOffset    += axisGetSrcOffset   (from);
	into->dstOffset    += axisGetDstOffset   (from);
	into->dstArgOffset += axisGetDstArgOffset(from);
	into->len          *= axisGetLen         (from);
	
	return 1;
}

/**
 * @brief Check whether we can add another reduction axis or free axis
 *        to the hardware axis list for either the primary or secondary kernel.
 */

static int        reduxCanAppendHwAxis          (redux_ctx*  ctx,
                                                 int         kernelType,
                                                 int         axisType){
	int kernelNdh  = kernelType == KERNEL_PRIMARY ? ctx->pri.ndh  : ctx->aux.ndh;
	int kernelNdhr = kernelType == KERNEL_PRIMARY ? ctx->pri.ndhr : ctx->aux.ndhr;
	int kernelNdhd = kernelType == KERNEL_PRIMARY ? ctx->pri.ndhd : ctx->aux.ndhd;

	if (kernelNdh >= MAX_HW_DIMS){
		return 0;
	}else{
		return axisType == AXIS_REDUX ? kernelNdhr < ctx->ndr:
		                                kernelNdhd < ctx->ndd;
	}
}

/**
 * @brief Append the largest reduction axis or free axis that isn't yet
 *        in the hardware axis list for either the primary or secondary kernel
 *        into said hardware axis list.
 */

static void       reduxAppendLargestAxisToHwList(redux_ctx*  ctx,
                                                 int         kernelType,
                                                 int         axisType){
	int    maxI = 0, i, isInHwList, isInReduxList, isInDesiredList, isLargestSoFar;
	int*   hwAxisList, * ndh, * ndhr, * ndhd;
	size_t v, maxV = 0;

	/* Get pointers to the correct kernel's variables */
	hwAxisList = kernelType == KERNEL_PRIMARY ?  ctx->pri.axisList:
	                                             ctx->aux.axisList;
	ndh        = kernelType == KERNEL_PRIMARY ? &ctx->pri.ndh:
	                                            &ctx->aux.ndh;
	ndhr       = kernelType == KERNEL_PRIMARY ? &ctx->pri.ndhr:
	                                            &ctx->aux.ndhr;
	ndhd       = kernelType == KERNEL_PRIMARY ? &ctx->pri.ndhd:
	                                            &ctx->aux.ndhd;

	/* Find */
	for (i=0;i<ctx->nds;i++){
		isInHwList      = axisInSet(i, hwAxisList,     *ndh,     0);
		isInReduxList   = axisInSet(i, ctx->reduxList, ctx->ndr, 0);
		isInDesiredList = axisType == AXIS_REDUX ?  isInReduxList:
		                                           !isInReduxList;
		v               = ctx->src->dimensions[i];
		isLargestSoFar  = v >= maxV;

		if (!isInHwList && isInDesiredList && isLargestSoFar){
			maxV = v;
			maxI = i;
		}
	}

	/* Append */
	hwAxisList[(*ndh)++] = maxI;
	if (axisType == AXIS_REDUX){
		(*ndhr)++;
	}else{
		(*ndhd)++;
	}
}

/**
 * @brief Initialize the context.
 * 
 * After this function, calling reduxCleanup() becomes safe.
 */

static int        reduxInit                     (redux_ctx*  ctx){
	int i;

	/**
	 * We initialize certain parts of the context.
	 */

	ctx->wsDst         = NULL;
	ctx->wsDstArg      = NULL;
	ctx->srcAxisList   = NULL;
	ctx->dstDims       = NULL;
	ctx->gpuCtx        = NULL;

	ctx->srcTypeStr    = ctx->dstTypeStr    = ctx->dstArgTypeStr =
	ctx->accTypeStr    = ctx->idxTypeStr    = NULL;
	ctx->initValK      = NULL;
	ctx->sourceCode    = NULL;
	ctx->errorString0  = NULL;
	ctx->errorString1  = NULL;
	ctx->errorString2  = NULL;

	ctx->splitWarpAxis = -1;
	ctx->numStages     =  1;
	ctx->prodWarpAxes  =  1;
	ctx->prodAllAxes   = ctx->prodRdxAxes   = ctx->prodFreeAxes  = 1;
	strb_init(&ctx->s);
	srcbInit (&ctx->srcGen, &ctx->s);

	for (i=0;i<MAX_HW_DIMS;i++){
		ctx->aux.axisList[i] = ctx->pri.axisList[i] = 0;
		ctx->aux.bs      [i] = ctx->pri.bs      [i] = 1;
		ctx->aux.gs      [i] = ctx->pri.gs      [i] = 1;
		ctx->aux.cs      [i] = ctx->pri.cs      [i] = 1;
	}

	ctx->srcStepsGD      = ctx->srcSizeGD       =
	ctx->dstStepsGD      = ctx->dstArgStepsGD   =
	ctx->pri.chunkSizeGD = ctx->aux.chunkSizeGD = NULL;

	return reduxInferProperties(ctx);
}

/**
 * @brief Begin inferring the properties of the reduction.
 */

static int        reduxInferProperties          (redux_ctx*  ctx){
	axis_desc* a;
	int        i, j, retT, retK;
	size_t     d;


	/* Source code buffer preallocation failed? */
	if (strb_ensure(&ctx->s, 4*1024) != 0){
		return reduxCleanupMsg(ctx, GA_MEMORY_ERROR,
		    "Could not preallocate source code buffer!\n");
	}


	/* Insane src, reduxLen, dst or dstArg? */
	if       (!ctx->src){
		return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
		    "src is NULL!\n");
	}else if (ctx->src->nd  <= 0){
		return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
		    "src has less than 1 dimensions!\n");
	}else if (ctx->reduxLen <= 0){
		return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
		    "List of dimensions to be reduced is empty!\n");
	}else if (ctx->src->nd  <  (unsigned)ctx->reduxLen){
		return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
		    "src has fewer dimensions than there are dimensions to reduce!\n");
	}else if (reduxRequiresDst   (ctx) && !ctx->dst){
		return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
		    "dst is NULL, but reduction requires it!\n");
	}else if (reduxRequiresDstArg(ctx) && !ctx->dstArg){
		return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
		    "dstArg is NULL, but reduction requires it!\n");
	}else if (ctx->dst    && ctx->dst->nd   +ctx->reduxLen != ctx->src->nd){
		return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
		    "dst is of incorrect dimensionality for this reduction!\n");
	}else if (ctx->dstArg && ctx->dstArg->nd+ctx->reduxLen != ctx->src->nd){
		return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
		    "dstArg is of incorrect dimensionality for this reduction!\n");
	}
	ctx->nds = ctx->src->nd;
	ctx->ndr = ctx->reduxLen;
	ctx->ndd = ctx->nds - ctx->ndr;
	ctx->ndw = 0;
	ctx->ndp = 0;
	ctx->ndf = 0;
	ctx->ndt = ctx->ndd + 1;
	
	/* Insane reduxList? */
	for (i=0;i<ctx->ndr;i++){
		j = ctx->reduxList[i];
		if (j < -ctx->nds || j >= ctx->nds){
			return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
			    "Insane axis number %d! Should be [%d, %d)!\n",
			    j, -ctx->nds, ctx->nds);
		}
		j = j<0 ? ctx->nds+j : j;
		d                 = ctx->src->dimensions[j];
		ctx->zeroRdxAxes += !d;
		ctx->prodRdxAxes *=  d?d:1;
	}


	/**
	 * Insane shape?
	 * 
	 * The source tensor is allowed to be empty (its shape may contain 0s).
	 * However, all axes that are of length 0 must be reduction axes.
	 * 
	 * The reason for this is that a reduction cannot store any output into an
	 * empty destination tensor (whose dimensions are the free axes), because
	 * it has 0 space. The operation cannot then fulfill its contract.
	 * 
	 * On the other hand, when some or all reduction axes of a tensor are of
	 * length 0, the reduction can be interpreted as initializing the
	 * destination tensor to the identity value of the operation. For lack of a
	 * better idea, the destination argument tensor can then be zeroed.
	 */

	for (i=0;i<ctx->nds;i++){
		d                  = ctx->src->dimensions[i];
		ctx->zeroAllAxes += !d;
		ctx->prodAllAxes *=  d?d:1;
	}
	if (ctx->zeroAllAxes != ctx->zeroRdxAxes){
		return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
		    "Source tensor has length-0 dimensions that are not reduced!");
	}
	ctx->prodFreeAxes = ctx->prodAllAxes/ctx->prodRdxAxes;


	/**
	 * GPU context non-existent, or cannot read its properties?
	 */

	ctx->gpuCtx = GpuArray_context(ctx->src);
	if (!ctx->gpuCtx                                                                           ||
	    gpucontext_property(ctx->gpuCtx, GA_CTX_PROP_NUMPROCS,  &ctx->numProcs) != GA_NO_ERROR ||
	    gpucontext_property(ctx->gpuCtx, GA_CTX_PROP_MAXLSIZE,  &ctx->maxLg)    != GA_NO_ERROR ||
	    gpudata_property(ctx->src->data, GA_CTX_PROP_MAXLSIZE0, &ctx->maxLs[0]) != GA_NO_ERROR ||
	    gpudata_property(ctx->src->data, GA_CTX_PROP_MAXLSIZE1, &ctx->maxLs[1]) != GA_NO_ERROR ||
	    gpudata_property(ctx->src->data, GA_CTX_PROP_MAXLSIZE2, &ctx->maxLs[2]) != GA_NO_ERROR ||
	    gpudata_property(ctx->src->data, GA_CTX_PROP_MAXGSIZE,  &ctx->maxGg)    != GA_NO_ERROR ||
	    gpudata_property(ctx->src->data, GA_CTX_PROP_MAXGSIZE0, &ctx->maxGs[0]) != GA_NO_ERROR ||
	    gpudata_property(ctx->src->data, GA_CTX_PROP_MAXGSIZE1, &ctx->maxGs[1]) != GA_NO_ERROR ||
	    gpudata_property(ctx->src->data, GA_CTX_PROP_MAXGSIZE2, &ctx->maxGs[2]) != GA_NO_ERROR ){
		/* gpukernel_property(ctx->kernel.k,     GA_KERNEL_PROP_PREFLSIZE, &warpSize); */
		return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
		    "Error obtaining one or more properties from GPU context!\n");
	}
	ctx->warpSize = 32;


	/**
	 * Type management.
	 * 
	 * - Deal with the various typecodes.
	 * - Determine initializer and error out if reduction unsupported on that
	 *   datatype.
	 */

	ctx->srcTypeCode    = ctx->src->typecode;
	ctx->dstTypeCode    = ctx->srcTypeCode;
	ctx->dstArgTypeCode = GA_SSIZE;
	ctx->idxTypeCode    = GA_SSIZE;
	switch (ctx->srcTypeCode){
		case GA_HALF:
		  ctx->accTypeCode = GA_FLOAT;
		break;
		case GA_HALF2:
		  ctx->accTypeCode = GA_FLOAT2;
		break;
		case GA_HALF4:
		  ctx->accTypeCode = GA_FLOAT4;
		break;
		case GA_HALF8:
		  ctx->accTypeCode = GA_FLOAT8;
		break;
		case GA_HALF16:
		  ctx->accTypeCode = GA_FLOAT16;
		break;
		default:
		  ctx->accTypeCode = ctx->srcTypeCode;
	}
	ctx->srcTypeStr     = gpuarray_get_type(ctx->srcTypeCode)   ->cluda_name;
	ctx->dstTypeStr     = gpuarray_get_type(ctx->dstTypeCode)   ->cluda_name;
	ctx->dstArgTypeStr  = gpuarray_get_type(ctx->dstArgTypeCode)->cluda_name;
	ctx->idxTypeStr     = gpuarray_get_type(ctx->idxTypeCode)   ->cluda_name;
	ctx->accTypeStr     = gpuarray_get_type(ctx->accTypeCode)   ->cluda_name;
	if (!ctx->srcTypeStr    ||
	    !ctx->dstTypeStr    ||
	    !ctx->dstArgTypeStr ||
	    !ctx->idxTypeStr    ||
	    !ctx->accTypeStr    ){
		return reduxCleanup(ctx, GA_INVALID_ERROR);
	}
	switch (ctx->op){
		case GA_REDUCE_SUM:
		  retT = reduxGetSumInit (ctx->dstTypeCode, &ctx->initValT);
		  retK = reduxGetSumInit (ctx->accTypeCode, &ctx->initValK);
		break;
		case GA_REDUCE_PRODNZ:
		case GA_REDUCE_PROD:
		  retT = reduxGetProdInit(ctx->dstTypeCode, &ctx->initValT);
		  retK = reduxGetProdInit(ctx->accTypeCode, &ctx->initValK);
		break;
		case GA_REDUCE_MINANDARGMIN:
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_MIN:
		  retT = reduxGetMinInit (ctx->dstTypeCode, &ctx->initValT);
		  retK = reduxGetMinInit (ctx->accTypeCode, &ctx->initValK);
		break;
		case GA_REDUCE_MAXANDARGMAX:
		case GA_REDUCE_ARGMAX:
		case GA_REDUCE_MAX:
		  retT = reduxGetMaxInit (ctx->dstTypeCode, &ctx->initValT);
		  retK = reduxGetMaxInit (ctx->accTypeCode, &ctx->initValK);
		break;
		case GA_REDUCE_ALL:
		case GA_REDUCE_AND:
		  retT = reduxGetAndInit (ctx->dstTypeCode, &ctx->initValT);
		  retK = reduxGetAndInit (ctx->accTypeCode, &ctx->initValK);
		break;
		case GA_REDUCE_ANY:
		case GA_REDUCE_XOR:
		case GA_REDUCE_OR:
		  retT = reduxGetOrInit  (ctx->dstTypeCode, &ctx->initValT);
		  retK = reduxGetOrInit  (ctx->accTypeCode, &ctx->initValK);
		break;
		default:
		  retT = GA_UNSUPPORTED_ERROR;
		  retK = GA_UNSUPPORTED_ERROR;
	}
	if (retT != GA_NO_ERROR){
		return reduxCleanupMsg(ctx, retT,
		    "Problem selecting types to be used in reduction!\n");
	}
	if (retK != GA_NO_ERROR){
		return reduxCleanupMsg(ctx, retK,
		    "Problem selecting types to be used in reduction!\n");
	}


	/**
	 * Allocate and construct source-tensor axis-description lists.
	 * 
	 * While constructing the descriptions of each axis, verify that:
	 * 
	 *   1. reduxLen has no duplicates.
	 *   2. dst and/or dstArg's dimensions match src's dimensions, stripped of
	 *      the reduction axes.
	 */

	ctx->xdSrc     = calloc(ctx->nds, sizeof(*ctx->xdSrc));
	ctx->xdSrcPtrs = calloc(ctx->nds, sizeof(*ctx->xdSrcPtrs));
	ctx->xdSrcFlat = calloc(ctx->nds, sizeof(*ctx->xdSrcFlat));
	ctx->xdTmp     = calloc(ctx->ndt, sizeof(*ctx->xdTmp));
	if (!ctx->xdSrc || !ctx->xdSrcPtrs || !ctx->xdSrcFlat || !ctx->xdTmp){
		return reduxCleanup(ctx, GA_MEMORY_ERROR);
	}
	for (i=0;i<ctx->nds;i++){
		axisInit(&ctx->xdSrc[i],
		         ctx->src->dimensions[i],
		         ctx->src->strides[i]);
	}
	for (i=0;i<ctx->ndr;i++){
		j = ctx->reduxList[i];
		j = j<0 ? ctx->nds+j : j;
		a = reduxGetSrcAxis(ctx, j);
		if (axisIsReduced(a)){
			return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
			                       "Axis %d appears multiple times in the "
			                       "reduction axis list!\n",
			                       j);
		}
		axisMarkReduced(a, i);
	}
	for (i=j=0;i<ctx->nds;i++){
		axis_desc* a      = reduxGetSrcAxis(ctx, i);
		size_t     srcLen = axisGetLen(a), dstLen, dstArgLen;
		
		if (axisIsReduced(a)){continue;}
		if (reduxRequiresDst(ctx)){
			dstLen = ctx->dst->dimensions[j];
			
			if(srcLen != dstLen){
				return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
				                       "Source axis %d has length %zu, but "
				                       "corresponding destination axis %d has length %zu!\n",
				                       i, srcLen, j, dstLen);
			}
			
			a->dstStride    = ctx->dst->strides[j];
		}
		if (reduxRequiresDstArg(ctx)){
			dstArgLen = ctx->dstArg->dimensions[j];
			
			if(srcLen != dstArgLen){
				return reduxCleanupMsg(ctx, GA_INVALID_ERROR,
				                       "Source axis %d has length %zu, but "
				                       "corresponding destination-argument axis %d has length %zu!\n",
				                       i, srcLen, j, dstArgLen);
			}
			
			a->dstArgStride = ctx->dstArg->strides[j];
		}
		
		j++;
	}


	/**
	 * Begin flattening the source tensor.
	 */

	return reduxFlattenSource(ctx);
}

/**
 * @brief Flatten the source tensor as much as is practical.
 * 
 * This makes the axis lengths as long as possible and the tensor itself as
 * contiguous as possible.
 */

static int        reduxFlattenSource            (redux_ctx*  ctx){
	axis_desc* axis, *flatAxis, *sortAxis;
	int        i, j, isSensitive;
	
	/**
	 * Copy source axis descriptions list to flattened source axis description
	 * list, in preparation for attempts at flattening.
	 */
	
	memcpy(ctx->xdSrcFlat, ctx->xdSrc, ctx->nds*sizeof(*ctx->xdSrcFlat));
	ctx->ndf = ctx->nds;

	/**
	 * Pass 1: Flatten out 0-length dimensions. We already know that
	 * 
	 *         a) There are no 0-length free dimensions, because that
	 *            constitutes an invalid input, and
	 *         b) How many 0-length reduction dimensions there are, because
	 *            we counted them in the error-checking code.
	 * 
	 * So if there are any 0-length axes, we can delete all reduction axes and
	 * replace them with a single one.
	 */
	
	if (ctx->zeroRdxAxes > 0){
		for (i=j=0;i<ctx->ndf;i++){
			axis = reduxGetSrcFlatAxis(ctx, i);
			
			if (!axisIsReduced(axis)){
				*reduxGetSrcFlatAxis(ctx, j++) = *axis;
			}
		}
		
		axisInit       (reduxGetSrcFlatAxis(ctx, j), 0, 0);
		axisMarkReduced(reduxGetSrcFlatAxis(ctx, j), 0);
		j++;
		ctx->ndf = j;
	}
	
	/**
	 * Pass 2: Flatten out 1-length dimensions, since they can always be
	 *         ignored; They are always indexed at [0].
	 */
	
	for (i=j=0;i<ctx->ndf;i++){
		axis = reduxGetSrcFlatAxis(ctx, i);
		
		if (axisGetLen(axis) != 1){
			*reduxGetSrcFlatAxis(ctx, j++) = *axis;
		}
	}
	ctx->ndf = j;
	
	/**
	 * Pass 3: Flatten out continuous dimensions, where strides and sensitivity
	 *         allows it.
	 */
	
	isSensitive = reduxIsSensitive(ctx);
	
	qsort(ctx->xdSrcFlat, ctx->ndf, sizeof(*ctx->xdSrcFlat),
		  isSensitive ? reduxSortFlatSensitive : reduxSortFlatInsensitive);
	
	for (i=j=1;i<ctx->ndf;i++){
		flatAxis = reduxGetSrcFlatAxis(ctx, j-1);
		sortAxis = reduxGetSrcFlatAxis(ctx, i);
		
		if (!reduxTryFlattenInto(ctx, flatAxis, sortAxis)){
			*reduxGetSrcFlatAxis(ctx, j++) = *sortAxis;
		}
	}
	ctx->ndf = j;

	return reduxSelectWarpAxes(ctx);
}

/**
 * @brief Select the warp axes in such a way as to maximize memory bandwidth.
 */

static int        reduxSelectWarpAxes           (redux_ctx*  ctx){
	axis_desc* a;
	int        i;
	size_t     aL;


	/**
	 * NOTE: At this point it is possible for there to be no axes
	 * (ctx->ndf == 0), but this will only occur if all axes of the original
	 * tensor were length-1 (i.e., if this was a scalar masquerading as a
	 * multidimensional tensor).
	 * 
	 * We check for this case and simulate a 1-dimensional, 1-length tensor.
	 */

	if(ctx->ndf == 0){
		axisInit       (reduxGetSrcFlatAxis(ctx, ctx->ndf), 1, 0);
		axisMarkReduced(reduxGetSrcFlatAxis(ctx, ctx->ndf), 0);
		ctx->ndf = 1;
	}


	/**
	 * Select Warp Axes.
	 * 
	 * Using a particular heuristic order (*), sort the axis list by
	 * suitability for belonging to the warp. Then, pick the first few axes,
	 * until the product of their lengths exceeds the warp size.
	 * 
	 * (*) See documentation of value-comparison function.
	 */

	for(i=0;i<ctx->ndf;i++){
		ctx->xdSrcPtrs[i] = reduxGetSrcFlatAxis(ctx, i);
	}

	qsort(ctx->xdSrcPtrs, ctx->ndf, sizeof(*ctx->xdSrcPtrs), reduxSortWarp);

	for (i=0;i<ctx->ndf;i++){
		a  = reduxGetSrcSortAxis(ctx, i);
		aL = axisGetLen(a);
		if (aL <= 1){break;}
		
		ctx->prodWarpAxes *= aL;
		if (ctx->prodWarpAxes <= ctx->warpSize){
			axisMarkWarp(a, aL);
			ctx->ndw++;
		}else{
			/**
			 * The product of warp lengths just exceeded warpSize. We backtrack
			 * by undoing the multiplication by aL. We then check whether we
			 * can "split" this axis by extracting at least a factor of 2 into
			 * warpLen. If yes, we mark is as the (only) warp axis that is
			 * split by setting its warpLen to something neither 0 nor len.
			 */
			
			ctx->prodWarpAxes /= aL;
			aL = ctx->warpSize/ctx->prodWarpAxes;
			if (aL >= 2){
				axisMarkWarp(a, aL);
				ctx->prodWarpAxes  *= aL;
				ctx->splitWarpAxis  = i;
				ctx->ndw++;
				ctx->ndp++;
			}
			break;
		}
	}


	return reduxSelectNumStages(ctx);
}

/**
 * @brief Select the number of stages of the reduction.
 * 
 * This depends a lot on the GPU and the specific size of the reduction.
 */

static int        reduxSelectNumStages          (redux_ctx*  ctx){
	size_t parallelism = 2 * ctx->numProcs * ctx->maxLg;
	
	if(ctx->zeroRdxAxes                     || /* Reduction is empty? */
	   ctx->prodFreeAxes > ctx->prodRdxAxes || /* Large # of destination elements? */
	   ctx->prodFreeAxes > parallelism      ){ /* # of destination elements large enough to fill available parallelism? */
		ctx->numStages = 1;
	}else{
		ctx->numStages = 2;
	}
	
	return reduxSelectHwAxes(ctx);
}

/**
 * @brief Select which axes (up to MAX_HW_DIMS) will be assigned to hardware
 *        dimensions for both the primary and auxiliary kernels.
 *
 * LARGE code model: Up to the MAX_HW_DIMS largest free axes are selected.
 *                   Because the primary reduction kernel does everything, it's
 *                   not necessary to compute an auxiliary kernel axis
 *                   selection (or at least, one distinct from the primary
 *                   kernel's).
 *
 * SMALL code model: For the primary reduction kernel, up to MAX_HW_DIMS
 *                   reduction axes (largest-to-smallest) are selected. If less
 *                   than MAX_HW_DIMS axes were selected, free axes are
 *                   selected until MAX_HW_DIMS total axes are selected, or no
 *                   free axes are left.
 *
 *                   For the auxiliary reduction kernel, up to the MAX_HW_DIMS
 *                   largest free axes are selected.
 */

static int        reduxSelectHwAxes             (redux_ctx*  ctx){
	int ret;
	
	ctx->srcAxisList = malloc(ctx->nds * sizeof(unsigned));
	ctx->dstDims     = malloc(ctx->ndd * sizeof(size_t));
	if (!ctx->srcAxisList ||
	    !ctx->dstDims     ){
		return reduxCleanup(ctx, GA_MEMORY_ERROR);
	}

	ctx->largeCodeModel = 1;/* BUG: Erase when small code model fixed. */
	/**
	 * *** IT IS NOW SAFE TO CALL: ***
	 *       - reduxIsLargeModel()
	 *       - reduxIsSmallModel()
	 *       - reduxKernelRequiresDst()
	 *       - reduxKernelRequiresDstArg()
	 */


	/**
	 * Allocate workspaces.
	 *
	 * Certain reductions may require a workspace that isn't provided by the user.
	 * For instance, **when using the small code model**, argmin/argmax require
	 * a dst buffer, but the user didn't supply one (as he would have for
	 * maxandargmax/minandargmin). We must allocate and deallocate it ourselves.
	 *
	 * Otherwise we use the user-supplied buffers.
	 */

	if (!reduxRequiresDst   (ctx) && reduxKernelRequiresDst(ctx)){
		ctx->wsDst    = malloc(sizeof(*ctx->wsDst));
		if (!ctx->wsDst){
			return reduxCleanup(ctx, GA_MEMORY_ERROR);
		}

		ret = GpuArray_empty(ctx->wsDst, ctx->gpuCtx,  ctx->dstTypeCode,
		                     ctx->ndd,   ctx->dstDims, GA_C_ORDER);
		if (ret != GA_NO_ERROR){
			return reduxCleanup(ctx, ret);
		}
	}else{
		ctx->wsDst    = ctx->dst;
	}
	if (!reduxRequiresDstArg(ctx) && reduxKernelRequiresDstArg(ctx)){
		ctx->wsDstArg = malloc(sizeof(*ctx->wsDstArg));
		if (!ctx->wsDstArg){
			return reduxCleanup(ctx, GA_MEMORY_ERROR);
		}

		ret = GpuArray_empty(ctx->wsDstArg, ctx->gpuCtx,  ctx->dstArgTypeCode,
		                     ctx->ndd,      ctx->dstDims, GA_C_ORDER);
		if (ret != GA_NO_ERROR){
			return reduxCleanup(ctx, ret);
		}
	}else{
		ctx->wsDstArg = ctx->dstArg;
	}


	if (reduxIsLargeCodeModel(ctx)){
		while (reduxCanAppendHwAxis       (ctx, KERNEL_PRIMARY,   AXIS_FREE)){
			reduxAppendLargestAxisToHwList(ctx, KERNEL_PRIMARY,   AXIS_FREE);
		}
	}else{
		while (reduxCanAppendHwAxis       (ctx, KERNEL_PRIMARY,   AXIS_REDUX)){
			reduxAppendLargestAxisToHwList(ctx, KERNEL_PRIMARY,   AXIS_REDUX);
		}
		while (reduxCanAppendHwAxis       (ctx, KERNEL_PRIMARY,   AXIS_FREE)){
			reduxAppendLargestAxisToHwList(ctx, KERNEL_PRIMARY,   AXIS_FREE);
		}

		while (reduxCanAppendHwAxis       (ctx, KERNEL_AUXILIARY, AXIS_FREE)){
			reduxAppendLargestAxisToHwList(ctx, KERNEL_AUXILIARY, AXIS_FREE);
		}
	}

	return reduxComputeAxisList(ctx);
}

/**
 * @brief Compute the axis list.
 *
 * The axis list describes the mapping between the nested loops of the kernel
 * as well as their accompanying indices (i0*, i1*, ..., in*) on one hand, and
 * the axes of the source tensor. The first axis in the list corresponds to the
 * outermost loop and the last axis in the list to the innermost.
 *
 * The first ctx->ndd axes correspond to the outer loops that iterate over
 * each destination element. The last ctx->ndr axes correspond to the inner
 * loops that iterate over the dimensions of elements that are to be reduced.
 */

static int   reduxComputeAxisList          (redux_ctx*  ctx){
	int i, f=0;

	for (i=0;i<ctx->nds;i++){
		if (!axisInSet(i, ctx->reduxList, ctx->ndr, 0)){
			ctx->srcAxisList[f++] = i;
		}
	}
	memcpy(&ctx->srcAxisList[f], ctx->reduxList, ctx->ndr * sizeof(*ctx->reduxList));


	return reduxGenSource(ctx);
}

/**
 * @brief Generate the kernel code for the reduction.
 *
 * @return GA_MEMORY_ERROR if not enough memory left; GA_NO_ERROR otherwise.
 */

static int   reduxGenSource                (redux_ctx*  ctx){
	reduxAppendSource(ctx);
	ctx->sourceCodeLen = ctx->s.l;
	ctx->sourceCode    = strb_cstr(&ctx->s);
	if (!ctx->sourceCode){
		return reduxCleanup(ctx, GA_MEMORY_ERROR);
	}

	return reduxCompile(ctx);
}
static void  reduxAppendSource             (redux_ctx*  ctx){
	reduxAppendIncludes         (ctx);
	reduxAppendMacroDefs        (ctx);
	reduxAppendTypedefs         (ctx);
	reduxAppendGetInitValFns    (ctx);
	reduxAppendWriteBackFn      (ctx);
	reduxAppendReduxKernel       (ctx);
	if (reduxIsSmallCodeModel(ctx)){
		reduxAppendInitKernel    (ctx);
		reduxAppendPostKernel   (ctx);
	}
}
static void  reduxAppendTensorDeclArgs     (redux_ctx*  ctx,
                                            const char* type,
                                            const char* baseName){
	srcbAppendElemf(&ctx->srcGen, "%s* %sPtr",             type, baseName);
	srcbAppendElemf(&ctx->srcGen, "const X %sOff",               baseName);
	srcbAppendElemf(&ctx->srcGen, "const GLOBAL_MEM X* %sSteps", baseName);
	(void)reduxAppendTensorCallArgs;/* Silence unused warning */
}
static void  reduxAppendTensorCallArgs     (redux_ctx*  ctx,
                                            const char* baseName){
	srcbAppendElemf(&ctx->srcGen, "%sPtr",   baseName);
	srcbAppendElemf(&ctx->srcGen, "%sOff",   baseName);
	srcbAppendElemf(&ctx->srcGen, "%sSteps", baseName);
}
static void  reduxAppendMacroDefs          (redux_ctx*  ctx){
	int i;

	srcbAppends    (&ctx->srcGen, "#define FOROVER(idx)    for (i##idx = i##idx##Start; i##idx < i##idx##End; i##idx++)\n");
	srcbAppends    (&ctx->srcGen, "#define ESCAPE(idx)     if (i##idx >= i##idx##Dim){continue;}\n");

	/* srcVal indexer */
	srcbAppends    (&ctx->srcGen, "#define srcVal          (*(const GLOBAL_MEM S*)(");
	srcbBeginList  (&ctx->srcGen, "+", "0");
	srcbAppendElemf(&ctx->srcGen, "(const GLOBAL_MEM char*)srcPtr");
	srcbAppendElemf(&ctx->srcGen, "srcOff");
	for (i=0;i<ctx->nds;i++){
		srcbAppendElemf(&ctx->srcGen, "i%d*i%dSStep", i, i);
	}
	srcbEndList    (&ctx->srcGen);
	srcbAppends    (&ctx->srcGen, "))\n");

	/* dstVal indexer */
	if (reduxKernelRequiresDst(ctx)){
		srcbAppends    (&ctx->srcGen, "#define dstVal          (*(GLOBAL_MEM T*)(");
		srcbBeginList  (&ctx->srcGen, "+", "0");
		srcbAppendElemf(&ctx->srcGen, "(GLOBAL_MEM char*)dstPtr");
		srcbAppendElemf(&ctx->srcGen, "dstOff");
		for (i=0;i<ctx->ndd;i++){
			srcbAppendElemf(&ctx->srcGen, "i%d*i%dDStep", i, i);
		}
		srcbEndList    (&ctx->srcGen);
		srcbAppends    (&ctx->srcGen, "))\n");
	}

	/* dstArgVal indexer */
	if (reduxKernelRequiresDstArg(ctx)){
		srcbAppends    (&ctx->srcGen, "#define dstArgVal       (*(GLOBAL_MEM A*)(");
		srcbBeginList  (&ctx->srcGen, "+", "0");
		srcbAppendElemf(&ctx->srcGen, "(GLOBAL_MEM char*)dstArgPtr");
		srcbAppendElemf(&ctx->srcGen, "dstArgOff");
		for (i=0;i<ctx->ndd;i++){
			srcbAppendElemf(&ctx->srcGen, "i%d*i%dAStep", i, i);
		}
		srcbEndList    (&ctx->srcGen);
		srcbAppends    (&ctx->srcGen, "))\n");
	}

	/* rdxIdx indexer */
	srcbAppends    (&ctx->srcGen, "#define rdxIdx          (");
	srcbBeginList  (&ctx->srcGen, "+", "0");
	for (i=ctx->ndd;i<ctx->nds;i++){
		srcbAppendElemf(&ctx->srcGen, "i%d*i%dPDim", i, i);
	}
	srcbEndList    (&ctx->srcGen);
	srcbAppends    (&ctx->srcGen, ")\n");
}
static void  reduxAppendIncludes           (redux_ctx*  ctx){
	strb_appends(&ctx->s, "/* Includes */\n");
	strb_appends(&ctx->s, "#include \"cluda.h\"\n");
	strb_appends(&ctx->s, "\n");
	strb_appends(&ctx->s, "\n");
	strb_appends(&ctx->s, "\n");
}
static void  reduxAppendTypedefs           (redux_ctx*  ctx){
	strb_appendf(&ctx->s, "typedef %s S;\n", ctx->srcTypeStr);   /* The type of the source array. */
	strb_appendf(&ctx->s, "typedef %s T;\n", ctx->dstTypeStr);   /* The type of the destination array. */
	strb_appendf(&ctx->s, "typedef %s A;\n", ctx->dstArgTypeStr);/* The type of the destination argument array. */
	strb_appendf(&ctx->s, "typedef %s X;\n", ctx->idxTypeStr);   /* The type of the indices: signed 32/64-bit. */
	strb_appendf(&ctx->s, "typedef %s K;\n", ctx->accTypeStr);   /* The type of the accumulator variable. */
}
static void  reduxAppendGetInitValFns      (redux_ctx*  ctx){
	/**
	 * Initial value functions.
	 */

	strb_appendf(&ctx->s, "WITHIN_KERNEL T    getInitValTFn(void){\n"
	                      "\treturn (%s);\n"
	                      "}\n\n\n\n"
	                      "WITHIN_KERNEL K    getInitValKFn(void){\n"
	                      "\treturn (%s);\n"
	                      "}\n\n\n\n", ctx->initValT, ctx->initValK);
}
static void  reduxAppendWriteBackFn        (redux_ctx*  ctx){
	/**
	 * Global memory value reduction function.
	 *
	 * Responsible for either:
	 *   1) Safe writeback of final value to memory, or
	 *   2) Safe atomic reduction of partial value into memory.
	 */

	srcbAppends    (&ctx->srcGen, "WITHIN_KERNEL void writeBackFn(");
	srcbBeginList  (&ctx->srcGen, ", ", "void");
	if (reduxKernelRequiresDst(ctx)){
		srcbAppendElemf(&ctx->srcGen, "GLOBAL_MEM T* d_");
		srcbAppendElemf(&ctx->srcGen, "T d");
	}
	if (reduxKernelRequiresDstArg(ctx)){
		srcbAppendElemf(&ctx->srcGen, "GLOBAL_MEM A* a_");
		srcbAppendElemf(&ctx->srcGen, "A a");
	}
	srcbEndList    (&ctx->srcGen);
	srcbAppends    (&ctx->srcGen, "){\n");

	if (reduxIsLargeCodeModel(ctx)){
		if (reduxKernelRequiresDst   (ctx)){
			srcbAppends    (&ctx->srcGen, "\t*d_ = d;\n");
		}
		if (reduxKernelRequiresDstArg(ctx)){
			srcbAppends    (&ctx->srcGen, "\t*a_ = a;\n");
		}
	}else{
		/* BUG: Implement the atomic reduction, one or two CAS loops. */
		if       ( reduxKernelRequiresDst   (ctx) && !reduxKernelRequiresDstArg(ctx)){

		}else if (!reduxKernelRequiresDst   (ctx) &&  reduxKernelRequiresDstArg(ctx)){

		}else if ( reduxKernelRequiresDst   (ctx) &&  reduxKernelRequiresDstArg(ctx)){

		}
	}

	/* Close off function. */
	strb_appends(&ctx->s, "}\n\n\n\n");
}
static void  reduxAppendReduxKernel        (redux_ctx*  ctx){
	reduxAppendPrototype        (ctx);
	strb_appends                (&ctx->s, "{\n");
	reduxAppendIndexDeclarations(ctx);
	reduxAppendRangeCalculations(ctx);
	reduxAppendLoops            (ctx);
	strb_appends                (&ctx->s, "}\n");
}
static void  reduxAppendPrototype          (redux_ctx*  ctx){
	srcbAppends    (&ctx->srcGen, "KERNEL void reduxKer(");
	srcbBeginList  (&ctx->srcGen, ", ", "void");
	reduxAppendTensorDeclArgs(ctx, "S", "src");
	srcbAppendElemf(&ctx->srcGen, "const GLOBAL_MEM X*        srcSize");
	srcbAppendElemf(&ctx->srcGen, "const GLOBAL_MEM X*        chunkSize");
	if (reduxKernelRequiresDst(ctx)){
		reduxAppendTensorDeclArgs(ctx, "T", "dst");
	}
	if (reduxKernelRequiresDstArg(ctx)){
		reduxAppendTensorDeclArgs(ctx, "A", "dstArg");
	}
	srcbEndList    (&ctx->srcGen);
	srcbAppends    (&ctx->srcGen, ")");
}
static void  reduxAppendIndexDeclarations  (redux_ctx*  ctx){
	int i;
	strb_appends(&ctx->s, "\t/* GPU kernel coordinates. Always 3D in OpenCL/CUDA. */\n");

	strb_appends(&ctx->s, "\tX bi0 = GID_0,        bi1 = GID_1,        bi2 = GID_2;\n");
	strb_appends(&ctx->s, "\tX bd0 = LDIM_0,       bd1 = LDIM_1,       bd2 = LDIM_2;\n");
	strb_appends(&ctx->s, "\tX ti0 = LID_0,        ti1 = LID_1,        ti2 = LID_2;\n");
	strb_appends(&ctx->s, "\tX gi0 = bi0*bd0+ti0,  gi1 = bi1*bd1+ti1,  gi2 = bi2*bd2+ti2;\n");
	if (ctx->pri.ndh>0){
		strb_appends(&ctx->s, "\tX ");
		for (i=0;i<ctx->pri.ndh;i++){
			strb_appendf(&ctx->s, "ci%u = chunkSize[%u]%s",
			             i, i, (i==ctx->pri.ndh-1) ? ";\n" : ", ");
		}
	}
	strb_appends(&ctx->s, "\t\n\t\n");
	strb_appends(&ctx->s, "\t/* Free indices & Reduction indices */\n");
	if (ctx->nds >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->nds, "",        ";\n");}
	if (ctx->nds >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->nds, "Dim",     ";\n");}
	if (ctx->nds >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->nds, "Start",   ";\n");}
	if (ctx->nds >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->nds, "End",     ";\n");}
	if (ctx->nds >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->nds, "SStep",   ";\n");}
	if (ctx->ndd >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->ndd, "DStep",   ";\n");}
	if (ctx->ndd >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->ndd, "AStep",   ";\n");}
	if (ctx->nds > ctx->ndd){appendIdxes (&ctx->s, "\tX ", "i", ctx->ndd, ctx->nds, "PDim",    ";\n");}
	strb_appends(&ctx->s, "\t\n\t\n");
}
static void  reduxAppendRangeCalculations  (redux_ctx*  ctx){
	size_t hwDim;
	int    i;

	strb_appends(&ctx->s, "\t/* Compute ranges for this thread. */\n");

	for (i=0;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "\ti%dDim     = srcSize[%d];\n",  i, ctx->srcAxisList[i]);
	}
	for (i=0;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "\ti%dSStep   = srcSteps[%d];\n", i, ctx->srcAxisList[i]);
	}
	if (reduxKernelRequiresDst(ctx)){
		for (i=0;i<ctx->ndd;i++){
			strb_appendf(&ctx->s, "\ti%dDStep   = dstSteps[%d];\n", i, i);
		}
	}
	if (reduxKernelRequiresDstArg(ctx)){
		for (i=0;i<ctx->ndd;i++){
			strb_appendf(&ctx->s, "\ti%dAStep   = dstArgSteps[%d];\n", i, i);
		}
	}
	for (i=ctx->nds-1;i>=ctx->ndd;i--){
		/**
		 * If this is the last index, it's the first cumulative dimension
		 * product we generate, and thus we initialize to 1.
		 */

		if (i == ctx->nds-1){
			strb_appendf(&ctx->s, "\ti%dPDim    = 1;\n", i);
		}else{
			strb_appendf(&ctx->s, "\ti%dPDim    = i%dPDim * i%dDim;\n", i, i+1, i+1);
		}
	}
	for (i=0;i<ctx->nds;i++){
		/**
		 * Up to MAX_HW_DIMS dimensions get to rely on hardware loops.
		 * The others, if any, have to use software looping beginning at 0.
		 */

		if (axisInSet(ctx->srcAxisList[i], ctx->pri.axisList, ctx->pri.ndh, &hwDim)){
			strb_appendf(&ctx->s, "\ti%dStart   = gi%d * ci%d;\n", i, hwDim, hwDim);
		}else{
			strb_appendf(&ctx->s, "\ti%dStart   = 0;\n", i);
		}
	}
	for (i=0;i<ctx->nds;i++){
		/**
		 * Up to MAX_HW_DIMS dimensions get to rely on hardware loops.
		 * The others, if any, have to use software looping beginning at 0.
		 */

		if (axisInSet(ctx->srcAxisList[i], ctx->pri.axisList, ctx->pri.ndh, &hwDim)){
			strb_appendf(&ctx->s, "\ti%dEnd     = i%dStart + ci%d;\n", i, i, hwDim);
		}else{
			strb_appendf(&ctx->s, "\ti%dEnd     = i%dStart + i%dDim;\n", i, i, i);
		}
	}

	strb_appends(&ctx->s, "\t\n\t\n");
}
static void  reduxAppendLoops              (redux_ctx*  ctx){
	int i;

	for (i=0;i<ctx->ndd;i++){
		srcbAppendf(&ctx->srcGen, "\tFOROVER(%d){ESCAPE(%d)\n", i, i);
	}

	srcbAppends    (&ctx->srcGen, "\t\tT rdxT;\n");
	srcbAppends    (&ctx->srcGen, "\t\tK rdxK = getInitValKFn();\n");
	if (reduxKernelRequiresDstArg(ctx)){
		srcbAppends(&ctx->srcGen, "\t\tX rdxA = 0;\n");
	}
	srcbAppends    (&ctx->srcGen, "\t\t\n");

	for (i=ctx->ndd;i<ctx->nds;i++){
		srcbAppendf    (&ctx->srcGen, "\t\tFOROVER(%d){ESCAPE(%d)\n", i, i);
	}

	srcbAppends    (&ctx->srcGen, "\t\t\tS s = srcVal;\n");

	/**
	 * Prescalar transformations go here. They transform and coerce the S-typed
	 * value s into the K-typed value k.
	 */

	srcbAppends    (&ctx->srcGen, "\t\t\tK k = s;\n");

	switch (ctx->op){
		case GA_REDUCE_SUM:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK += k;\n");
		break;
		case GA_REDUCE_PROD:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK *= k;\n");
		break;
		case GA_REDUCE_PRODNZ:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK *= k==0 ? getInitValKFn() : k;\n");
		break;
		case GA_REDUCE_MIN:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK  = min(rdxK, k);\n");
		break;
		case GA_REDUCE_MAX:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK  = max(rdxK, k);\n");
		break;
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_MINANDARGMIN:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK  = min(rdxK, k);\n"
		                            "\t\t\tif (rdxK == k){\n"
		                            "\t\t\t\trdxA = rdxIdx;\n"
		                            "\t\t\t}\n");
		break;
		case GA_REDUCE_ARGMAX:
		case GA_REDUCE_MAXANDARGMAX:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK  = max(rdxK, k);\n"
		                            "\t\t\tif (rdxK == k){\n"
		                            "\t\t\t\trdxA = rdxIdx;\n"
		                            "\t\t\t}\n");
		break;
		case GA_REDUCE_AND:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK &= k;\n");
		break;
		case GA_REDUCE_OR:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK |= k;\n");
		break;
		case GA_REDUCE_XOR:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK ^= k;\n");
		break;
		case GA_REDUCE_ALL:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK  = rdxK && k;\n");
		break;
		case GA_REDUCE_ANY:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK  = rdxK || k;\n");
		break;
	}

	for (i=ctx->ndd;i<ctx->nds;i++){
		srcbAppends(&ctx->srcGen, "\t\t}\n");
	}
	srcbAppends(&ctx->srcGen, "\t\t\n");

	/**
	 * Large code model: Postscalar transformations go here, coercing the
	 * K-typed value rdxK to the T-typed value rdxT
	 */

	srcbAppends    (&ctx->srcGen, "\t\trdxT = rdxK;\n");

	/* Final writeback. */
	srcbAppends    (&ctx->srcGen, "\t\twriteBackFn(");
	srcbBeginList  (&ctx->srcGen, ", ", "");
	if (reduxKernelRequiresDst(ctx)){
		srcbAppendElemf(&ctx->srcGen, "&dstVal");
		srcbAppendElemf(&ctx->srcGen, "rdxT");
	}
	if (reduxKernelRequiresDstArg(ctx)){
		srcbAppendElemf(&ctx->srcGen, "&dstArgVal");
		srcbAppendElemf(&ctx->srcGen, "rdxA");
	}
	srcbEndList    (&ctx->srcGen);
	srcbAppends    (&ctx->srcGen, ");\n");

	for (i=0;i<ctx->ndd;i++){
		srcbAppends(&ctx->srcGen, "\t}\n");
	}
}
static void  reduxAppendInitKernel         (redux_ctx*  ctx){
	/* BUG: Implement this for small code model. */
}
static void  reduxAppendPostKernel         (redux_ctx*  ctx){
	/* BUG: Implement this for small code model. */
}

/**
 * @brief Compile the kernel from source code.
 */

static int   reduxCompile                  (redux_ctx*  ctx){
	int    ret, i = 0;
	int    PRI_TYPECODES[11];
	size_t PRI_TYPECODES_LEN;
	int*   AUX_TYPECODES;
	size_t AUX_TYPECODES_LEN;


	/**
	 * Construct Argument Typecode Lists.
	 */

	PRI_TYPECODES[i++] = GA_BUFFER; /* srcPtr */
	PRI_TYPECODES[i++] = GA_SIZE;   /* srcOff */
	PRI_TYPECODES[i++] = GA_BUFFER; /* srcSteps */
	PRI_TYPECODES[i++] = GA_BUFFER; /* srcSize */
	PRI_TYPECODES[i++] = GA_BUFFER; /* chnkSize */
	if (reduxKernelRequiresDst(ctx)){
		PRI_TYPECODES[i++] = GA_BUFFER; /* dstPtr */
		PRI_TYPECODES[i++] = GA_SIZE;   /* dstOff */
		PRI_TYPECODES[i++] = GA_BUFFER; /* dstSteps */
	}
	if (reduxKernelRequiresDstArg(ctx)){
		PRI_TYPECODES[i++] = GA_BUFFER; /* dstArgPtr */
		PRI_TYPECODES[i++] = GA_SIZE;   /* dstArgOff */
		PRI_TYPECODES[i++] = GA_BUFFER; /* dstArgSteps */
	}
	PRI_TYPECODES_LEN  = i;
	AUX_TYPECODES      = &PRI_TYPECODES[3];
	AUX_TYPECODES_LEN  = PRI_TYPECODES_LEN-3;


	/**
	 * Compile the kernels.
	 */

	{
		ret  = GpuKernel_init(&ctx->kernel,
		                      ctx->gpuCtx,
		                      1,
		                      (const char**)&ctx->sourceCode,
		                      &ctx->sourceCodeLen,
		                      "reduxKer",
		                      PRI_TYPECODES_LEN,
		                      PRI_TYPECODES,
		                      GA_USE_CLUDA,
		                      &ctx->errorString0);
		if (ret != GA_NO_ERROR){
			return reduxCleanup(ctx, ret);
		}
	}
	if (reduxIsSmallCodeModel(ctx)){
		ret  = GpuKernel_init(&ctx->kernel,
		                      ctx->gpuCtx,
		                      1,
		                      (const char**)&ctx->sourceCode,
		                      &ctx->sourceCodeLen,
		                      "initKer",
		                      AUX_TYPECODES_LEN,
		                      AUX_TYPECODES,
		                      GA_USE_CLUDA,
		                      &ctx->errorString1);
		if (ret != GA_NO_ERROR){
			return reduxCleanup(ctx, ret);
		}
		ret  = GpuKernel_init(&ctx->kernel,
		                      ctx->gpuCtx,
		                      1,
		                      (const char**)&ctx->sourceCode,
		                      &ctx->sourceCodeLen,
		                      "postKer",
		                      AUX_TYPECODES_LEN,
		                      AUX_TYPECODES,
		                      GA_USE_CLUDA,
		                      &ctx->errorString2);
		if (ret != GA_NO_ERROR){
			return reduxCleanup(ctx, ret);
		}
	}

	return reduxSchedule(ctx);
}

/**
 * @brief Compute a good thread block size / grid size / software chunk size
 *        for the primary/auxilliary kernels.
 */

static int   reduxSchedule                 (redux_ctx*  ctx){
	int      i, priNdims = 0, auxNdims = 0;
	uint64_t maxLgRdx = 0, maxLgPre = 0, maxLgPost = 0;
	uint64_t maxLgPri = 0, maxLgAux = 0;
	uint64_t maxLs  [MAX_HW_DIMS];
	uint64_t maxGg;
	uint64_t maxGs  [MAX_HW_DIMS];
	uint64_t priDims[MAX_HW_DIMS];
	uint64_t auxDims[MAX_HW_DIMS];
	uint64_t bs     [MAX_HW_DIMS];
	uint64_t gs     [MAX_HW_DIMS];
	uint64_t cs     [MAX_HW_DIMS];
	size_t   warpSize,
	         maxL, maxL0, maxL1, maxL2,
	         maxG, maxG0, maxG1, maxG2;


	/**
	 * Obtain the constraints of our problem.
	 */

	gpudata_property  (ctx->src->data,    GA_CTX_PROP_MAXLSIZE0,    &maxL0);
	gpudata_property  (ctx->src->data,    GA_CTX_PROP_MAXLSIZE1,    &maxL1);
	gpudata_property  (ctx->src->data,    GA_CTX_PROP_MAXLSIZE2,    &maxL2);
	gpudata_property  (ctx->src->data,    GA_CTX_PROP_MAXGSIZE,     &maxG);
	gpudata_property  (ctx->src->data,    GA_CTX_PROP_MAXGSIZE0,    &maxG0);
	gpudata_property  (ctx->src->data,    GA_CTX_PROP_MAXGSIZE1,    &maxG1);
	gpudata_property  (ctx->src->data,    GA_CTX_PROP_MAXGSIZE2,    &maxG2);
	gpukernel_property(ctx->kernel.k,     GA_KERNEL_PROP_PREFLSIZE, &warpSize);
	gpukernel_property(ctx->kernel.k,     GA_KERNEL_PROP_MAXLSIZE,  &maxL);
	maxLgRdx  = maxL;
	maxLgPri  = maxLgRdx;
	if (reduxIsSmallCodeModel(ctx)){
		gpukernel_property(ctx->preKernel.k,  GA_KERNEL_PROP_MAXLSIZE,  &maxL);
		maxLgPre  = maxL;
		gpukernel_property(ctx->postKernel.k, GA_KERNEL_PROP_MAXLSIZE,  &maxL);
		maxLgPost = maxL;
		maxLgAux  = maxLgPre<maxLgPost ? maxLgPre : maxLgPost;
	}

	priNdims  = ctx->pri.ndh;
	maxGs[0]  = maxG0;
	maxGs[1]  = maxG1;
	maxGs[2]  = maxG2;
	maxGg     = maxG;
	maxLs[0]  = maxL0;
	maxLs[1]  = maxL1;
	maxLs[2]  = maxL2;
	for (i=0;i<priNdims;i++){
		priDims[i] = ctx->src->dimensions[ctx->pri.axisList[i]];
	}
	if (reduxIsSmallCodeModel(ctx)){
		auxNdims  = ctx->aux.ndh;
		for (i=0;i<auxNdims;i++){
			auxDims[i] = ctx->src->dimensions[ctx->aux.axisList[i]];
		}
	}


	/**
	 * Apply the solver.
	 */

	{
		reduxScheduleKernel(priNdims,
		                    priDims,
		                    warpSize,
		                    maxLgPri, maxLs,
		                    maxGg,    maxGs,
		                    bs, gs, cs);
		for (i=0;i<priNdims;i++){
			ctx->pri.bs[i] = bs[i];
			ctx->pri.gs[i] = gs[i];
			ctx->pri.cs[i] = cs[i];
		}
		if (priNdims <= 0){
			ctx->pri.bs[i] = ctx->pri.gs[i] = ctx->pri.cs[i] = 1;
		}
	}
	if (reduxIsSmallCodeModel(ctx)){
		reduxScheduleKernel(auxNdims,
		                    auxDims,
		                    warpSize,
		                    maxLgAux, maxLs,
		                    maxGg,    maxGs,
		                    bs, gs, cs);
		for (i=0;i<auxNdims;i++){
			ctx->aux.bs[i] = bs[i];
			ctx->aux.gs[i] = gs[i];
			ctx->aux.cs[i] = cs[i];
		}
		if (auxNdims <= 0){
			ctx->aux.bs[i] = ctx->aux.gs[i] = ctx->aux.cs[i] = 1;
		}
	}

	return reduxInvoke(ctx);
}

/**
 * @brief Given the parameters of a kernel scheduling problem, solve it as
 *        optimally as possible.
 *
 * NB: This is the only function in this entire file that should have
 *     anything to do with the integer factorization APIs.
 */

static void  reduxScheduleKernel           (int         ndims,
                                            uint64_t*   dims,
                                            uint64_t    warpSize,
                                            uint64_t    maxLg,
                                            uint64_t*   maxLs,
                                            uint64_t    maxGg,
                                            uint64_t*   maxGs,
                                            uint64_t*   bs,
                                            uint64_t*   gs,
                                            uint64_t*   cs){
	uint64_t       warpMod, bestWarpMod  = 1;
	int            i,       bestWarpAxis = 0;
	uint64_t       roundedDims[MAX_HW_DIMS];
	double         slack      [MAX_HW_DIMS];
	ga_factor_list factBS     [MAX_HW_DIMS];
	ga_factor_list factGS     [MAX_HW_DIMS];
	ga_factor_list factCS     [MAX_HW_DIMS];


	/**
	 * Quick check for scalar case.
	 */

	if (ndims <= 0){
		return;
	}


	/**
	 * Identify the dimension to which the warp factor will be given.
	 *
	 * The current heuristic is to find the dimension that is either
	 *   1) Evenly divided by the warp size, or
	 *   2) As close to filling the last warp as possible.
	 */

	for (i=0;i<ndims;i++){
		roundedDims[i] = dims[i];
		slack      [i] = 1.1;
		gaIFLInit(&factBS[i]);
		gaIFLInit(&factGS[i]);
		gaIFLInit(&factCS[i]);

		warpMod = roundedDims[i] % warpSize;
		if (bestWarpMod>0 && (warpMod==0 || warpMod>=bestWarpMod)){
			bestWarpAxis = i;
			bestWarpMod  = warpMod;
		}
	}

	if (ndims > 0){
		roundedDims[bestWarpAxis] = (roundedDims[bestWarpAxis] + warpSize - 1)/warpSize;
		gaIFactorize(warpSize, 0, 0, &factBS[bestWarpAxis]);
	}

	/**
	 * Factorization job. We'll steadily increase the slack in case of failure
	 * in order to ensure we do get a factorization, which we place into
	 * chunkSize.
	 */

	for (i=0;i<ndims;i++){
		while (!gaIFactorize(roundedDims[i],
		                     roundedDims[i]*slack[i],
		                     maxLs      [i],
		                     &factCS    [i])){
			/**
			 * Error! Failed to factorize dimension i with given slack and
			 * k-smoothness constraints! Increase slack. Once slack reaches
			 * 2.0 it will factorize guaranteed.
			 */

			slack[i] += 0.1;
		}
	}

	/**
	 * Invoke the scheduler.
	 *
	 * The scheduler will move some factors from chunkSize into blockSize and
	 * gridSize, improving performance.
	 */

	gaIFLSchedule(ndims, maxLg, maxLs, maxGg, maxGs, factBS, factGS, factCS);
	for (i=0;i<ndims;i++){
		bs[i] = gaIFLGetProduct(&factBS[i]);
		gs[i] = gaIFLGetProduct(&factGS[i]);
		cs[i] = gaIFLGetProduct(&factCS[i]);
	}
}

/**
 * Invoke the kernel.
 */

static int   reduxInvoke                   (redux_ctx*  ctx){
	void* priArgs[11];
	void* auxArgs[ 8];
	int   ret, i = 0;
	int   failedDstSteps     = 0;
	int   failedDstArgSteps  = 0;
	int   failedAuxChunkSize = 0;


	/**
	 * Argument Marshalling. This the grossest gross thing in here.
	 */

	const int flags      = GA_BUFFER_READ_ONLY|GA_BUFFER_INIT;
	ctx->srcStepsGD      = gpudata_alloc(ctx->gpuCtx, ctx->nds     * sizeof(size_t),
	                                     ctx->src->strides,     flags, 0);
	ctx->srcSizeGD       = gpudata_alloc(ctx->gpuCtx, ctx->nds     * sizeof(size_t),
	                                     ctx->src->dimensions,  flags, 0);
	ctx->pri.chunkSizeGD = gpudata_alloc(ctx->gpuCtx, ctx->pri.ndh * sizeof(size_t),
	                                     ctx->pri.cs,           flags, 0);

	priArgs[i++] = (void*) ctx->src->data;
	priArgs[i++] = (void*)&ctx->src->offset;
	priArgs[i++] = (void*) ctx->srcStepsGD;
	priArgs[i++] = (void*) ctx->srcSizeGD;
	priArgs[i++] = (void*) ctx->pri.chunkSizeGD;
	if (reduxKernelRequiresDst   (ctx)){
		ctx->dstStepsGD      = gpudata_alloc(ctx->gpuCtx, ctx->ndd * sizeof(size_t),
		                                     ctx->wsDst->strides,    flags, 0);
		priArgs[i++]         = (void*) ctx->wsDst->data;
		priArgs[i++]         = (void*)&ctx->wsDst->offset;
		priArgs[i++]         = (void*) ctx->dstStepsGD;
		failedDstSteps       =        !ctx->dstStepsGD;
	}
	if (reduxKernelRequiresDstArg(ctx)){
		ctx->dstArgStepsGD   = gpudata_alloc(ctx->gpuCtx, ctx->ndd * sizeof(size_t),
		                                     ctx->wsDstArg->strides, flags, 0);
		priArgs[i++]         = (void*) ctx->wsDstArg->data;
		priArgs[i++]         = (void*)&ctx->wsDstArg->offset;
		priArgs[i++]         = (void*) ctx->dstArgStepsGD;
		failedDstArgSteps    =        !ctx->dstArgStepsGD;
	}
	if (reduxIsSmallCodeModel    (ctx)){
		/**
		 * The auxiliary kernel's args are identical to the primary kernel's,
		 * except that the first three arguments are deleted and the fifth
		 * argument (now second), called chunkSize, is different.
		 */

		memcpy(auxArgs, &priArgs[3], sizeof(auxArgs));
		ctx->aux.chunkSizeGD = gpudata_alloc(ctx->gpuCtx, ctx->aux.ndh * sizeof(size_t),
		                                     ctx->aux.cs,           flags, 0);
		auxArgs[ 1 ]         = (void*) ctx->aux.chunkSizeGD;
		failedAuxChunkSize   =        !ctx->aux.chunkSizeGD;
	}


	/**
	 * One or three kernels is now invoked, depending on the code model.
	 */

	if (ctx->srcStepsGD      &&
	    ctx->srcSizeGD       &&
	    ctx->pri.chunkSizeGD &&
	    !failedDstSteps      &&
	    !failedDstArgSteps   &&
	    !failedAuxChunkSize){
		/* Pre-kernel invocation, if necessary */
		if (reduxIsSmallCodeModel(ctx)){
			ret = GpuKernel_call(&ctx->preKernel,
			                     ctx->aux.ndh>0 ? ctx->aux.ndh : 1,
			                     ctx->aux.gs,
			                     ctx->aux.bs,
			                     0,
			                     auxArgs);
			if (ret != GA_NO_ERROR){
				return reduxCleanup(ctx, ret);
			}
		}

		/* Reduction kernel invocation */
		ret = GpuKernel_call(&ctx->kernel,
		                     ctx->pri.ndh>0 ? ctx->pri.ndh : 1,
		                     ctx->pri.gs,
		                     ctx->pri.bs,
		                     0,
		                     priArgs);
		if (ret != GA_NO_ERROR){
			return reduxCleanup(ctx, ret);
		}

		/* Post-kernel invocation, if necessary */
		if (reduxIsSmallCodeModel(ctx)){
			ret = GpuKernel_call(&ctx->postKernel,
			                     ctx->aux.ndh>0 ? ctx->aux.ndh : 1,
			                     ctx->aux.gs,
			                     ctx->aux.bs,
			                     0,
			                     auxArgs);
			if (ret != GA_NO_ERROR){
				return reduxCleanup(ctx, ret);
			}
		}

		return reduxCleanup(ctx, ret);
	}else{
		return reduxCleanup(ctx, GA_MEMORY_ERROR);
	}
}

/**
 * Cleanup
 */

static int        reduxCleanup                  (redux_ctx*  ctx, int ret){
	if (ctx->dst    != ctx->wsDst){
		if(ctx->wsDst){
			GpuArray_clear(ctx->wsDst);
		}
		free(ctx->wsDst);
		ctx->wsDst    = NULL;
	}
	if (ctx->dstArg != ctx->wsDstArg){
		if(ctx->wsDstArg){
			GpuArray_clear(ctx->wsDstArg);
		}
		free(ctx->wsDstArg);
		ctx->wsDstArg = NULL;
	}

	free(ctx->srcAxisList);
	free(ctx->dstDims);
	free(ctx->sourceCode);
	free(ctx->errorString0);
	free(ctx->errorString1);
	free(ctx->errorString2);
	ctx->srcAxisList  = NULL;
	ctx->dstDims      = NULL;
	ctx->sourceCode   = NULL;
	ctx->errorString0 = NULL;
	ctx->errorString1 = NULL;
	ctx->errorString2 = NULL;

	gpudata_release(ctx->srcStepsGD);
	gpudata_release(ctx->srcSizeGD);
	gpudata_release(ctx->dstStepsGD);
	gpudata_release(ctx->dstArgStepsGD);
	gpudata_release(ctx->pri.chunkSizeGD);
	gpudata_release(ctx->aux.chunkSizeGD);
	ctx->srcStepsGD      = ctx->srcSizeGD       =
	ctx->dstStepsGD      = ctx->dstArgStepsGD   =
	ctx->pri.chunkSizeGD = ctx->aux.chunkSizeGD = NULL;

	return ret;
}

static int   reduxCleanupMsg               (redux_ctx*  ctx, int ret,
                                            const char* fmt, ...){
#if DEBUG
	FILE* fp = stderr;
	
	va_list ap;
	va_start(ap, fmt);
	vfprintf(fp, fmt, ap);
	va_end(ap);
	fflush(fp);
#else
	(void)fmt;
#endif
	
	return reduxCleanup(ctx, ret);
}
