/* Includes */
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
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
 *                    Reduction Kernel Generator.
 *
 * The generator produces a kernel from one of two "code models":
 *   - Large
 *   - Small
 * Which one is used depends on the size of the destination tensor and the
 * number of reductions for each destination element. A destination tensor
 * with more than SMALL_REDUX_THRESHOLD elements or more elements than
 * reductions for each element will result in use of the large code model;
 * Otherwise the small code model is used.
 *
 *
 *                         LARGE CODE MODEL:
 *
 * In the large code model, each destination element is processed by a
 * single thread.
 *
 * Each thread begins with an initial value in a register, reads from all
 * source elements contributing to the reduction, computes the result and
 * writes it to the destination element.
 *
 * A single kernel is generated that performs prescalar transformations, the
 * reduction itself, postscalar transformations and the write to global memory.
 *
 *
 *                         SMALL CODE MODEL:
 *
 * In the small code model, each destination element is processed by
 * multiple threads.
 *
 * The destination tensor is first initialized with the initial value. Then,
 * one several threads cooperate to perform the reduction atomically on each
 * destination element. Lastly, postscalar transformations are applied
 * in-place.
 *
 * Two or three kernels are generated: The initialization kernel, the main
 * kernel that performs prescalar transformations and the reduction itself, and
 * possibly also a postscalar transformation kernel when it is required.
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
	GpuArray*       wsDst;
	GpuArray*       wsDstArg;
	int*            srcAxisList;
	size_t*         dstDims;
	gpucontext*     gpuCtx;

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
	int             ndd;
	int             ndr;
	int             nds;
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



/* Function prototypes */
static int   reduxGetSumInit               (int typecode, const char** property);
static int   reduxGetProdInit              (int typecode, const char** property);
static int   reduxGetMinInit               (int typecode, const char** property);
static int   reduxGetMaxInit               (int typecode, const char** property);
static int   reduxGetAndInit               (int typecode, const char** property);
static int   reduxGetOrInit                (int typecode, const char** property);
static int   axisInSet                     (int                v,
                                            const int*         set,
                                            size_t             setLen,
                                            size_t*            where);
static void  appendIdxes                   (strb*              s,
                                            const char*        prologue,
                                            const char*        prefix,
                                            int                startIdx,
                                            int                endIdx,
                                            const char*        suffix,
                                            const char*        epilogue);
static int   reduxCheckargs                (redux_ctx*  ctx);
static void  reduxSelectTypes              (redux_ctx*  ctx);
static int   reduxIsSmallCodeModel         (redux_ctx*  ctx);
static int   reduxIsLargeCodeModel         (redux_ctx*  ctx);
static int   reduxRequiresDst              (redux_ctx*  ctx);
static int   reduxRequiresDstArg           (redux_ctx*  ctx);
static int   reduxKernelRequiresDst        (redux_ctx*  ctx);
static int   reduxKernelRequiresDstArg     (redux_ctx*  ctx);
static int   reduxCanAppendHwAxis          (redux_ctx* ctx,
                                            int        kernelType,
                                            int        axisType);
static void  reduxAppendLargestAxisToHwList(redux_ctx* ctx,
                                            int        kernelType,
                                            int        axisType);
static int   reduxSelectHwAxes             (redux_ctx*  ctx);
static int   reduxComputeAxisList          (redux_ctx*  ctx);
static int   reduxGenSource                (redux_ctx*  ctx);
static void  reduxAppendSource             (redux_ctx*  ctx);
static void  reduxAppendIncludes           (redux_ctx*  ctx);
static void  reduxAppendTensorDeclArgs     (redux_ctx*  ctx,
                                            const char* type,
                                            const char* baseName);
static void  reduxAppendTensorCallArgs     (redux_ctx*  ctx,
                                            const char* baseName);
static void  reduxAppendMacroDefs          (redux_ctx*  ctx);
static void  reduxAppendTypedefs           (redux_ctx*  ctx);
static void  reduxAppendGetInitValFns      (redux_ctx*  ctx);
static void  reduxAppendWriteBackFn        (redux_ctx*  ctx);
static void  reduxAppendReduxKernel        (redux_ctx*  ctx);
static void  reduxAppendPrototype          (redux_ctx*  ctx);
static void  reduxAppendIndexDeclarations  (redux_ctx*  ctx);
static void  reduxAppendRangeCalculations  (redux_ctx*  ctx);
static void  reduxAppendLoops              (redux_ctx*  ctx);
static void  reduxAppendInitKernel         (redux_ctx*  ctx);
static void  reduxAppendPostKernel         (redux_ctx*  ctx);
static int   reduxCompile                  (redux_ctx*  ctx);
static int   reduxSchedule                 (redux_ctx*  ctx);
static void  reduxScheduleKernel           (int         ndims,
                                            uint64_t*   dims,
                                            uint64_t    warpSize,
                                            uint64_t    maxLg,
                                            uint64_t*   maxLs,
                                            uint64_t    maxGg,
                                            uint64_t*   maxGs,
                                            uint64_t*   bs,
                                            uint64_t*   gs,
                                            uint64_t*   cs);
static int   reduxInvoke                   (redux_ctx*  ctx);
static int   reduxCleanup                  (redux_ctx*  ctx, int ret);


/* Function implementation */
GPUARRAY_PUBLIC int  GpuArray_sum         (GpuArray*       dst,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_SUM,
	                          dst,  NULL,   src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_prod        (GpuArray*       dst,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_PROD,
	                          dst,  NULL,   src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_prodnz      (GpuArray*       dst,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_PRODNZ,
	                          dst,  NULL,   src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_min         (GpuArray*       dst,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_MIN,
	                          dst,  NULL,   src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_max         (GpuArray*       dst,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_MAX,
	                          dst,  NULL,   src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_argmin      (GpuArray*       dstArg,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_ARGMIN,
	                          NULL, dstArg, src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_argmax      (GpuArray*       dstArg,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_ARGMAX,
	                          NULL, dstArg, src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_minandargmin(GpuArray*       dst,
                                           GpuArray*       dstArg,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_MINANDARGMIN,
	                          dst,  dstArg, src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_maxandargmax(GpuArray*       dst,
                                           GpuArray*       dstArg,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_MAXANDARGMAX,
	                          dst,  dstArg, src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_and         (GpuArray*       dst,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_AND,
	                          dst,  NULL,   src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_or          (GpuArray*       dst,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_OR,
	                          dst,  NULL,   src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_xor         (GpuArray*       dst,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_XOR,
	                          dst,  NULL,   src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_all         (GpuArray*       dst,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_ALL,
	                          dst,  NULL,   src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_any         (GpuArray*       dst,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	return GpuArray_reduction(GA_REDUCE_ANY,
	                          dst,  NULL,   src, reduxLen, reduxList);
}
GPUARRAY_PUBLIC int  GpuArray_reduction   (ga_reduce_op    op,
                                           GpuArray*       dst,
                                           GpuArray*       dstArg,
                                           const GpuArray* src,
                                           unsigned        reduxLen,
                                           const unsigned* reduxList){
	redux_ctx  ctxSTACK = {op, dst, dstArg, src,
	                       (int)reduxLen, (const int*)reduxList};
	redux_ctx *ctx      = &ctxSTACK;

	return reduxCheckargs(ctx);
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

static int   reduxGetSumInit               (int typecode, const char** property){
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

static int   reduxGetProdInit              (int typecode, const char** property){
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

static int   reduxGetMinInit               (int typecode, const char** property){
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

static int   reduxGetMaxInit               (int typecode, const char** property){
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

static int   reduxGetAndInit               (int typecode, const char** property){
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

static int   reduxGetOrInit                (int typecode, const char** property){
	if (typecode == GA_POINTER ||
	    typecode == GA_BUFFER){
		return GA_UNSUPPORTED_ERROR;
	}
	*property = "0";
	return GA_NO_ERROR;
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

static int   axisInSet                     (int                v,
                                            const int*         set,
                                            size_t             setLen,
                                            size_t*            where){
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

static void  appendIdxes                   (strb*              s,
                                            const char*        prologue,
                                            const char*        prefix,
                                            int                startIdx,
                                            int                endIdx,
                                            const char*        suffix,
                                            const char*        epilogue){
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

/**
 * @brief Check the sanity of the arguments in agreement with the
 *        documentation for GpuArray_reduction().
 *
 *        Also initialize certain parts of the context, allocate memory
 *        buffers and fail out if at any point the environment gives us
 *        a problem.
 *
 * @return GA_INVALID_ERROR if arguments invalid; GA_NO_MEMORY if out of
 *         memory, GA_NO_ERROR otherwise.
 */

static int   reduxCheckargs                (redux_ctx*  ctx){
	int      i, j, ret, retT, retK;
	unsigned numProcs;
	size_t   localSize;
	size_t   dstNumElem = 1, reduxPerElem = 1;

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
	ctx->initValK       = NULL;
	ctx->pri.ndh       = ctx->aux.ndh  = 0;
	ctx->pri.ndhd      = ctx->aux.ndhd = 0;
	ctx->pri.ndhr      = ctx->aux.ndhr = 0;
	ctx->sourceCode    = NULL;
	ctx->sourceCodeLen = 0;
	ctx->errorString0  = NULL;
	ctx->errorString1  = NULL;
	ctx->errorString2  = NULL;
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
	/* *** IT IS NOW SAFE TO CALL reduxCleanup() *** */


	/* Insane src, reduxLen, dst or dstArg? */
	if (!ctx->src                                                      ||
	    (reduxRequiresDst   (ctx) && !ctx->dst)                        ||
	    (reduxRequiresDstArg(ctx) && !ctx->dstArg)                     ||
	    (ctx->src->nd  <= 0)                                           ||
	    (ctx->reduxLen <= 0)                                           ||
	    (ctx->src->nd  <  (unsigned)ctx->reduxLen)                     ||
	    (ctx->dst    && ctx->dst->nd   +ctx->reduxLen != ctx->src->nd) ||
	    (ctx->dstArg && ctx->dstArg->nd+ctx->reduxLen != ctx->src->nd) ){
		return reduxCleanup(ctx, GA_INVALID_ERROR);
	}


	/* Insane or duplicate list entry? */
	for (i=0;i<ctx->reduxLen;i++){
		if (ctx->reduxList[i] <  0                            ||
		    ctx->reduxList[i] >= (int)ctx->src->nd            ||
		    axisInSet(ctx->reduxList[i], ctx->reduxList, i, 0)){
			return reduxCleanup(ctx, GA_INVALID_ERROR);
		}
	}


	/* GPU context non-existent? */
	ctx->gpuCtx     = GpuArray_context(ctx->src);
	if (!ctx->gpuCtx){
		return reduxCleanup(ctx, GA_INVALID_ERROR);
	}


	/* Unknown type? */
	reduxSelectTypes(ctx);
	if (!ctx->srcTypeStr || !ctx->dstTypeStr || !ctx->dstArgTypeStr ||
	    !ctx->accTypeStr){
		return reduxCleanup(ctx, GA_INVALID_ERROR);
	}


	/* Determine initializer, and error out if reduction unsupported. */
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
		return reduxCleanup(ctx, retT);
	}
	if (retK != GA_NO_ERROR){
		return reduxCleanup(ctx, retK);
	}


	/**
	 * We initialize some more parts of the context, using the guarantees
	 * we now have about the sanity of the arguments.
	 */

	ctx->nds = ctx->src->nd;
	ctx->ndr = ctx->reduxLen;
	ctx->ndd = ctx->nds - ctx->ndr;
	strb_ensure(&ctx->s, 3*1024);


	/**
	 * And make a few small dynamic memory allocations for the benefit of the
	 * rest of the code, allowing error checking to happen early and fail fast.
	 */

	ctx->srcAxisList = malloc(ctx->nds * sizeof(unsigned));
	ctx->dstDims     = malloc(ctx->ndd * sizeof(size_t));
	if (!ctx->srcAxisList ||
	    !ctx->dstDims     ){
		return reduxCleanup(ctx, GA_MEMORY_ERROR);
	}


	/**
	 * Query device for approximate total level of parallelism. If destination
	 * tensor is so big it can keep all threads busy on individual elements,
	 * use large code model; Otherwise use small code model, where threads will
	 * have to cooperate.
	 *
	 *    - Large (Destination tensor >= SMALL_REDUX_THRESHOLD elements, or
	 *             destination tensor size >= # of reductions per destination
	 *             tensor element):
	 *        All destination elements have their own thread.
	 *    - Small (otherwise):
	 *        Multiple threads cooperate on a single destination element.
	 */

	ret = gpucontext_property(ctx->gpuCtx, GA_CTX_PROP_NUMPROCS, &numProcs);
	if (ret != GA_NO_ERROR){
		return reduxCleanup(ctx, ret);
	}
	ret = gpucontext_property(ctx->gpuCtx, GA_CTX_PROP_MAXLSIZE, &localSize);
	if (ret != GA_NO_ERROR){
		return reduxCleanup(ctx, ret);
	}

	for (i=j=0;i<ctx->nds;i++){
		if (axisInSet(i, ctx->reduxList, ctx->nds, NULL)){
			reduxPerElem *= ctx->src->dimensions[i];
		}else{
			dstNumElem   *= ctx->src->dimensions[i];
			ctx->dstDims[j++] = ctx->src->dimensions[i];;
		}
	}

	ctx->largeCodeModel = dstNumElem >= numProcs*localSize ||
	                      dstNumElem >= reduxPerElem
	                      || 1;/* BUG: Erase when small code model implemented. */
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
		if(ret != GA_NO_ERROR){
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
		if(ret != GA_NO_ERROR){
			return reduxCleanup(ctx, ret);
		}
	}else{
		ctx->wsDstArg = ctx->dstArg;
	}



	return reduxSelectHwAxes(ctx);
}

/**
 * @brief Select types for the reduction kernel's implementation.
 *
 * There are 5 types of relevance:
 *   - Source                   (S=Source)
 *   - Destination              (T=Target)
 *   - Destination Argument     (A=Arg)
 *   - Index                    (X=indeX)
 *   - Accumulator              (K=aKKumulator/reduction)
 */

static void  reduxSelectTypes              (redux_ctx*  ctx){
	/* Deal with the various typecodes. */
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

	/* Get the string version as well. */
	ctx->srcTypeStr     = gpuarray_get_type(ctx->srcTypeCode)   ->cluda_name;
	ctx->dstTypeStr     = gpuarray_get_type(ctx->dstTypeCode)   ->cluda_name;
	ctx->dstArgTypeStr  = gpuarray_get_type(ctx->dstArgTypeCode)->cluda_name;
	ctx->idxTypeStr     = gpuarray_get_type(ctx->idxTypeCode)   ->cluda_name;
	ctx->accTypeStr     = gpuarray_get_type(ctx->accTypeCode)   ->cluda_name;
}

/**
 * @brief Returns whether we are using the small code model or not.
 */

static int   reduxIsSmallCodeModel         (redux_ctx*  ctx){
	return !reduxIsLargeCodeModel(ctx);
}

/**
 * @brief Returns whether we are using the large code model or not.
 */

static int   reduxIsLargeCodeModel         (redux_ctx*  ctx){
	return ctx->largeCodeModel;
}

/**
 * @brief Returns whether the reduction interface requires a dst argument.
 */

static int   reduxRequiresDst              (redux_ctx*  ctx){
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

static int   reduxRequiresDstArg           (redux_ctx*  ctx){
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

static int   reduxKernelRequiresDst        (redux_ctx*  ctx){
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

static int   reduxKernelRequiresDstArg     (redux_ctx*  ctx){
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
 * @brief Check whether we can add another reduction axis or free axis
 *        to the hardware axis list for either the primary or secondary kernel.
 */

static int   reduxCanAppendHwAxis          (redux_ctx* ctx,
                                            int        kernelType,
                                            int        axisType){
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

static void  reduxAppendLargestAxisToHwList(redux_ctx* ctx,
                                            int        kernelType,
                                            int        axisType){
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

static int   reduxSelectHwAxes             (redux_ctx*  ctx){
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

	srcbAppends    (&ctx->srcGen, "#define FOROVER(idx)    for(i##idx = i##idx##Start; i##idx < i##idx##End; i##idx++)\n");
	srcbAppends    (&ctx->srcGen, "#define ESCAPE(idx)     if(i##idx >= i##idx##Dim){continue;}\n");

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
	if(reduxKernelRequiresDst(ctx)){
		reduxAppendTensorDeclArgs(ctx, "T", "dst");
	}
	if(reduxKernelRequiresDstArg(ctx)){
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
	if(reduxKernelRequiresDst(ctx)){
		for (i=0;i<ctx->ndd;i++){
			strb_appendf(&ctx->s, "\ti%dDStep   = dstSteps[%d];\n", i, i);
		}
	}
	if(reduxKernelRequiresDstArg(ctx)){
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
		                            "\t\t\tif(rdxK == k){\n"
		                            "\t\t\t\trdxA = rdxIdx;\n"
		                            "\t\t\t}\n");
		break;
		case GA_REDUCE_ARGMAX:
		case GA_REDUCE_MAXANDARGMAX:
		  srcbAppends(&ctx->srcGen, "\t\t\trdxK  = max(rdxK, k);\n"
		                            "\t\t\tif(rdxK == k){\n"
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
	int      i, priNdims, auxNdims;
	uint64_t maxLgRdx, maxLgPre, maxLgPost;
	uint64_t maxLgPri, maxLgAux;
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

static int   reduxCleanup                  (redux_ctx*  ctx, int ret){
	if (ctx->dst    != ctx->wsDst){
		GpuArray_clear(ctx->wsDst);
		free(ctx->wsDst);
		ctx->wsDst    = NULL;
	}
	if (ctx->dstArg != ctx->wsDstArg){
		GpuArray_clear(ctx->wsDstArg);
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
