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
#include "util/integerfactoring.h"


/* Defines */
#define  MAX_HW_DIMS                   3



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
 *     //Includes
 *     #include <limits.h>
 *     #include <math.h>
 *     #include <stdint.h>
 *     
 *     
 *     //Typedefs:
 *     typedef  float    T
 *     typedef  int64_t  X
 *     
 *     
 *     //Initializer (in case initial T cannot be expressed as a literal)
 *     static T    getInitVal(void){
 *         return ...
 *     }
 *     
 *     
 *     //Reduce into global memory destination a value.
 *     static void atomicRedux(GLOBAL_MEM T* dst, T val){
 *         ...
 *     }
 *     
 *     
 *     //Load data from source and apply pre-operations.
 *     static T loadVal(X i0, X i1, ..., X iN,
 *                      const GLOBAL_MEM T* src,
 *                      const GLOBAL_MEM X* srcSteps,
 *                      ...?){
 *         return ...
 *     }
 *     
 *     
 *     //Initialization kernel,
 *     KERNEL void preRedux(const GLOBAL_MEM X*        srcSize,
 *                          const GLOBAL_MEM X*        chunkSize,
 *                          GLOBAL_MEM T*              dst,
 *                          const X                    dstOff,
 *                          const GLOBAL_MEM X*        dstSteps){
 *         //OFFSETS
 *         dst += dstOff;
 *         
 *         //Initialize
 *         dst[...] = getInitVal();
 *     }
 *     
 *     
 *     //Reduction Kernel.
 *     KERNEL void redux(const GLOBAL_MEM T*        src,
 *                       const X                    srcOff,
 *                       const GLOBAL_MEM X*        srcSteps,
 *                       const GLOBAL_MEM X*        srcSize,
 *                       const GLOBAL_MEM X*        chunkSize,
 *                       GLOBAL_MEM T*              dst,
 *                       const X                    dstOff,
 *                       const GLOBAL_MEM X*        dstSteps,
 *                       GLOBAL_MEM X*              dstArg,
 *                       const X                    dstArgOff,
 *                       const GLOBAL_MEM X*        dstArgSteps){
 *         //OFFSETS
 *         src    += srcOff
 *         dst    += dstOff
 *         dstArg += dstArgOff
 *         
 *         //Declare Indices
 *         //Compute Ranges
 *         
 *         //Define macros
 *         //Outer Loops
 *            //Inner Loops
 *         //Undefine macros
 *     }
 *     
 *     
 *     //Post-scalar kernel,
 *     KERNEL void postRedux(const GLOBAL_MEM X*        srcSize,
 *                           const GLOBAL_MEM X*        chunkSize,
 *                           GLOBAL_MEM T*              dst,
 *                           const X                    dstOff,
 *                           const GLOBAL_MEM X*        dstSteps){
 *         //OFFSETS
 *         dst += dstOff;
 *         
 *         //Initialize
 *         dst[...] = getInitVal();
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
	int*            axisList;
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
	const char*     initVal;
	int             ndd;
	int             ndr;
	int             nds;
	int             ndh;
	int             ndhd;
	int             ndhr;
	int             largeCodeModel;
	strb            s;
	char*           sourceCode;
	GpuKernel       preKernel;
	GpuKernel       kernel;
	GpuKernel       postKernel;

	/* Scheduler */
	int             hwAxisList[MAX_HW_DIMS];
	size_t          blockSize [MAX_HW_DIMS];
	size_t          gridSize  [MAX_HW_DIMS];
	size_t          chunkSize [MAX_HW_DIMS];

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
static int   reduxSelectModel              (redux_ctx*  ctx);
static int   reduxIsSmallCodeModel         (redux_ctx*  ctx);
static int   reduxIsLargeCodeModel         (redux_ctx*  ctx);
static int   reduxHasDst                   (redux_ctx*  ctx);
static int   reduxHasDstArg                (redux_ctx*  ctx);
static int   reduxKernelRequiresDst        (redux_ctx*  ctx);
static int   reduxKernelRequiresDstArg     (redux_ctx*  ctx);
static int   reduxCanAppendHwAxis          (redux_ctx* ctx, int wantReductionAxis);
static void  reduxAppendLargestAxisToHwList(redux_ctx* ctx, int wantReductionAxis);
static int   reduxSelectHwAxes             (redux_ctx*  ctx);
static int   reduxComputeAxisList          (redux_ctx*  ctx);
static int   reduxGenSource                (redux_ctx*  ctx);
static void  reduxAppendSource             (redux_ctx*  ctx);
static void  reduxAppendIncludes           (redux_ctx*  ctx);
static void  reduxAppendTypedefs           (redux_ctx*  ctx);
static void  reduxAppendFuncGetInitVal     (redux_ctx*  ctx);
static void  reduxAppendFuncLoadVal        (redux_ctx*  ctx);
static void  reduxAppendFuncReduxVal       (redux_ctx*  ctx);
static void  reduxAppendFuncPreKernel      (redux_ctx*  ctx);
static void  reduxAppendFuncKernel         (redux_ctx*  ctx);
static void  reduxAppendFuncPostKernel     (redux_ctx*  ctx);
static void  reduxAppendPrototype          (redux_ctx*  ctx);
static void  reduxAppendOffsets            (redux_ctx*  ctx);
static void  reduxAppendIndexDeclarations  (redux_ctx*  ctx);
static void  reduxAppendRangeCalculations  (redux_ctx*  ctx);
static void  reduxAppendLoops              (redux_ctx*  ctx);
static void  reduxAppendLoopMacroDefs      (redux_ctx*  ctx);
static void  reduxAppendLoopOuter          (redux_ctx*  ctx);
static void  reduxAppendLoopInner          (redux_ctx*  ctx);
static void  reduxAppendLoopMacroUndefs    (redux_ctx*  ctx);
static int   reduxCompileLarge             (redux_ctx*  ctx);
static int   reduxCompileSmall             (redux_ctx*  ctx);
static int   reduxScheduleLarge            (redux_ctx*  ctx);
static int   reduxInvokeLarge              (redux_ctx*  ctx);
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
		case GA_BYTE:               *property = "SCHAR_MIN"; break;
		case GA_SHORT2:
		case GA_SHORT3:
		case GA_SHORT4:
		case GA_SHORT8:
		case GA_SHORT16:
		case GA_SHORT:              *property = "SHRT_MIN"; break;
		case GA_INT2:
		case GA_INT3:
		case GA_INT4:
		case GA_INT8:
		case GA_INT16:
		case GA_INT:                *property = "INT_MIN"; break;
		case GA_LONG2:
		case GA_LONG3:
		case GA_LONG4:
		case GA_LONG8:
		case GA_LONG16:
		case GA_LONG:               *property = "LONG_MIN"; break;
		case GA_LONGLONG:           *property = "LLONG_MIN"; break;
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
		case GA_SIZE:               *property = "0"; break;
		case GA_HALF:
		case GA_FLOAT:
		case GA_DOUBLE:
		case GA_QUAD:               *property = "NAN"; break;
		default:      return GA_UNSUPPORTED_ERROR;
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
		case GA_BOOL:               *property = "1"; break;
		case GA_BYTE2:
		case GA_BYTE3:
		case GA_BYTE4:
		case GA_BYTE8:
		case GA_BYTE16:
		case GA_BYTE:               *property = "SCHAR_MAX"; break;
		case GA_UBYTE2:
		case GA_UBYTE3:
		case GA_UBYTE4:
		case GA_UBYTE8:
		case GA_UBYTE16:
		case GA_UBYTE:              *property = "UCHAR_MAX"; break;
		case GA_SHORT2:
		case GA_SHORT3:
		case GA_SHORT4:
		case GA_SHORT8:
		case GA_SHORT16:
		case GA_SHORT:              *property = "SHRT_MAX"; break;
		case GA_USHORT2:
		case GA_USHORT3:
		case GA_USHORT4:
		case GA_USHORT8:
		case GA_USHORT16:
		case GA_USHORT:             *property = "USHRT_MAX"; break;
		case GA_INT2:
		case GA_INT3:
		case GA_INT4:
		case GA_INT8:
		case GA_INT16:
		case GA_INT:                *property = "INT_MAX"; break;
		case GA_UINT2:
		case GA_UINT3:
		case GA_UINT4:
		case GA_UINT8:
		case GA_UINT16:
		case GA_UINT:               *property = "UINT_MAX"; break;
		case GA_LONG2:
		case GA_LONG3:
		case GA_LONG4:
		case GA_LONG8:
		case GA_LONG16:
		case GA_LONG:               *property = "LONG_MAX"; break;
		case GA_ULONG2:
		case GA_ULONG3:
		case GA_ULONG4:
		case GA_ULONG8:
		case GA_ULONG16:
		case GA_ULONG:              *property = "ULONG_MAX"; break;
		case GA_LONGLONG:           *property = "LLONG_MAX"; break;
		case GA_ULONGLONG:          *property = "ULLONG_MAX"; break;
		case GA_HALF:
		case GA_FLOAT:
		case GA_DOUBLE:
		case GA_QUAD:               *property = "NAN"; break;
		default:      return GA_UNSUPPORTED_ERROR;
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
 * @brief Check the sanity of the arguments, in agreement with the
 *        documentation for GpuArray_reduction().
 *
 *        Also initialize certain parts of the context.
 *
 * @return GA_INVALID_ERROR if arguments invalid; GA_NO_ERROR otherwise.
 */

static int   reduxCheckargs                (redux_ctx*  ctx){
	int i, ret;

	/**
	 * We initialize certain parts of the context.
	 */

	ctx->axisList      = NULL;
	ctx->gpuCtx        = NULL;

	ctx->srcTypeStr    = ctx->dstTypeStr    = ctx->dstArgTypeStr =
	ctx->accTypeStr    = ctx->idxTypeStr    = NULL;
	ctx->initVal       = NULL;
	ctx->ndh           = 0;
	ctx->ndhd          = 0;
	ctx->ndhr          = 0;
	ctx->sourceCode    = NULL;
	strb_init(&ctx->s);

	for (i=0;i<MAX_HW_DIMS;i++){
		ctx->hwAxisList[i] = 0;
		ctx->blockSize [i] = 1;
		ctx->gridSize  [i] = 1;
		ctx->chunkSize [i] = 1;
	}

	ctx->srcStepsGD = ctx->srcSizeGD     = ctx->chunkSizeGD   =
	ctx->dstStepsGD = ctx->dstArgStepsGD = NULL;
	/* *** IT IS NOW SAFE TO CALL reduxCleanup() *** */


	/* Insane src, reduxLen, dst or dstArg? */
	if (!ctx->src || ctx->src->nd <= 0 || ctx->reduxLen == 0 ||
	    ctx->reduxLen > (int)ctx->src->nd){
		return reduxCleanup(ctx, GA_INVALID_ERROR);
	}
	if ((reduxHasDst   (ctx) && !ctx->dst)   ||
	    (reduxHasDstArg(ctx) && !ctx->dstArg)){
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
		case GA_REDUCE_SUM:  ret = reduxGetSumInit (ctx->accTypeCode, &ctx->initVal); break;
		case GA_REDUCE_PRODNZ:
		case GA_REDUCE_PROD: ret = reduxGetProdInit(ctx->accTypeCode, &ctx->initVal); break;
		case GA_REDUCE_MINANDARGMIN:
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_MIN:  ret = reduxGetMinInit (ctx->accTypeCode, &ctx->initVal); break;
		case GA_REDUCE_MAXANDARGMAX:
		case GA_REDUCE_ARGMAX:
		case GA_REDUCE_MAX:  ret = reduxGetMaxInit (ctx->accTypeCode, &ctx->initVal); break;
		case GA_REDUCE_ALL:
		case GA_REDUCE_AND:  ret = reduxGetAndInit (ctx->accTypeCode, &ctx->initVal); break;
		case GA_REDUCE_ANY:
		case GA_REDUCE_XOR:
		case GA_REDUCE_OR:   ret = reduxGetOrInit  (ctx->accTypeCode, &ctx->initVal); break;
		default:             ret = GA_UNSUPPORTED_ERROR; break;
	}
	if (ret != GA_NO_ERROR){
		return reduxCleanup(ctx, ret);
	}


	/**
	 * We initialize some more parts of the context, using the guarantees
	 * we now have about the sanity of the arguments.
	 */

	ctx->nds = ctx->src->nd;
	ctx->ndr = ctx->reduxLen;
	ctx->ndd = ctx->nds - ctx->ndr;
	strb_ensure(&ctx->s, 5*1024);



	return reduxSelectModel(ctx);
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
		case GA_HALF:   ctx->accTypeCode = GA_FLOAT;
		case GA_HALF2:  ctx->accTypeCode = GA_FLOAT2;
		case GA_HALF4:  ctx->accTypeCode = GA_FLOAT4;
		case GA_HALF8:  ctx->accTypeCode = GA_FLOAT8;
		case GA_HALF16: ctx->accTypeCode = GA_FLOAT16;
		default:        ctx->accTypeCode = ctx->srcTypeCode;
	}
	
	/* Get the string version as well. */
	ctx->srcTypeStr     = gpuarray_get_type(ctx->srcTypeCode)   ->cluda_name;
	ctx->dstTypeStr     = gpuarray_get_type(ctx->dstTypeCode)   ->cluda_name;
	ctx->dstArgTypeStr  = gpuarray_get_type(ctx->dstArgTypeCode)->cluda_name;
	ctx->idxTypeStr     = gpuarray_get_type(ctx->idxTypeCode)   ->cluda_name;
	ctx->accTypeStr     = gpuarray_get_type(ctx->accTypeCode)   ->cluda_name;
}

/**
 * @brief Select which code model will be used:
 * 
 *        - Large (Destination tensor >= SMALL_REDUX_THRESHOLD elements, or
 *                 destination tensor size >= # of reductions per destination
 *                 tensor element):
 *            All destination elements have their own thread.
 *        - Small (otherwise):
 *            Multiple threads cooperate on a single destination element.
 */

static int   reduxSelectModel              (redux_ctx*  ctx){
	int      i, ret;
	unsigned numProcs;
	size_t   localSize;
	size_t   dstNumElem = 1, reduxPerElem = 1;


	/**
	 * Query device for approximate total level of parallelism. If destination
	 * tensor is so big it can keep all threads busy on individual elements,
	 * use large code model; Otherwise use small code model, where threads will
	 * have to cooperate.
	 */
	
	ret = gpucontext_property(ctx->gpuCtx, GA_CTX_PROP_NUMPROCS, &numProcs);
	if (ret != GA_NO_ERROR){
		return reduxCleanup(ctx, ret);
	}
	ret = gpucontext_property(ctx->gpuCtx, GA_CTX_PROP_MAXLSIZE, &localSize);
	if (ret != GA_NO_ERROR){
		return reduxCleanup(ctx, ret);
	}


	/**
	 * Compute #elems in dst and # reductions per dst element.
	 */

	for (i=0;i<ctx->nds;i++){
		if (axisInSet(i, ctx->reduxList, ctx->nds, NULL)){
			reduxPerElem *= ctx->src->dimensions[i];
		}else{
			dstNumElem   *= ctx->src->dimensions[i];
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
	
	
	return reduxSelectHwAxes(ctx);
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

static int   reduxHasDst                   (redux_ctx*  ctx){
	switch (ctx->op){
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_ARGMAX:       return 0;
		default:                     return 1;
	}
}

/**
 * @brief Returns whether the reduction interface requires a dstArg argument.
 */

static int   reduxHasDstArg                (redux_ctx*  ctx){
	switch (ctx->op){
		case GA_REDUCE_MINANDARGMIN:
		case GA_REDUCE_MAXANDARGMAX:
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_ARGMAX:       return 1;
		default:                     return 0;
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
		case GA_REDUCE_ARGMAX:       return reduxIsSmallCodeModel(ctx);
		default:                     return 1;
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
	
	return reduxHasDstArg(ctx);
}

/**
 * @brief Check whether we can add another reduction axis
 *        (wantReductionAxis=1) or destination axis (wantReductionAxis=0) to
 *        the hardware axis list.
 */

static int   reduxCanAppendHwAxis          (redux_ctx* ctx, int wantReductionAxis){
	if (ctx->ndh >= MAX_HW_DIMS){
		return 0;
	}else{
		return wantReductionAxis ? ctx->ndhr < ctx->ndr:
		                           ctx->ndhd < ctx->ndd;
	}
}

/**
 * @brief Append the largest reduction axis (wantReductionAxis=1) or
 *        destination axis (wantReductionAxis=0) that isn't yet in the hardware
 *        axis list into said hardware axis list.
 */

static void  reduxAppendLargestAxisToHwList(redux_ctx* ctx, int wantReductionAxis){
	int    maxI = 0, i, isInHwList, isInReduxList, isInDesiredList, isLargestSoFar;
	size_t maxV = 0;
	
	/* Find */
	for (i=0;i<ctx->nds;i++){
		isInHwList      = axisInSet(i, ctx->hwAxisList, ctx->ndh, 0);
		isInReduxList   = axisInSet(i, ctx->reduxList,  ctx->ndr, 0);
		isInDesiredList = wantReductionAxis ? isInReduxList : !isInReduxList;
		isLargestSoFar  = ctx->src->dimensions[i] >= maxV;
		
		if (!isInHwList && isInDesiredList && isLargestSoFar){
			maxV = ctx->src->dimensions[i];
			maxI = i;
		}
	}
	
	/* Append */
	ctx->hwAxisList[ctx->ndh++] = maxI;
	if (wantReductionAxis){
		ctx->ndhr++;
	}else{
		ctx->ndhd++;
	}
}

/**
 * @brief Select which axes (up to MAX_HW_DIMS) will be assigned to hardware
 *        dimensions.
 * 
 * For the "large" code model: The up-to-MAX_HW_DIMS largest destination tensor
 *                             dimensions are selected.
 * For the "small" code model: Up to MAX_HW_DIMS reduction dimensions (largest-
 *                             to-smallest) are selected. If less than
 *                             MAX_HW_DIMS dimensions were selected,
 *                             destination tensor dimensions are selected until
 *                             MAX_HW_DIMS total dimensions are selected, or no
 *                             destination tensors are left.
 */

static int   reduxSelectHwAxes             (redux_ctx*  ctx){
	if (reduxIsSmallCodeModel(ctx)){
		while(reduxCanAppendHwAxis(ctx, 1)){
			reduxAppendLargestAxisToHwList(ctx, 1);
		}
	}
	
	while(reduxCanAppendHwAxis(ctx, 0)){
		reduxAppendLargestAxisToHwList(ctx, 0);
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
 * 
 * @return GA_MEMORY_ERROR if allocating the list failed; Otherwise, returns
 *         GA_NO_ERROR.
 */

static int   reduxComputeAxisList          (redux_ctx*  ctx){
	int i, f=0;
	
	ctx->axisList = malloc(ctx->nds * sizeof(unsigned));
	if (!ctx->axisList){
		return reduxCleanup(ctx, GA_MEMORY_ERROR);
	}

	for (i=0;i<ctx->nds;i++){
		if (!axisInSet(i, ctx->reduxList, ctx->ndr, 0)){
			ctx->axisList[f++] = i;
		}
	}
	memcpy(&ctx->axisList[f], ctx->reduxList, ctx->ndr * sizeof(*ctx->reduxList));
	
	
	return reduxGenSource(ctx);
}

/**
 * @brief Generate the kernel code for the reduction.
 *
 * @return GA_MEMORY_ERROR if not enough memory left; GA_NO_ERROR otherwise.
 */

static int   reduxGenSource                (redux_ctx*  ctx){
	reduxAppendSource(ctx);
	ctx->sourceCode = strb_cstr(&ctx->s);
	if (!ctx->sourceCode){
		return reduxCleanup(ctx, GA_MEMORY_ERROR);
	}
	
	return reduxIsLargeCodeModel(ctx) ? reduxCompileLarge(ctx):
	                                    reduxCompileSmall(ctx);
}
static void  reduxAppendSource             (redux_ctx*  ctx){
	reduxAppendIncludes         (ctx);
	reduxAppendTypedefs         (ctx);
	reduxAppendFuncGetInitVal   (ctx);
	reduxAppendFuncLoadVal      (ctx);
	reduxAppendFuncReduxVal     (ctx);
	reduxAppendFuncPreKernel    (ctx);
	reduxAppendFuncKernel       (ctx);
	reduxAppendFuncPostKernel   (ctx);
}
static void  reduxAppendIncludes           (redux_ctx*  ctx){
	strb_appends(&ctx->s, "/* Includes */\n");
	strb_appends(&ctx->s, "#include \"cluda.h\"\n");
	strb_appends(&ctx->s, "\n");
	strb_appends(&ctx->s, "\n");
	strb_appends(&ctx->s, "\n");
}
static void  reduxAppendTypedefs           (redux_ctx*  ctx){
	strb_appends(&ctx->s, "/* Typedefs */\n");
	strb_appendf(&ctx->s, "typedef %s     S;/* The type of the source array. */\n",                ctx->srcTypeStr);
	strb_appendf(&ctx->s, "typedef %s     T;/* The type of the destination array. */\n",           ctx->dstTypeStr);
	strb_appendf(&ctx->s, "typedef %s     A;/* The type of the destination argument array. */\n",  ctx->dstArgTypeStr);
	strb_appendf(&ctx->s, "typedef %s     X;/* The type of the indices: signed 32/64-bit. */\n",   ctx->idxTypeStr);
	strb_appendf(&ctx->s, "typedef %s     K;/* The type of the accumulator variable. */\n",        ctx->accTypeStr);
	strb_appends(&ctx->s, "\n\n\n");
}
static void  reduxAppendFuncGetInitVal     (redux_ctx*  ctx){
	strb_appends(&ctx->s, "/**\n");
	strb_appends(&ctx->s, " * Initial value function.\n");
	strb_appends(&ctx->s, " */\n\n");
	strb_appends(&ctx->s, "WITHIN_KERNEL K    getInitVal(void){\n");
	strb_appendf(&ctx->s, "\treturn (%s);\n", ctx->initVal);
	strb_appends(&ctx->s, "}\n\n\n\n");
}
static void  reduxAppendFuncLoadVal        (redux_ctx*  ctx){
	int i;
	
	strb_appends(&ctx->s, "/**\n");
	strb_appends(&ctx->s, " * Multidimensional source element loader.\n");
	strb_appends(&ctx->s, " *\n");
	strb_appends(&ctx->s, " * Also implements prescalar transformations if any.\n");
	strb_appends(&ctx->s, " */\n");
	strb_appends(&ctx->s, "\n");
	appendIdxes (&ctx->s, "WITHIN_KERNEL K    loadVal(", "X i", 0, ctx->nds, "", "");
	if (ctx->nds > 0){
		strb_appends(&ctx->s, ", ");
	}
	strb_appends(&ctx->s, "const GLOBAL_MEM S* src, const GLOBAL_MEM X* srcSteps){\n");
	strb_appends(&ctx->s, "\tS v = (*(const GLOBAL_MEM S*)((const GLOBAL_MEM char*)src + ");
	for (i=0;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "i%d*srcSteps[%d] + \\\n\t                                                            ", i, ctx->axisList[i]);
	}
	strb_appends(&ctx->s, "0));\n");
	
	/* Prescalar transformations go here... */
	
	/* Return the value. */
	strb_appends(&ctx->s, "\treturn v;\n");
	strb_appends(&ctx->s, "}\n\n\n\n");
}
static void  reduxAppendFuncReduxVal       (redux_ctx*  ctx){
	int i, anyArgsEmitted = 0;
	
	/* Function Signature. */
	strb_appends(&ctx->s, "/**\n");
	strb_appends(&ctx->s, " * Global memory value reduction function.\n");
	strb_appends(&ctx->s, " *\n");
	strb_appends(&ctx->s, " * Responsible for either:\n");
	strb_appends(&ctx->s, " *   1) Safe writeback of final value to memory, or\n");
	strb_appends(&ctx->s, " *   2) Safe atomic reduction of partial value into memory.\n");
	strb_appends(&ctx->s, " */\n");
	strb_appends(&ctx->s, "\n");
	appendIdxes (&ctx->s, "WITHIN_KERNEL void reduxVal(", "X i", 0, ctx->ndd, "", "");
	anyArgsEmitted = ctx->ndd>0;
	if (reduxKernelRequiresDst   (ctx)){
		if (anyArgsEmitted){
			strb_appends(&ctx->s, ", ");
		}
		anyArgsEmitted = 1;
		strb_appends(&ctx->s, "GLOBAL_MEM T* dst,    const GLOBAL_MEM X* dstSteps,    K v");
	}
	if (reduxKernelRequiresDstArg(ctx)){
		if (anyArgsEmitted){
			strb_appends(&ctx->s, ", ");
		}
		anyArgsEmitted = 1;
		strb_appends(&ctx->s, "GLOBAL_MEM A* dstArg, const GLOBAL_MEM X* dstArgSteps, X i");
	}
	strb_appends(&ctx->s, "){\n");
	
	
	/* Post-scalar transformations go here. */
	
	
	/* Write to memory. */
	if (reduxIsLargeCodeModel(ctx)){
		/* Large code model. Easy: just write out the data, since it's safe. */
		if (reduxKernelRequiresDst   (ctx)){
			strb_appends(&ctx->s, "\t(*(GLOBAL_MEM T*)((GLOBAL_MEM char*)dst + ");
			for (i=0;i<ctx->ndd;i++){
				strb_appendf(&ctx->s, "i%d*dstSteps[%d] +\n\t                                          ", i, i);
			}
			strb_appends(&ctx->s, "0)) = v;\n");
		}
		if (reduxKernelRequiresDstArg(ctx)){
			strb_appends(&ctx->s, "\t(*(GLOBAL_MEM A*)((GLOBAL_MEM char*)dstArg + ");
			for (i=0;i<ctx->ndd;i++){
				strb_appendf(&ctx->s, "i%d*dstArgSteps[%d] +\n\t                                             ", i, i);
			}
			strb_appends(&ctx->s, "0)) = i;\n");
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
static void  reduxAppendFuncPreKernel      (redux_ctx*  ctx){
	
}
static void  reduxAppendFuncKernel         (redux_ctx*  ctx){
	reduxAppendPrototype        (ctx);
	strb_appends                (&ctx->s, "{\n");
	reduxAppendOffsets          (ctx);
	reduxAppendIndexDeclarations(ctx);
	reduxAppendRangeCalculations(ctx);
	reduxAppendLoops            (ctx);
	strb_appends                (&ctx->s, "}\n");
}
static void  reduxAppendFuncPostKernel     (redux_ctx*  ctx){
	
}
static void  reduxAppendPrototype          (redux_ctx*  ctx){
	strb_appends(&ctx->s, "/**\n");
	strb_appends(&ctx->s, " * Reduction Kernel.\n");
	strb_appends(&ctx->s, " *\n");
	strb_appends(&ctx->s, " * Implements actual reduction operation.\n");
	strb_appends(&ctx->s, " */\n\n");
	strb_appends(&ctx->s, "KERNEL void redux(const GLOBAL_MEM S*        src,\n");
	strb_appends(&ctx->s, "                  const X                    srcOff,\n");
	strb_appends(&ctx->s, "                  const GLOBAL_MEM X*        srcSteps,\n");
	strb_appends(&ctx->s, "                  const GLOBAL_MEM X*        srcSize,\n");
	strb_appends(&ctx->s, "                  const GLOBAL_MEM X*        chunkSize,\n");
	strb_appends(&ctx->s, "                  GLOBAL_MEM T*              dst,\n");
	strb_appends(&ctx->s, "                  const X                    dstOff,\n");
	strb_appends(&ctx->s, "                  const GLOBAL_MEM X*        dstSteps,\n");
	strb_appends(&ctx->s, "                  GLOBAL_MEM A*              dstArg,\n");
	strb_appends(&ctx->s, "                  const X                    dstArgOff,\n");
	strb_appends(&ctx->s, "                  const GLOBAL_MEM X*        dstArgSteps)");
}
static void  reduxAppendOffsets            (redux_ctx*  ctx){
	strb_appends(&ctx->s, "\t/* Add offsets */\n");
	strb_appends(&ctx->s, "\tsrc    = (const GLOBAL_MEM T*)((const GLOBAL_MEM char*)src    + srcOff);\n");
	if (reduxKernelRequiresDst(ctx)){
		strb_appends(&ctx->s, "\tdst    = (GLOBAL_MEM T*)      ((GLOBAL_MEM char*)      dst    + dstOff);\n");
	}
	if (reduxKernelRequiresDstArg(ctx)){
		strb_appends(&ctx->s, "\tdstArg = (GLOBAL_MEM X*)      ((GLOBAL_MEM char*)      dstArg + dstArgOff);\n");
	}
	strb_appends(&ctx->s, "\t\n\t\n");
}
static void  reduxAppendIndexDeclarations  (redux_ctx*  ctx){
	int i;
	strb_appends(&ctx->s, "\t/* GPU kernel coordinates. Always 3D in OpenCL/CUDA. */\n");

	strb_appends(&ctx->s, "\tX bi0 = GID_0,        bi1 = GID_1,        bi2 = GID_2;\n");
	strb_appends(&ctx->s, "\tX bd0 = LDIM_0,       bd1 = LDIM_1,       bd2 = LDIM_2;\n");
	strb_appends(&ctx->s, "\tX ti0 = LID_0,        ti1 = LID_1,        ti2 = LID_2;\n");
	strb_appends(&ctx->s, "\tX gi0 = bi0*bd0+ti0,  gi1 = bi1*bd1+ti1,  gi2 = bi2*bd2+ti2;\n");
	if (ctx->ndh>0){
		strb_appends(&ctx->s, "\tX ");
		for (i=0;i<ctx->ndh;i++){
			strb_appendf(&ctx->s, "ci%u = chunkSize[%u]%s",
			             i, i, (i==ctx->ndh-1) ? ";\n" : ", ");
		}
	}

	strb_appends(&ctx->s, "\t\n\t\n");
	strb_appends(&ctx->s, "\t/* Free indices & Reduction indices */\n");

	if (ctx->nds >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->nds, "",        ";\n");}
	if (ctx->nds >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->nds, "Dim",     ";\n");}
	if (ctx->nds >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->nds, "Start",   ";\n");}
	if (ctx->nds >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->nds, "End",     ";\n");}
	if (ctx->nds >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->nds, "SStep",   ";\n");}
	if (ctx->ndd >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->ndd, "MStep",   ";\n");}
	if (ctx->ndd >        0){appendIdxes (&ctx->s, "\tX ", "i", 0,        ctx->ndd, "AStep",   ";\n");}
	if (ctx->nds > ctx->ndd){appendIdxes (&ctx->s, "\tX ", "i", ctx->ndd, ctx->nds, "PDim",    ";\n");}

	strb_appends(&ctx->s, "\t\n\t\n");
}
static void  reduxAppendRangeCalculations  (redux_ctx*  ctx){
	size_t hwDim;
	int    i;

	/* Use internal remapping when computing the ranges for this thread. */
	strb_appends(&ctx->s, "\t/* Compute ranges for this thread. */\n");

	for (i=0;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "\ti%dDim     = srcSize[%d];\n", i, ctx->axisList[i]);
	}
	for (i=0;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "\ti%dSStep   = srcSteps[%d];\n", i, ctx->axisList[i]);
	}
	for (i=0;i<ctx->ndd;i++){
		strb_appendf(&ctx->s, "\ti%dMStep   = dstSteps[%d];\n", i, i);
	}
	for (i=0;i<ctx->ndd;i++){
		strb_appendf(&ctx->s, "\ti%dAStep   = dstArgSteps[%d];\n", i, i);
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

		if (axisInSet(ctx->axisList[i], ctx->hwAxisList, ctx->ndh, &hwDim)){
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

		if (axisInSet(ctx->axisList[i], ctx->hwAxisList, ctx->ndh, &hwDim)){
			strb_appendf(&ctx->s, "\ti%dEnd     = i%dStart + ci%d;\n", i, i, hwDim);
		}else{
			strb_appendf(&ctx->s, "\ti%dEnd     = i%dStart + i%dDim;\n", i, i, i);
		}
	}

	strb_appends(&ctx->s, "\t\n\t\n");
}
static void  reduxAppendLoops              (redux_ctx*  ctx){
	strb_appends(&ctx->s, "\t/**\n");
	strb_appends(&ctx->s, "\t * FREE LOOPS.\n");
	strb_appends(&ctx->s, "\t */\n");
	strb_appends(&ctx->s, "\t\n");

	reduxAppendLoopMacroDefs  (ctx);
	reduxAppendLoopOuter      (ctx);
	reduxAppendLoopMacroUndefs(ctx);
}
static void  reduxAppendLoopMacroDefs      (redux_ctx*  ctx){
	int i;

	/**
	 * FOROVER Macro
	 */

	strb_appends(&ctx->s, "#define FOROVER(idx)    for(i##idx = i##idx##Start; i##idx < i##idx##End; i##idx++)\n");

	/**
	 * ESCAPE Macro
	 */

	strb_appends(&ctx->s, "#define ESCAPE(idx)     if(i##idx >= i##idx##Dim){continue;}\n");

	/**
	 * RDXINDEXER Macro
	 */

	appendIdxes (&ctx->s, "#define RDXINDEXER(", "i", ctx->ndd, ctx->nds, "", ")              (");
	for (i=ctx->ndd;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "i%d*i%dPDim + \\\n                                        ", i, i);
	}
	strb_appends(&ctx->s, "0)\n");
}
static void  reduxAppendLoopOuter          (redux_ctx*  ctx){
	int i;

	/**
	 * Outer Loop Header Generation
	 */

	for (i=0;i<ctx->ndd;i++){
		strb_appendf(&ctx->s, "\tFOROVER(%d){ESCAPE(%d)\n", i, i);
	}

	/**
	 * Inner Loop Generation
	 */

	reduxAppendLoopInner(ctx);

	/**
	 * Outer Loop Trailer Generation
	 */

	for (i=0;i<ctx->ndd;i++){
		strb_appends(&ctx->s, "\t}\n");
	}
}
static void  reduxAppendLoopInner          (redux_ctx*  ctx){
	int i;

	/**
	 * Inner Loop Prologue
	 */

	strb_appends(&ctx->s, "\t\t/**\n");
	strb_appends(&ctx->s, "\t\t * Reduction initialization.\n");
	strb_appends(&ctx->s, "\t\t */\n");
	strb_appends(&ctx->s, "\t\t\n");
	strb_appends(&ctx->s, "\t\tK rdxV = getInitVal();\n");
	if (reduxKernelRequiresDstArg(ctx)){
		strb_appends(&ctx->s, "\t\tX argI = 0;\n");
	}
	strb_appends(&ctx->s, "\t\t\n");
	strb_appends(&ctx->s, "\t\t/**\n");
	strb_appends(&ctx->s, "\t\t * REDUCTION LOOPS.\n");
	strb_appends(&ctx->s, "\t\t */\n");
	strb_appends(&ctx->s, "\t\t\n");

	/**
	 * Inner Loop Header Generation
	 */

	for (i=ctx->ndd;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "\t\tFOROVER(%d){ESCAPE(%d)\n", i, i);
	}

	/**
	 * Inner Loop Body Generation
	 */

	appendIdxes (&ctx->s, "\t\t\tK v = loadVal(", "i", 0, ctx->nds, "", "");
	if (ctx->nds > 0){
		strb_appends(&ctx->s, ", ");
	}
	strb_appends(&ctx->s, "src, srcSteps);\n");
	strb_appends(&ctx->s, "\t\t\t\n");
	switch (ctx->op){
		case GA_REDUCE_SUM:          strb_appends(&ctx->s, "\t\t\trdxV += v;\n"); break;
		case GA_REDUCE_PROD:         strb_appends(&ctx->s, "\t\t\trdxV *= v;\n"); break;
		case GA_REDUCE_PRODNZ:       strb_appends(&ctx->s, "\t\t\trdxV *= v==0 ? getInitVal() : v;\n"); break;
		case GA_REDUCE_MIN:          strb_appends(&ctx->s, "\t\t\trdxV  = min(rdxV, v);\n"); break;
		case GA_REDUCE_MAX:          strb_appends(&ctx->s, "\t\t\trdxV  = max(rdxV, v);\n"); break;
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_MINANDARGMIN:
			strb_appends(&ctx->s, "\t\t\trdxV  = min(rdxV, v);\n");
			strb_appends(&ctx->s, "\t\t\tif(v == rdxV){\n");
			appendIdxes (&ctx->s, "\t\t\t\targI = RDXINDEXER(", "i", ctx->ndd, ctx->nds, "", ");\n");
			strb_appends(&ctx->s, "\t\t\t}\n");
		break;
		case GA_REDUCE_ARGMAX:
		case GA_REDUCE_MAXANDARGMAX:
			strb_appends(&ctx->s, "\t\t\trdxV  = max(rdxV, v);\n");
			strb_appends(&ctx->s, "\t\t\tif(v == rdxV){\n");
			appendIdxes (&ctx->s, "\t\t\t\targI = RDXINDEXER(", "i", ctx->ndd, ctx->nds, "", ");\n");
			strb_appends(&ctx->s, "\t\t\t}\n");
		break;
		case GA_REDUCE_AND:          strb_appends(&ctx->s, "\t\t\trdxV &= v;\n"); break;
		case GA_REDUCE_OR:           strb_appends(&ctx->s, "\t\t\trdxV |= v;\n"); break;
		case GA_REDUCE_XOR:          strb_appends(&ctx->s, "\t\t\trdxV ^= v;\n"); break;
		case GA_REDUCE_ALL:          strb_appends(&ctx->s, "\t\t\trdxV  = rdxV && v;\n"); break;
		case GA_REDUCE_ANY:          strb_appends(&ctx->s, "\t\t\trdxV  = rdxV || v;\n"); break;
	}

	/**
	 * Inner Loop Trailer Generation
	 */

	for (i=ctx->ndd;i<ctx->nds;i++){
		strb_appends(&ctx->s, "\t\t}\n");
	}
	strb_appends(&ctx->s, "\t\t\n");

	/**
	 * Inner Loop Epilogue Generation
	 */

	strb_appends(&ctx->s, "\t\t/**\n");
	strb_appends(&ctx->s, "\t\t * Destination writeback.\n");
	strb_appends(&ctx->s, "\t\t */\n");
	strb_appends(&ctx->s, "\t\t\n");
	if       ( reduxKernelRequiresDst   (ctx) && !reduxKernelRequiresDstArg(ctx)){
		appendIdxes (&ctx->s, "\t\treduxVal(", "i", 0, ctx->ndd, "", "");
		if (ctx->ndd > 0){
			strb_appends(&ctx->s, ", ");
		}
		strb_appends(&ctx->s, "dst, dstSteps, rdxV);\n");
	}else if (!reduxKernelRequiresDst   (ctx) &&  reduxKernelRequiresDstArg(ctx)){
		appendIdxes (&ctx->s, "\t\treduxVal(", "i", 0, ctx->ndd, "", "");
		if (ctx->ndd > 0){
			strb_appends(&ctx->s, ", ");
		}
		strb_appends(&ctx->s, "dstArg, dstArgSteps, argI);\n");
	}else if ( reduxKernelRequiresDst   (ctx) &&  reduxKernelRequiresDstArg(ctx)){
		appendIdxes (&ctx->s, "\t\treduxVal(", "i", 0, ctx->ndd, "", "");
		if (ctx->ndd > 0){
			strb_appends(&ctx->s, ", ");
		}
		strb_appends(&ctx->s, "dst, dstSteps, rdxV, dstArg, dstArgSteps, argI);\n");
	}
}
static void  reduxAppendLoopMacroUndefs    (redux_ctx*  ctx){
	strb_appends(&ctx->s, "#undef FOROVER\n");
	strb_appends(&ctx->s, "#undef ESCAPE\n");
	strb_appends(&ctx->s, "#undef RDXINDEXER\n");
}

/**
 * @brief Compile the kernel from source code.
 *
 * @return
 */

static int   reduxCompileLarge             (redux_ctx*  ctx){
	const int    ARG_TYPECODES[]   = {
	    GA_BUFFER, /* src */
	    GA_SIZE,   /* srcOff */
	    GA_BUFFER, /* srcSteps */
	    GA_BUFFER, /* srcSize */
	    GA_BUFFER, /* chnkSize */
	    GA_BUFFER, /* dst */
	    GA_SIZE,   /* dstOff */
	    GA_BUFFER, /* dstSteps */
	    GA_BUFFER, /* dstArg */
	    GA_SIZE,   /* dstArgOff */
	    GA_BUFFER  /* dstArgSteps */
	};
	const size_t ARG_TYPECODES_LEN = sizeof(ARG_TYPECODES)/sizeof(*ARG_TYPECODES);
	const char*  SRCS[1]           = {ctx->sourceCode};
	const size_t SRC_LENS[1]       = {strlen(ctx->sourceCode)};
	const size_t SRCS_LEN          = sizeof(SRCS)/sizeof(*SRCS);

	int ret  = GpuKernel_init(&ctx->kernel,
	                          ctx->gpuCtx,
	                          SRCS_LEN,
	                          SRCS,
	                          SRC_LENS,
	                          "redux",
	                          ARG_TYPECODES_LEN,
	                          ARG_TYPECODES,
	                          0,
	                          (char**)0);

	if (ret != GA_NO_ERROR){
		return reduxCleanup(ctx, ret);
	}else{
		return reduxScheduleLarge(ctx);
	}
}
static int   reduxCompileSmall             (redux_ctx*  ctx){
	/* BUG: Implement small code model. */
	return reduxCompileLarge(ctx);
}

/**
 * Compute a good thread block size / grid size / software chunk size for Nvidia.
 */

static int   reduxScheduleLarge            (redux_ctx*  ctx){
	int            i;
	size_t         warpMod;
	size_t         bestWarpMod  = 1;
	unsigned       bestWarpAxis = 0;
	uint64_t       maxLg;
	uint64_t       maxLs[MAX_HW_DIMS];
	uint64_t       maxGg;
	uint64_t       maxGs [MAX_HW_DIMS];
	uint64_t       dims  [MAX_HW_DIMS];
	double         slack [MAX_HW_DIMS];
	ga_factor_list factBS[MAX_HW_DIMS];
	ga_factor_list factGS[MAX_HW_DIMS];
	ga_factor_list factCS[MAX_HW_DIMS];


	/**
	 * Obtain the constraints of our problem.
	 */

	size_t warpSize,
	       maxL, maxL0, maxL1, maxL2,  /* Maximum total and per-dimension thread/block sizes */
	       maxG, maxG0, maxG1, maxG2;  /* Maximum total and per-dimension block /grid  sizes */
	gpukernel_property(ctx->kernel.k,  GA_KERNEL_PROP_PREFLSIZE, &warpSize);
	gpukernel_property(ctx->kernel.k,  GA_KERNEL_PROP_MAXLSIZE,  &maxL);
	gpudata_property  (ctx->src->data, GA_CTX_PROP_MAXLSIZE0,    &maxL0);
	gpudata_property  (ctx->src->data, GA_CTX_PROP_MAXLSIZE1,    &maxL1);
	gpudata_property  (ctx->src->data, GA_CTX_PROP_MAXLSIZE2,    &maxL2);
	gpudata_property  (ctx->src->data, GA_CTX_PROP_MAXGSIZE0,    &maxG0);
	maxG = maxG0;
	gpudata_property  (ctx->src->data, GA_CTX_PROP_MAXGSIZE1,    &maxG1);
	gpudata_property  (ctx->src->data, GA_CTX_PROP_MAXGSIZE2,    &maxG2);

	/**
	 * Prepare inputs to the solver.
	 *
	 * This involves, amongst others,
	 * - Initializing the blockSize, gridSize and chunkSize factor lists for all
	 *   hardware dimensions.
	 * - Finding on which hardware axis is it optimal to place the warpSize factor.
	 */

	maxLg    = maxL;
	maxLs[0] = maxL0, maxLs[1]=maxL1, maxLs[2]=maxL2;
	maxGg    = maxG;
	maxGs[0] = maxG0, maxGs[1]=maxG1, maxGs[2]=maxG2;
	dims[0]  = dims[1]  = dims[2]  = 1;
	slack[0] = slack[1] = slack[2] = 1.1;

	for (i=0;i<ctx->ndh;i++){
		dims[i] = ctx->src->dimensions[ctx->hwAxisList[i]];
		gaIFLInit(&factBS[i]);
		gaIFLInit(&factGS[i]);
		gaIFLInit(&factCS[i]);

		warpMod = dims[i]%warpSize;
		if (bestWarpMod>0 && (warpMod==0 || warpMod>=bestWarpMod)){
			bestWarpAxis = i;
			bestWarpMod  = warpMod;
		}
	}

	if (ctx->ndh > 0){
		dims[bestWarpAxis] = (dims[bestWarpAxis] + warpSize - 1)/warpSize;
		gaIFactorize(warpSize, 0, 0, &factBS[bestWarpAxis]);
	}

	/**
	 * Factorization job. We'll steadily increase the slack in case of failure
	 * in order to ensure we do get a factorization, which we place into
	 * chunkSize.
	 */

	for (i=0;i<ctx->ndh;i++){
		while (!gaIFactorize(dims[i], (uint64_t)(dims[i]*slack[i]), maxLs[i], &factCS[i])){
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

	gaIFLSchedule(ctx->ndh, maxLg, maxLs, maxGg, maxGs, factBS, factGS, factCS);

	/* Output. */
	for (i=0;i<ctx->ndh;i++){
		ctx->blockSize[i] = gaIFLGetProduct(&factBS[i]);
		ctx->gridSize [i] = gaIFLGetProduct(&factGS[i]);
		ctx->chunkSize[i] = gaIFLGetProduct(&factCS[i]);
	}

	/* Return. */
	return reduxInvokeLarge(ctx);
}

/**
 * Invoke the kernel.
 */

static int   reduxInvokeLarge              (redux_ctx*  ctx){
	void* args[11];
	int   ret;

	/**
	 * Argument Marshalling. This the grossest gross thing in here.
	 */

	const int flags    = GA_BUFFER_READ_ONLY|GA_BUFFER_INIT;
	ctx->srcStepsGD    = gpudata_alloc(ctx->gpuCtx, ctx->nds * sizeof(size_t),
	                                   ctx->src->strides,    flags, 0);
	ctx->srcSizeGD     = gpudata_alloc(ctx->gpuCtx, ctx->nds * sizeof(size_t),
	                                   ctx->src->dimensions, flags, 0);
	ctx->chunkSizeGD   = gpudata_alloc(ctx->gpuCtx, ctx->ndh * sizeof(size_t),
	                                   ctx->chunkSize,       flags, 0);
	if (reduxKernelRequiresDst(ctx)){
		ctx->dstStepsGD    = gpudata_alloc(ctx->gpuCtx, ctx->ndd * sizeof(size_t),
		                                   ctx->dst->strides,    flags, 0);
	}
	if (reduxKernelRequiresDstArg(ctx)){
		ctx->dstArgStepsGD = gpudata_alloc(ctx->gpuCtx, ctx->ndd * sizeof(size_t),
		                                   ctx->dstArg->strides, flags, 0);
	}
	args[ 0] = (void*) ctx->src->data;
	args[ 1] = (void*)&ctx->src->offset;
	args[ 2] = (void*) ctx->srcStepsGD;
	args[ 3] = (void*) ctx->srcSizeGD;
	args[ 4] = (void*) ctx->chunkSizeGD;
	if       ( reduxKernelRequiresDst   (ctx) &&  reduxKernelRequiresDstArg(ctx)){
		args[ 5] = (void*) ctx->dst->data;
		args[ 6] = (void*)&ctx->dst->offset;
		args[ 7] = (void*) ctx->dstStepsGD;
		args[ 8] = (void*) ctx->dstArg->data;
		args[ 9] = (void*)&ctx->dstArg->offset;
		args[10] = (void*) ctx->dstArgStepsGD;
	}else if ( reduxKernelRequiresDst   (ctx) && !reduxKernelRequiresDstArg(ctx)){
		args[ 5] = (void*) ctx->dst->data;
		args[ 6] = (void*)&ctx->dst->offset;
		args[ 7] = (void*) ctx->dstStepsGD;
	}else if (!reduxKernelRequiresDst   (ctx) &&  reduxKernelRequiresDstArg(ctx)){
		args[ 5] = (void*) ctx->dstArg->data;
		args[ 6] = (void*)&ctx->dstArg->offset;
		args[ 7] = (void*) ctx->dstArgStepsGD;
	}

	if (ctx->srcStepsGD   &&
	    ctx->srcSizeGD    &&
	    ctx->chunkSizeGD  &&
	    ctx->dstStepsGD   &&
	    ctx->dstArgStepsGD){
		ret = GpuKernel_call(&ctx->kernel,
		                     ctx->ndh>0 ? ctx->ndh : 1,
		                     ctx->gridSize,
		                     ctx->blockSize,
		                     0,
		                     args);
		return reduxCleanup(ctx, ret);
	}else{
		return reduxCleanup(ctx, GA_MEMORY_ERROR);
	}
}

/**
 * Cleanup
 */

static int   reduxCleanup                  (redux_ctx*  ctx, int ret){
	free(ctx->axisList);
	free(ctx->sourceCode);
	ctx->axisList   = NULL;
	ctx->sourceCode = NULL;

	gpudata_release(ctx->srcStepsGD);
	gpudata_release(ctx->srcSizeGD);
	gpudata_release(ctx->chunkSizeGD);
	gpudata_release(ctx->dstStepsGD);
	gpudata_release(ctx->dstArgStepsGD);
	ctx->srcStepsGD = ctx->srcSizeGD     = ctx->chunkSizeGD   =
	ctx->dstStepsGD = ctx->dstArgStepsGD = NULL;

	return ret;
}
