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
#define  DIVIDECEIL(a,b) (((a)+(b)-1)/(b))
#define  MAX_HW_DIMS                   3



/* Datatypes */

/**
 * @brief Axis Description.
 */

struct axis_desc{
	int      reduxNum;
	int      ibNum;
	unsigned ibp;
	unsigned isReduced : 1;
	unsigned isIntra   : 1;
	size_t   len;
	size_t   splitLen;
	size_t   pdim;
	ssize_t  srcStride;
	ssize_t  dstStride;
	ssize_t  dstArgStride;
};
typedef struct axis_desc axis_desc;

/**
 *                    Reduction Kernel Invoker.
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
 *   4. Ensuring there are no more than 5 blocks per multiprocessor.
 *   5. Minimizing the workspace size (if it is required).
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
 *  14. Size of workspace tensor
 *  15. Intrablock/split/free/reduced axes
 *  16. Source code
 * 
 * Rationale for dependencies:
 * 
 *   1) Get the GPU context and its properties immediately, since an invalid
 *      context is a likely error and we want to fail fast.
 *   2) The type and initializer of the accumulator should be determined after
 *      the context's properties have been retrieved since they provide
 *      information about the device's natively-supported types and operations
 *      (e.g. half-precision float)
 */

struct redux_ctx{
	/* Function Arguments. */
	GpuReduction*   gr;
	ga_reduce_op    op;
	GpuArray*       dst;
	GpuArray*       dstArg;
	const GpuArray* src;
	int             reduxLen;
	const int*      reduxList;
	int             flags;

	/* General. */
	int             nds;          /* # Source              dimensions */
	int             ndr;          /* # Reduced             dimensions */
	int             ndd;          /* # Destination         dimensions */
	int             ndfs;         /* # Flattened source    dimensions */
	int             ndfr;         /* # Flattened source    dimensions */
	int             ndfd;         /* # Flattened source    dimensions */
	int             ndib;         /* # Intra-block         dimensions */
	int             zeroAllAxes;  /* # of zero-length                   axes in source tensor */
	int             zeroRdxAxes;  /* # of zero-length         reduction axes in source tensor */
	size_t          prodAllAxes;  /* Product of length of all           axes in source tensor */
	size_t          prodRdxAxes;  /* Product of length of all reduction axes in source tensor */
	size_t          prodFreeAxes; /* Product of length of all free      axes in source tensor */
	
	/* Flattening */
	axis_desc*      xdSrc;
	axis_desc**     xdSrcPtrs;
	axis_desc**     xdTmpPtrs;

	/* Invoker */
	int             phase;
	size_t          U;
	size_t          V;
	size_t          B;
	unsigned        D;
	unsigned        H;
	unsigned        splitReduce;
	unsigned        splitFree;
	
	axis_desc*      xdSplit;
	
	size_t*         l;
	size_t*         lPDim;
	ssize_t*        sJ;
	ssize_t*        dJ;
	ssize_t*        aJ;
	
	gpudata*        flatSrcData;
	ssize_t         flatSrcOffset;
	gpudata*        flatDstData;
	ssize_t         flatDstOffset;
	gpudata*        flatDstArgData;
	ssize_t         flatDstArgOffset;
	
	gpudata*        w;
	size_t          SHMEM;
	ssize_t         wdOff;
	ssize_t         pdOff;
	ssize_t         waOff;
	ssize_t         paOff;
	
	unsigned*       ibs;
	unsigned*       ibp;
	size_t*         iblPDim;
	ssize_t*        ibsOff;
	ssize_t*        ibdOff;
	ssize_t*        ibaOff;
	
	void**          kArgs;
	
	
	/* Scheduler */
	size_t          bs;
	size_t          gs;
};
typedef struct redux_ctx redux_ctx;


/**
 *                    Reduction Operator.
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
 *   4. Ensuring there are no more than 5 blocks per multiprocessor.
 *   5. Minimizing the workspace size (if it is required).
 * 
 * 
 * REFERENCES
 * 
 * http://lpgpu.org/wp/wp-content/uploads/2013/05/poster_andresch_acaces2014.pdf
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

struct GpuReduction{
	/* Function Arguments. */
	gpucontext*      gpuCtx;
	ga_reduce_op     op;
	int              ndd;
	int              ndr;
	int              srcTypeCode;
	int              flags;
	
	/* Misc */
	int              nds;
	
	/* Source code Generator. */
	strb             s;
	srcb             srcGen;
	char*            kSourceCode;
	size_t           kSourceCodeLen;
	int              dstTypeCode;
	int              dstArgTypeCode;
	int              idxTypeCode;
	int              accTypeCode;
	const char*      srcTypeStr;
	const char*      dstTypeStr;
	const char*      dstArgTypeStr;
	const char*      idxTypeStr;
	const char*      accTypeStr;
	const char*      initVal;
	
	/* Compile */
	int              log2MaxL;
	int              kNumArgs;
	int*             kArgTypeCodes;
	char*            kErrorString;
	GpuKernel        k;
	
	/* Scheduling */
	unsigned         numProcs;
	size_t           maxLg;
	size_t           maxL0;
	size_t           maxGg;
	size_t           maxG0;
	size_t           maxLM;
	size_t           maxLK;
};


/* Typedefs */
typedef void (*GpuReductionIterFn)(GpuReduction* gr,
                                   int           typecode,
                                   const char*   typeName,
                                   const char*   baseName,
                                   int           num,
                                   void*         user);


/* Static Function prototypes */
/* Utilities */
static int        reduxGetSumInit               (int typecode, const char** property);
static int        reduxGetProdInit              (int typecode, const char** property);
static int        reduxGetMinInit               (int typecode, const char** property);
static int        reduxGetMaxInit               (int typecode, const char** property);
static int        reduxGetAndInit               (int typecode, const char** property);
static int        reduxGetOrInit                (int typecode, const char** property);
static int        reduxIsSensitive              (int               typecode);
static int        reduxSortFlatSensitive        (const void* a, const void* b);
static int        reduxSortFlatInsensitive      (const void* a, const void* b);
static int        reduxSortPtrIBSrcRdSelect     (const void* a, const void* b);
static int        reduxSortPtrByReduxNum        (const void* a, const void* b);
static int        reduxSortPtrIBDstWrSelect     (const void* a, const void* b);
static int        reduxSortPtrIBDstArgWrSelect  (const void* a, const void* b);
static int        reduxSortPtrInsertFinalOrder  (const void* a, const void* b);

/* Axis Description API */
static void       axisInit                      (axis_desc*           axis,
                                                 ssize_t              len,
                                                 ssize_t              srcStride);
static void       axisMarkReduced               (axis_desc*           axis, int    reduxNum);
static void       axisMarkIntraBlock            (axis_desc*           axis,
                                                 int                  ibNum,
                                                 size_t               ibLen);
static int        axisGetReduxNum               (const axis_desc*     axis);
static size_t     axisGetLen                    (const axis_desc*     axis);
static size_t     axisGetIntraLen               (const axis_desc*     axis);
static size_t     axisGetInterLen               (const axis_desc*     axis);
static size_t     axisGetIntraInterLen          (const axis_desc*     axis);
static ssize_t    axisGetSrcStride              (const axis_desc*     axis);
static size_t     axisGetSrcAbsStride           (const axis_desc*     axis);
static ssize_t    axisGetDstStride              (const axis_desc*     axis);
static size_t     axisGetDstAbsStride           (const axis_desc*     axis);
static ssize_t    axisGetDstArgStride           (const axis_desc*     axis);
static size_t     axisGetDstArgAbsStride        (const axis_desc*     axis);
static unsigned   axisGetIBP                    (const axis_desc*     axis);
static int        axisGetIBNum                  (const axis_desc*     axis);
static void       axisSetIBP                    (axis_desc*           axis,
                                                 unsigned             ibp);
static size_t     axisGetPDim                   (const axis_desc*     axis);
static void       axisSetPDim                   (axis_desc*           axis,
                                                 size_t               pdim);
static int        axisIsReduced                 (const axis_desc*     axis);
static int        axisIsIntra                   (const axis_desc*     axis);
static int        axisIsInter                   (const axis_desc*     axis);
static int        axisIsSplit                   (const axis_desc*     axis);

/* Reduction Context API */
/*     Generator Control Flow */
static int        reduxGenInit                  (GpuReduction*        gr);
static int        reduxGenInferProperties       (GpuReduction*        gr);
static void       reduxGenIterArgs              (GpuReduction*        gr,
                                                 GpuReductionIterFn   fn,
                                                 void*                user);
static int        reduxGenSrc                   (GpuReduction*        gr);
static void       reduxGenSrcAppend             (GpuReduction*        gr);
static void       reduxGenSrcAppendIncludes     (GpuReduction*        gr);
static void       reduxGenSrcAppendMacroDefs    (GpuReduction*        gr);
static void       reduxGenSrcAppendTypedefs     (GpuReduction*        gr);
static void       reduxGenSrcAppendReduxKernel  (GpuReduction*        gr);
static void       reduxGenSrcAppendPrototype    (GpuReduction*        gr);
static void       reduxGenSrcAppendBlockDecode  (GpuReduction*        gr);
static void       reduxGenSrcAppendThreadDecode (GpuReduction*        gr);
static void       reduxGenSrcAppendPhase0       (GpuReduction*        gr);
static void       reduxGenSrcAppendLoops        (GpuReduction*        gr,
                                                 int                  freeMaybeSplit,
                                                 int                  reduceMaybeSplit);
static void       reduxGenSrcAppendLoop         (GpuReduction*        gr,
                                                 int                  initial,
                                                 int                  freeMaybeSplit,
                                                 int                  reduceMaybeSplit);
static void       reduxGenSrcAppendDecrement    (GpuReduction*        gr);
static void       reduxGenSrcAppendVertical     (GpuReduction*        gr,
                                                 int                  freeMaybeSplit,
                                                 int                  reduceMaybeSplit);
static void       reduxGenSrcAppendIncrement    (GpuReduction*        gr,
                                                 int                  axis,
                                                 int                  initial,
                                                 int                  freeMaybeSplit,
                                                 int                  reduceMaybeSplit);
static void       reduxGenSrcAppendDstWrite     (GpuReduction*        gr,
                                                 int                  initial,
                                                 int                  freeMaybeSplit,
                                                 int                  reduceMaybeSplit);
static void       reduxGenSrcAppendPhase1       (GpuReduction*        gr);
static int        reduxGenCompile               (GpuReduction*        gr);
static int        reduxGenComputeLaunchBounds   (GpuReduction*        gr);
static int        reduxGenCleanup               (GpuReduction*        gr,  int ret);
static int        reduxGenCleanupMsg            (GpuReduction*        gr,  int ret,
                                                 const char*          fmt, ...);

/*     Generator Utilities */
static void       reduxGenCountArgs             (GpuReduction*        gr,
                                                 int                  typecode,
                                                 const char*          typeName,
                                                 const char*          baseName,
                                                 int                  num,
                                                 void*                user);
static void       reduxGenSaveArgTypecodes      (GpuReduction*        gr,
                                                 int                  typecode,
                                                 const char*          typeName,
                                                 const char*          baseName,
                                                 int                  num,
                                                 void*                user);
static void       reduxGenAppendArg             (GpuReduction*        gr,
                                                 int                  typecode,
                                                 const char*          typeName,
                                                 const char*          baseName,
                                                 int                  num,
                                                 void*                user);
static void       reduxInvMarshalArg            (GpuReduction*        gr,
                                                 int                  typecode,
                                                 const char*          typeName,
                                                 const char*          baseName,
                                                 int                  num,
                                                 void*                user);
static size_t     reduxGenEstimateParallelism   (const GpuReduction*  gr);
static int        reduxGenRequiresDst           (const GpuReduction*  gr);
static int        reduxGenRequiresDstArg        (const GpuReduction*  gr);
static int        reduxGenKernelRequiresDst     (const GpuReduction*  gr);
static int        reduxGenKernelRequiresDstArg  (const GpuReduction*  gr);
static int        reduxGenAxisMaybeSplit        (const GpuReduction*  gr, int axis);
static size_t     reduxGenGetReduxStateSize     (const GpuReduction*  gr);
static size_t     reduxGenGetMaxLocalSize       (const GpuReduction*  gr);
static size_t     reduxGenGetSHMEMSize          (const GpuReduction*  gr, size_t bs);
static size_t     reduxGenGetSHMEMDstOff        (const GpuReduction*  gr, size_t bs);
static size_t     reduxGenGetSHMEMDstArgOff     (const GpuReduction*  gr, size_t bs);
static size_t     reduxGenGetWMEMSize           (const GpuReduction*  gr, size_t bs);
static size_t     reduxGenGetWMEMDstOff         (const GpuReduction*  gr, size_t bs);
static size_t     reduxGenGetWMEMDstArgOff      (const GpuReduction*  gr, size_t bs);

/*     Invoker Control Flow */
static int        reduxInvInit                  (redux_ctx*           ctx);
static int        reduxInvInferProperties       (redux_ctx*           ctx);
static int        reduxInvFlattenSource         (redux_ctx*           ctx);
static int        reduxInvComputeKArgs          (redux_ctx*           ctx);
static int        reduxInvSchedule              (redux_ctx*           ctx);
static int        reduxInvoke                   (redux_ctx*           ctx);
static int        reduxInvCleanup               (redux_ctx*           ctx, int ret);
static int        reduxInvCleanupMsg            (redux_ctx*           ctx, int ret,
                                                 const char*          fmt, ...);

/*     Invoker Utilities */
static size_t     reduxInvEstimateParallelism   (const redux_ctx*  ctx);
static int        reduxInvRequiresDst           (const redux_ctx*  ctx);
static int        reduxInvRequiresDstArg        (const redux_ctx*  ctx);
static int        reduxInvKernelRequiresDst     (const redux_ctx*  ctx);
static unsigned   reduxInvGetSplitFree          (const redux_ctx*  ctx);
static unsigned   reduxInvGetSplitReduce        (const redux_ctx*  ctx);
static axis_desc* reduxInvGetSrcAxis            (const redux_ctx*  ctx, int i);
static axis_desc* reduxInvGetSrcSortAxis        (const redux_ctx*  ctx, int i);
static int        reduxTryFlattenOut            (const redux_ctx*  ctx,
                                                 const axis_desc*  out);
static int        reduxTryFlattenInto           (redux_ctx*        ctx,
                                                 axis_desc*        into,
                                                 const axis_desc*  from);
static void       reduxSortAxisPtrsBy           (axis_desc**       ptrs,
                                                 axis_desc*        axes,
                                                 size_t            numAxes,
                                                 int(*fn)(const void*, const void*));


/* Function Implementations */
/* Extern Functions */
GPUARRAY_PUBLIC int   GpuReduction_new   (GpuReduction**   grOut,
                                          gpucontext*      gpuCtx,
                                          ga_reduce_op     op,
                                          unsigned         ndf,
                                          unsigned         ndr,
                                          int              srcTypeCode,
                                          int              flags){
	if(!grOut){
		return GA_INVALID_ERROR;
	}
	
	*grOut = calloc(1, sizeof(**grOut));
	if(*grOut){
		(*grOut)->gpuCtx      = gpuCtx;
		(*grOut)->op          = op;
		(*grOut)->ndd         = (int)ndf;
		(*grOut)->ndr         = (int)ndr;
		(*grOut)->srcTypeCode = srcTypeCode;
		(*grOut)->flags       = flags;
		
		return reduxGenInit(*grOut);
	}else{
		return GA_MEMORY_ERROR;
	}
}
GPUARRAY_PUBLIC void  GpuReduction_free  (GpuReduction*    gr){
	reduxGenCleanup(gr, !GA_NO_ERROR);
}
GPUARRAY_PUBLIC int   GpuReduction_call  (GpuReduction*    gr,
                                          GpuArray*        dst,
                                          GpuArray*        dstArg,
                                          const GpuArray*  src,
                                          unsigned         reduxLen,
                                          const int*       reduxList,
                                          int              flags){
	redux_ctx ctxSTACK, *ctx = &ctxSTACK;
	memset(ctx, 0, sizeof(*ctx));

	ctx->gr        = gr;
	ctx->dst       = dst;
	ctx->dstArg    = dstArg;
	ctx->src       = src;
	ctx->reduxLen  = reduxLen;
	ctx->reduxList = reduxList;
	ctx->flags     = flags;

	return reduxInvInit(ctx);
}


/* Static Functions */

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

static int        reduxIsSensitive              (int               typecode){
	switch (typecode){
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
 * @brief Sort the axes into optimal order for contiguous memory access.
 * 
 * This means ascending order of absolute stride.
 */

static int        reduxSortPtrIBSrcRdSelect     (const void* a, const void* b){
	const axis_desc* xda  = *(const axis_desc* const*)a;
	const axis_desc* xdb  = *(const axis_desc* const*)b;
	
	if       (axisGetSrcAbsStride(xda)  <  axisGetSrcAbsStride(xdb)){
		return -1;
	}else if (axisGetSrcAbsStride(xda)  >  axisGetSrcAbsStride(xdb)){
		return +1;
	}

	return 0;
}
static int        reduxSortPtrByReduxNum        (const void* a, const void* b){
	const axis_desc* xda  = *(const axis_desc* const*)a;
	const axis_desc* xdb  = *(const axis_desc* const*)b;
	
	if       ( axisIsReduced(xda)  && !axisIsReduced(xdb)){
		return -1;
	}else if (!axisIsReduced(xda)  &&  axisIsReduced(xdb)){
		return +1;
	}
	
	if       (axisGetReduxNum(xda)  <  axisGetReduxNum(xdb)){
		return +1;
	}else if (axisGetReduxNum(xda)  >  axisGetReduxNum(xdb)){
		return -1;
	}

	return 0;
}
static int        reduxSortPtrIBDstWrSelect     (const void* a, const void* b){
	const axis_desc* xda  = *(const axis_desc* const*)a;
	const axis_desc* xdb  = *(const axis_desc* const*)b;
	
	/* All intra axes go first. */
	if       (axisIsIntra(xda)  &&  axisIsInter(xdb)){
		return -1;
	}else if (axisIsInter(xda)  &&  axisIsIntra(xdb)){
		return +1;
	}
	
	/* All free axes go first (for lower stride within SHMEM[H][D]). */
	if       ( axisIsReduced(xda)  && !axisIsReduced(xdb)){
		return +1;
	}else if (!axisIsReduced(xda)  &&  axisIsReduced(xdb)){
		return -1;
	}
	
	/* The split axis, if it is free, goes last within the free axes. */
	if       ( axisIsSplit(xda)  && !axisIsReduced(xda)){
		return +1;
	}else if ( axisIsSplit(xdb)  && !axisIsReduced(xdb)){
		return -1;
	}
	
	/* Otherwise it's sort by destination absolute stride. */
	if       (axisGetDstAbsStride(xda)  <  axisGetDstAbsStride(xdb)){
		return -1;
	}else if (axisGetDstAbsStride(xda)  >  axisGetDstAbsStride(xdb)){
		return +1;
	}

	return 0;
}
static int        reduxSortPtrIBDstArgWrSelect  (const void* a, const void* b){
	const axis_desc* xda  = *(const axis_desc* const*)a;
	const axis_desc* xdb  = *(const axis_desc* const*)b;
	
	/* All intra axes go first. */
	if       (axisIsIntra(xda)  &&  axisIsInter(xdb)){
		return -1;
	}else if (axisIsInter(xda)  &&  axisIsIntra(xdb)){
		return +1;
	}
	
	/* All free axes go first (for lower stride within SHMEM[H][D]). */
	if       ( axisIsReduced(xda)  && !axisIsReduced(xdb)){
		return +1;
	}else if (!axisIsReduced(xda)  &&  axisIsReduced(xdb)){
		return -1;
	}
	
	/* The split axis, if it is free, goes last within the free axes. */
	if       ( axisIsSplit(xda)  && !axisIsReduced(xda)){
		return +1;
	}else if ( axisIsSplit(xdb)  && !axisIsReduced(xdb)){
		return -1;
	}
	
	/* Otherwise it's sort by destination argument absolute stride. */
	if       (axisGetDstArgAbsStride(xda)  <  axisGetDstArgAbsStride(xdb)){
		return -1;
	}else if (axisGetDstArgAbsStride(xda)  >  axisGetDstArgAbsStride(xdb)){
		return +1;
	}

	return 0;
}
static int        reduxSortPtrInsertFinalOrder  (const void* a, const void* b){
	const axis_desc* xda  = *(const axis_desc* const*)a;
	const axis_desc* xdb  = *(const axis_desc* const*)b;
	
	
	/* All intra axes go first. */
	if       (axisIsIntra(xda)  &&  axisIsInter(xdb)){
		return -1;
	}else if (axisIsInter(xda)  &&  axisIsIntra(xdb)){
		return +1;
	}
	
	if(axisIsIntra(xda)){
		/**
		 * Intra axes sort between themselves by descending intra axis number.
		 */
		
		if       (axisGetIBNum(xda)  <  axisGetIBNum(xdb)){
			return +1;
		}else if (axisGetIBNum(xda)  >  axisGetIBNum(xdb)){
			return -1;
		}
		
		return 0;
	}else{
		/**
		 * Inter axes sort between themselves
		 * 
		 *   - Reduced axes first
		 *   - Then by ascending source tensor stride
		 */
		
		if       ( axisIsReduced(xda)  && !axisIsReduced(xdb)){
			return -1;
		}else if (!axisIsReduced(xda)  &&  axisIsReduced(xdb)){
			return +1;
		}
		
		if       (axisGetSrcAbsStride(xda)  <  axisGetSrcAbsStride(xdb)){
			return -1;
		}else if (axisGetSrcAbsStride(xda)  >  axisGetSrcAbsStride(xdb)){
			return +1;
		}
	}

	return 0;
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
	axis->ibNum           = -1;
	axis->ibp             = 0;
	axis->len             = len;
	axis->splitLen        = 1;
	axis->pdim            = 0;
	
	axis->srcStride       = srcStride;
	axis->dstStride       = 0;
	axis->dstArgStride    = 0;
}

/**
 * @brief Mark axis as reduction axis, with position reduxNum in the axis list.
 */

static void       axisMarkReduced               (axis_desc*       axis, int    reduxNum){
	axis->isReduced = 1;
	axis->reduxNum  = reduxNum;
}

/**
 * @brief Mark axis as (split) intrablock axis.
 */

static void       axisMarkIntraBlock            (axis_desc*       axis,
                                                 int              ibNum,
                                                 size_t           ibLen){
	axis->isIntra  = 1;
	axis->ibNum    = ibNum;
	axis->splitLen = ibLen;
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
static size_t     axisGetIntraLen               (const axis_desc* axis){
	if       (axisIsSplit(axis)){
		return axis->splitLen;
	}else if (axisIsIntra(axis)){
		return axis->len;
	}else{
		return 1;
	}
}
static size_t     axisGetInterLen               (const axis_desc* axis){
	if       (axisIsSplit(axis)){
		return DIVIDECEIL(axis->len, axis->splitLen);
	}else if (axisIsIntra(axis)){
		return 1;
	}else{
		return axis->len;
	}
}
static size_t     axisGetIntraInterLen          (const axis_desc* axis){
	return axisGetIntraLen(axis)*axisGetInterLen(axis);
}
static ssize_t    axisGetSrcStride              (const axis_desc* axis){
	return axisGetLen(axis) > 1 ? axis->srcStride : 0;
}
static size_t     axisGetSrcAbsStride           (const axis_desc* axis){
	return axisGetSrcStride(axis)<0 ? -(size_t)axisGetSrcStride(axis):
	                                  +(size_t)axisGetSrcStride(axis);
}
static ssize_t    axisGetDstStride              (const axis_desc* axis){
	return axisGetLen(axis) > 1 ? axis->dstStride : 0;
}
static size_t     axisGetDstAbsStride           (const axis_desc* axis){
	return axisGetDstStride(axis)<0 ? -(size_t)axisGetDstStride(axis):
	                                  +(size_t)axisGetDstStride(axis);
}
static ssize_t    axisGetDstArgStride           (const axis_desc* axis){
	return axisGetLen(axis) > 1 ? axis->dstArgStride : 0;
}
static size_t     axisGetDstArgAbsStride        (const axis_desc* axis){
	return axisGetDstArgStride(axis)<0 ? -(size_t)axisGetDstArgStride(axis):
	                                     +(size_t)axisGetDstArgStride(axis);
}
static unsigned   axisGetIBP                    (const axis_desc* axis){
	return axis->ibp;
}
static int        axisGetIBNum                  (const axis_desc* axis){
	return axis->ibNum;
}
static void       axisSetIBP                    (axis_desc*       axis,
                                                 unsigned         ibp){
	axis->ibp = ibp;
}
static size_t     axisGetPDim                   (const axis_desc*     axis){
	return axis->pdim;
}
static void       axisSetPDim                   (axis_desc*           axis,
                                                 size_t               pdim){
	axis->pdim = pdim;
}
static int        axisIsReduced                 (const axis_desc* axis){
	return axis->isReduced;
}
static int        axisIsIntra                   (const axis_desc* axis){
	return axis->isIntra;
}
static int        axisIsInter                   (const axis_desc* axis){
	return !axisIsIntra(axis);
}
static int        axisIsSplit                   (const axis_desc* axis){
	return axisIsIntra(axis) && axis->splitLen != axis->len;
}
static size_t     reduxInvEstimateParallelism   (const redux_ctx*  ctx){
	return reduxGenEstimateParallelism(ctx->gr);
}
static int        reduxInvRequiresDst           (const redux_ctx*  ctx){
	return reduxGenRequiresDst(ctx->gr);
}
static int        reduxInvRequiresDstArg        (const redux_ctx*  ctx){
	return reduxGenRequiresDstArg(ctx->gr);
}
static int        reduxInvKernelRequiresDst     (const redux_ctx*  ctx){
	return reduxGenKernelRequiresDst(ctx->gr);
}
static int        reduxInvKernelRequiresDstArg  (const redux_ctx*  ctx){
	return reduxGenKernelRequiresDstArg(ctx->gr);
}
static unsigned   reduxInvGetSplitFree          (const redux_ctx*  ctx){
	if(ctx->xdSplit && !axisIsReduced(ctx->xdSplit)){
		return axisGetIntraLen(ctx->xdSplit);
	}else{
		return 1;
	}
}
static unsigned   reduxInvGetSplitReduce        (const redux_ctx*  ctx){
	if(ctx->xdSplit && axisIsReduced(ctx->xdSplit)){
		return axisGetIntraLen(ctx->xdSplit);
	}else{
		return 1;
	}
}

/**
 * @brief Get description of source axis with given number.
 */

static axis_desc* reduxInvGetSrcAxis            (const redux_ctx*  ctx, int i){
	return &ctx->xdSrc[i];
}

/**
 * @brief Get description of source axis with given number in sort-order.
 */

static axis_desc* reduxInvGetSrcSortAxis        (const redux_ctx*  ctx, int i){
	return ctx->xdSrcPtrs[i];
}

/**
 * @brief Attempt to flatten out an axis from the context.
 * 
 * An axis can be flattened out if:
 * 
 *   1. The axis is of length 1.
 *   2. The axis is a reduction axis, and there exists at least one reduction
 *      axis of length 0 in the source tensor.
 * 
 * @return Non-zero if flattening attempt successful; Zero otherwise.
 */

static int        reduxTryFlattenOut            (const redux_ctx*  ctx,
                                                 const axis_desc*  out){
	if ((axisGetLen   (out) == 1                   )||
	    (axisIsReduced(out) && ctx->zeroRdxAxes > 0)){
		return 1;
	}else{
		return 0;
	}
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

static int        reduxTryFlattenInto           (redux_ctx*        ctx,
                                                 axis_desc*        into,
                                                 const axis_desc*  from){
	int signSrc    = 0, signDst    = 0, signDstArg    = 0,
	    reverseSrc = 0, reverseDst = 0, reverseDstArg = 0;
	
	if (axisIsReduced         (into) != axisIsReduced         (from)                 ||
	    axisGetSrcAbsStride   (into) != axisGetSrcAbsStride   (from)*axisGetLen(from)){
		return 0;
	}
	
	if (reduxInvRequiresDst   (ctx) &&
	    axisGetDstAbsStride   (into) != axisGetDstAbsStride   (from)*axisGetLen(from)){
		return 0;
	}
	
	if (reduxInvRequiresDstArg(ctx) &&
	    axisGetDstArgAbsStride(into) != axisGetDstArgAbsStride(from)*axisGetLen(from)){
		return 0;
	}
	
	signSrc       = (axisGetSrcStride   (into)^axisGetSrcStride   (from)) < 0;
	signDst       = (axisGetDstStride   (into)^axisGetDstStride   (from)) < 0;
	signDstArg    = (axisGetDstArgStride(into)^axisGetDstArgStride(from)) < 0;
	reverseSrc    = signSrc;
	reverseDst    = signDst    && reduxInvRequiresDst   (ctx);
	reverseDstArg = signDstArg && reduxInvRequiresDstArg(ctx);
	
	if (reduxIsSensitive(ctx->op)){
		if(reverseSrc || reverseDst || reverseDstArg){
			return 0;
		}
	}
	
	if (reduxInvRequiresDst   (ctx) &&
	    reduxInvRequiresDstArg(ctx) &&
	    reverseDst != reverseDstArg){
		/* Either both, or neither, of dst and dstArg must require reversal. */
		return 0;
	}
	
	if (reverseSrc){
		ctx->flatSrcOffset    += (ssize_t)(axisGetLen(from)-1)*axisGetSrcStride(from);
		into->srcStride        = -axisGetSrcStride   (from);
	}else{
		into->srcStride        =  axisGetSrcStride   (from);
	}
	
	if (reverseDst){
		ctx->flatDstOffset    += (ssize_t)(axisGetLen(from)-1)*axisGetDstStride(from);
		into->dstStride        = -axisGetDstStride   (from);
	}else{
		into->dstStride        =  axisGetDstStride   (from);
	}
	
	if (reverseDstArg){
		ctx->flatDstArgOffset += (ssize_t)(axisGetLen(from)-1)*axisGetDstArgStride(from);
		into->dstArgStride     = -axisGetDstArgStride(from);
	}else{
		into->dstArgStride     =  axisGetDstArgStride(from);
	}
	
	into->len *= axisGetLen(from);
	
	return 1;
}

/**
 * Sort an array of *pointers* to axes by the given comparison function, while
 * not touching the axes themselves.
 */

static void       reduxSortAxisPtrsBy           (axis_desc**       ptrs,
                                                 axis_desc*        axes,
                                                 size_t            numAxes,
                                                 int(*fn)(const void*, const void*)){
	size_t i;
	
	for(i=0;i<numAxes;i++){
		ptrs[i] = &axes[i];
	}
	
	qsort(ptrs, numAxes, sizeof(*ptrs), fn);
}


/**
 * @brief Initialize generator context.
 * 
 * After this function, calling reduxGenCleanup*() becomes safe.
 */

static int        reduxGenInit                  (GpuReduction*     gr){
	gr->kArgTypeCodes = NULL;
	gr->kSourceCode   = NULL;
	gr->kErrorString  = NULL;
	gr->kNumArgs      = 0;
	
	return reduxGenInferProperties(gr);
}

/**
 * @brief Begin inferring the properties of the reduction operator.
 */

static int        reduxGenInferProperties       (GpuReduction*     gr){
	int i, ret;
	
	
	/**
	 * Insane arguments?
	 */
	
	if(gr->ndr <= 0){
		return reduxGenCleanupMsg(gr, GA_INVALID_ERROR,
		       "No reduction axes!\n");
	}
	if(gr->ndd <  0){
		return reduxGenCleanupMsg(gr, GA_INVALID_ERROR,
		       "Destination has less than 0 dimensions!\n");
	}
	if(gr->flags != 0){
		return reduxGenCleanupMsg(gr, GA_INVALID_ERROR,
		       "\"flags\" must be set to 0!\n");
	}
	gr->nds = gr->ndr+gr->ndd;
	
	
	/**
	 * Source code buffer preallocation failed?
	 */
	
	if (strb_ensure(&gr->s, 32*1024) != 0){
		return reduxGenCleanupMsg(gr, GA_MEMORY_ERROR,
		       "Could not preallocate source code buffer!\n");
	}
	srcbInit(&gr->srcGen, &gr->s);
	
	
	/**
	 * GPU context non-existent, or cannot read its properties?
	 */
	
	if (!gr->gpuCtx                                                                          ||
	    gpucontext_property(gr->gpuCtx, GA_CTX_PROP_NUMPROCS,  &gr->numProcs) != GA_NO_ERROR ||
	    gpucontext_property(gr->gpuCtx, GA_CTX_PROP_MAXLSIZE,  &gr->maxLg)    != GA_NO_ERROR ||
	    gpucontext_property(gr->gpuCtx, GA_CTX_PROP_MAXLSIZE0, &gr->maxL0)    != GA_NO_ERROR ||
	    gpucontext_property(gr->gpuCtx, GA_CTX_PROP_MAXGSIZE,  &gr->maxGg)    != GA_NO_ERROR ||
	    gpucontext_property(gr->gpuCtx, GA_CTX_PROP_MAXGSIZE0, &gr->maxG0)    != GA_NO_ERROR ||
	    gpucontext_property(gr->gpuCtx, GA_CTX_PROP_LMEMSIZE,  &gr->maxLM)    != GA_NO_ERROR ){
		return reduxGenCleanupMsg(gr, GA_INVALID_ERROR,
		       "Error obtaining one or more properties from GPU context!\n");
	}
	
	
	/**
	 * Type management.
	 * 
	 * - Deal with the various typecodes.
	 * - Determine initializer and error out if reduction unsupported on that
	 *   datatype.
	 */

	gr->dstTypeCode    = gr->srcTypeCode;
	gr->dstArgTypeCode = GA_SSIZE;
	gr->idxTypeCode    = GA_SSIZE;
	switch (gr->srcTypeCode){
		case GA_HALF:
		  gr->accTypeCode = GA_FLOAT;
		break;
		case GA_HALF2:
		  gr->accTypeCode = GA_FLOAT2;
		break;
		case GA_HALF4:
		  gr->accTypeCode = GA_FLOAT4;
		break;
		case GA_HALF8:
		  gr->accTypeCode = GA_FLOAT8;
		break;
		case GA_HALF16:
		  gr->accTypeCode = GA_FLOAT16;
		break;
		default:
		  gr->accTypeCode = gr->srcTypeCode;
	}
	gr->srcTypeStr     = gpuarray_get_type(gr->srcTypeCode)   ->cluda_name;
	gr->dstTypeStr     = gpuarray_get_type(gr->dstTypeCode)   ->cluda_name;
	gr->dstArgTypeStr  = gpuarray_get_type(gr->dstArgTypeCode)->cluda_name;
	gr->idxTypeStr     = gpuarray_get_type(gr->idxTypeCode)   ->cluda_name;
	gr->accTypeStr     = gpuarray_get_type(gr->accTypeCode)   ->cluda_name;
	if (!gr->srcTypeStr    ||
	    !gr->dstTypeStr    ||
	    !gr->dstArgTypeStr ||
	    !gr->idxTypeStr    ||
	    !gr->accTypeStr    ){
		return reduxGenCleanupMsg(gr, GA_INVALID_ERROR,
		                          "Have typecode with no CLUDA name!\n");
	}
	switch (gr->op){
		case GA_REDUCE_SUM:
		  ret = reduxGetSumInit (gr->accTypeCode, &gr->initVal);
		break;
		case GA_REDUCE_PRODNZ:
		case GA_REDUCE_PROD:
		  ret = reduxGetProdInit(gr->accTypeCode, &gr->initVal);
		break;
		case GA_REDUCE_MINANDARGMIN:
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_MIN:
		  ret = reduxGetMinInit (gr->accTypeCode, &gr->initVal);
		break;
		case GA_REDUCE_MAXANDARGMAX:
		case GA_REDUCE_ARGMAX:
		case GA_REDUCE_MAX:
		  ret = reduxGetMaxInit (gr->accTypeCode, &gr->initVal);
		break;
		case GA_REDUCE_ALL:
		case GA_REDUCE_AND:
		  ret = reduxGetAndInit (gr->accTypeCode, &gr->initVal);
		break;
		case GA_REDUCE_ANY:
		case GA_REDUCE_XOR:
		case GA_REDUCE_OR:
		  ret = reduxGetOrInit  (gr->accTypeCode, &gr->initVal);
		break;
		default:
		  ret = GA_UNSUPPORTED_ERROR;
	}
	if (ret != GA_NO_ERROR){
		return reduxGenCleanupMsg(gr, ret,
		       "Problem selecting types to be used in reduction!\n");
	}
	
	
	/* Compute floor(log2(gr->log2MaxL)). */
	gr->log2MaxL = gr->maxLg-1;
	for(i=1;gr->log2MaxL & (gr->log2MaxL+1);i*=2){
		gr->log2MaxL |= gr->log2MaxL>>i;
	}
	for(i=0;gr->log2MaxL;i++){
		gr->log2MaxL >>= 1;
	}
	gr->log2MaxL = i?i:1;
	
	
	/**
	 * Compute number of kernel arguments and construct kernel argument
	 * typecode list.
	 */
	
	reduxGenIterArgs(gr, reduxGenCountArgs, 0);
	gr->kArgTypeCodes = calloc(gr->kNumArgs, sizeof(*gr->kArgTypeCodes));
	if(!gr->kArgTypeCodes){
		return reduxGenCleanupMsg(gr, GA_MEMORY_ERROR,
		                          "Failed to allocate memory for kernel arguments "
		                          "typecode list!\n");
	}
	i = 0;
	reduxGenIterArgs(gr, reduxGenSaveArgTypecodes, &i);
	
	
	/* Generate source code. */
	return reduxGenSrc(gr);
}

/**
 * Iterate over the arguments of the reduction operator.
 */

static void       reduxGenIterArgs              (GpuReduction*        gr,
                                                 GpuReductionIterFn   fn,
                                                 void*                user){
	int k;
	
	fn(gr, GA_INT,    "int",                      "phase",       0, user);
	fn(gr, GA_SIZE,   "TX",                       "U",           0, user);
	fn(gr, GA_SIZE,   "TX",                       "V",           0, user);
	fn(gr, GA_SIZE,   "TX",                       "B",           0, user);
	fn(gr, GA_UINT,   "unsigned",                 "D",           0, user);
	fn(gr, GA_UINT,   "unsigned",                 "H",           0, user);
	fn(gr, GA_UINT,   "unsigned",                 "splitFree",   0, user);
	fn(gr, GA_UINT,   "unsigned",                 "splitReduce", 0, user);
	for(k=0;k < gr->nds;k++){
		fn(gr, GA_SIZE,   "TX",                       "l%d",         k, user);
	}
	for(k=gr->ndd;k < gr->nds && reduxGenRequiresDstArg(gr);k++){
		fn(gr, GA_SIZE,   "TX",                       "l%dPDim",     k, user);
	}
	fn(gr, GA_BUFFER, "const GLOBAL_MEM char*",   "s",           0, user);
	fn(gr, GA_SSIZE,  "TX",                       "sOff",        0, user);
	for(k=0;k < gr->nds;k++){
		fn(gr, GA_SIZE,   "TX",                       "sJ%d",        k, user);
	}
	if(reduxGenRequiresDst   (gr)){
		fn(gr, GA_BUFFER, "GLOBAL_MEM char*",         "d",           0, user);
		fn(gr, GA_SSIZE,  "TX",                       "dOff",        0, user);
		for(k=0;k < gr->ndd;k++){
			fn(gr, GA_SIZE,   "TX",                       "dJ%d",        k, user);
		}
	}
	if(reduxGenRequiresDstArg(gr)){
		fn(gr, GA_BUFFER, "GLOBAL_MEM char*",         "a",           0, user);
		fn(gr, GA_SSIZE,  "TX",                       "aOff",        0, user);
		for(k=0;k < gr->ndd;k++){
			fn(gr, GA_SIZE,   "TX",                       "aJ%d",        k, user);
		}
	}
	fn(gr, GA_BUFFER, "GLOBAL_MEM char*",         "w",           0, user);
	if(reduxGenKernelRequiresDst   (gr)){
		fn(gr, GA_SSIZE,  "TX",                       "wdOff",       0, user);
		fn(gr, GA_SSIZE,  "TX",                       "pdOff",       0, user);
	}
	if(reduxGenKernelRequiresDstArg(gr)){
		fn(gr, GA_SSIZE,  "TX",                       "waOff",       0, user);
		fn(gr, GA_SSIZE,  "TX",                       "paOff",       0, user);
	}
	for(k=0;k < gr->log2MaxL;k++){
		fn(gr, GA_UINT,   "unsigned",                 "ibs%d",       k, user);
	}
	for(k=0;k < gr->log2MaxL;k++){
		fn(gr, GA_UINT,   "unsigned",                 "ibp%d",       k, user);
	}
	for(k=0;k < gr->log2MaxL && reduxGenRequiresDstArg(gr);k++){
		fn(gr, GA_SIZE,   "TX",                       "ibl%dPDim",   k, user);
	}
	for(k=0;k < gr->log2MaxL;k++){
		fn(gr, GA_SSIZE,  "TX",                       "ibsOff%d",    k, user);
	}
	for(k=0;k < gr->log2MaxL && reduxGenRequiresDst   (gr);k++){
		fn(gr, GA_SSIZE,  "TX",                       "ibdOff%d",    k, user);
	}
	for(k=0;k < gr->log2MaxL && reduxGenRequiresDstArg(gr);k++){
		fn(gr, GA_SSIZE,  "TX",                       "ibaOff%d",    k, user);
	}
}

/**
 * @brief Generate the kernel source code for the reduction.
 *
 * @return GA_MEMORY_ERROR if not enough memory left; GA_NO_ERROR otherwise.
 */

static int        reduxGenSrc                   (GpuReduction*     gr){
	reduxGenSrcAppend(gr);

	gr->kSourceCodeLen = gr->s.l;
	gr->kSourceCode    = strb_cstr(&gr->s);

	if (gr->kSourceCode){
		return reduxGenCompile(gr);
	}else{
		return reduxGenCleanupMsg(gr, GA_MEMORY_ERROR,
		                          "Failure in source code string buffer allocation "
		                          "during codegen!\n");
	}
}

/**
 * @brief Append source code to the string buffer.
 */

static void       reduxGenSrcAppend             (GpuReduction*     gr){
	reduxGenSrcAppendIncludes      (gr);
	reduxGenSrcAppendMacroDefs     (gr);
	reduxGenSrcAppendTypedefs      (gr);
	reduxGenSrcAppendReduxKernel   (gr);
}
static void       reduxGenSrcAppendIncludes     (GpuReduction*     gr){
	srcbAppends(&gr->srcGen, "/* Includes */\n");
	srcbAppends(&gr->srcGen, "#include \"cluda.h\"\n");
	srcbAppends(&gr->srcGen, "\n");
	srcbAppends(&gr->srcGen, "\n");
	srcbAppends(&gr->srcGen, "\n");
}
static void       reduxGenSrcAppendMacroDefs    (GpuReduction*     gr){
	int i;
	
	/**
	 * DECLREDUXSTATE, INITREDUXSTATE and SETREDUXSTATE macros.
	 */
	
	if       ( reduxGenKernelRequiresDst(gr) &&  reduxGenKernelRequiresDstArg(gr)){
		srcbAppendf(&gr->srcGen,
		            "#define DECLREDUXSTATE(V, I) TK V;TX I;\n"
		            "#define INITREDUXSTATE(V, I) do{(V) = %s;(I) = 0;}while(0)\n"
		            "#define SETREDUXSTATE(V, I, v, i)  do{(V) = (v);(I) = (i);}while(0)\n",
		            gr->initVal);
	}else if ( reduxGenKernelRequiresDst(gr) && !reduxGenKernelRequiresDstArg(gr)){
		srcbAppendf(&gr->srcGen,
		            "#define DECLREDUXSTATE(V, I) TK V;\n"
		            "#define INITREDUXSTATE(V, I) do{(V) = %s;}while(0)\n"
		            "#define SETREDUXSTATE(V, I, v, i)  do{(V) = (v);}while(0)\n",
		            gr->initVal);
	}else if (!reduxGenKernelRequiresDst(gr) &&  reduxGenKernelRequiresDstArg(gr)){
		srcbAppendf(&gr->srcGen,
		            "#define DECLREDUXSTATE(V, I) TX I;\n"
		            "#define INITREDUXSTATE(V, I) do{(I) = 0;}while(0)\n"
		            "#define SETREDUXSTATE(V, I, v, i)  do{(I) = (i);}while(0)\n");
	}
	
	
	/**
	 * LOADS(v, p) macro.
	 * 
	 * Loads a TK-typed value v from a TS-typed source pointer p.
	 */
	
	if (gr->srcTypeCode == GA_HALF && gr->accTypeCode == GA_FLOAT){
		srcbAppends(&gr->srcGen, "#define LOADS(v, p) do{(v) = (TK)load_half((TS*)(p));}while(0)\n");
	}else{
		srcbAppends(&gr->srcGen, "#define LOADS(v, p) do{(v) = (TK)*(TS*)(p);}while(0)\n");
	}
	
	
	/**
	 * GETIDX macro.
	 * 
	 * Expands to the current flattened index.
	 */
	
	srcbAppends    (&gr->srcGen, "#define GETIDX   (");
	srcbBeginList  (&gr->srcGen, " + ", "0");
	srcbAppendElemf(&gr->srcGen, "ti");
	for(i=gr->ndd;i<gr->nds;i++){
		srcbAppendElemf(&gr->srcGen, "i%d*l%dPDim", i, i);
	}
	srcbEndList    (&gr->srcGen);
	srcbAppends    (&gr->srcGen, ")\n");
	
	/**
	 * REDUX macro.
	 * 
	 * Performs a reduction operation, jointly reducing a datum v and its
	 * flattened index i into reduction states V and I respectively.
	 */
	
	srcbAppends(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n");
	switch (gr->op){
		case GA_REDUCE_SUM:
		  srcbAppendf(&gr->srcGen, "        (V) += (v);                     \\\n");
		break;
		case GA_REDUCE_PROD:
		  srcbAppendf(&gr->srcGen, "        (V) *= (v);                     \\\n");
		break;
		case GA_REDUCE_PRODNZ:
		  srcbAppendf(&gr->srcGen, "        (V) *= ((v) == 0 ? (%s) : (v)); \\\n", gr->initVal);
		break;
		case GA_REDUCE_MIN:
		  srcbAppendf(&gr->srcGen, "    (V)  = min((V), (v));           \\\n");
		break;
		case GA_REDUCE_MAX:
		  srcbAppendf(&gr->srcGen, "        (V)  = max((V), (v));           \\\n");
		break;
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_MINANDARGMIN:
		  srcbAppendf(&gr->srcGen, "        (V)  = min((V), (v));           \\\n"
		                           "        if((V) == (v)){                 \\\n"
		                           "            (I) = (i);                  \\\n"
		                           "        }                               \\\n");
		break;
		case GA_REDUCE_ARGMAX:
		case GA_REDUCE_MAXANDARGMAX:
		  srcbAppendf(&gr->srcGen, "        (V)  = max((V), (v));           \\\n"
		                           "        if((V) == (v)){                 \\\n"
		                           "            (I) = (i);                  \\\n"
		                           "        }                               \\\n");
		break;
		case GA_REDUCE_AND:
		  srcbAppendf(&gr->srcGen, "        (V) &= (v);                     \\\n");
		break;
		case GA_REDUCE_OR:
		  srcbAppendf(&gr->srcGen, "        (V) |= (v);                     \\\n");
		break;
		case GA_REDUCE_XOR:
		  srcbAppendf(&gr->srcGen, "        (V) ^= (v);                     \\\n");
		break;
		case GA_REDUCE_ALL:
		  srcbAppendf(&gr->srcGen, "        (V)  = (V) && (v);              \\\n");
		break;
		case GA_REDUCE_ANY:
		  srcbAppendf(&gr->srcGen, "        (V)  = (V) || (v);              \\\n");
		break;
	}
	srcbAppends(&gr->srcGen, "    }while(0)\n");
	
	
	/**
	 * HREDUX macro.
	 * 
	 * Performs a horizontal reduction operation, first intra-block permuting
	 * the data and its index and then reducing it till done.
	 */
	
	srcbAppends(&gr->srcGen,
	"#define HREDUX(pd, pa, tp, V, I)                                                    \\\n"
	"    do{                                                                             \\\n"
	"        /* Horizontal Reduction */                                                  \\\n"
	"        SETREDUXSTATE(pd[tp], pa[tp], accV, accI);                                  \\\n"
	"        local_barrier();                                                            \\\n"
	"                                                                                    \\\n"
	"        h = H;                                                                      \\\n"
	"        while(h>1){                                                                 \\\n"
	"            if((h&1) && (LID_0 < D)){                                               \\\n"
	"                REDUX(pd[LID_0], pa[LID_0], pd[LID_0 + D*h-D], pa[LID_0 + D*h-D]);  \\\n"
	"            }                                                                       \\\n"
	"            h >>= 1;                                                                \\\n"
	"            if(LID_0 < D*h){                                                        \\\n"
	"                REDUX(pd[LID_0], pa[LID_0], pd[LID_0 + D*h  ], pa[LID_0 + D*h  ]);  \\\n"
	"            }                                                                       \\\n"
	"            local_barrier();                                                        \\\n"
	"        }                                                                           \\\n"
	"    }while(0)\n");
	
	/**
	 * STORED macro.
	 * 
	 * Stores a TK-typed value v into a TS-typed destination pointer p.
	 */
	
	if (reduxGenRequiresDst(gr)){
		if (gr->dstTypeCode == GA_HALF && gr->accTypeCode == GA_FLOAT){
			srcbAppends(&gr->srcGen, "#define STORED(p, v) do{store_half((TD*)(p), (v));}while(0)\n");
		}else{
			srcbAppends(&gr->srcGen, "#define STORED(p, v) do{*(TD*)(p) = (v);}while(0)\n");
		}
	}else{
		srcbAppends(&gr->srcGen, "#define STORED(p, v) do{}while(0)\n");
	}
	
	
	/**
	 * STOREA macro.
	 * 
	 * Stores a TX-typed value v into a TA-typed destination pointer p.
	 */
	
	if (reduxGenRequiresDstArg(gr)){
		srcbAppends(&gr->srcGen, "#define STOREA(p, v) do{*(TA*)(p) = (v);}while(0)\n");
	}else{
		srcbAppends(&gr->srcGen, "#define STOREA(p, v) do{}while(0)\n");
	}
	
	
	/**
	 * DIVIDECEIL macro.
	 */
	
	srcbAppends(&gr->srcGen, "#define DIVIDECEIL(a,b) (((a)+(b)-1)/(b))\n");
	
	srcbAppends(&gr->srcGen, "\n\n\n\n");
}
static void       reduxGenSrcAppendTypedefs     (GpuReduction*     gr){
	srcbAppendf(&gr->srcGen, "typedef %-20s TS;\n", gr->srcTypeStr);
	srcbAppendf(&gr->srcGen, "typedef %-20s TD;\n", gr->dstTypeStr);
	srcbAppendf(&gr->srcGen, "typedef %-20s TA;\n", gr->dstArgTypeStr);
	srcbAppendf(&gr->srcGen, "typedef %-20s TX;\n", gr->idxTypeStr);
	srcbAppendf(&gr->srcGen, "typedef %-20s TK;\n", gr->accTypeStr);
	srcbAppendf(&gr->srcGen, "\n\n\n\n");
}
static void       reduxGenSrcAppendReduxKernel  (GpuReduction*     gr){
	reduxGenSrcAppendPrototype   (gr);
	srcbAppends                  (&gr->srcGen, "{\n");
	reduxGenSrcAppendBlockDecode (gr);
	reduxGenSrcAppendThreadDecode(gr);
	srcbAppends                  (&gr->srcGen, "    /**\n"
	                                           "     * PERFORM REDUCTION.\n"
	                                           "     * \n"
	                                           "     * We either perform Phase 0 or Phase 1 according to our argument.\n"
	                                           "     * \n"
	                                           "     * Phase 0 is the primary worker and, in special cases, is the only necessary phase.\n"
	                                           "     * However, it may occasionally do only part of a reduction, in which case it leaves\n"
	                                           "     * the partial reduction results in a workspace that is then read by Phase 1.\n"
	                                           "     * \n"
	                                           "     * Phase 1 is a fixup phase that collects any partial reduction results from Phase 0\n"
	                                           "     * and completes the reduction before writing to the final destination.\n"
	                                           "     */\n"
	                                           "    \n"
	                                           "    if(phase==0){\n");
	reduxGenSrcAppendPhase0      (gr);
	srcbAppends                  (&gr->srcGen, "    }else{\n");
	reduxGenSrcAppendPhase1      (gr);
	srcbAppends                  (&gr->srcGen, "    }\n");
	srcbAppends                  (&gr->srcGen, "}\n");
}
static void       reduxGenSrcAppendPrototype    (GpuReduction*     gr){
	int i=0;
	
	srcbAppends            (&gr->srcGen, "KERNEL void redux(");
	reduxGenIterArgs(gr, reduxGenAppendArg, &i);
	srcbAppends    (&gr->srcGen, ")");
}
static void       reduxGenSrcAppendBlockDecode  (GpuReduction*     gr){
	int i;
	
	srcbAppends(&gr->srcGen,
	"    GA_DECL_SHARED_BODY(char, SHMEM)\n"
	"    DECLREDUXSTATE(accV, accI)\n"
	"    DECLREDUXSTATE(tmpV, tmpI)\n"
	"    INITREDUXSTATE(accV, accI);\n"
	"    \n"
	"     /**\n"
	"      *  +-------------+-------------+------------+---------------------------------+\n"
	"      *  |  misalignL  |  misalignR  |  doFinish  |            DESCRIPTION          |\n"
	"      *  +-------------+-------------+------------+---------------------------------+\n"
	"      *  |      0      |       0     |      0     |  Impossible unless v == 0,      |\n"
	"      *  |             |             |            |  which is forbidden.            |\n"
	"      *  |             |             |            |                                 |\n"
	"      *  |      0      |       0     |      1     |  V % B == 0. Each block         |\n"
	"      *  |             |             |            |  handles integer number of      |\n"
	"      *  |             |             |            |  destination elements, no       |\n"
	"      *  |             |             |            |  partial results are required,  |\n"
	"      *  |             |             |            |  workspace is unused.           |\n"
	"      *  |             |             |            |                                 |\n"
	"      *  |      0      |       1     |      0     |  V < B. Block begins aligned    |\n"
	"      *  |             |             |            |  but ends misaligned, before    |\n"
	"      *  |             |             |            |  the end of its first element.  |\n"
	"      *  |             |             |            |  Partial result written to      |\n"
	"      *  |             |             |            |  right-half of array.           |\n"
	"      *  |             |             |            |                                 |\n"
	"      *  |      0      |       1     |      1     |  V > B, V % B != 0. Block       |\n"
	"      *  |             |             |            |  begins aligned but ends        |\n"
	"      *  |             |             |            |  misaligned, after the end of   |\n"
	"      *  |             |             |            |  its first element.             |\n"
	"      *  |             |             |            |  First 1 or more complete       |\n"
	"      *  |             |             |            |  elements written out directly  |\n"
	"      *  |             |             |            |  to destination.                |\n"
	"      *  |             |             |            |  Partial result of last element |\n"
	"      *  |             |             |            |  written to right-half of array.|\n"
	"      *  |             |             |            |                                 |\n"
	"      *  |      1      |       0     |      0     |  Impossible unless v == 0,      |\n"
	"      *  |             |             |            |  which is forbidden.            |\n"
	"      *  |             |             |            |                                 |\n"
	"      *  |      1      |       0     |      1     |  V % B != 0. Partial result of  |\n"
	"      *  |             |             |            |  first element written to left- |\n"
	"      *  |             |             |            |  half of array. Zero or more    |\n"
	"      *  |             |             |            |  complete reductions performed  |\n"
	"      *  |             |             |            |  and written directly to        |\n"
	"      *  |             |             |            |  destination. Block ends        |\n"
	"      *  |             |             |            |  aligned.                       |\n"
	"      *  |             |             |            |                                 |\n"
	"      *  |      1      |       1     |      0     |  V < B. Block begins misaligned |\n"
	"      *  |             |             |            |  and ends misaligned, before    |\n"
	"      *  |             |             |            |  the end of its first element.  |\n"
	"      *  |             |             |            |  Partial result written to at   |\n"
	"      *  |             |             |            |  least right-half of array.     |\n"
	"      *  |             |             |            |                                 |\n"
	"      *  |      1      |       1     |      1     |  V % B != 0. Block begins       |\n"
	"      *  |             |             |            |  misaligned and ends misaligned,|\n"
	"      *  |             |             |            |  after the end of its first     |\n"
	"      *  |             |             |            |  element.                       |\n"
	"      *  |             |             |            |  Partial result of first element|\n"
	"      *  |             |             |            |  written to left-half of array. |\n"
	"      *  |             |             |            |  Partial result of last element |\n"
	"      *  |             |             |            |  written to right-half of array.|\n"
	"      *  |             |             |            |  0 or more complete elements    |\n"
	"      *  |             |             |            |  written out directly to        |\n"
	"      *  |             |             |            |  destination.                   |\n"
	"      *  +-------------+-------------+------------+---------------------------------+\n"
	"      * \n"
	"      * Possible configurations of blocks:\n"
	"      *   If V % B == 0:  001\n"
	"      *   If V < B:       010, 110, 111, 101\n"
	"      *   If V > B:       011, 111, 101\n"
	"      * \n"
	"      * Possible configurations for collector blocks (responsible for gathering of\n"
	"      * results to the left):\n"
	"      *   101, 111          (misalignL && doFinish)\n"
	"      * \n"
	"      * Possible configurations for left-neighbours of collector blocks\n"
	"      *   110 (any number 0+), then exactly one of:\n"
	"      *   010, 011, 111\n"
	"      * \n"
	"      * Conclusion:\n"
	"      *     - In Phase 0:\n"
	"      *         - Always make a right-write if misalignR                (010, 011, 110, 111).\n"
	"      *         - Make        a left -write at least if collector block (101, 111).\n"
	"      *     - In Phase 1:\n"
	"      *         - Exit if not collector block (101, 111)\n"
	"      *         - If collector block,\n"
	"      *             - Left -read from self\n"
	"      *             - Right-read from all left-neighbours with same write-target.\n"
	"      * \n"
	"      * Code Structure perfectly satisfying conclusion:\n"
	"      * \n"
	"      * if(misalignL){\n"
	"      *     while(v > 0){\n"
	"      *         v--;\n"
	"      *         REDUX();\n"
	"      *         ReduxLoopIncs_CONTINUE;\n"
	"      *         HREDUX();\n"
	"      *         WSLeftWrite();\n"
	"      *         REINIT();\n"
	"      *         FreeLoopIncs_BREAK;\n"
	"      *         BREAK;\n"
	"      *     }\n"
	"      * }\n"
	"      * while(v > 0){\n"
	"      *     v--;\n"
	"      *     REDUX();\n"
	"      *     ReduxLoopIncs_CONTINUE;\n"
	"      *     HREDUX();\n"
	"      *     DstWrite();\n"
	"      *     REINIT();\n"
	"      *     FreeLoopIncs_CONTINUE;\n"
	"      *     BREAK;\n"
	"      * }\n"
	"      * if(misalignR){\n"
	"      *     HREDUX();\n"
	"      *     WSRightWrite();\n"
	"      * }\n"
	"      * \n"
	"      * Code Walkthrough:\n"
	"      * \n"
	"      * 000, 100: Impossible, can be ignored.\n"
	"      * 001:      Only master loop entered, handles exact integer number of destinations.\n"
	"      * 010:      Master loop entered but broken on vcount before HREDUX() reached.\n"
	"      *           No reinit executed on breakout. HREDUX(), followed by WSRightWrite() of\n"
	"      *           partial result.\n"
	"      * 011:      Master loop entered for at least 1 full destination, then broken on\n"
	"      *           vcount before HREDUX() reached. No reinit executed on breakout. HREDUX()\n"
	"      *           followed by WSRightWrite() of partial result.\n"
	"      * 101:      Left-misalign loop entered and completes a reduction. HREDUX()\n"
	"      *           performed, WSLeftWrite() performed, reinitialization, bump of outer\n"
	"      *           loop counters, then breakout. Master loop entered for 0 or more complete\n"
	"      *           destination elements involving full writeouts to destination and reinit.\n"
	"      *           Aligned on both misalignL and master loop breakouts. No entry into\n"
	"      *           misalignR fixup.\n"
	"      * 110:      Left-misalign loop entered, breaks on vcount before HREDUX(). No reinit\n"
	"      *           executed on breakout. Master loop not entered. HREDUX(), followed by\n"
	"      *           WSRightWrite() of partial result.\n"
	"      * 111:      Left-misalign loop entered and completes a reduction. HREDUX() performed,\n"
	"      *           WSLeftWrite() performed, reinit, bump of outer loop counters, breakout.\n"
	"      *           Master loop entered for 0 or more complete destination elements\n"
	"      *           involving full writeout to destination and reinit.\n"
	"      *           Master loop broken on vcount before HREDUX(). misalignR fixup entered,\n"
	"      *           HREDUX(), WSRightWrite().\n"
	"      */\n"
	"    \n"
	"    TX      start        = GID_0 * V;\n"
	"    if(start >= U){return;}\n"
	"    TX      v            = U-start < V ? U-start : V;\n"
	"    \n"
	"    int     misalignL    = (start+0)%B != 0;\n"
	"    int     misalignR    = (start+v)%B != 0;\n"
	"    int     doFinish     = (start+0)/B != (start+v)/B;\n"
	"    \n"
	"    /**\n"
	"     * Decode BLOCK start point.\n"
	"     * \n"
	"     * For the purpose of decoding the start point, the split axis's \"length\"\n"
	"     * is divided by either splitReduce or splitFree and rounded up. Therefore,\n"
	"     * for those axes the true computed initial starting point must be\n"
	"     * multiplied by either splitReduce or splitFree.\n"
	"     * \n"
	"     * Since we provide not strides but \"jumps\" to the kernel (to move as many\n"
	"     * things as possible into constant memory and out of the fast path), we\n"
	"     * must also convert jumps to strides in preparation for offsetting the\n"
	"     * base pointers to their starting point.\n"
	"     */\n"
	"    \n"
	"    TX          z, h, k;\n"
	"    unsigned    Dunit    = D/splitFree;\n");
	if(gr->ndd > 0){
		srcbAppendf(&gr->srcGen,
		"    TX          l%dDiv    = DIVIDECEIL(l%d, splitFree);\n",
		            gr->ndd-1, gr->ndd-1);
	}
	if(gr->ndr > 0){
		srcbAppendf(&gr->srcGen,
		"    TX          l%dDiv    = DIVIDECEIL(l%d, splitReduce);\n",
		            gr->nds-1, gr->nds-1);
	}
	srcbAppends(&gr->srcGen,
	"    \n"
	"    z                    = start;\n");
	for(i=gr->nds-1;i>=0;i--){
		if(i == gr->nds-1 || i == gr->ndd-1){
			srcbAppendf(&gr->srcGen,
			"    TX          i%d       = z %% l%dDiv;z /= l%dDiv;\n",
			            i, i, i);
		}else{
			srcbAppendf(&gr->srcGen,
			"    TX          i%d       = z %% l%d;   z /= l%d;\n",
			            i, i, i);
		}
	}
	srcbAppends(&gr->srcGen, "    \n");
	for(i=gr->nds-1;i>=0;i--){
		if(i == gr->nds-1){
			srcbAppendf(&gr->srcGen,
			"    TX          sS%d      = sJ%d;\n",
			            i, i);
		}else{
			srcbAppendf(&gr->srcGen,
			"    TX          sS%d      = sJ%d + l%d%s*sS%d;\n",
			            i, i, i+1,
			            reduxGenAxisMaybeSplit(gr, i+1) ? "Div" : "   ", i+1);
		}
	}
	if (reduxGenRequiresDst(gr)){
		srcbAppends(&gr->srcGen, "    \n");
		for(i=gr->ndd-1;i>=0;i--){
			if(i == gr->ndd-1){
				srcbAppendf(&gr->srcGen,
				"    TX          dS%d      = dJ%d;\n",
				            i, i);
			}else{
				srcbAppendf(&gr->srcGen,
				"    TX          dS%d      = dJ%d + l%d%s*dS%d;\n",
				            i, i, i+1,
				            reduxGenAxisMaybeSplit(gr, i+1) ? "Div" : "   ", i+1);
			}
		}
	}
	if (reduxGenRequiresDstArg(gr)){
		srcbAppends(&gr->srcGen, "    \n");
		for(i=gr->ndd-1;i>=0;i--){
			if(i == gr->ndd-1){
				srcbAppendf(&gr->srcGen,
				"    TX          aS%d      = aJ%d;\n",
				            i, i);
			}else{
				srcbAppendf(&gr->srcGen,
				"    TX          aS%d      = aJ%d + l%d%s*aS%d;\n",
				            i, i, i+1,
				            reduxGenAxisMaybeSplit(gr, i+1) ? "Div" : "   ", i+1);
			}
		}
	}
	srcbAppends(&gr->srcGen, "    \n");
	srcbAppends(&gr->srcGen, "    sOff                += ");
	srcbBeginList(&gr->srcGen, " + ", "0");
	for(i=0;i<gr->nds;i++){
		srcbAppendElemf(&gr->srcGen, "(TX)i%d*sS%d", i, i);
	}
	srcbEndList(&gr->srcGen);
	srcbAppends(&gr->srcGen, ";\n");
	if (reduxGenRequiresDst(gr)){
		srcbAppends(&gr->srcGen, "    dOff                += ");
		srcbBeginList(&gr->srcGen, " + ", "0");
		for(i=0;i<gr->ndd;i++){
			srcbAppendElemf(&gr->srcGen, "(TX)i%d*dS%d", i, i);
		}
		srcbEndList(&gr->srcGen);
		srcbAppends(&gr->srcGen, ";\n");
	}
	if (reduxGenRequiresDstArg(gr)){
		srcbAppends(&gr->srcGen, "    aOff                += ");
		srcbBeginList(&gr->srcGen, " + ", "0");
		for(i=0;i<gr->ndd;i++){
			srcbAppendElemf(&gr->srcGen, "(TX)i%d*aS%d", i, i);
		}
		srcbEndList(&gr->srcGen);
		srcbAppends(&gr->srcGen, ";\n");
	}
	srcbAppends(&gr->srcGen, "    \n");
	if(gr->ndd > 0){
		srcbAppendf(&gr->srcGen,
		"    i%d                  *= splitFree;\n",
		            gr->ndd-1);
	}
	if(gr->ndr > 0){
		srcbAppendf(&gr->srcGen,
		"    i%d                  *= splitReduce;\n",
	                gr->nds-1);
	}
	srcbAppends(&gr->srcGen, "    \n");
	if(reduxGenKernelRequiresDst(gr)){
		srcbAppends(&gr->srcGen,
		"    TK*         wd       = (TK*)(w     + wdOff);\n"
		"    TK*         wdL      = &wd[0];\n"
		"    TK*         wdR      = &wd[GDIM_0*D];\n"
		"    TK*         pd       = (TK*)(SHMEM + pdOff);\n");
	}
	if(reduxGenKernelRequiresDstArg(gr)){
		srcbAppends(&gr->srcGen,
		"    TA*         wa       = (TA*)(w     + waOff);\n"
		"    TA*         waL      = &wa[0];\n"
		"    TA*         waR      = &wa[GDIM_0*D];\n"
		"    TA*         pa       = (TA*)(SHMEM + paOff);\n");
	}
	srcbAppends(&gr->srcGen, "    \n");
}
static void       reduxGenSrcAppendThreadDecode (GpuReduction*     gr){
	int i;

	srcbAppends(&gr->srcGen,
	"    /**\n"
	"     * Decode THREAD start point.\n"
	"     * \n"
	"     * This involves computing the intra-block coordinate of a thread in a\n"
	"     * up-to-log2(MAX_BLOCK_THREADS)-dimensional coordinate system, then using\n"
	"     * those coordinates to compute private source/destination/destination\n"
	"     * argument pointers, argument indices and permute targets.\n"
	"     */\n"
	"    \n"
	"    unsigned    iSplit   = LID_0/(LDIM_0/(splitFree*splitReduce));\n"
	"    z                    = LID_0;\n");
	
	for(i=gr->log2MaxL-1;i>=0;i--){
		srcbAppendf(&gr->srcGen,
		"    int         t%d       = z %% ibs%d;z /= ibs%d;\n",
		            i, i, i);
	}
	if(reduxGenRequiresDstArg(gr)){
		srcbAppends(&gr->srcGen, "    TX          ti       = ");
		srcbBeginList(&gr->srcGen, " + ", "0");
		for(i=0;i<gr->log2MaxL;i++){
			srcbAppendElemf(&gr->srcGen, "t%d*ibl%dPDim", i, i);
		}
		srcbEndList(&gr->srcGen);
		srcbAppends(&gr->srcGen, ";\n");
	}
	srcbAppends(&gr->srcGen, "    unsigned    tp       = ");
	srcbBeginList(&gr->srcGen, " + ", "0");
	for(i=0;i<gr->log2MaxL;i++){
		srcbAppendElemf(&gr->srcGen, "t%d*    ibp%d", i, i);
	}
	srcbEndList(&gr->srcGen);
	srcbAppends(&gr->srcGen, ";\n");
	srcbAppends(&gr->srcGen, "    \n"
	                         "    sOff                += ");
	srcbBeginList(&gr->srcGen, " + ", "0");
	for(i=0;i<gr->log2MaxL;i++){
		srcbAppendElemf(&gr->srcGen, "t%d*ibsOff%d ", i, i);
	}
	srcbEndList(&gr->srcGen);
	srcbAppends(&gr->srcGen, ";\n");
	if(reduxGenRequiresDst(gr)){
		srcbAppends(&gr->srcGen, "    \n"
		                         "    dOff                += ");
		srcbBeginList(&gr->srcGen, " + ", "0");
		for(i=0;i<gr->log2MaxL;i++){
			srcbAppendElemf(&gr->srcGen, "t%d*ibdOff%d ", i, i);
		}
		srcbEndList(&gr->srcGen);
		srcbAppends(&gr->srcGen, ";\n");
		srcbAppends(&gr->srcGen, "    ((TX*)SHMEM)[tp]     = dOff;\n"
		                         "    local_barrier();\n"
		                         "    dOff                 = ((TX*)SHMEM)[LID_0];\n"
		                         "    local_barrier();\n");
	}
	if(reduxGenRequiresDstArg(gr)){
		srcbAppends(&gr->srcGen, "    \n"
		                         "    aOff                += ");
		srcbBeginList(&gr->srcGen, " + ", "0");
		for(i=0;i<gr->log2MaxL;i++){
			srcbAppendElemf(&gr->srcGen, "t%d*ibaOff%d ", i, i);
		}
		srcbEndList(&gr->srcGen);
		srcbAppends(&gr->srcGen, ";\n");
		srcbAppends(&gr->srcGen, "    ((TX*)SHMEM)[tp]     = aOff;\n"
		                         "    local_barrier();\n"
		                         "    aOff                 = ((TX*)SHMEM)[LID_0];\n"
		                         "    local_barrier();\n");
	}
	srcbAppends(&gr->srcGen, "    \n"
	                         "    const char* ts       = s + sOff;\n");
	if(reduxGenRequiresDst(gr)){
		srcbAppends(&gr->srcGen, "    char*       td       = d + dOff;\n");
	}
	if(reduxGenRequiresDstArg(gr)){
		srcbAppends(&gr->srcGen, "    char*       ta       = a + aOff;\n");
	}
	srcbAppends(&gr->srcGen, "    \n"
	                         "    \n");
}
static void       reduxGenSrcAppendPhase0       (GpuReduction*     gr){
	srcbAppends(&gr->srcGen,
	"        /* PHASE 0 */\n"
	"        \n"
	"        /* Loop Cores. */\n");
	if (gr->ndd == 0){
		/**
		 * Special case: If ndd == 0, we know this is an all-reduce or nearly, so
		 * we know that the only split axis, if any, is going to be a reduction axis.
		 * Therefore, splitFree will always be 1, and we only need to generate one
		 * set of loops.
		 */
		
		reduxGenSrcAppendLoops(gr, 0, 1);
	}else{
		srcbAppends(&gr->srcGen, "        if(splitReduce == 1){\n"
		                         "            /* Free   axis possibly split. */\n");
		reduxGenSrcAppendLoops(gr, 1, 0);
		srcbAppends(&gr->srcGen, "        }else{\n"
		                         "            /* Reduce axis possibly split. */\n");
		reduxGenSrcAppendLoops(gr, 0, 1);
		srcbAppends(&gr->srcGen, "        }\n");
	}
}
static void       reduxGenSrcAppendLoops        (GpuReduction*        gr,
                                                 int                  freeMaybeSplit,
                                                 int                  reduceMaybeSplit){
	srcbAppends(&gr->srcGen, "            if(misalignL){\n");
	reduxGenSrcAppendLoop(gr, 1, freeMaybeSplit, reduceMaybeSplit);
	srcbAppends(&gr->srcGen, "            }\n");
	reduxGenSrcAppendLoop(gr, 0, freeMaybeSplit, reduceMaybeSplit);
	srcbAppends(&gr->srcGen,
	"            \n"
	"            /**\n"
	"             * Are we misaligned on the right? If so, we have a partial reduction\n"
	"             * to save.\n"
	"             */\n"
	"            \n"
	"            if(misalignR){\n"
	"                HREDUX(pd, pa, tp, accV, accI);\n"
	"                \n"
	"                /* Right-write partial reduction to workspace. */\n"
	"                if(LID_0 < D){\n"
	"                    SETREDUXSTATE(wdR[GID_0*D+LID_0], waR[GID_0*D+LID_0], pd[LID_0], pa[LID_0]);\n"
	"                }\n"
	"            }\n");
}
static void       reduxGenSrcAppendLoop         (GpuReduction*        gr,
                                                 int                  initial,
                                                 int                  freeMaybeSplit,
                                                 int                  reduceMaybeSplit){
	int i;
	
	srcbAppends(&gr->srcGen, "            while(v > 0){\n");
	reduxGenSrcAppendDecrement(gr);
	reduxGenSrcAppendVertical (gr, freeMaybeSplit, reduceMaybeSplit);
	srcbAppends(&gr->srcGen, "                /* Reduction Increments */\n");
	for(i=gr->nds-1;i >= gr->ndd;i--){
		reduxGenSrcAppendIncrement(gr, i, initial, freeMaybeSplit, reduceMaybeSplit);
	}
	srcbAppends(&gr->srcGen, "                /* Horizontal Reduction */\n"
	                         "                HREDUX(pd, pa, tp, accV, accI);\n"
	                         "                \n");
	reduxGenSrcAppendDstWrite(gr, initial, freeMaybeSplit, reduceMaybeSplit);
	srcbAppends(&gr->srcGen, "                /* Reinitialize accumulators */\n"
	                         "                INITREDUXSTATE(accV, accI);\n"
	                         "                \n");
	srcbAppends(&gr->srcGen, "                /* Free Increments */\n");
	for(i=gr->ndd-1;i >= 0;i--){
		reduxGenSrcAppendIncrement(gr, i, initial, freeMaybeSplit, reduceMaybeSplit);
	}
	srcbAppends(&gr->srcGen, "                /* Exit loop */\n"
	                         "                break;\n"
	                         "            }\n");
}
static void       reduxGenSrcAppendDecrement    (GpuReduction*        gr){
	srcbAppends(&gr->srcGen, "                /* Decrement. */\n"
	                         "                v--;\n"
	                         "                \n");
}
static void       reduxGenSrcAppendVertical     (GpuReduction*        gr,
                                                 int                  freeMaybeSplit,
                                                 int                  reduceMaybeSplit){
	int i;
	
	if(!freeMaybeSplit && !reduceMaybeSplit){
		srcbAppends(&gr->srcGen, "                /* Vertical Reductions */\n"
		                         "                LOADS(tmpV, ts);\n"
		                         "                REDUX(accV, accI, tmpV, GETIDX);\n"
		                         "                \n");
	}else{
		i = freeMaybeSplit ? gr->ndd-1 : gr->nds-1;
		srcbAppendf(&gr->srcGen, "                /* Vertical Reductions */\n"
		                         "                if(i%d+iSplit < l%d){\n"
		                         "                    LOADS(tmpV, ts);\n"
		                         "                    REDUX(accV, accI, tmpV, GETIDX);\n"
		                         "                }\n"
		                         "                \n", i, i);
	}
}
static void       reduxGenSrcAppendIncrement    (GpuReduction*        gr,
                                                 int                  axis,
                                                 int                  initial,
                                                 int                  freeMaybeSplit,
                                                 int                  reduceMaybeSplit){
	const char* breakOrCont = (initial) && (axis < gr->ndd) ? "break" : "continue";
	
	if       (freeMaybeSplit   && axis == gr->ndd-1){
		srcbAppendf(&gr->srcGen,
		"                i%d += splitFree;\n"
		"                ts += sJ%d;",
		            axis, axis);
		if(reduxGenRequiresDst(gr)){
			srcbAppendf(&gr->srcGen, "td += dJ%d;", axis);
		}
		if(reduxGenRequiresDstArg(gr)){
			srcbAppendf(&gr->srcGen, "ta += aJ%d;", axis);
		}
		srcbAppends(&gr->srcGen, "\n");
		srcbAppendf(&gr->srcGen,
		"                if  (i%d < l%d){%s;}\n"
		"                else         {i%d = 0;}\n"
		"                \n",
		            axis, axis, breakOrCont, axis);
	}else if (reduceMaybeSplit && axis == gr->nds-1){
		srcbAppendf(&gr->srcGen,
		"                i%d += splitReduce;\n"
		"                ts += sJ%d;\n"
		"                if  (i%d < l%d){%s;}\n"
		"                else         {i%d = 0;}\n"
		"                \n",
		            axis, axis, axis, axis, breakOrCont, axis);
	}else{
		srcbAppendf(&gr->srcGen,
		"                i%d++;\n"
		"                ts += sJ%d;",
		            axis, axis);
		if(axis < gr->ndd){
			if(reduxGenRequiresDst(gr)){
				srcbAppendf(&gr->srcGen, "td += dJ%d;", axis);
			}
			if(reduxGenRequiresDstArg(gr)){
				srcbAppendf(&gr->srcGen, "ta += aJ%d;", axis);
			}
		}
		srcbAppends(&gr->srcGen, "\n");
		srcbAppendf(&gr->srcGen,
		"                if  (i%d < l%d){%s;}\n"
		"                else         {i%d = 0;}\n"
		"                \n",
		            axis, axis, breakOrCont, axis);
	}
}
static void       reduxGenSrcAppendDstWrite     (GpuReduction*        gr,
                                                 int                  initial,
                                                 int                  freeMaybeSplit,
                                                 int                  reduceMaybeSplit){
	if(initial){
		srcbAppends(&gr->srcGen, "                /* Workspace Left-Write */\n"
		                         "                if(LID_0 < D){\n"
		                         "                    SETREDUXSTATE(wdL[GID_0*D + LID_0], waL[GID_0*D + LID_0], pd[LID_0], pa[LID_0]);\n"
		                         "                }\n"
		                         "                \n");
	}else{
		if(!freeMaybeSplit){
			srcbAppends(&gr->srcGen, "                /* Destination Write */\n"
			                         "                if(LID_0 < D){\n"
			                         "                    STORED(td, pd[LID_0]);\n"
			                         "                    STOREA(ta, pa[LID_0]);\n"
			                         "                }\n"
			                         "                \n");
		}else{
			if(gr->ndd > 0){
				srcbAppendf(&gr->srcGen, "                /* Destination Write */\n"
				                         "                if(LID_0 < (l%d-i%d<splitFree ? (l%d-i%d)*Dunit : D)){\n"
				                         "                    STORED(td, pd[LID_0]);\n"
				                         "                    STOREA(ta, pa[LID_0]);\n"
				                         "                }\n"
				                         "                \n",
				            gr->ndd-1, gr->ndd-1, gr->ndd-1, gr->ndd-1);
			}else{
				srcbAppendf(&gr->srcGen, "                STORED(td, pd[LID_0]);\n"
				                         "                STOREA(ta, pa[LID_0]);\n");
			}
		}
	}
}
static void       reduxGenSrcAppendPhase1       (GpuReduction*        gr){
	srcbAppends(&gr->srcGen,
	"        /* PHASE 1 */\n"
	"        \n"
	"        /**\n"
	"         * If we are a collector block, gather all partial results for the\n"
	"         * same point to the left of the current position in our workspace\n"
	"         * and accumulate them into our partial result, then write out to\n"
	"         * destination/destination argument.\n"
	"         * We perform a left-read of our workspace and a right-read of the\n"
	"         * other blocks' workspace.\n"
	"         */\n"
	"        \n"
	"        if(misalignL && doFinish && LID_0 < D){\n"
	"            SETREDUXSTATE(accV, accI, wdL[(GID_0+0)*D+LID_0], waL[(GID_0+0)*D+LID_0]);\n"
	"            \n"
	"            for(k=-1;                /* Starting with the first block to our left... */\n"
	"                (start      +0)/B == /* Is our write target the same as that of */\n"
	"                (start+k*V+V-1)/B;   /* the target k blocks to our left? */\n"
	"                k--){                /* Try moving one more to the left. */\n"
	"                REDUX(accV, accI, wdR[(GID_0+k)*D+LID_0], waR[(GID_0+k)*D+LID_0]);\n"
	"            }\n"
	"            \n");
	if(gr->ndd > 0){
		srcbAppendf(&gr->srcGen,
		"            if(LID_0 < (l%d-i%d<splitFree ? (l%d-i%d)*Dunit : D)){\n"
		"                STORED(td, accV);\n"
		"                STOREA(ta, accI);\n"
		"            }\n",
		            gr->ndd-1, gr->ndd-1, gr->ndd-1, gr->ndd-1);
	}else{
		srcbAppends(&gr->srcGen,
		            "            STORED(td, accV);\n"
		            "            STOREA(ta, accI);\n");
	}
	srcbAppends(&gr->srcGen,
	            "        }\n");
}

/**
 * @brief Compile the generated kernel.
 */

static int        reduxGenCompile               (GpuReduction*     gr){
	int ret;
	
	ret  = GpuKernel_init(&gr->k,
	                      gr->gpuCtx,
	                      1,
	                      (const char**)&gr->kSourceCode,
	                      &gr->kSourceCodeLen,
	                      "redux",
	                      gr->kNumArgs,
	                      gr->kArgTypeCodes,
	                      GA_USE_CLUDA,
	                      &gr->kErrorString);

	if (ret != GA_NO_ERROR){
		return reduxGenCleanupMsg(gr, ret,
		       "Failed to compile reduction kernel!\n"
		       "Error code   is: %d\n"
		       "Error string is:\n"
		       "%s\n"
		       "Source code  is:\n"
		       "%s\n",
		       ret, gr->kErrorString, gr->kSourceCode);
	}
	
	return reduxGenComputeLaunchBounds(gr);
}

/**
 * @brief Compute the maximum number of threads this reduction operator will
 *        support launching.
 */

static int        reduxGenComputeLaunchBounds   (GpuReduction*        gr){
	int    ret;
	size_t a,b,c;
	
	/**
	 * Compute the maximum number of threads this kernel will support,
	 * since this is critical to the scheduling and will not change now
	 * that the kernel is compiled.
	 * 
	 * This depends on several exhaustible resources and isn't necessarily
	 * trivial to compute due to the complicated rules we must follow to
	 * align shared memory, possibly slightly increasing consumption.
	 */
	
	ret = gpukernel_property(gr->k.k, GA_KERNEL_PROP_MAXLSIZE,  &gr->maxLK);
	if(ret != GA_NO_ERROR){
		return reduxGenCleanupMsg(gr, ret,
		       "Failed to read max local size for compiled kernel!\n");
	}
	a         = gr->maxL0;
	b         = gr->maxLg;
	c         = gr->maxLM/reduxGenGetReduxStateSize(gr);
	                                       /* Kernel register use              */
	gr->maxLK = gr->maxLK<a ? gr->maxLK: a;/* Maximum block size on axis 0     */
	gr->maxLK = gr->maxLK<b ? gr->maxLK: b;/* Maximum total block size         */
	gr->maxLK = gr->maxLK<c ? gr->maxLK: c;/* Shared memory per thread.        */
	
	/**
	 * We now have a tight bound on the maximum block size, but due to memory
	 * alignment rules the memory consumption may be slightly higher than we
	 * initially computed, and thus the shared memory use can still be
	 * excessive. The following loop will almost certainly decrement at most
	 * once, unless type alignments are very wierd.
	 */
	
	while(reduxGenGetSHMEMSize(gr, gr->maxLK) > gr->maxLM){
		gr->maxLK--;
	}
	
	return reduxGenCleanup(gr, GA_NO_ERROR);
}

/**
 * @brief Cleanup generator context.
 */

static int        reduxGenCleanup               (GpuReduction*     gr,  int ret){
	if(ret != GA_NO_ERROR){
		free(gr->kArgTypeCodes);
		free(gr->kSourceCode);
		free(gr->kErrorString);
	
		memset(gr, 0, sizeof(*gr));
		free(gr);
	}

	return ret;
}
static int        reduxGenCleanupMsg            (GpuReduction*     gr,  int ret,
                                                 const char*       fmt, ...){
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
	
	return reduxGenCleanup(gr, ret);
}

/**
 * Count # of arguments as determined by iterator.
 */

static void       reduxGenCountArgs             (GpuReduction*        gr,
                                                 int                  typecode,
                                                 const char*          typeName,
                                                 const char*          baseName,
                                                 int                  num,
                                                 void*                user){
	(void)typecode;
	(void)typeName;
	(void)baseName;
	(void)num;
	(void)user;
	
	gr->kNumArgs++;
}

/**
 * Record the typecodes in the arguments typecode array.
 */

static void       reduxGenSaveArgTypecodes      (GpuReduction*        gr,
                                                 int                  typecode,
                                                 const char*          typeName,
                                                 const char*          baseName,
                                                 int                  num,
                                                 void*                user){
	(void)typeName;
	(void)baseName;
	(void)num;
	(void)user;
	
	gr->kArgTypeCodes[(*(int*)user)++] = typecode;
}

/**
 * Append an argument declaration to prototype.
 */

static void       reduxGenAppendArg             (GpuReduction*        gr,
                                                 int                  typecode,
                                                 const char*          typeName,
                                                 const char*          baseName,
                                                 int                  num,
                                                 void*                user){
	(void)user;
	(void)typecode;
	
	if((*(int*)user)++ > 0){
		srcbAppends(&gr->srcGen, ",\n                  ");
	}
	srcbAppendf(&gr->srcGen, "%-25s ", typeName);
	srcbAppendf(&gr->srcGen, baseName, num);
}

/**
 * Marshall argument declaration during invocation.
 */

static void       reduxInvMarshalArg            (GpuReduction*        gr,
                                                 int                  typecode,
                                                 const char*          typeName,
                                                 const char*          baseName,
                                                 int                  k,
                                                 void*                user){
	redux_ctx* ctx;
	int*       i;
	
	(void)typecode;
	(void)typeName;
	
	ctx = (redux_ctx*)(((void**)user)[0]);
	i   = (int      *)(((void**)user)[1]);
	
	if       (strcmp(baseName, "phase") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->phase;
	}else if (strcmp(baseName, "U") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->U;
	}else if (strcmp(baseName, "V") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->V;
	}else if (strcmp(baseName, "B") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->B;
	}else if (strcmp(baseName, "D") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->D;
	}else if (strcmp(baseName, "H") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->H;
	}else if (strcmp(baseName, "splitFree") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->splitFree;
	}else if (strcmp(baseName, "splitReduce") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->splitReduce;
	}else if (strcmp(baseName, "l%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->l[k];
	}else if (strcmp(baseName, "l%dPDim") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->lPDim[k-gr->ndd];
	}else if (strcmp(baseName, "s") == 0){
		ctx->kArgs[(*i)++] = (void*) ctx->flatSrcData;
	}else if (strcmp(baseName, "sOff") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->flatSrcOffset;
	}else if (strcmp(baseName, "sJ%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->sJ[k];
	}else if (strcmp(baseName, "d") == 0){
		ctx->kArgs[(*i)++] = (void*) ctx->flatDstData;
	}else if (strcmp(baseName, "dOff") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->flatDstOffset;
	}else if (strcmp(baseName, "dJ%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->dJ[k];
	}else if (strcmp(baseName, "a") == 0){
		ctx->kArgs[(*i)++] = (void*) ctx->flatDstArgData;
	}else if (strcmp(baseName, "aOff") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->flatDstArgOffset;
	}else if (strcmp(baseName, "aJ%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->aJ[k];
	}else if (strcmp(baseName, "w") == 0){
		ctx->kArgs[(*i)++] = (void*) ctx->w;
	}else if (strcmp(baseName, "wdOff") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->wdOff;
	}else if (strcmp(baseName, "pdOff") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->pdOff;
	}else if (strcmp(baseName, "waOff") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->waOff;
	}else if (strcmp(baseName, "paOff") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->paOff;
	}else if (strcmp(baseName, "ibs%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->ibs[k];
	}else if (strcmp(baseName, "ibp%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->ibp[k];
	}else if (strcmp(baseName, "ibl%dPDim") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->iblPDim[k];
	}else if (strcmp(baseName, "ibsOff%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->ibsOff[k];
	}else if (strcmp(baseName, "ibdOff%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->ibdOff[k];
	}else if (strcmp(baseName, "ibaOff%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->ibaOff[k];
	}
}


/**
 * @brief Estimate the level of parallelism available in the GPU context of
 *        this reduction operator.
 * 
 * This is a rough target number of threads.  It would definitely fill the
 * device, plus some substantial margin.
 */

static size_t     reduxGenEstimateParallelism   (const GpuReduction*  gr){
	/**
	 * An arbitrary margin factor ensuring there will be a few thread blocks
	 * per SMX.
	 * 
	 * E.g. on Kepler, each SMX can handle up to two 1024-thread blocks
	 * simultaneously, so a margin of 6/SMX should ensure with very high
	 * likelyhood that all SMXes will be fed and kept busy.
	 */
	
	size_t marginFactor = 6;
	return marginFactor * gr->numProcs * gr->maxLg;
}

/**
 * @brief Returns whether the reduction interface requires a dst argument.
 */

static int        reduxGenRequiresDst           (const GpuReduction*  gr){
	switch (gr->op){
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

static int        reduxGenRequiresDstArg        (const GpuReduction*  gr){
	switch (gr->op){
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
 *        workspace.
 *
 * This is semantically subtly different from reduxGenRequiresDst(). The main
 * difference is in the implementation of the GA_REDUCE_ARGMIN/ARGMAX
 * reductions; both require a dst workspace buffer for the min/max values
 * associated with the indices that they return, even though they will be
 * discarded.
 * 
 * As of now, all reductions use a dst workspace internally.
 */

static int        reduxGenKernelRequiresDst     (const GpuReduction*  gr){
	return 1;
}

/**
 * @brief Returns whether the generated kernel internally requires a dstArg
 *        workspace.
 *
 * This is semantically subtly different from reduxHasDstArg(), since it asks
 * whether the reduction, even though it might not accept a dstArg argument,
 * still requires a dstArg workspace internally.
 * 
 * Currently, there exist no operations that require a dstArg workspace
 * internally but which is not also part of the external interface.
 */

static int        reduxGenKernelRequiresDstArg  (const GpuReduction*  gr){
	return reduxGenRequiresDstArg(gr);
}

/**
 * @brief Whether or not an axis is maybe split.
 * 
 * An axis is possibly split if it is the last free or last reduction axis.
 */

static int        reduxGenAxisMaybeSplit        (const GpuReduction*  gr, int axis){
	return axis == gr->ndd-1 || axis == gr->nds-1;
}

/**
 * @brief Get the number of bytes of workspace per (partial) reduction per thread.
 */

static size_t     reduxGenGetReduxStateSize     (const GpuReduction*  gr){
	size_t total = 0, idxSize = gpuarray_get_elsize(gr->idxTypeCode);
	
	/* The accumulator and index types can be wider than dst/dstArg's types. */
	total += reduxGenKernelRequiresDst(gr)           ?
	         gpuarray_get_elsize(gr->accTypeCode)    :
	         0;
	total += reduxGenKernelRequiresDstArg(gr)        ?
	         gpuarray_get_elsize(gr->idxTypeCode)    :
	         0;
	
	/* At minimum, there must be space for the offset permute. */
	total  = total < idxSize ? idxSize : total;
	          
	
	/* Return the calculated amount of space. */
	return total;
}

/**
 * @brief Get the maximum number of threads this operator's kernel can handle.
 */

static size_t     reduxGenGetMaxLocalSize       (const GpuReduction*  gr){
	return gr->maxLK;
}

/**
 * @brief Get the shared memory consumption for a given block size.
 * 
 * This is non-trivial since it requires ensuring alignment of datatypes.
 */

static size_t     reduxGenGetSHMEMSize          (const GpuReduction*  gr, size_t bs){
	const gpuarray_type* type;
	size_t               total = 0, permuteSpace;
	
	if(reduxGenKernelRequiresDst(gr)){
		type   = gpuarray_get_type(gr->accTypeCode);
		total  = DIVIDECEIL(total, type->align)*type->align;
		total += bs*type->size;
	}
	if(reduxGenKernelRequiresDstArg(gr)){
		type   = gpuarray_get_type(gr->idxTypeCode);
		total  = DIVIDECEIL(total, type->align)*type->align;
		total += bs*type->size;
	}
	
	/* Ensure space for pointer permute. */
	permuteSpace = gpuarray_get_type(gr->idxTypeCode)->size * bs;
	if(total < permuteSpace){
		total = permuteSpace;
	}
	
	return total;
}

/**
 * @brief Get the shared memory byte offset for dst.
 */

static size_t     reduxGenGetSHMEMDstOff        (const GpuReduction*  gr, size_t bs){
	return 0;
}

/**
 * @brief Get the shared memory byte offset for dstArg.
 */

static size_t     reduxGenGetSHMEMDstArgOff     (const GpuReduction*  gr, size_t bs){
	const gpuarray_type* type;
	size_t               total = 0;
	
	if(reduxGenKernelRequiresDst(gr) && reduxGenKernelRequiresDstArg(gr)){
		type   = gpuarray_get_type(gr->accTypeCode);
		total  = DIVIDECEIL(total, type->align)*type->align;
		total += bs*type->size;
		type   = gpuarray_get_type(gr->idxTypeCode);
		total  = DIVIDECEIL(total, type->align)*type->align;
		
		return total;
	}else{
		return 0;
	}
}

/**
 * Get the amount of Workspace memory required.
 * 
 * NOT necessarily the same as amount of SHMEM! The workspace is NOT used for
 * intrablock offset permutes, for instance.
 */

static size_t     reduxGenGetWMEMSize           (const GpuReduction*  gr, size_t bs){
	const gpuarray_type* type;
	size_t               total = 0;
	
	if(reduxGenKernelRequiresDst(gr)){
		type   = gpuarray_get_type(gr->accTypeCode);
		total  = DIVIDECEIL(total, type->align)*type->align;
		total += bs*type->size;
	}
	if(reduxGenKernelRequiresDstArg(gr)){
		type   = gpuarray_get_type(gr->idxTypeCode);
		total  = DIVIDECEIL(total, type->align)*type->align;
		total += bs*type->size;
	}
	
	return total;
}

/**
 * @brief Get the workspace memory byte offset for dst.
 */

static size_t     reduxGenGetWMEMDstOff         (const GpuReduction*  gr, size_t bs){
	return reduxGenGetSHMEMDstOff(gr, bs);
}

/**
 * @brief Get the workspace memory byte offset for dstArg.
 */

static size_t     reduxGenGetWMEMDstArgOff      (const GpuReduction*  gr, size_t bs){
	return reduxGenGetSHMEMDstArgOff(gr, bs);
}

/**
 * @brief Initialize the context.
 * 
 * After this function, calling reduxInvCleanup*() becomes safe.
 */

static int        reduxInvInit                  (redux_ctx*  ctx){
	/**
	 * We initialize certain parts of the context.
	 */
	
	ctx->l                 = NULL;
	ctx->lPDim             = NULL;
	ctx->sJ                = NULL;
	ctx->dJ                = NULL;
	ctx->aJ                = NULL;
	ctx->ibs               = NULL;
	ctx->ibp               = NULL;
	ctx->iblPDim           = NULL;
	ctx->ibsOff            = NULL;
	ctx->ibdOff            = NULL;
	ctx->ibaOff            = NULL;
	ctx->kArgs             = NULL;
	ctx->xdSrc             = NULL;
	ctx->xdSrcPtrs         = NULL;
	ctx->xdTmpPtrs         = NULL;
	ctx->xdSplit           = NULL;
	
	ctx->w                 = NULL;
	
	ctx->prodAllAxes       = ctx->prodRdxAxes   = ctx->prodFreeAxes  = 1;
	ctx->bs                = ctx->gs            = 1;

	return reduxInvInferProperties(ctx);
}

/**
 * @brief Begin inferring the properties of the reduction invocation.
 */

static int        reduxInvInferProperties       (redux_ctx*  ctx){
	axis_desc* a;
	int        i, j;
	size_t     d;


	/* Insane src, reduxLen, dst or dstArg? */
	if(!ctx->reduxList){
		ctx->reduxLen = ctx->src->nd;
	}
	if       (!ctx->src){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "src is NULL!\n");
	}else if (ctx->src->nd  <= 0){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "src is a scalar, cannot reduce it!\n");
	}else if (ctx->reduxLen <  0){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "Length of list of dimensions to be reduced is less than 0!\n");
	}else if (ctx->src->nd  <  (unsigned)ctx->reduxLen){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "src has fewer dimensions than there are dimensions to reduce!\n");
	}else if (reduxInvRequiresDst   (ctx) && !ctx->dst){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "dst is NULL, but reduction requires it!\n");
	}else if (reduxInvRequiresDstArg(ctx) && !ctx->dstArg){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "dstArg is NULL, but reduction requires it!\n");
	}else if (ctx->dst    && ctx->dst->nd   +ctx->reduxLen != ctx->src->nd){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "dst is of incorrect dimensionality for this reduction!\n");
	}else if (ctx->dstArg && ctx->dstArg->nd+ctx->reduxLen != ctx->src->nd){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "dstArg is of incorrect dimensionality for this reduction!\n");
	}
	ctx->nds  = ctx->src->nd;
	ctx->ndr  = ctx->reduxLen;
	ctx->ndd  = ctx->nds - ctx->ndr;
	ctx->ndfs = ctx->ndfr = ctx->ndfd = 0;
	
	/* Insane reduxList? */
	for (i=0;i<ctx->ndr;i++){
		j = ctx->reduxList ? ctx->reduxList[i] : i;
		if (j < -ctx->nds || j >= ctx->nds){
			return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
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
		d                 =  ctx->src->dimensions[i];
		ctx->zeroAllAxes += !d;
		ctx->prodAllAxes *=  d?d:1;
	}
	if (ctx->zeroAllAxes != ctx->zeroRdxAxes){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "Source tensor has length-0 dimensions that are not reduced!\n");
	}
	ctx->prodFreeAxes = ctx->prodAllAxes/ctx->prodRdxAxes;


	/**
	 * Allocate and construct source-tensor axis-description lists.
	 * 
	 * While constructing the descriptions of each axis, verify that:
	 * 
	 *   1. reduxLen has no duplicates.
	 *   2. dst and/or dstArg's dimensions match src's dimensions, stripped of
	 *      the reduction axes.
	 */

	ctx->xdSrc     = calloc(ctx->nds,   sizeof(*ctx->xdSrc));
	ctx->xdSrcPtrs = calloc(ctx->nds+1, sizeof(*ctx->xdSrcPtrs));
	if (!ctx->xdSrc || !ctx->xdSrcPtrs){
		return reduxInvCleanup(ctx, GA_MEMORY_ERROR);
	}
	for (i=0;i<ctx->nds;i++){
		axisInit(&ctx->xdSrc[i],
		         ctx->src->dimensions[i],
		         ctx->src->strides[i]);
	}
	for (i=0;i<ctx->ndr;i++){
		j = ctx->reduxList ? ctx->reduxList[i] : i;
		j = j<0 ? ctx->nds+j : j;
		a = reduxInvGetSrcAxis(ctx, j);
		if (axisIsReduced(a)){
			return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
			       "Axis %d appears multiple times in the "
			       "reduction axis list!\n",
			       j);
		}
		axisMarkReduced(a, i);
	}
	for (i=j=0;i<ctx->nds;i++){
		axis_desc* a      = reduxInvGetSrcAxis(ctx, i);
		size_t     srcLen = axisGetLen(a), dstLen, dstArgLen;
		
		if (axisIsReduced(a)){continue;}
		if (reduxInvRequiresDst(ctx)){
			dstLen = ctx->dst->dimensions[j];
			
			if(srcLen != dstLen){
				return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
				       "Source axis %d has length %zu, but "
				       "corresponding destination axis %d has length %zu!\n",
				       i, srcLen, j, dstLen);
			}
			
			a->dstStride    = ctx->dst->strides[j];
		}
		if (reduxInvRequiresDstArg(ctx)){
			dstArgLen = ctx->dstArg->dimensions[j];
			
			if(srcLen != dstArgLen){
				return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
				       "Source axis %d has length %zu, but "
				       "corresponding destination-argument axis %d has length %zu!\n",
				       i, srcLen, j, dstArgLen);
			}
			
			a->dstArgStride = ctx->dstArg->strides[j];
		}
		
		j++;
	}
	
	
	/**
	 * Grab gpudata buffers and byte offsets before we begin flattening the
	 * tensors. As we flatten the tensor, we may reverse some axes, leading to
	 * a bump of the byte offset.
	 */
	
	ctx->flatSrcData       = ctx->src->data;
	ctx->flatSrcOffset     = ctx->src->offset;
	if(reduxInvRequiresDst(ctx)){
		ctx->flatDstData       = ctx->dst->data;
		ctx->flatDstOffset     = ctx->dst->offset;
	}
	if(reduxInvRequiresDstArg(ctx)){
		ctx->flatDstArgData    = ctx->dstArg->data;
		ctx->flatDstArgOffset  = ctx->dstArg->offset;
	}

	return reduxInvFlattenSource(ctx);
}

/**
 * @brief Flatten the source tensor as much as is practical.
 * 
 * This makes the axis lengths as long as possible and the tensor itself as
 * contiguous as possible.
 */

static int        reduxInvFlattenSource         (redux_ctx*  ctx){
	axis_desc* axis, *flatAxis, *sortAxis;
	int        i, j, k, isSensitive;

	ctx->ndfs = ctx->nds;

	/**
	 * Pass 1: Flatten out 0- and 1-length dimensions. We already know that
	 * 
	 *         a) There are no 0-length free dimensions, because that
	 *            constitutes an invalid input, and
	 *         b) How many 0-length reduction dimensions there are, because
	 *            we counted them in the error-checking code.
	 * 
	 * So if there are any 0-length axes, we can delete all reduction axes and
	 * replace them with a single one.
	 * 
	 * We can also delete 1-length axes outright, since they can always be
	 * ignored; They are always indexed at [0].
	 */

	for (i=j=0;i<ctx->ndfs;i++){
		axis = reduxInvGetSrcAxis(ctx, i);
		if (!reduxTryFlattenOut(ctx, axis)){
			*reduxInvGetSrcAxis(ctx, j++) = *axis;
		}
	}
	if(ctx->zeroRdxAxes > 0){
		/* New reduction axis of 0 length. */
		axisInit       (reduxInvGetSrcAxis(ctx, j), 0, 0);
		axisMarkReduced(reduxInvGetSrcAxis(ctx, j), 0);
		j++;
	}
	ctx->ndfs = j;


	/**
	 * Pass 2: Flatten out continuous dimensions, where strides and sensitivity
	 *         allows it.
	 */
	
	k           = ctx->ndfs;
	isSensitive = reduxIsSensitive(ctx->op);
	qsort(ctx->xdSrc, ctx->ndfs, sizeof(*ctx->xdSrc),
	      isSensitive ? reduxSortFlatSensitive : reduxSortFlatInsensitive);
	for (i=j=1;i<ctx->ndfs;i++){
		flatAxis = reduxInvGetSrcAxis(ctx, j-1);
		sortAxis = reduxInvGetSrcAxis(ctx, i);
		
		if (reduxTryFlattenInto(ctx, flatAxis, sortAxis)){
			k--;
		}else{
			*reduxInvGetSrcAxis(ctx, j++) = *sortAxis;
		}
	}
	ctx->ndfs = k;


	/**
	 * Compute number of free and reduced dimensions.
	 */

	for(ctx->ndfr=ctx->ndfd=i=0;i<ctx->ndfs;i++){
		if(axisIsReduced(reduxInvGetSrcAxis(ctx, i))){
			ctx->ndfr++;
		}else{
			ctx->ndfd++;
		}
	}

	return reduxInvComputeKArgs(ctx);
}

/**
 * @brief Compute the arguments to the kernel.
 * 
 * This is a multistep process and involves a lot of axis sorting on various
 * criteria.
 */

static int        reduxInvComputeKArgs          (redux_ctx*  ctx){
	axis_desc* axis, *prevAxis;
	size_t     target, aL, aLS;
	int        i, j, k, haveSplitFreeAxis, haveSplitReducedAxis;


	/**
	 * STEP 0: Default Kernel Argument Values.
	 * 
	 * They should be valid for a "scalar" job. In particular, for any
	 * non-existent axis, assume length 1.
	 */
	
	ctx->phase       = 0;
	ctx->U           = 1;
	ctx->V           = 1;
	ctx->B           = 1;
	ctx->D           = 1;
	ctx->H           = 1;
	ctx->splitFree   = 1;
	ctx->splitReduce = 1;
	ctx->xdSplit     = NULL;
	ctx->l           = calloc(ctx->gr->nds,      sizeof(*ctx->l));
	ctx->lPDim       = calloc(ctx->gr->ndr,      sizeof(*ctx->lPDim));
	ctx->sJ          = calloc(ctx->gr->nds,      sizeof(*ctx->sJ));
	ctx->dJ          = calloc(ctx->gr->ndd,      sizeof(*ctx->dJ));
	ctx->aJ          = calloc(ctx->gr->ndd,      sizeof(*ctx->aJ));
	ctx->wdOff       = 0;
	ctx->pdOff       = 0;
	ctx->waOff       = 0;
	ctx->paOff       = 0;
	ctx->ibs         = calloc(ctx->gr->log2MaxL, sizeof(*ctx->ibs));
	ctx->ibp         = calloc(ctx->gr->log2MaxL, sizeof(*ctx->ibp));
	ctx->iblPDim     = calloc(ctx->gr->log2MaxL, sizeof(*ctx->iblPDim));
	ctx->ibsOff      = calloc(ctx->gr->log2MaxL, sizeof(*ctx->ibsOff));
	ctx->ibdOff      = calloc(ctx->gr->log2MaxL, sizeof(*ctx->ibdOff));
	ctx->ibaOff      = calloc(ctx->gr->log2MaxL, sizeof(*ctx->ibaOff));
	ctx->bs          = 1;
	ctx->gs          = 1;
	ctx->kArgs       = calloc(ctx->gr->kNumArgs, sizeof(*ctx->kArgs));
	
	if(!ctx->l      || !ctx->lPDim  || !ctx->sJ     || !ctx->dJ       ||
	   !ctx->aJ     || !ctx->ibs    || !ctx->ibp    || !ctx->iblPDim  ||
	   !ctx->ibsOff || !ctx->ibdOff || !ctx->ibaOff || !ctx->kArgs){
		return reduxInvCleanupMsg(ctx, GA_MEMORY_ERROR,
		       "Failed to allocate memory for kernel invocation arguments!\n");
	}
	for(i=0;i<ctx->gr->nds;i++){
		ctx->l[i] = 1;
	}
	for(i=0;i<ctx->gr->log2MaxL;i++){
		ctx->ibs[i] = 1;
	}


	/**
	 * STEP 1: Select Intra-Block Axes.
	 * 
	 * Sort the axes in the order likely to maximize contiguity of source
	 * memory accesses, then tag them to the kernel block size limit, possibly
	 * splitting an axis in the process.
	 */
	
	reduxSortAxisPtrsBy(ctx->xdSrcPtrs, ctx->xdSrc, ctx->ndfs,
	                    reduxSortPtrIBSrcRdSelect);
	target = reduxGenGetMaxLocalSize(ctx->gr);
	
	for(i=0;i<ctx->ndfs && i<ctx->gr->log2MaxL;i++){
		axis = reduxInvGetSrcSortAxis(ctx, i);
		aL   = axisGetLen(axis);
		
		if(ctx->bs*aL <= target){
			ctx->bs     *= aL;
			axisMarkIntraBlock(axis, i, aL);
		}else{
			if(target/ctx->bs >= 2){
				aLS          = target/ctx->bs;
				ctx->bs     *= aLS;
				axisMarkIntraBlock(axis, i, aLS);
				ctx->xdSplit = axis;
				i++;
			}
			break;
		}
	}
	ctx->ndib = i;


	/**
	 * STEP 2: Compute values dependent only on the intrablock axis selection.
	 * 
	 * For instance, the splitFree/splitReduce factors depend only on the split
	 * axis, if any.
	 * 
	 * The shared memory consumption and shared memory offsets depend only
	 * on block size.
	 */

	ctx->splitFree   = reduxInvGetSplitFree     (ctx);
	ctx->splitReduce = reduxInvGetSplitReduce   (ctx);
	ctx->SHMEM       = reduxGenGetSHMEMSize     (ctx->gr, ctx->bs);
	ctx->pdOff       = reduxGenGetSHMEMDstOff   (ctx->gr, ctx->bs);
	ctx->paOff       = reduxGenGetSHMEMDstArgOff(ctx->gr, ctx->bs);


	/**
	 * STEP 3: Compute U, B, D, H
	 */
	
	for (i=0;i<ctx->ndfs;i++){
		axis    = reduxInvGetSrcAxis(ctx, i);
		ctx->U *= axisGetInterLen(axis);
		ctx->B *= axisIsReduced(axis) ? axisGetInterLen(axis) : 1;
		ctx->H *= axisIsReduced(axis) ? axisGetIntraLen(axis) : 1;
	}
	ctx->D = ctx->bs/ctx->H;
	
	
	/**
	 * STEP 4: Compute PDim values.
	 * 
	 * This will be used for index calculation.
	 */
	
	reduxSortAxisPtrsBy(ctx->xdSrcPtrs, ctx->xdSrc, ctx->ndfs,
	                    reduxSortPtrByReduxNum);
	for (i=0;i<ctx->ndfs;i++){
		axis = reduxInvGetSrcSortAxis(ctx, i);
		
		if(axisIsReduced(axis)){
			if(i==0){
				axisSetPDim(axis, 1);
			}else{
				prevAxis = reduxInvGetSrcSortAxis(ctx, i-1);
				axisSetPDim(axis, axisGetPDim(prevAxis)*axisGetLen(prevAxis));
			}
		}
	}
	
	
	/**
	 * STEP 5: Compute Intra-Block Permute Core.
	 * 
	 * Sort the axes in the order most likely to maximize contiguity of
	 * destination/destination argument memory accesses, then compute the
	 * permutation that achieves the highest-bandwidth,
	 * post-horizontal-reduction destination writes.
	 */
	
	reduxSortAxisPtrsBy(ctx->xdSrcPtrs, ctx->xdSrc, ctx->ndfs,
	                    reduxInvRequiresDst(ctx)    ?
	                    reduxSortPtrIBDstWrSelect   :
	                    reduxSortPtrIBDstArgWrSelect);
	for(i=0;i<ctx->ndfs;i++){
		axis = reduxInvGetSrcSortAxis(ctx, i);
		
		if(axisIsIntra(axis)){
			if(i==0){
				axisSetIBP(axis, 1);
			}else{
				prevAxis = reduxInvGetSrcSortAxis(ctx, i-1);
				axisSetIBP(axis, axisGetIBP(prevAxis)*axisGetIntraLen(prevAxis));
			}
		}
	}
	
	/**
	 * STEP 6. Place the intra axis arguments
	 * 
	 *              ibs, ibp, iblPDim, ibsOff, ibdOff, ibaOff
	 * 
	 * For this we need the axes in final order of insertion.
	 */
	
	reduxSortAxisPtrsBy(ctx->xdSrcPtrs, ctx->xdSrc, ctx->ndfs,
	                    reduxSortPtrInsertFinalOrder);
	for(i=0;i<ctx->ndib;i++){
		axis = reduxInvGetSrcSortAxis(ctx,  i);
		
		ctx->ibs    [i] = axisGetIntraLen    (axis);
		ctx->ibp    [i] = axisGetIBP         (axis);
		ctx->iblPDim[i] = axisGetPDim        (axis);
		ctx->ibsOff [i] = axisGetSrcStride   (axis);
		ctx->ibdOff [i] = axisGetDstStride   (axis);
		ctx->ibaOff [i] = axisGetDstArgStride(axis);
	}
	
	/**
	 * STEP 7. Place the inter axis arguments
	 * 
	 *              lN, lNPDim, sJN, dJN, aJN
	 * 
	 * , where N in [0, ctx->gr->ndd) are free axes,
	 *         N in [ctx->gr->ndd, ctx->gr->nds) are reduced axes,
	 * and ctx->xdSrcPtr[...] are sorted in the reverse of that order for
	 * insertion, and excludes any split axis.
	 * 
	 * How precisely the insertion is done depends closely on whether there is
	 * a split axis and if so whether it is free or reduced.
	 * 
	 * - If there is a split axis and it is free, then it should be inserted as
	 *   the first free axis. Its jumps should be
	 *             sJN = -sSM*intrainterLenM + sSN*splitFree
	 *             dJN = -dSM*intrainterLenM + dSN*splitFree
	 *             aJN = -aSM*intrainterLenM + aSN*splitFree
	 * - If there is a split axis and it is reduced, then it should be inserted
	 *   as the first reduced axis. Its jump should be
	 *             sJN = -sSM*intrainterLenM + sSN*splitReduced
	 * - If there is no split axis, proceed normally in filling the axes.
	 */
	
	haveSplitFreeAxis    = ctx->xdSplit && !axisIsReduced(ctx->xdSplit);
	haveSplitReducedAxis = ctx->xdSplit &&  axisIsReduced(ctx->xdSplit);
	
	/* If we have a reduced split axis, insert it before any other reduced axis. */
	j  = ctx->gr->nds-1;
	k  = ctx->gr->ndr-1;
	if(haveSplitReducedAxis && k>=0){
		ctx->l      [j]  =           axisGetLen          (ctx->xdSplit);
		ctx->lPDim  [k]  =           axisGetPDim         (ctx->xdSplit);
		ctx->sJ     [j] +=  (ssize_t)axisGetSrcStride    (ctx->xdSplit)*
		                    (ssize_t)axisGetIntraLen     (ctx->xdSplit);
		if(j>0){
			ctx->sJ   [j-1] -=  (ssize_t)axisGetSrcStride    (ctx->xdSplit)*
			                    (ssize_t)axisGetIntraInterLen(ctx->xdSplit);
		}
		j--;
		k--;
	}
	
	/* Insert rest of reduced axes. */
	for(;i<ctx->ndfs && k>=0;i++,j--,k--){
		axis = reduxInvGetSrcSortAxis(ctx, i);
		if(!axisIsReduced(axis)){
			break;
		}
		
		ctx->l      [j]  =           axisGetLen          (axis);
		ctx->lPDim  [k]  =           axisGetPDim         (axis);
		ctx->sJ     [j] +=  (ssize_t)axisGetSrcStride    (axis)*
		                    (ssize_t)axisGetIntraLen     (axis);
		if(j>0){
			ctx->sJ   [j-1] -=  (ssize_t)axisGetSrcStride    (axis)*
			                    (ssize_t)axisGetIntraInterLen(axis);
		}
	}
	
	/* If we have a free split axis, insert it before any other free axis. */
	k = ctx->gr->ndd-1;
	if(haveSplitFreeAxis && k>=0){
		ctx->l      [k]  =           axisGetLen          (ctx->xdSplit);
		ctx->sJ     [k] +=  (ssize_t)axisGetSrcStride    (ctx->xdSplit)*
		                    (ssize_t)axisGetIntraLen     (ctx->xdSplit);
		ctx->dJ     [k] +=  (ssize_t)axisGetDstStride    (ctx->xdSplit)*
		                    (ssize_t)axisGetIntraLen     (ctx->xdSplit);
		ctx->aJ     [k] +=  (ssize_t)axisGetDstArgStride (ctx->xdSplit)*
		                    (ssize_t)axisGetIntraLen     (ctx->xdSplit);
		if(k>0){
			ctx->sJ  [k-1] -=  (ssize_t)axisGetSrcStride    (ctx->xdSplit)*
			                   (ssize_t)axisGetIntraInterLen(ctx->xdSplit);
			ctx->dJ  [k-1] -=  (ssize_t)axisGetDstStride    (ctx->xdSplit)*
			                   (ssize_t)axisGetIntraInterLen(ctx->xdSplit);
			ctx->aJ  [k-1] -=  (ssize_t)axisGetDstArgStride (ctx->xdSplit)*
			                   (ssize_t)axisGetIntraInterLen(ctx->xdSplit);
		}
		k--;
	}
	
	/* Insert rest of free axes. */
	for(;i<ctx->ndfs && k>=0;i++,k--){
		axis = reduxInvGetSrcSortAxis(ctx, i);
		if(axisIsReduced(axis)){
			break;
		}
		
		ctx->l      [k]  =           axisGetLen          (axis);
		ctx->sJ     [k] +=  (ssize_t)axisGetSrcStride    (axis)*
		                    (ssize_t)axisGetIntraLen     (axis);
		ctx->dJ     [k] +=  (ssize_t)axisGetDstStride    (axis)*
		                    (ssize_t)axisGetIntraLen     (axis);
		ctx->aJ     [k] +=  (ssize_t)axisGetDstArgStride (axis)*
		                    (ssize_t)axisGetIntraLen     (axis);
		if(k>0){
			ctx->sJ  [k-1] -=  (ssize_t)axisGetSrcStride    (axis)*
			                   (ssize_t)axisGetIntraInterLen(axis);
			ctx->dJ  [k-1] -=  (ssize_t)axisGetDstStride    (axis)*
			                   (ssize_t)axisGetIntraInterLen(axis);
			ctx->aJ  [k-1] -=  (ssize_t)axisGetDstArgStride (axis)*
			                   (ssize_t)axisGetIntraInterLen(axis);
		}
	}

	return reduxInvSchedule(ctx);
}

#if 0
static void       reduxScheduleKernel           (int                  ndims,
                                                 uint64_t*            dims,
                                                 uint64_t             warpSize,
                                                 uint64_t             maxLg,
                                                 uint64_t*            maxLs,
                                                 uint64_t             maxGg,
                                                 uint64_t*            maxGs,
                                                 uint64_t*            bs,
                                                 uint64_t*            gs,
                                                 uint64_t*            cs);

/**
 * @brief Given the parameters of a kernel scheduling problem, solve it as
 *        optimally as possible.
 *
 * NB: This is the only function in this entire file that should have
 *     anything to do with the integer factorization APIs.
 */

static void       reduxScheduleKernel           (int         ndims,
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
#endif

/**
 * @brief With nearly all parameters of the kernel computed, schedule the
 *        kernel for maximum performance.
 * 
 * The thread block size has already been chosen; We only have to choose
 * 
 *   1. ctx->gs: The grid size, which is the number of thread blocks.
 *   2. ctx->V:  The number of vertical reductions per thread block.
 * 
 * Two factors drive the scheduling:
 * 
 *   1. We want to keep all multiprocessors of the device busy; For this we use
 *      an estimate of the level of parallelism of the device.
 *   2. If V can be chosen such that V % B == 0, then only a single kernel
 *      phase is necessary.
 * 
 * Once the scheduling is performed, the workspace can be allocated and
 * workspace offsets can be computed.
 */

static int        reduxInvSchedule              (redux_ctx*           ctx){
	const int flags = GA_BUFFER_READ_WRITE;
	size_t    WSPACESIZE;
	
	/**
	 * Get enough blocks to fill available device parallelism to capacity.
	 * Then, compute corresponding V.
	 */
	
	ctx->gs    = DIVIDECEIL(reduxInvEstimateParallelism(ctx),
	                        reduxGenGetMaxLocalSize(ctx->gr));
	ctx->V     = DIVIDECEIL(ctx->U, ctx->gs);
	
	/**
	 * Allocate required workspace.
	 */
	
	ctx->wdOff = reduxGenGetWMEMDstOff   (ctx->gr, 2*ctx->gs*ctx->D);
	ctx->waOff = reduxGenGetWMEMDstArgOff(ctx->gr, 2*ctx->gs*ctx->D);
	WSPACESIZE = reduxGenGetWMEMSize     (ctx->gr, 2*ctx->gs*ctx->D);
	ctx->w     = gpudata_alloc(ctx->gr->gpuCtx, WSPACESIZE, 0, flags, 0);
	if(!ctx->w){
		return reduxInvCleanupMsg(ctx, GA_MEMORY_ERROR,
		       "Could not allocate %zu-byte workspace for reduction!\n",
		       WSPACESIZE);
	}
	
	return reduxInvoke(ctx);
}

/**
 * @brief Invoke the kernel.
 */

static int        reduxInvoke                   (redux_ctx*           ctx){
	int   ret, i=0;
	void* ptrs[2] = {ctx, &i};
	
	/**
	 * Argument Marshalling.
	 */
	
	reduxGenIterArgs(ctx->gr, reduxInvMarshalArg, ptrs);



	/**
	 * The kernel is now invoked once or twice, for phase 0 or 1.
	 * 
	 * Phase 1 is sometimes optional.
	 */

	ctx->phase = 0;
	ret = GpuKernel_call(&ctx->gr->k, 1, &ctx->gs, &ctx->bs, ctx->SHMEM, ctx->kArgs);
	if (ret != GA_NO_ERROR){
		return reduxInvCleanupMsg(ctx, ret,
		                          "Failure in kernel call, Phase 0!\n");
	}
	
	if(ctx->V%ctx->B != 0){
		ctx->phase = 1;
		ret = GpuKernel_call(&ctx->gr->k, 1, &ctx->gs, &ctx->bs, ctx->SHMEM, ctx->kArgs);
		if (ret != GA_NO_ERROR){
			return reduxInvCleanupMsg(ctx, ret,
			                          "Failure in kernel call, Phase 1!\n");
		}
	}
	
	/* Success! */
	return reduxInvCleanup(ctx, GA_NO_ERROR);
}

/**
 * Cleanup
 */

static int        reduxInvCleanup               (redux_ctx*        ctx, int ret){
	free(ctx->l);
	free(ctx->lPDim);
	free(ctx->sJ);
	free(ctx->dJ);
	free(ctx->aJ);
	free(ctx->ibs);
	free(ctx->ibp);
	free(ctx->iblPDim);
	free(ctx->ibsOff);
	free(ctx->ibdOff);
	free(ctx->ibaOff);
	free(ctx->kArgs);
	free(ctx->xdSrc);
	free(ctx->xdSrcPtrs);
	free(ctx->xdTmpPtrs);
	
	gpudata_release(ctx->w);
	
	ctx->l                 = NULL;
	ctx->lPDim             = NULL;
	ctx->sJ                = NULL;
	ctx->dJ                = NULL;
	ctx->aJ                = NULL;
	ctx->ibs               = NULL;
	ctx->ibp               = NULL;
	ctx->iblPDim           = NULL;
	ctx->ibsOff            = NULL;
	ctx->ibdOff            = NULL;
	ctx->ibaOff            = NULL;
	ctx->kArgs             = NULL;
	ctx->xdSrc             = NULL;
	ctx->xdSrcPtrs         = NULL;
	ctx->xdTmpPtrs         = NULL;
	
	ctx->w                 = NULL;

	return ret;
}
static int        reduxInvCleanupMsg            (redux_ctx*        ctx, int ret,
                                                 const char*       fmt, ...){
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
	
	return reduxInvCleanup(ctx, ret);
}
