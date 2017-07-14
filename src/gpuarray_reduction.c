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

/**
 * Template Selector
 * 
 * This is a bitfield interpreted as follows:
 * 
 *     0b000x: Phase 1 processing (Phase 0)
 *     0b00x0: Split axis is free (Reduced)
 *     0bxx00: Huge axis is:
 *             00: Nonexistent
 *             01: Same as split axis
 *             10: Same type (free/reduced) as split axis
 *             11: Opposite type (free/reduced) to split axis
 */

#define SELECTOR_PHASE1              0x01
#define SELECTOR_SPLIT_FREE          0x02
#define SELECTOR_HUGE_AXIS           0x0C
#define SELECTOR_HUGE_NONE           0x00
#define SELECTOR_HUGE_IS_SPLIT       0x04
#define SELECTOR_HUGE_SAME_TYPE      0x08
#define SELECTOR_HUGE_OPPOSITE_TYPE  0x0C


/* Datatypes */

/**
 * @brief Axis Description.
 */

struct axis_desc{
	int      reduxNum;
	int      ibNum;
	unsigned perm;
	unsigned isReduced : 1;
	unsigned isIntra   : 1;
	size_t   len;
	size_t   splitLen;
	ssize_t  s0S;
	ssize_t  d0S;
	ssize_t  d1S;
	size_t   i0S;
};
typedef struct axis_desc axis_desc;

/**
 *                    Reduction Kernel Invoker.
 */

struct redux_ctx{
	/* Function Arguments. */
	const GpuReduction* gr;
	GpuArray*           d0;
	GpuArray*           d1;
	const GpuArray*     s0;
	int                 reduxLen;
	const int*          reduxList;
	int                 flags;

	/* General. */
	int                 nds0;         /* # Source                           axes */
	int                 nds0r;        /* # Reduced                          axes */
	int                 ndd0;         /* # Destination                      axes */
	int                 ndfs0;        /* # Flattened source                 axes */
	int                 ndib;         /* # Intra-block                      axes */
	int                 zeroAllAxes;  /* # of zero-length                   axes in source tensor */
	int                 zeroRdxAxes;  /* # of zero-length         reduction axes in source tensor */
	size_t              prodAllAxes;  /* Product of length of all           axes in source tensor */
	size_t              prodRdxAxes;  /* Product of length of all reduction axes in source tensor */
	size_t              prodFreeAxes; /* Product of length of all free      axes in source tensor */
	
	/* Flattening */
	axis_desc*          xdSrc;
	axis_desc**         xdSrcPtrs;
	axis_desc*          xdSplit;

	/* Invoker */
	uint32_t            selector;
	uint64_t            U;
	uint64_t            V;
	uint64_t            B;
	uint32_t            D;
	uint32_t            Dunit;
	uint32_t            H;
	
	uint32_t            LSlice;
	uint64_t            LPadded;
	uint64_t*           L;
	uint32_t*           Li;
	gpudata*            S0Data;
	int64_t             S0Off;
	int64_t*            S0J, *S0Si;
	gpudata*            D0Data;
	int64_t             D0Off;
	int64_t*            D0J, *D0Si;
	gpudata*            D1Data;
	int64_t             D1Off;
	int64_t*            D1J, *D1Si;
	int64_t*            I0J, *I0Si;
	
	gpudata*            W;
	int64_t             W0Off;
	ssize_t             W1Off;
	size_t              shmemBytes;
	ssize_t             SHMEMK0Off;
	ssize_t             SHMEMK1Off;
	
	unsigned*           perm;
	
	void**              kArgs;
	
	/* Scheduler */
	size_t              bs;
	size_t              gs;
};
typedef struct redux_ctx redux_ctx;



/**
 *                    Reduction Operator Attributes.
 */

struct GpuReductionAttr{
	gpucontext*   gpuCtx;
	unsigned      numProcs;
	size_t        maxLg, maxL0, maxGg, maxG0, maxLM;
	
	ga_reduce_op  op;
	int           maxSrcDims;
	int           maxDstDims;
	int           s0Typecode, d0Typecode, d1Typecode, i0Typecode;
};


/**
 *                    Reduction Operator.
 * 
 * INTRO
 * 
 * Generates the source code for a reduction kernel over arbitrarily-ranked,
 * -shaped and -typed tensors.
 * 
 * It is assumed that at most one axis will ever be of length > 2**31-1. The
 * assumption is believed safe because no GPU or similar accelerator presently
 * on Earth has the capacity to store or process 2**62-element tensors.
 * 
 * 
 * TYPE NAMES
 * 
 *     TS0:  Type of s0 tensor
 *     TPS0: Promoted type of s0 tensor
 *     TD0:  Type of d0 tensor
 *     TD1:  Type of d1 tensor
 *     TS32: Type of 32-bit narrow, signed,   2's complement integer
 *     TU32: Type of 32-bit narrow, unsigned, 2's complement integer
 *     TS64: Type of 64-bit wide,   signed,   2's complement integer
 *     TU64: Type of 64-bit wide,   unsigned, 2's complement integer
 *     TK0:  Type of reduction accumulator
 *     TK1:  Type of flattened index
 * 
 * But note however that: 
 *   - TS0 is not necessarily the same as TPS0/TD0/TD1
 *   - TD1 is not necessarily TS32/TU32/TS64/TU64/TK1
 *   - TK1 is not necessarily TU64
 *   - TK0 is not necessarily the same as TS0 or TPS0. Moreover, since it may
 *         be a "custom" type that exists only within the kernel, it might not
 *         necessarily have a gpuarray_type typecode associated with it.
 * 
 *         Example 1: TK0 might eventually become a double-TS0 struct for Kahan
 *         compensated summation. No typecode exists for a struct of two TS0
 *         values.
 * 
 *         Example 2: If doing a Kahan summation of a GA_HALF array, the
 *         following might be the case:
 *             TS0  == GA_HALF
 *             TPS0 == GA_FLOAT
 *             TK0  == struct{GA_FLOAT,GA_FLOAT}
 * 
 * 
 * NOTES
 * 
 * Information elements required to generate source code:
 * 
 *   1. Maximum rank and dtype of s0 tensor
 *   2. Maximum rank and dtype of d0/d1 tensors
 *   3. GPU context
 *   4. Number of processors
 *   5. Warp size
 *   6. Maximum size of block
 *   7. Maximum size of block axis X
 *   8. Maximum size of grid
 *   9. Maximum size of grid  axis X
 *  10. Dtype and initializer of accumulator
 * 
 * Rationale for some dependencies:
 * 
 *   1) Get the GPU context and its properties immediately, since an invalid
 *      context is a likely error and we want to fail fast.
 *   2) The type and initializer of the accumulator should be determined after
 *      the context's properties have been retrieved since they provide
 *      information about the device's natively-supported types and operations
 *      (e.g. half-precision float)
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
	GpuReductionAttr grAttr;
	gpucontext*      gpuCtx;
	ga_reduce_op     op;
	int              nds;
	int              ndd;
	int              ndr;
	
	/* Source code Generator. */
	strb             s;
	srcb             srcGen;
	char             kName[256];
	char*            kSourceCode;
	size_t           kSourceCodeLen;
	int              TS0tc;
	int              TPS0tc;
	int              TD0tc;
	int              TD1tc;
	int              TI0tc;
	int              TS32tc;
	int              TU32tc;
	int              TS64tc;
	int              TU64tc;
	struct{
		size_t       size;
		size_t       align;
		char         defn[256];
		char         init[256];
	} TK0, TK1;
	int              idxTypeCode;
	int              accTypeCode;
	const char*      srcTypeStr;
	const char*      dstTypeStr;
	const char*      dstArgTypeStr;
	const char*      idxTypeStr;
	
	/* Compile */
	int              kNumArgs;
	int*             kArgTypeCodes;
	char*            kErrorString;
	GpuKernel        k;
	
	/* Scheduling */
	size_t           maxLK;
	size_t           maxBS;
	int              log2MaxBS;
};


/* Typedefs */
typedef void (*GpuReductionIterFn)(const GpuReduction* gr,
                                   int                 typecode,
                                   const char*         typeName,
                                   const char*         baseName,
                                   int                 num,
                                   void*               user);


/* Static Function prototypes */
/* Utilities */
static int         reduxGetSumInit                  (int typecode, const char** property);
static int         reduxGetProdInit                 (int typecode, const char** property);
static int         reduxGetMinInit                  (int typecode, const char** property);
static int         reduxGetMaxInit                  (int typecode, const char** property);
static int         reduxGetAndInit                  (int typecode, const char** property);
static int         reduxGetOrInit                   (int typecode, const char** property);
static int         reduxIsFloatingPoint             (int typecode);
static unsigned    reduxCeilLog2                    (uint64_t x);
static uint64_t    reduxNextPow2                    (uint64_t x);
static int         reduxSortFlatInsensitive         (const void* a, const void* b);
static int         reduxSortFlatSensitive           (const void* a, const void* b);
static int         reduxSortPtrS0AbsStride          (const void* a, const void* b);
static int         reduxSortPtrByReduxNum           (const void* a, const void* b);
static int         reduxSortPtrD0WrSelect           (const void* a, const void* b);
static int         reduxSortPtrD1WrSelect           (const void* a, const void* b);
static int         reduxSortPtrInsertFinalOrder     (const void* a, const void* b);

/* Axis Description API */
static void        axisInit                         (axis_desc*           axis,
                                                     ssize_t              len,
                                                     ssize_t              s0S);
static void        axisMarkReduced                  (axis_desc*           axis, int    reduxNum);
static void        axisMarkIntraBlock               (axis_desc*           axis,
                                                     int                  ibNum,
                                                     size_t               ibLen);
static int         axisGetReduxNum                  (const axis_desc*     axis);
static size_t      axisGetLen                       (const axis_desc*     axis);
static size_t      axisGetIntraLen                  (const axis_desc*     axis);
static size_t      axisGetInterLen                  (const axis_desc*     axis);
static size_t      axisGetIntraInterLen             (const axis_desc*     axis);
static ssize_t     axisGetS0Stride                  (const axis_desc*     axis);
static size_t      axisGetS0AbsStride               (const axis_desc*     axis);
static ssize_t     axisGetD0Stride                  (const axis_desc*     axis);
static size_t      axisGetD0AbsStride               (const axis_desc*     axis);
static ssize_t     axisGetD1Stride                  (const axis_desc*     axis);
static size_t      axisGetD1AbsStride               (const axis_desc*     axis);
static size_t      axisGetI0Stride                  (const axis_desc*     axis);
static void        axisSetI0Stride                  (axis_desc*           axis,
                                                     size_t               pdim);
static unsigned    axisGetPerm                      (const axis_desc*     axis);
static int         axisGetIBNum                     (const axis_desc*     axis);
static void        axisSetPerm                      (axis_desc*           axis,
                                                     unsigned             ibp);
static int         axisIsReduced                    (const axis_desc*     axis);
static int         axisIsIntra                      (const axis_desc*     axis);
static int         axisIsInter                      (const axis_desc*     axis);
static int         axisIsSplit                      (const axis_desc*     axis);

/* Reduction Context API */
/*     Generator Control Flow */
static int         reduxGenInit                     (GpuReduction*        gr);
static int         reduxGenInferProperties          (GpuReduction*        gr);
static void        reduxGenSetMaxBS                 (GpuReduction*        gr);
static void        reduxGenSetKTypes                (GpuReduction*        gr);
static void        reduxGenIterArgs                 (const GpuReduction*  gr,
                                                     GpuReductionIterFn   fn,
                                                     void*                user);
static int         reduxGenSrc                      (GpuReduction*        gr);
static void        reduxGenSrcAppend                (GpuReduction*        gr);
static void        reduxGenSrcAppendIncludes        (GpuReduction*        gr);
static void        reduxGenSrcAppendMacroTypedefs   (GpuReduction*        gr);
static void        reduxGenSrcAppendReduxKernel     (GpuReduction*        gr);
static void        reduxGenSrcAppendPrototype       (GpuReduction*        gr);
static void        reduxGenSrcAppendDecode          (GpuReduction*        gr);
static void        reduxGenSrcAppendPhase0          (GpuReduction*        gr,
                                                     uint32_t             selector);
static void        reduxGenSrcAppendLoop            (GpuReduction*        gr,
                                                     uint32_t             selector,
                                                     int                  initial);
static void        reduxGenSrcAppendVertical        (GpuReduction*        gr,
                                                     uint32_t             selector);
static void        reduxGenSrcAppendIncrement       (GpuReduction*        gr,
                                                     uint32_t             selector,
                                                     int                  initial,
                                                     int                  axis);
static void        reduxGenSrcAppendDstWrite        (GpuReduction*        gr,
                                                     uint32_t             selector,
                                                     int                  initial);
static void        reduxGenSrcAppendPhase1          (GpuReduction*        gr);
static int         reduxGenSrcAxisIsHuge            (GpuReduction*        gr,
                                                     uint32_t             selector,
                                                     int                  axis);
static int         reduxGenSrcAxisIsSplit           (GpuReduction*        gr,
                                                     uint32_t             selector,
                                                     int                  axis);
static int         reduxGenCompile                  (GpuReduction*        gr);
static int         reduxGenComputeLaunchBounds      (GpuReduction*        gr);
static int         reduxGenCleanup                  (GpuReduction*        gr,  int ret);
static int         reduxGenCleanupMsg               (GpuReduction*        gr,  int ret,
                                                     const char*          fmt, ...);

/*     Generator Utilities */
static void        reduxGenCountArgs                (const GpuReduction*  gr,
                                                     int                  typecode,
                                                     const char*          typeName,
                                                     const char*          baseName,
                                                     int                  num,
                                                     void*                user);
static void        reduxGenSaveArgTypecodes         (const GpuReduction*  gr,
                                                     int                  typecode,
                                                     const char*          typeName,
                                                     const char*          baseName,
                                                     int                  num,
                                                     void*                user);
static void        reduxGenAppendArg                (const GpuReduction*  gr,
                                                     int                  typecode,
                                                     const char*          typeName,
                                                     const char*          baseName,
                                                     int                  num,
                                                     void*                user);
static void        reduxInvMarshalArg               (const GpuReduction*  gr,
                                                     int                  typecode,
                                                     const char*          typeName,
                                                     const char*          baseName,
                                                     int                  num,
                                                     void*                user);
static size_t      reduxGenEstimateParallelism      (const GpuReduction*  gr);
static int         reduxGenRequiresS0               (const GpuReduction*  gr);
static int         reduxGenRequiresD0               (const GpuReduction*  gr);
static int         reduxGenRequiresD1               (const GpuReduction*  gr);
static int         reduxGenKernelRequiresLatticeS0  (const GpuReduction*  gr);
static int         reduxGenKernelRequiresLatticeD0  (const GpuReduction*  gr);
static int         reduxGenKernelRequiresLatticeD1  (const GpuReduction*  gr);
static int         reduxGenKernelRequiresLatticeI0  (const GpuReduction*  gr);
static int         reduxGenKernelRequiresStateK0    (const GpuReduction*  gr);
static int         reduxGenKernelRequiresStateK1    (const GpuReduction*  gr);
static int         reduxGenKernelRequiresWspace     (const GpuReduction*  gr);
static size_t      reduxGenGetK0Size                (const GpuReduction*  gr);
static size_t      reduxGenGetK0Align               (const GpuReduction*  gr);
static size_t      reduxGenGetK1Size                (const GpuReduction*  gr);
static size_t      reduxGenGetK1Align               (const GpuReduction*  gr);
static size_t      reduxGenGetReduxStateSize        (const GpuReduction*  gr);
static size_t      reduxGenGetMaxLocalSize          (const GpuReduction*  gr);
static size_t      reduxGenGetSHMEMSize             (const GpuReduction*  gr, size_t cells);
static size_t      reduxGenGetSHMEMK0Off            (const GpuReduction*  gr, size_t cells);
static size_t      reduxGenGetSHMEMK1Off            (const GpuReduction*  gr, size_t cells);
static size_t      reduxGenGetWMEMSize              (const GpuReduction*  gr, size_t cells);
static size_t      reduxGenGetWMEMK0Off             (const GpuReduction*  gr, size_t cells);
static size_t      reduxGenGetWMEMK1Off             (const GpuReduction*  gr, size_t cells);

/*     Invoker Control Flow */
static int         reduxInvInit                     (redux_ctx*           ctx);
static int         reduxInvInferProperties          (redux_ctx*           ctx);
static int         reduxInvFlattenSource            (redux_ctx*           ctx);
static int         reduxInvComputeKernelArgs        (redux_ctx*           ctx);
static int         reduxInvSchedule                 (redux_ctx*           ctx);
static int         reduxInvoke                      (redux_ctx*           ctx);
static int         reduxInvCleanup                  (redux_ctx*           ctx, int ret);
static int         reduxInvCleanupMsg               (redux_ctx*           ctx, int ret,
                                                     const char*          fmt, ...);

/*     Invoker Utilities */
static size_t      reduxInvEstimateParallelism      (const redux_ctx*       ctx);
static int         reduxInvRequiresS0               (const redux_ctx*       ctx);
static int         reduxInvRequiresD0               (const redux_ctx*       ctx);
static int         reduxInvRequiresD1               (const redux_ctx*       ctx);
static axis_desc*  reduxInvGetSrcAxis               (const redux_ctx*       ctx, int i);
static axis_desc*  reduxInvGetSrcSortAxis           (const redux_ctx*       ctx, int i);
static int         reduxTryFlattenOut               (const redux_ctx*       ctx,
                                                     const axis_desc*       axis);
static int         reduxTryFlattenInto              (redux_ctx*             ctx,
                                                     axis_desc*             into,
                                                     const axis_desc*       from);
static void        reduxSortAxisPtrsBy              (axis_desc**            ptrs,
                                                     axis_desc*             axes,
                                                     size_t                 numAxes,
                                                     int(*fn)(const void*, const void*));


/* Function Implementations */
/* Extern Functions */
GPUARRAY_PUBLIC int   GpuReductionAttr_new          (GpuReductionAttr**         grAttr,
                                                     gpucontext*                gpuCtx){
	if (!grAttr){
		return GA_INVALID_ERROR;
	}
	if (!gpuCtx){
		*grAttr = NULL;
		return GA_INVALID_ERROR;
	}
	*grAttr = calloc(1, sizeof(**grAttr));
	if (!*grAttr){
		return GA_MEMORY_ERROR;
	}
	
	(*grAttr)->gpuCtx     = gpuCtx;
	if (gpucontext_property(gpuCtx, GA_CTX_PROP_NUMPROCS,  &(*grAttr)->numProcs) != GA_NO_ERROR ||
	    gpucontext_property(gpuCtx, GA_CTX_PROP_MAXLSIZE,  &(*grAttr)->maxLg)    != GA_NO_ERROR ||
	    gpucontext_property(gpuCtx, GA_CTX_PROP_MAXLSIZE0, &(*grAttr)->maxL0)    != GA_NO_ERROR ||
	    gpucontext_property(gpuCtx, GA_CTX_PROP_MAXGSIZE,  &(*grAttr)->maxGg)    != GA_NO_ERROR ||
	    gpucontext_property(gpuCtx, GA_CTX_PROP_MAXGSIZE0, &(*grAttr)->maxG0)    != GA_NO_ERROR ||
	    gpucontext_property(gpuCtx, GA_CTX_PROP_LMEMSIZE,  &(*grAttr)->maxLM)    != GA_NO_ERROR ){
		free(*grAttr);
		return GA_INVALID_ERROR;
	}
	(*grAttr)->op         = GA_REDUCE_SUM;
	(*grAttr)->maxSrcDims = 1;
	(*grAttr)->maxDstDims = 1;
	(*grAttr)->s0Typecode = GA_FLOAT;
	(*grAttr)->d0Typecode = GA_FLOAT;
	(*grAttr)->d1Typecode = GA_ULONG;
	(*grAttr)->i0Typecode = GA_ULONG;
	
	return GA_NO_ERROR;
}
GPUARRAY_PUBLIC int   GpuReductionAttr_setop        (GpuReductionAttr*          grAttr,
                                                     ga_reduce_op               op){
	grAttr->op = op;
	
	return GA_NO_ERROR;
}
GPUARRAY_PUBLIC int   GpuReductionAttr_setdims      (GpuReductionAttr*          grAttr,
                                                     unsigned                   maxSrcDims,
                                                     unsigned                   maxDstDims){
	grAttr->maxSrcDims = maxSrcDims;
	grAttr->maxDstDims = maxDstDims;
	
	return GA_NO_ERROR;
}
GPUARRAY_PUBLIC int   GpuReductionAttr_sets0type    (GpuReductionAttr*          grAttr,
                                                     int                        s0Typecode){
	switch (grAttr->op){
		case GA_REDUCE_AND:
		case GA_REDUCE_OR:
		case GA_REDUCE_XOR:
			if (reduxIsFloatingPoint(s0Typecode)){
				/* Bitwise operations not applicable to floating-point datatypes! */
				return GA_INVALID_ERROR;
			}
		break;
		default:
		break;
	}
	
	grAttr->s0Typecode = s0Typecode;
	
	return GA_NO_ERROR;
}
GPUARRAY_PUBLIC int   GpuReductionAttr_setd0type    (GpuReductionAttr*          grAttr,
                                                     int                        d0Typecode){
	grAttr->d0Typecode = d0Typecode;
	
	return GA_NO_ERROR;
}
GPUARRAY_PUBLIC int   GpuReductionAttr_setd1type    (GpuReductionAttr*          grAttr,
                                                     int                        d1Typecode){
	grAttr->d1Typecode = d1Typecode;
	
	return GA_NO_ERROR;
}
GPUARRAY_PUBLIC int   GpuReductionAttr_seti0type    (GpuReductionAttr*          grAttr,
                                                     int                        i0Typecode){
	grAttr->i0Typecode = i0Typecode;
	
	return GA_NO_ERROR;
}
GPUARRAY_PUBLIC int   GpuReductionAttr_appendopname (GpuReductionAttr*          grAttr,
                                                     size_t                     n,
                                                     char*                      name){
	switch (grAttr->op){
		case GA_REDUCE_COPY:         return snprintf(name, n, "Copy_%d",            grAttr->maxSrcDims);
		case GA_REDUCE_SUM:          return snprintf(name, n, "Sum_%d_%d",          grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_PROD:         return snprintf(name, n, "Prod_%d_%d",         grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_PRODNZ:       return snprintf(name, n, "ProdNonZero_%d_%d",  grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_MIN:          return snprintf(name, n, "Min_%d_%d",          grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_MAX:          return snprintf(name, n, "Max_%d_%d",          grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_ARGMIN:       return snprintf(name, n, "Argmin_%d_%d",       grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_ARGMAX:       return snprintf(name, n, "Argmax_%d_%d",       grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_MINANDARGMIN: return snprintf(name, n, "MinAndArgmin_%d_%d", grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_MAXANDARGMAX: return snprintf(name, n, "MaxAndArgmax_%d_%d", grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_AND:          return snprintf(name, n, "And_%d_%d",          grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_OR:           return snprintf(name, n, "Or_%d_%d",           grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_XOR:          return snprintf(name, n, "Xor_%d_%d",          grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_ALL:          return snprintf(name, n, "All_%d_%d",          grAttr->maxSrcDims, grAttr->maxDstDims);
		case GA_REDUCE_ANY:          return snprintf(name, n, "Any_%d_%d",          grAttr->maxSrcDims, grAttr->maxDstDims);
		default:                     if (name && n>0){*name = '\0';} return GA_INVALID_ERROR;
	}
}
GPUARRAY_PUBLIC int   GpuReductionAttr_issensitive  (const GpuReductionAttr*    grAttr){
	/**
	 * @brief Returns whether the reduction is "sensitive".
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
	
	switch (grAttr->op){
		case GA_REDUCE_MINANDARGMIN:
		case GA_REDUCE_MAXANDARGMAX:
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_ARGMAX:
		  return 1;
		default:
		  return 0;
	}
}
GPUARRAY_PUBLIC int   GpuReductionAttr_requiresS0   (const GpuReductionAttr*    grAttr){
	switch (grAttr->op){
		default: return 1;
	}
}
GPUARRAY_PUBLIC int   GpuReductionAttr_requiresD0   (const GpuReductionAttr*    grAttr){
	switch (grAttr->op){
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_ARGMAX:
		  return 0;
		default:
		  return 1;
	}
}
GPUARRAY_PUBLIC int   GpuReductionAttr_requiresD1   (const GpuReductionAttr*    grAttr){
	switch (grAttr->op){
		case GA_REDUCE_MINANDARGMIN:
		case GA_REDUCE_MAXANDARGMAX:
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_ARGMAX:
		  return 1;
		default:
		  return 0;
	}
}
GPUARRAY_PUBLIC void  GpuReductionAttr_free         (GpuReductionAttr*          grAttr){
	free(grAttr);
}
GPUARRAY_PUBLIC int   GpuReduction_new              (GpuReduction**             gr,
                                                     const GpuReductionAttr*    grAttr){
	if (!gr){
		return GA_INVALID_ERROR;
	}
	if (!grAttr){
		*gr = NULL;
		return GA_INVALID_ERROR;
	}
	
	*gr = calloc(1, sizeof(**gr));
	if (*gr){
		(*gr)->grAttr = *grAttr;
		(*gr)->gpuCtx = grAttr->gpuCtx;
		(*gr)->op     = grAttr->op;
		(*gr)->nds    = (int)grAttr->maxSrcDims;
		(*gr)->ndd    = (int)grAttr->maxDstDims;
		(*gr)->ndr    = (int)(grAttr->maxSrcDims-grAttr->maxDstDims);
		
		return reduxGenInit(*gr);
	}else{
		return GA_MEMORY_ERROR;
	}
}
GPUARRAY_PUBLIC void  GpuReduction_free             (GpuReduction*              gr){
	reduxGenCleanup(gr, !GA_NO_ERROR);
}
GPUARRAY_PUBLIC int   GpuReduction_call             (const GpuReduction*        gr,
                                                     GpuArray*                  d0,
                                                     GpuArray*                  d1,
                                                     const GpuArray*            s0,
                                                     unsigned                   reduxLen,
                                                     const int*                 reduxList,
                                                     int                        flags){
	redux_ctx ctxSTACK, *ctx = &ctxSTACK;
	memset(ctx, 0, sizeof(*ctx));

	ctx->gr        = gr;
	ctx->d0        = d0;
	ctx->d1        = d1;
	ctx->s0        = s0;
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

static int         reduxGetSumInit                  (int typecode, const char** property){
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

static int         reduxGetProdInit                 (int typecode, const char** property){
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

static int         reduxGetMinInit                  (int typecode, const char** property){
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

static int         reduxGetMaxInit                  (int typecode, const char** property){
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

static int         reduxGetAndInit                  (int typecode, const char** property){
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

static int         reduxGetOrInit                   (int typecode, const char** property){
	if (typecode == GA_POINTER ||
	    typecode == GA_BUFFER){
		return GA_UNSUPPORTED_ERROR;
	}
	*property = "0";
	return GA_NO_ERROR;
}

/**
 * Whether or not the typecode is a floating-point type.
 */

static int         reduxIsFloatingPoint             (int typecode){
	switch (typecode){
		case GA_HALF:
		case GA_HALF2:
		case GA_HALF4:
		case GA_HALF8:
		case GA_HALF16:
		case GA_FLOAT:
		case GA_FLOAT2:
		case GA_FLOAT4:
		case GA_FLOAT8:
		case GA_FLOAT16:
		case GA_DOUBLE:
		case GA_DOUBLE2:
		case GA_DOUBLE4:
		case GA_DOUBLE8:
		case GA_DOUBLE16:
		case GA_QUAD:
		case GA_CFLOAT:
		case GA_CDOUBLE:
		case GA_CQUAD:
		  return 1;
		default:
		  return 0;
	}
}

/**
 * Compute ceil(log2(x)).
 */

static unsigned    reduxCeilLog2                    (uint64_t x){
	int i;
	
	if (x <= 1){
		return 1;
	}
	for (i=0,x--;x;i++,x>>=1){}
	return i;
}

/**
 * Compute next power of 2.
 * 
 * If x is a power of two already, return x.
 */

static uint64_t    reduxNextPow2                    (uint64_t x){
	if (x & (x-1)){
		x |= x >>  1;
		x |= x >>  2;
		x |= x >>  4;
		x |= x >>  8;
		x |= x >> 16;
		x |= x >> 32;
		return x+1;
	}else{
		return x;
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

static int         reduxSortFlatInsensitive         (const void* a, const void* b){
	const axis_desc* xda  = (const axis_desc*)a;
	const axis_desc* xdb  = (const axis_desc*)b;

	if       ( axisIsReduced(xda)      && !axisIsReduced(xdb)){
		return +1;
	}else if (!axisIsReduced(xda)      &&  axisIsReduced(xdb)){
		return -1;
	}
	
	if       (axisGetS0AbsStride(xda)  <  axisGetS0AbsStride(xdb)){
		return +1;
	}else if (axisGetS0AbsStride(xda)  >  axisGetS0AbsStride(xdb)){
		return -1;
	}

	return 0;
}
static int         reduxSortFlatSensitive           (const void* a, const void* b){
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
		if       (axisGetS0AbsStride(xda)  <  axisGetS0AbsStride(xdb)){
			return +1;
		}else if (axisGetS0AbsStride(xda)  >  axisGetS0AbsStride(xdb)){
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

static int         reduxSortPtrS0AbsStride          (const void* a, const void* b){
	const axis_desc* xda  = *(const axis_desc* const*)a;
	const axis_desc* xdb  = *(const axis_desc* const*)b;
	
	if       (axisGetS0AbsStride(xda)  <  axisGetS0AbsStride(xdb)){
		return -1;
	}else if (axisGetS0AbsStride(xda)  >  axisGetS0AbsStride(xdb)){
		return +1;
	}

	return 0;
}
static int         reduxSortPtrByReduxNum           (const void* a, const void* b){
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
static int         reduxSortPtrD0WrSelect           (const void* a, const void* b){
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
	if       (axisGetD0AbsStride(xda)  <  axisGetD0AbsStride(xdb)){
		return -1;
	}else if (axisGetD0AbsStride(xda)  >  axisGetD0AbsStride(xdb)){
		return +1;
	}

	return 0;
}
static int         reduxSortPtrD1WrSelect           (const void* a, const void* b){
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
	if       (axisGetD1AbsStride(xda)  <  axisGetD1AbsStride(xdb)){
		return -1;
	}else if (axisGetD1AbsStride(xda)  >  axisGetD1AbsStride(xdb)){
		return +1;
	}

	return 0;
}
static int         reduxSortPtrInsertFinalOrder     (const void* a, const void* b){
	const axis_desc* xda  = *(const axis_desc* const*)a;
	const axis_desc* xdb  = *(const axis_desc* const*)b;
	
	
	/* All intra axes go first. */
	if       (axisIsIntra(xda)  &&  axisIsInter(xdb)){
		return -1;
	}else if (axisIsInter(xda)  &&  axisIsIntra(xdb)){
		return +1;
	}
	
	if (axisIsIntra(xda)){
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
		
		if       (axisGetS0AbsStride(xda)  <  axisGetS0AbsStride(xdb)){
			return -1;
		}else if (axisGetS0AbsStride(xda)  >  axisGetS0AbsStride(xdb)){
			return +1;
		}
	}

	return 0;
}


/* Axis Description API */

/**
 * @brief Initialize Axis Description.
 */

static void        axisInit                         (axis_desc*           axis,
                                                     ssize_t              len,
                                                     ssize_t              s0S){
	memset(axis, 0, sizeof(*axis));
	
	axis->reduxNum = -1;
	axis->ibNum    = -1;
	axis->perm     = 0;
	axis->len      = len;
	axis->splitLen = 1;
	axis->i0S      = 0;
	
	axis->s0S      = s0S;
	axis->d0S      = 0;
	axis->d1S      = 0;
}

/**
 * @brief Mark axis as reduction axis, with position reduxNum in the axis list.
 */

static void        axisMarkReduced                  (axis_desc*           axis, int    reduxNum){
	axis->isReduced = 1;
	axis->reduxNum  = reduxNum;
}

/**
 * @brief Mark axis as (split) intrablock axis.
 */

static void        axisMarkIntraBlock               (axis_desc*           axis,
                                                     int                  ibNum,
                                                     size_t               ibLen){
	axis->isIntra  = 1;
	axis->ibNum    = ibNum;
	axis->splitLen = ibLen;
}

/**
 * @brief Get properties of an axis.
 */

static int         axisGetReduxNum                  (const axis_desc*     axis){
	return axis->reduxNum;
}
static size_t      axisGetLen                       (const axis_desc*     axis){
	return axis->len;
}
static size_t      axisGetIntraLen                  (const axis_desc*     axis){
	if       (axisIsSplit(axis)){
		return axis->splitLen;
	}else if (axisIsIntra(axis)){
		return axis->len;
	}else{
		return 1;
	}
}
static size_t      axisGetInterLen                  (const axis_desc*     axis){
	if       (axisIsSplit(axis)){
		return DIVIDECEIL(axis->len, axis->splitLen);
	}else if (axisIsIntra(axis)){
		return 1;
	}else{
		return axis->len;
	}
}
static size_t      axisGetIntraInterLen             (const axis_desc*     axis){
	return axisGetIntraLen(axis)*axisGetInterLen(axis);
}
static ssize_t     axisGetS0Stride                  (const axis_desc*     axis){
	return axisGetLen(axis) > 1 ? axis->s0S : 0;
}
static size_t      axisGetS0AbsStride               (const axis_desc*     axis){
	return axisGetS0Stride(axis)<0 ? -(size_t)axisGetS0Stride(axis):
	                                  +(size_t)axisGetS0Stride(axis);
}
static ssize_t     axisGetD0Stride                  (const axis_desc*     axis){
	return axisGetLen(axis) > 1 ? axis->d0S : 0;
}
static size_t      axisGetD0AbsStride               (const axis_desc*     axis){
	return axisGetD0Stride(axis)<0 ? -(size_t)axisGetD0Stride(axis):
	                                  +(size_t)axisGetD0Stride(axis);
}
static ssize_t     axisGetD1Stride                  (const axis_desc*     axis){
	return axisGetLen(axis) > 1 ? axis->d1S : 0;
}
static size_t      axisGetD1AbsStride               (const axis_desc*     axis){
	return axisGetD1Stride(axis)<0 ? -(size_t)axisGetD1Stride(axis):
	                                     +(size_t)axisGetD1Stride(axis);
}
static size_t      axisGetI0Stride                  (const axis_desc*     axis){
	return axis->i0S;
}
static void        axisSetI0Stride                  (axis_desc*           axis,
                                                     size_t               i0S){
	axis->i0S = i0S;
}
static unsigned    axisGetPerm                      (const axis_desc*     axis){
	return axis->perm;
}
static int         axisGetIBNum                     (const axis_desc*     axis){
	return axis->ibNum;
}
static void        axisSetPerm                      (axis_desc*           axis,
                                                     unsigned             perm){
	axis->perm = perm;
}
static int         axisIsReduced                    (const axis_desc*     axis){
	return axis->isReduced;
}
static int         axisIsIntra                      (const axis_desc*     axis){
	return axis->isIntra;
}
static int         axisIsInter                      (const axis_desc*     axis){
	return !axisIsIntra(axis);
}
static int         axisIsSplit                      (const axis_desc*     axis){
	return axisIsIntra(axis) && axis->splitLen != axis->len;
}
static size_t      reduxInvEstimateParallelism      (const redux_ctx*     ctx){
	return reduxGenEstimateParallelism(ctx->gr);
}
static int         reduxInvRequiresS0               (const redux_ctx*     ctx){
	return reduxGenRequiresS0(ctx->gr);
}
static int         reduxInvRequiresD0               (const redux_ctx*     ctx){
	return reduxGenRequiresD0(ctx->gr);
}
static int         reduxInvRequiresD1               (const redux_ctx*     ctx){
	return reduxGenRequiresD1(ctx->gr);
}

/**
 * @brief Get description of source axis with given number.
 */

static axis_desc*  reduxInvGetSrcAxis               (const redux_ctx*       ctx, int i){
	return &ctx->xdSrc[i];
}

/**
 * @brief Get description of source axis with given number in sort-order.
 */

static axis_desc*  reduxInvGetSrcSortAxis           (const redux_ctx*       ctx, int i){
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

static int         reduxTryFlattenOut               (const redux_ctx*       ctx,
                                                     const axis_desc*       axis){
	if ((axisGetLen   (axis) == 1                   )||
	    (axisIsReduced(axis) && ctx->zeroRdxAxes > 0)){
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

static int         reduxTryFlattenInto              (redux_ctx*             ctx,
                                                     axis_desc*             into,
                                                     const axis_desc*       from){
	int signS0    = 0, signD0    = 0, signD1    = 0,
	    reverseS0 = 0, reverseD0 = 0, reverseD1 = 0;
	
	if (axisIsReduced     (into) != axisIsReduced     (from)                 ||
	    axisGetS0AbsStride(into) != axisGetS0AbsStride(from)*axisGetLen(from)){
		return 0;
	}
	
	if (reduxInvRequiresD0(ctx)  &&
	    axisGetD0AbsStride(into) != axisGetD0AbsStride(from)*axisGetLen(from)){
		return 0;
	}
	
	if (reduxInvRequiresD1(ctx)  &&
	    axisGetD1AbsStride(into) != axisGetD1AbsStride(from)*axisGetLen(from)){
		return 0;
	}
	
	signS0    = (axisGetS0Stride(into)^axisGetS0Stride(from)) < 0;
	signD0    = (axisGetD0Stride(into)^axisGetD0Stride(from)) < 0;
	signD1    = (axisGetD1Stride(into)^axisGetD1Stride(from)) < 0;
	reverseS0 = signS0;
	reverseD0 = signD0 && reduxInvRequiresD0(ctx);
	reverseD1 = signD1 && reduxInvRequiresD1(ctx);
	
	if (GpuReductionAttr_issensitive(&ctx->gr->grAttr)){
		if (reverseS0 || reverseD0 || reverseD1){
			return 0;
		}
	}
	
	if (reduxInvRequiresD0(ctx) &&
	    reduxInvRequiresD1(ctx) &&
	    reverseD0 != reverseD1){
		/* Either both, or neither, of dst and dstArg must require reversal. */
		return 0;
	}
	
	if (reverseS0){
		ctx->S0Off += (ssize_t)(axisGetLen(from)-1)*axisGetS0Stride(from);
		into->s0S   = -axisGetS0Stride(from);
	}else{
		into->s0S   =  axisGetS0Stride(from);
	}
	
	if (reverseD0){
		ctx->D0Off += (ssize_t)(axisGetLen(from)-1)*axisGetD0Stride(from);
		into->d0S   = -axisGetD0Stride(from);
	}else{
		into->d0S   =  axisGetD0Stride(from);
	}
	
	if (reverseD1){
		ctx->D1Off += (ssize_t)(axisGetLen(from)-1)*axisGetD1Stride(from);
		into->d1S   = -axisGetD1Stride(from);
	}else{
		into->d1S   =  axisGetD1Stride(from);
	}
	
	into->len *= axisGetLen(from);
	
	return 1;
}

/**
 * Sort an array of *pointers* to axes by the given comparison function, while
 * not touching the axes themselves.
 */

static void        reduxSortAxisPtrsBy              (axis_desc**            ptrs,
                                                     axis_desc*             axes,
                                                     size_t                 numAxes,
                                                     int(*fn)(const void*, const void*)){
	size_t i;
	
	for (i=0;i<numAxes;i++){
		ptrs[i] = &axes[i];
	}
	
	qsort(ptrs, numAxes, sizeof(*ptrs), fn);
}


/**
 * @brief Initialize generator context.
 * 
 * After this function, calling reduxGenCleanup*() becomes safe.
 */

static int         reduxGenInit                     (GpuReduction*        gr){
	gr->kArgTypeCodes = NULL;
	gr->kSourceCode   = NULL;
	gr->kErrorString  = NULL;
	gr->kNumArgs      = 0;
	
	return reduxGenInferProperties(gr);
}

/**
 * @brief Begin inferring the properties of the reduction operator.
 */

static int         reduxGenInferProperties          (GpuReduction*        gr){
	int i;
	
	/**
	 * Source code buffer preallocation failed?
	 */
	
	if (strb_ensure(&gr->s, 32*1024) != 0){
		return reduxGenCleanupMsg(gr, GA_MEMORY_ERROR,
		       "Could not preallocate source code buffer!\n");
	}
	srcbInit(&gr->srcGen, &gr->s);
	
	
	/**
	 * Type management.
	 * 
	 * Read out the various typecodes from the attributes.
	 */
	
	gr->TS0tc  = gr->grAttr.s0Typecode;
	gr->TD0tc  = gr->grAttr.d0Typecode;
	gr->TD1tc  = gr->grAttr.d1Typecode;
	gr->TI0tc  = gr->grAttr.i0Typecode;
	gr->TS32tc = GA_INT;
	gr->TU32tc = GA_UINT;
	gr->TS64tc = GA_LONG;
	gr->TU64tc = GA_ULONG;
	reduxGenSetKTypes(gr);
	
	
	/**
	 * Compute number of kernel arguments and construct kernel argument
	 * typecode list.
	 */
	
	reduxGenSetMaxBS(gr);
	reduxGenIterArgs(gr, reduxGenCountArgs, &gr->kNumArgs);
	gr->kArgTypeCodes = calloc(gr->kNumArgs, sizeof(*gr->kArgTypeCodes));
	if (!gr->kArgTypeCodes){
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
 * Compute maximum block size we shall support in generated kernels.
 */

static void        reduxGenSetMaxBS                 (GpuReduction*        gr){
	gr->maxBS = gr->grAttr.maxLM/reduxGenGetReduxStateSize(gr);
	gr->maxBS = gr->maxBS < gr->grAttr.maxLg ? gr->maxBS : gr->grAttr.maxLg;
	gr->maxBS = gr->maxBS < gr->grAttr.maxL0 ? gr->maxBS : gr->grAttr.maxL0;
	
	/**
	 * In practice we want a moderate amount of blocks, not just one monolith
	 * that occupies a processor for its entire lifetime. E.g. An NVIDIA GPU
	 * supports 1024 threads / block, but we shall gun for less than that.
	 * 
	 * Our heuristic shall be to divide by 4 the maximum number of threads per
	 * block, so that there's 4 times more blocks than normally there would be.
	 * This helps on many fronts:
	 * 
	 *   - A smaller "tail effect" when the last huge block must wait its turn
	 *     and then delays the completion of the entire grid
	 *   - The horizontal reductions take less time per block, and sometimes
	 *     horizontal reduction time can dominate performance.
	 *   - Less time taken for across-thread synchronization; And whenever a
	 *     block's threads are stalled waiting for synchronization, another
	 *     block's threads can fill in with their global memory requests.
	 */
	
	if (gr->maxBS >= 16){
		gr->maxBS /= 4;
	}
	
	/* Since ceil(log2(maxBS)) is also heavily used, compute it here */
	gr->log2MaxBS = reduxCeilLog2(gr->maxBS);
}

/**
 * Decide on the TK* accumulator types and initializers we will use.
 * 
 * Currently, the only special thing we do is to promote the accumulator type
 * to GA_FLOATx if the source type is GA_HALFx:
 * 
 *     TPS0 = promotion(TS0)
 * 
 * Therefore, it is currently always the case that TK0 == TPS0.
 * 
 * In the future this might become wierder when the accumulator is a Kahan
 * summation, for instance, and then TK0 != promoted(TS0).
 * 
 * If the user guaranteed to us through gr->grAttr that TK1 can be made
 * narrower than 64-bit, this is also where we'd take this into account.
 * For now we default TK1 to exactly TI0.
 */

static void        reduxGenSetKTypes                (GpuReduction*        gr){
	const gpuarray_type *TK0     = NULL, *TK1     = NULL, *TPS0    = NULL;
	const char*          TK0init = NULL;
	
	/**
	 * Handle TPS0 type promotion....
	 */
	
	switch (gr->TS0tc){
		case GA_HALF:
		  TPS0 = gpuarray_get_type(GA_FLOAT);
		break;
		case GA_HALF2:
		  TPS0 = gpuarray_get_type(GA_FLOAT2);
		break;
		case GA_HALF4:
		  TPS0 = gpuarray_get_type(GA_FLOAT4);
		break;
		case GA_HALF8:
		  TPS0 = gpuarray_get_type(GA_FLOAT8);
		break;
		case GA_HALF16:
		  TPS0 = gpuarray_get_type(GA_FLOAT16);
		break;
		default:
		  TPS0 = gpuarray_get_type(gr->TS0tc);
	}
	gr->TPS0tc = TPS0->typecode;
	
	
	/**
	 * Each operator may define and initialize TK0 and/or TK1 any way
	 * they want.
	 */
	
	switch (gr->grAttr.op){
		case GA_REDUCE_SUM:
		  TK0 = TPS0;
		  reduxGetSumInit (TK0->typecode, &TK0init);
		  gr->TK0.align = TK0->align;
		  gr->TK0.size  = TK0->size;
		  sprintf(gr->TK0.defn, "%s", TK0->cluda_name);
		  sprintf(gr->TK0.init, "%s", TK0init);
		break;
		case GA_REDUCE_PRODNZ:
		case GA_REDUCE_PROD:
		  TK0 = TPS0;
		  reduxGetProdInit(TK0->typecode, &TK0init);
		  gr->TK0.align = TK0->align;
		  gr->TK0.size  = TK0->size;
		  sprintf(gr->TK0.defn, "%s", TK0->cluda_name);
		  sprintf(gr->TK0.init, "%s", TK0init);
		break;
		case GA_REDUCE_MINANDARGMIN:
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_MIN:
		  TK0 = TPS0;
		  TK1 = gpuarray_get_type(gr->TI0tc);
		  reduxGetMinInit (TK0->typecode, &TK0init);
		  gr->TK0.align = TK0->align;
		  gr->TK0.size  = TK0->size;
		  sprintf(gr->TK0.defn, "%s", TK0->cluda_name);
		  sprintf(gr->TK0.init, "%s", TK0init);
		  gr->TK1.align = TK1->align;
		  gr->TK1.size  = TK1->size;
		  sprintf(gr->TK1.defn, "%s", TK1->cluda_name);
		  sprintf(gr->TK1.init, "0");
		break;
		case GA_REDUCE_MAXANDARGMAX:
		case GA_REDUCE_ARGMAX:
		case GA_REDUCE_MAX:
		  TK0 = TPS0;
		  TK1 = gpuarray_get_type(gr->TI0tc);
		  reduxGetMaxInit (TK0->typecode, &TK0init);
		  gr->TK0.align = TK0->align;
		  gr->TK0.size  = TK0->size;
		  sprintf(gr->TK0.defn, "%s", TK0->cluda_name);
		  sprintf(gr->TK0.init, "%s", TK0init);
		  gr->TK1.align = TK1->align;
		  gr->TK1.size  = TK1->size;
		  sprintf(gr->TK1.defn, "%s", TK1->cluda_name);
		  sprintf(gr->TK1.init, "0");
		break;
		case GA_REDUCE_ALL:
		case GA_REDUCE_AND:
		  TK0 = TPS0;
		  reduxGetAndInit (TK0->typecode, &TK0init);
		  gr->TK0.align = TK0->align;
		  gr->TK0.size  = TK0->size;
		  sprintf(gr->TK0.defn, "%s", TK0->cluda_name);
		  sprintf(gr->TK0.init, "%s", TK0init);
		break;
		case GA_REDUCE_ANY:
		case GA_REDUCE_XOR:
		case GA_REDUCE_OR:
		  TK0 = TPS0;
		  reduxGetOrInit  (TK0->typecode, &TK0init);
		  gr->TK0.align = TK0->align;
		  gr->TK0.size  = TK0->size;
		  sprintf(gr->TK0.defn, "%s", TK0->cluda_name);
		  sprintf(gr->TK0.init, "%s", TK0init);
		break;
		default:
		  ;/* Unreachable */
	}
}

/**
 * Iterate over the arguments of the reduction operator.
 */

static void        reduxGenIterArgs                 (const GpuReduction*  gr,
                                                     GpuReductionIterFn   fn,
                                                     void*                user){
	int k;
	
	/**
	 * Template selector
	 */
	
	fn(gr, gr->TU32tc, "TU32",                              "selector",    0, user);
	
	/**
	 * "Universal" parameters describing the partitioning of the problem.
	 */
	
	fn(gr, gr->TU64tc, "TU64",                              "U",           0, user);
	fn(gr, gr->TU64tc, "TU64",                              "V",           0, user);
	fn(gr, gr->TU64tc, "TU64",                              "B",           0, user);
	fn(gr, gr->TU32tc, "TU32",                              "D",           0, user);
	fn(gr, gr->TU32tc, "TU32",                              "Dunit",       0, user);
	fn(gr, gr->TU32tc, "TU32",                              "H",           0, user);
	
	/* Global Lattice Coordinates */
	fn(gr, gr->TU32tc, "TU32",                              "LSlice",      0, user);
	fn(gr, gr->TU32tc, "TU64",                              "LPadded",     0, user);
	for (k=0;k < gr->nds;k++){
		fn(gr, gr->TU64tc, "TU64",                              "L%d",         k, user);
	}
	for (k=0;k < gr->log2MaxBS;k++){
		fn(gr, gr->TU32tc, "TU32",                              "L%di",        k, user);
	}
	
	/* S0 Lattice */
	if (reduxGenKernelRequiresLatticeS0(gr)){
		fn(gr, GA_BUFFER,  "const GLOBAL_MEM char* restrict",   "S0",          0, user);
		fn(gr, gr->TS64tc, "TS64",                              "S0Off",       0, user);
		for (k=0;k < gr->nds;k++){
			fn(gr, gr->TS64tc, "TS64",                              "S0J%d",       k, user);
		}
		for (k=0;k < gr->log2MaxBS;k++){
			fn(gr, gr->TS64tc, "TS64",                              "S0S%di",      k, user);
		}
	}
	
	/* d0 Lattice */
	if (reduxGenKernelRequiresLatticeD0(gr)){
		fn(gr, GA_BUFFER,  "GLOBAL_MEM char* restrict",         "D0",          0, user);
		fn(gr, gr->TS64tc, "TS64",                              "D0Off",       0, user);
		for (k=0;k < gr->ndd;k++){
			fn(gr, gr->TS64tc, "TS64",                              "D0J%d",       k, user);
		}
		for (k=0;k < gr->log2MaxBS;k++){
			fn(gr, gr->TS64tc, "TS64",                              "D0S%di",      k, user);
		}
	}
	
	/* D1 Lattice */
	if (reduxGenKernelRequiresLatticeD1(gr)){
		fn(gr, GA_BUFFER,  "GLOBAL_MEM char* restrict",         "D1",          0, user);
		fn(gr, gr->TS64tc, "TS64",                              "D1Off",       0, user);
		for (k=0;k < gr->ndd;k++){
			fn(gr, gr->TS64tc, "TS64",                              "D1J%d",       k, user);
		}
		for (k=0;k < gr->log2MaxBS;k++){
			fn(gr, gr->TS64tc, "TS64",                              "D1S%di",      k, user);
		}
	}
	
	/* I0 Lattice */
	if (reduxGenKernelRequiresLatticeI0(gr)){
		for (k=0;k < gr->nds;k++){
			fn(gr, gr->TS64tc, "TS64",                              "I0J%d",       k, user);
		}
		for (k=0;k < gr->log2MaxBS;k++){
			fn(gr, gr->TS64tc, "TS64",                              "I0S%di",      k, user);
		}
	}
	
	/* Workspace */
	if (reduxGenKernelRequiresWspace(gr)){
		fn(gr, GA_BUFFER,  "GLOBAL_MEM char* restrict",         "W",           0, user);
		if (reduxGenKernelRequiresStateK0(gr)){
			fn(gr, gr->TS64tc, "TS64",                              "W0Off",       0, user);
			fn(gr, gr->TS64tc, "TS64",                              "SHMEMK0Off",  0, user);
		}
		if (reduxGenKernelRequiresStateK1(gr)){
			fn(gr, gr->TS64tc, "TS64",                              "W1Off",       0, user);
			fn(gr, gr->TS64tc, "TS64",                              "SHMEMK1Off",  0, user);
		}
	}
	
	/* Intra-Block Permute Core */
	for (k=0;k < gr->log2MaxBS;k++){
		fn(gr, gr->TU32tc, "TU32",                              "perm%di",     k, user);
	}
}

/**
 * @brief Generate the kernel source code for the reduction.
 *
 * @return GA_MEMORY_ERROR if not enough memory left; GA_NO_ERROR otherwise.
 */

static int         reduxGenSrc                      (GpuReduction*        gr){
	GpuReductionAttr_appendopname(&gr->grAttr, sizeof(gr->kName), gr->kName);
	
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

static void        reduxGenSrcAppend                (GpuReduction*        gr){
	reduxGenSrcAppendIncludes     (gr);
	reduxGenSrcAppendMacroTypedefs(gr);
	reduxGenSrcAppendReduxKernel  (gr);
}
static void        reduxGenSrcAppendIncludes        (GpuReduction*        gr){
	srcbAppends(&gr->srcGen, "/* Includes */\n");
	srcbAppends(&gr->srcGen, "#include \"cluda.h\"\n");
	srcbAppends(&gr->srcGen, "\n");
	srcbAppends(&gr->srcGen, "\n");
	srcbAppends(&gr->srcGen, "\n");
}
static void        reduxGenSrcAppendMacroTypedefs   (GpuReduction*        gr){
	/**
	 * Typedefs of various types.
	 */
	
	if (reduxGenRequiresS0(gr)){
		srcbAppendf(&gr->srcGen, "typedef %-20s TS0;\n",  gpuarray_get_type(gr->TS0tc )->cluda_name);
		srcbAppendf(&gr->srcGen, "typedef %-20s TPS0;\n", gpuarray_get_type(gr->TPS0tc)->cluda_name);
	}
	if (reduxGenRequiresD0(gr)){
		srcbAppendf(&gr->srcGen, "typedef %-20s TD0;\n",  gpuarray_get_type(gr->TD0tc )->cluda_name);
	}
	if (reduxGenRequiresD1(gr)){
		srcbAppendf(&gr->srcGen, "typedef %-20s TD1;\n",  gpuarray_get_type(gr->TD1tc )->cluda_name);
	}
	if (reduxGenKernelRequiresLatticeI0(gr)){
		srcbAppendf(&gr->srcGen, "typedef %-20s TI0;\n",  gpuarray_get_type(gr->TI0tc )->cluda_name);
	}
	srcbAppendf(&gr->srcGen, "typedef %-20s TS32;\n", gpuarray_get_type(gr->TS32tc)->cluda_name);
	srcbAppendf(&gr->srcGen, "typedef %-20s TU32;\n", gpuarray_get_type(gr->TU32tc)->cluda_name);
	srcbAppendf(&gr->srcGen, "typedef %-20s TS64;\n", gpuarray_get_type(gr->TS64tc)->cluda_name);
	srcbAppendf(&gr->srcGen, "typedef %-20s TU64;\n", gpuarray_get_type(gr->TU64tc)->cluda_name);
	if (reduxGenKernelRequiresStateK0(gr)){
		srcbAppendf(&gr->srcGen, "typedef %-20s TK0;\n", gr->TK0.defn);
	}
	if (reduxGenKernelRequiresStateK1(gr)){
		srcbAppendf(&gr->srcGen, "typedef %-20s TK1;\n", gr->TK1.defn);
	}
	srcbAppendf(&gr->srcGen, "\n\n\n\n");
	
	
	/**
	 * DECLREDUXSTATE, INITREDUXSTATE and SETREDUXSTATE macros.
	 */
	
	if       ( reduxGenKernelRequiresStateK0(gr) &&  reduxGenKernelRequiresStateK1(gr)){
		srcbAppendf(&gr->srcGen,
		            "#define DECLREDUXSTATE(V, I) TK0 V;TK1 I;\n"
		            "#define INITREDUXSTATE(V, I) do{(V) = (%s);(I) = (%s);}while(0)\n"
		            "#define SETREDUXSTATE(V, I, v, i)  do{(V) = (v);(I) = (i);}while(0)\n",
		            gr->TK0.init, gr->TK1.init);
	}else if ( reduxGenKernelRequiresStateK0(gr) && !reduxGenKernelRequiresStateK1(gr)){
		srcbAppendf(&gr->srcGen,
		            "#define DECLREDUXSTATE(V, I) TK0 V;\n"
		            "#define INITREDUXSTATE(V, I) do{(V) = (%s);}while(0)\n"
		            "#define SETREDUXSTATE(V, I, v, i)  do{(V) = (v);}while(0)\n",
		            gr->TK0.init);
	}else if (!reduxGenKernelRequiresStateK0(gr) &&  reduxGenKernelRequiresStateK1(gr)){
		srcbAppendf(&gr->srcGen,
		            "#define DECLREDUXSTATE(V, I) TK1 I;\n"
		            "#define INITREDUXSTATE(V, I) do{(I) = (%s);}while(0)\n"
		            "#define SETREDUXSTATE(V, I, v, i)  do{(I) = (i);}while(0)\n",
		            gr->TK1.init);
	}
	
	
	/**
	 * LOADS0(v, p) macro.
	 * 
	 * Loads a TK0-typed value v from a TS-typed source pointer p, promoting
	 * through type TPS0.
	 * 
	 * In some future, TK0 will not equal TPS0, and so a cast as done below will not
	 * necessarily be valid. Instead it may require an assignment to a struct member.
	 */
	
	if (reduxGenKernelRequiresLatticeS0(gr)){
		if (gr->TS0tc == GA_HALF && gr->TPS0tc == GA_FLOAT){
			srcbAppends(&gr->srcGen, "#define LOADS0(v, p) do{(v) = (TK0)(TPS0)load_half((const TS0* restrict)(p));}while(0)\n");
		}else{
			srcbAppends(&gr->srcGen, "#define LOADS0(v, p) do{(v) = (TK0)(TPS0)*(const TS0* restrict)(p);}while(0)\n");
		}
	}else{
		srcbAppends(&gr->srcGen, "#define LOADS0(p, v) do{}while(0)\n");
	}
	
	
	/**
	 * REDUX macro.
	 * 
	 * Performs a reduction operation, jointly reducing a datum v and its
	 * flattened index i into reduction states V and I respectively.
	 */
	
	switch (gr->grAttr.op){
		case GA_REDUCE_SUM:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V) += (v);                     \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_PROD:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V) *= (v);                     \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_PRODNZ:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        if((v) != 0){(V) *= (v);}       \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_MIN:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V)  = min((V), (v));           \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_MAX:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V)  = max((V), (v));           \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_ARGMIN:
		case GA_REDUCE_MINANDARGMIN:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V)  = min((V), (v));           \\\n"
		                           "        if((V) == (v)){                 \\\n"
		                           "            (I) = (i);                  \\\n"
		                           "        }                               \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_ARGMAX:
		case GA_REDUCE_MAXANDARGMAX:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V)  = max((V), (v));           \\\n"
		                           "        if((V) == (v)){                 \\\n"
		                           "            (I) = (i);                  \\\n"
		                           "        }                               \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_AND:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V) &= (v);                     \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_OR:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V) |= (v);                     \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_XOR:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V) ^= (v);                     \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_ALL:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V)  = (V) && (v);              \\\n"
		                           "    }while(0)\n");
		break;
		case GA_REDUCE_ANY:
		  srcbAppendf(&gr->srcGen, "#define REDUX(V, I, v, i) do{           \\\n"
		                           "        (V)  = (V) || (v);              \\\n"
		                           "    }while(0)\n");
		break;
		default:
		  /* Unreachable */
		break;
	}
	
	
	/**
	 * HREDUX macro.
	 * 
	 * Performs a horizontal reduction operation, first intra-block permuting
	 * the data and its index and then reducing it till done.
	 * 
	 *   - If D==LDIM_0, then no horizontal (across-block) reductions are
	 *     really needed. In this case, the permutation tp:
	 *       - Is fully in-bounds (tp < LDIM_0 for all threads)
	 *       - Exists firstly  to make it easy to mask writes     (hard).
	 *       - Exists secondly to optimize memory write bandwidth (soft).
	 *     and the value H should be equal to D and to LDIM_0
	 *   - If D<LDIM_0,  then horizontal reductions are needed. In this case,
	 *     the permutation tp:
	 *       - *May* be partially out-of-bounds (tp >= LDIM_0 for some threads)
	 *       - Exists firstly  to make it easy to mask writes     (hard).
	 *       - Exists secondly to enable a tree reduction         (hard).
	 *       - Exists thirdly  to optimize memory write bandwidth (soft).
	 *     and the value H must be a power of 2 and shall be set to nextPow2(bs).
	 * 
	 * E.g. Suppose that a block configuration was D=999, H=1 (bs=999). A
	 *      permutation we might want is
	 *          [0,...,332,333,...,665,666,...,998]
	 *      and we want H = 999.
	 * E.g. Suppose that a block configuration was D=257, H=3 (bs=771). A
	 *      permutation we might want is
	 *          [0,...,256,512,...,768,1024,...,1280]
	 *      and we want H = 1024.
	 * E.g. Suppose that a block configuration was D=33, H=17 (bs=561). A
	 *      permutation we might want is
	 *          [0,...,32,64,...,96,128,...,160,...,960,...,992,1024,...,1056]
	 *      and we want H = 1024.
	 * E.g. Suppose that a block configuration was D=16, H=16 (bs=256). A
	 *      permutation we might want is
	 *          [0,...255]
	 *      and we want H = 256.
	 * 
	 */
	
	srcbAppends(&gr->srcGen,
	"#define HREDUX(SHMEMK0, SHMEMK1, perm, k0, k1)     \\\n"
	"    do{                                            \\\n"
	"        if(D < LDIM_0){                            \\\n"
	"            /* SPECIAL FIRST REDUCTION: */         \\\n"
	"            h = H;                                 \\\n"
	"                                                   \\\n"
	"            /* LO Half */                          \\\n"
	"            if(perm < h){                          \\\n"
	"                SETREDUXSTATE(SHMEMK0[perm],       \\\n"
	"                              SHMEMK1[perm],       \\\n"
	"                              k0,                  \\\n"
	"                              k1);                 \\\n"
	"            }                                      \\\n"
	"            local_barrier();                       \\\n"
	"                                                   \\\n"
	"            /* HI Half */                          \\\n"
	"            if(perm >= h){                         \\\n"
	"                REDUX        (SHMEMK0[perm-h],     \\\n"
	"                              SHMEMK1[perm-h],     \\\n"
	"                              k0,                  \\\n"
	"                              k1);                 \\\n"
	"            }                                      \\\n"
	"            local_barrier();                       \\\n"
	"                                                   \\\n"
	"            /* Follow-up reductions */             \\\n"
	"            while((h >>= 1) >= D){                 \\\n"
	"                if(LID_0 < h){                     \\\n"
	"                    REDUX(SHMEMK0[LID_0],          \\\n"
	"                          SHMEMK1[LID_0],          \\\n"
	"                          SHMEMK0[LID_0+h],        \\\n"
	"                          SHMEMK1[LID_0+h]);       \\\n"
	"                }                                  \\\n"
	"                local_barrier();                   \\\n"
	"            }                                      \\\n"
	"        }else{                                     \\\n"
	"            /* All-permute */                      \\\n"
	"            SETREDUXSTATE(SHMEMK0[perm],           \\\n"
	"                          SHMEMK1[perm],           \\\n"
	"                          k0,                      \\\n"
	"                          k1);                     \\\n"
	"            local_barrier();                       \\\n"
	"        }                                          \\\n"
	"    }while(0)\n");
	
	/**
	 * STORED0 macro.
	 * 
	 * Stores a TK0-typed value v into a TD0-typed destination pointer p.
	 */
	
	if (reduxGenKernelRequiresLatticeD0(gr)){
		if (gr->TD0tc == GA_HALF && gr->TPS0tc == GA_FLOAT){
			srcbAppends(&gr->srcGen, "#define STORED0(p, v) do{store_half((TD0* restrict)(p), (v));}while(0)\n");
		}else{
			srcbAppends(&gr->srcGen, "#define STORED0(p, v) do{*(TD0* restrict)(p) = (v);}while(0)\n");
		}
	}else{
		srcbAppends(&gr->srcGen, "#define STORED0(p, v) do{}while(0)\n");
	}
	
	
	/**
	 * STORED1 macro.
	 * 
	 * Stores a TK1-typed value v into a TD1-typed destination pointer p.
	 */
	
	if (reduxGenKernelRequiresLatticeD1(gr)){
		srcbAppends(&gr->srcGen, "#define STORED1(p, v) do{*(TD1* restrict)(p) = (v);}while(0)\n");
	}else{
		srcbAppends(&gr->srcGen, "#define STORED1(p, v) do{}while(0)\n");
	}
	
	
	/**
	 * DIVIDECEIL macro.
	 */
	
	srcbAppends(&gr->srcGen, "#define DIVIDECEIL(a,b) (((a)+(b)-1)/(b))\n\n\n\n\n");
}
static void        reduxGenSrcAppendReduxKernel     (GpuReduction*        gr){
	reduxGenSrcAppendPrototype   (gr);
	srcbAppends                  (&gr->srcGen, "{\n");
	reduxGenSrcAppendDecode      (gr);
	
	/**
	 * PERFORM REDUCTION.
	 * 
	 * We either perform Phase 0 or Phase 1 according to the selector argument.
	 * 
	 * Phase 0 is the primary worker and, in special cases, is the only
	 * necessary phase. However, it may occasionally do only part of a
	 * reduction, in which case it leaves the partial reduction results in a
	 * workspace that is then read by Phase 1.
	 * 
	 * Phase 1 is a fixup phase that collects any partial reduction results
	 * from Phase 0 and completes the reduction before writing to the final
	 * destination.
	 * 
	 * The template selector indicates one of several specialized versions of
	 * the kernel to be executed. It indicates phase, which is the split axis,
	 * and which axis if any is "huge".
	 */
	
	srcbAppends                  (&gr->srcGen, "    if(selector&1){\n");
	reduxGenSrcAppendPhase1      (gr);
	srcbAppends                  (&gr->srcGen, "    }else if(selector ==  0){\n");
	reduxGenSrcAppendPhase0      (gr,   0);
	srcbAppends                  (&gr->srcGen, "    }else if(selector ==  2){\n");
	reduxGenSrcAppendPhase0      (gr,   2);
	srcbAppends                  (&gr->srcGen, "    }else if(selector ==  4){\n");
	reduxGenSrcAppendPhase0      (gr,   4);
	srcbAppends                  (&gr->srcGen, "    }else if(selector ==  6){\n");
	reduxGenSrcAppendPhase0      (gr,   6);
	srcbAppends                  (&gr->srcGen, "    }else if(selector ==  8){\n");
	reduxGenSrcAppendPhase0      (gr,   8);
	srcbAppends                  (&gr->srcGen, "    }else if(selector == 10){\n");
	reduxGenSrcAppendPhase0      (gr,  10);
	srcbAppends                  (&gr->srcGen, "    }else if(selector == 12){\n");
	reduxGenSrcAppendPhase0      (gr,  12);
	srcbAppends                  (&gr->srcGen, "    }else if(selector == 14){\n");
	reduxGenSrcAppendPhase0      (gr,  14);
	srcbAppends                  (&gr->srcGen, "    }\n");
	srcbAppends                  (&gr->srcGen, "}\n");
}
static void        reduxGenSrcAppendPrototype       (GpuReduction*        gr){
	int i=0;

	srcbAppendf(&gr->srcGen,
	"KERNEL void\n"
	"#if defined(__CUDACC__)\n"
	"__launch_bounds__(%d, 8)\n"
	"#endif\n",
	            gr->maxBS);
	srcbAppendf(&gr->srcGen,
	"%s(\n                  ",
	            gr->kName);
	reduxGenIterArgs(gr, reduxGenAppendArg, &i);
	srcbAppends(&gr->srcGen, ")");
}
static void        reduxGenSrcAppendDecode          (GpuReduction*        gr){
	int i;

	srcbAppends(&gr->srcGen,
	"    GA_DECL_SHARED_BODY(char, SHMEM)\n");
	if (reduxGenKernelRequiresLatticeI0(gr)){
		srcbAppends(&gr->srcGen,
		"    TI0 I0;\n");
	}
	srcbAppends(&gr->srcGen,
	"    TK0 tmpK0;\n"
	"    DECLREDUXSTATE(K0,    K1)\n"
	"    INITREDUXSTATE(K0,    K1);\n"
	"    \n"
	"    TU64        z, h, k;\n"
	"    \n"
	/**
	 *  +-------------+-------------+------------+---------------------------------+
	 *  |  misalignL  |  misalignR  |  doFinish  |            DESCRIPTION          |
	 *  +-------------+-------------+------------+---------------------------------+
	 *  |      0      |       0     |      0     |  Impossible unless v == 0,      |
	 *  |             |             |            |  which is forbidden.            |
	 *  |             |             |            |                                 |
	 *  |      0      |       0     |      1     |  V % B == 0. Each block         |
	 *  |             |             |            |  handles integer number of      |
	 *  |             |             |            |  destination elements, no       |
	 *  |             |             |            |  partial results are required,  |
	 *  |             |             |            |  workspace is unused.           |
	 *  |             |             |            |                                 |
	 *  |      0      |       1     |      0     |  V < B. Block begins aligned    |
	 *  |             |             |            |  but ends misaligned, before    |
	 *  |             |             |            |  the end of its first element.  |
	 *  |             |             |            |  Partial result written to      |
	 *  |             |             |            |  right-half of array.           |
	 *  |             |             |            |                                 |
	 *  |      0      |       1     |      1     |  V > B, V % B != 0. Block       |
	 *  |             |             |            |  begins aligned but ends        |
	 *  |             |             |            |  misaligned, after the end of   |
	 *  |             |             |            |  its first element.             |
	 *  |             |             |            |  First 1 or more complete       |
	 *  |             |             |            |  elements written out directly  |
	 *  |             |             |            |  to destination.                |
	 *  |             |             |            |  Partial result of last element |
	 *  |             |             |            |  written to right-half of array.|
	 *  |             |             |            |                                 |
	 *  |      1      |       0     |      0     |  Impossible unless v == 0,      |
	 *  |             |             |            |  which is forbidden.            |
	 *  |             |             |            |                                 |
	 *  |      1      |       0     |      1     |  V % B != 0. Partial result of  |
	 *  |             |             |            |  first element written to left- |
	 *  |             |             |            |  half of array. Zero or more    |
	 *  |             |             |            |  complete reductions performed  |
	 *  |             |             |            |  and written directly to        |
	 *  |             |             |            |  destination. Block ends        |
	 *  |             |             |            |  aligned.                       |
	 *  |             |             |            |                                 |
	 *  |      1      |       1     |      0     |  V < B. Block begins misaligned |
	 *  |             |             |            |  and ends misaligned, before    |
	 *  |             |             |            |  the end of its first element.  |
	 *  |             |             |            |  Partial result written to at   |
	 *  |             |             |            |  least right-half of array.     |
	 *  |             |             |            |                                 |
	 *  |      1      |       1     |      1     |  V % B != 0. Block begins       |
	 *  |             |             |            |  misaligned and ends misaligned,|
	 *  |             |             |            |  after the end of its first     |
	 *  |             |             |            |  element.                       |
	 *  |             |             |            |  Partial result of first element|
	 *  |             |             |            |  written to left-half of array. |
	 *  |             |             |            |  Partial result of last element |
	 *  |             |             |            |  written to right-half of array.|
	 *  |             |             |            |  0 or more complete elements    |
	 *  |             |             |            |  written out directly to        |
	 *  |             |             |            |  destination.                   |
	 *  +-------------+-------------+------------+---------------------------------+
	 *
	 * Possible configurations of blocks:
	 *   If V % B == 0:  001
	 *   If V < B:       010, 110, 111, 101
	 *   If V > B:       011, 111, 101
	 *
	 * Possible configurations for collector blocks (responsible for gathering of
	 * results to their right):
	 *   010, 011, 111       (misalignR && (!misalignL || doFinish))
	 *
	 * Possible configurations for right-neighbours of collector blocks
	 *   110 (any number 0+), then exactly one of:
	 *   101, 111
	 *
	 * Conclusion:
	 *     - In Phase 0:
	 *         - Always make a right-write if collector block (010, 011, 111).
	 *         - Always make a left -write if misalignL       (101, 110, 111).
	 *     - In Phase 1:
	 *         - Exit if not collector block (010, 011, 111)
	 *         - If collector block,
	 *             - Right-read from self
	 *             - Left -read from all right-neighbours with same write-target.
	 *
	 * Code Structure perfectly satisfying conclusion:
	 *
	 * if(misalignR){
	 *     while(v > 0){
	 *         v--;
	 *         REDUX();
	 *         ReduxLoopDecs_CONTINUE;
	 *         HREDUX();
	 *         WSRightWrite();
	 *         REINIT();
	 *         FreeLoopDecs_BREAK;
	 *         BREAK;
	 *     }
	 * }
	 * while(v > 0){
	 *     v--;
	 *     REDUX();
	 *     ReduxLoopDecs_CONTINUE;
	 *     HREDUX();
	 *     DstWrite();
	 *     REINIT();
	 *     FreeLoopDecs_CONTINUE;
	 *     BREAK;
	 * }
	 * if(misalignL){
	 *     HREDUX();
	 *     WSLeftWrite();
	 * }
	 *
	 * Code Walkthrough:
	 *
	 * 000, 100: --  Impossible, can be ignored.
	 * 001:      --  Only master loop entered, handles exact integer number of destinations.
	 * 010:      -R  Right-misalign loop entered, completes a reduction. HREDUX, partial
	 *               result right-written to workspace, reinit, bump of free loop counters,
	 *               break simultaneously on vcount and free loop breaks.
	 *               Master loop not entered. Left-misalign fixup not entered.
	 * 011:      -R  Right-misalign loop entered, completes a reduction. HREDUX, partial
	 *               result right-written to workspace, reinit, bump of free loop counters,
	 *               break on free loop breaks. Master loop entered for 1+ complete
	 *               destination elements written direct to destination. Break on vcount.
	 *               Left-misalign fixup not entered.
	 * 101:      L-  Master loop entered for 0+ complete destination elements written
	 *               directly to destination. Master loop broken on vcount. Left-misalign
	 *               fixup entered, HREDUX, partial result left-written to workspace.
	 * 110:      L-  Right-misalign loop entered, broken on vcount before HREDUX. No
	 *               reinit. Master loop not entered. Left-misalign fixup entered, HREDUX,
	 *               partial result left-written to workspace.
	 * 111:      LR  Right-misalign loop entered and completes a reduction. HREDUX, partial
	 *               result right-written to workspace, reinit, bump of free loop counters,
	 *               breakout. Master loop entered for 0 or more complete destination
	 *               elements written directly to destination. Master loop broken on vcount
	 *               before HREDUX. Right-misalign fixup entered, HREDUX, partial result
	 *               left-written to workspace.
	 */
	"    \n"
	"    TU64        left         = GID_0 * V;\n"
	"    if(left >= U){return;}\n"
	"    TU64        v            = U-left < V ? U-left : V;\n"
	"    \n"
	"    TS32        misalignL    = (left+0)%B != 0;\n"
	"    TS32        misalignR    = (left+v)%B != 0;\n"
	"    TS32        doFinish     = (left+0)/B != (left+v)/B;\n"
	"    TS32        collector    = misalignR && (!misalignL || doFinish);\n"
	"    \n"
	"    TU32        iSplit       = LID_0/(LDIM_0/LSlice);\n"
	"    \n");
	/**
	 * Decode Intra-/Inter-Block start point.
	 *
	 * For the purpose of decoding the start point, the split axis's \"length\"
	 * is divided by either splitReduce or splitFree and rounded up. Therefore,
	 * for those axes the true computed initial starting point must be
	 * multiplied by either splitReduce or splitFree.
	 *
	 * Since we provide not strides but \"jumps\" to the kernel (to move as many
	 * things as possible into constant memory and out of the fast path), we
	 * must also convert jumps to strides in preparation for offsetting the
	 * base pointers to their starting point.
	 *
	 * This also involves computing the intra-block coordinate of a thread in a
	 * up-to-log2(MAX_BLOCK_THREADS)-rank coordinate system, then using
	 * those coordinates to compute intrablock S0/D0/D1/I0/permute targets.
	 */

	for (i=gr->nds-1;i>=0;i--){
		if       (i == gr->nds-1 && i == gr->ndd-1){
			srcbAppendf(&gr->srcGen,
			"    TU64        _L%d          = DIVIDECEIL(L%d, LSlice);\n", i, i);
		}else if (i == gr->nds-1){
			srcbAppendf(&gr->srcGen,
			"    TU64        _L%d          = DIVIDECEIL(L%d, (selector&2) ? 1 : LSlice);\n", i, i);
		}else if (i == gr->ndd-1){
			srcbAppendf(&gr->srcGen,
			"    TU64        _L%d          = DIVIDECEIL(L%d, (selector&2) ? LSlice : 1);\n", i, i);
		}else{
			srcbAppendf(&gr->srcGen,
			"    TU64        _L%d          = L%d;\n", i, i);
		}
	}
	srcbAppends(&gr->srcGen,
	"    \n"
	"    z                        = left+v-1;\n");
	for (i=gr->nds-1;i>=0;i--){
		srcbAppendf(&gr->srcGen,
		"    TS64        _i%d          = z %% _L%d;  z /= _L%d;\n",  i, i, i);
	}
	srcbAppends(&gr->srcGen,
	"    z                        = LID_0;\n");
	for (i=gr->log2MaxBS-1;i>=0;i--){
		srcbAppendf(&gr->srcGen,
		"    TS32        _i%di         = z %%  L%di; z /=  L%di;\n", i, i, i);
	}


	/* Compute Intrablock Permute Core, since it will be used soon */
	srcbAppends(&gr->srcGen, "    \n");
	srcbAppends(&gr->srcGen, "    const TU32  perm         = ");
	srcbBeginList(&gr->srcGen, " + ", "0");
	for (i=0;i<gr->log2MaxBS;i++){
		srcbAppendElemf(&gr->srcGen, "_i%di*perm%di", i, i);
	}
	srcbEndList(&gr->srcGen);
	srcbAppends(&gr->srcGen, ";\n");


	/* S0 Lattice */
	if (reduxGenKernelRequiresLatticeS0(gr)){
		srcbAppends(&gr->srcGen, "    \n");
		for (i=gr->nds-1;i>=0;i--){
			if (i == gr->nds-1){
				srcbAppendf(&gr->srcGen,
				"    TS64        _S0S%d        = S0J%d;\n", i, i);
			}else{
				srcbAppendf(&gr->srcGen,
				"    TS64        _S0S%d        = S0J%d + _L%d*_S0S%d;\n", i, i, i+1, i+1);
			}
		}
		srcbAppends(&gr->srcGen, "    S0Off                   += ");
		srcbBeginList(&gr->srcGen, " + ", "0");
		for (i=0;i<gr->nds;i++){
			srcbAppendElemf(&gr->srcGen, "_i%d*_S0S%d", i, i);
		}
		for (i=0;i<gr->log2MaxBS;i++){
			srcbAppendElemf(&gr->srcGen, "_i%di*S0S%di", i, i);
		}
		srcbEndList(&gr->srcGen);
		srcbAppends(&gr->srcGen, ";\n"
		                         "    S0                      += S0Off;\n");
	}


	/* D0 Lattice */
	if (reduxGenKernelRequiresLatticeD0(gr)){
		srcbAppends(&gr->srcGen, "    \n");
		for (i=gr->ndd-1;i>=0;i--){
			if (i == gr->ndd-1){
				srcbAppendf(&gr->srcGen,
				"    TS64        _D0S%d        = D0J%d;\n", i, i);
			}else{
				srcbAppendf(&gr->srcGen,
				"    TS64        _D0S%d        = D0J%d + _L%d*_D0S%d;\n", i, i, i+1, i+1);
			}
		}
		srcbAppends(&gr->srcGen, "    D0Off                   += ");
		srcbBeginList(&gr->srcGen, " + ", "0");
		for (i=0;i<gr->ndd;i++){
			srcbAppendElemf(&gr->srcGen, "_i%d*_D0S%d", i, i);
		}
		for (i=0;i<gr->log2MaxBS;i++){
			srcbAppendElemf(&gr->srcGen, "_i%di*D0S%di", i, i);
		}
		srcbEndList(&gr->srcGen);
		srcbAppends(&gr->srcGen, ";\n"
		                         "    local_barrier();\n"
		                         "    if(perm < D){\n"
		                         "        ((TS64*)SHMEM)[perm]  = D0Off;\n"
		                         "    }\n"
		                         "    if(LID_0 >= D){\n"
		                         "        ((TS64*)SHMEM)[LID_0] = 0;\n"
		                         "    }\n"
		                         "    local_barrier();\n"
		                         "    D0Off                     = ((TS64*)SHMEM)[LID_0];\n"
		                         "    D0                       += D0Off;\n"
		                         "    local_barrier();\n");
	}


	/* D1 Lattice */
	if (reduxGenKernelRequiresLatticeD1(gr)){
		srcbAppends(&gr->srcGen, "    \n");
		for (i=gr->ndd-1;i>=0;i--){
			if (i == gr->ndd-1){
				srcbAppendf(&gr->srcGen,
				"    TS64        _D1S%d        = D1J%d;\n", i, i);
			}else{
				srcbAppendf(&gr->srcGen,
				"    TS64        _D1S%d        = D1J%d + _L%d*_D1S%d;\n", i, i, i+1, i+1);
			}
		}
		srcbAppends(&gr->srcGen, "    D1Off                   += ");
		srcbBeginList(&gr->srcGen, " + ", "0");
		for (i=0;i<gr->ndd;i++){
			srcbAppendElemf(&gr->srcGen, "_i%d*_D1S%d", i, i);
		}
		for (i=0;i<gr->log2MaxBS;i++){
			srcbAppendElemf(&gr->srcGen, "_i%di*D1S%di", i, i);
		}
		srcbEndList(&gr->srcGen);
		srcbAppends(&gr->srcGen, ";\n"
		                         "    local_barrier();\n"
		                         "    if(perm < D){\n"
		                         "        ((TS64*)SHMEM)[perm]  = D1Off;\n"
		                         "    }\n"
		                         "    if(LID_0 >= D){\n"
		                         "        ((TS64*)SHMEM)[LID_0] = 0;\n"
		                         "    }\n"
		                         "    local_barrier();\n"
		                         "    D1Off                     = ((TS64*)SHMEM)[LID_0];\n"
		                         "    D1                       += D1Off;\n"
		                         "    local_barrier();\n");
	}


	/* I0 Lattice */
	if (reduxGenKernelRequiresLatticeI0(gr)){
		srcbAppends(&gr->srcGen, "    \n");
		for (i=gr->nds-1;i>=0;i--){
			if (i == gr->nds-1){
				srcbAppendf(&gr->srcGen,
				"    TS64        _I0S%d        = I0J%d;\n", i, i);
			}else{
				srcbAppendf(&gr->srcGen,
				"    TS64        _I0S%d        = I0J%d + _L%d*_I0S%d;\n", i, i, i+1, i+1);
			}
		}
		srcbAppends(&gr->srcGen, "    I0                       = ");
		srcbBeginList(&gr->srcGen, " + ", "0");
		for (i=0;i<gr->nds;i++){
			srcbAppendElemf(&gr->srcGen, "_i%d*_I0S%d", i, i);
		}
		for (i=0;i<gr->log2MaxBS;i++){
			srcbAppendElemf(&gr->srcGen, "_i%di*I0S%di", i, i);
		}
		srcbEndList(&gr->srcGen);
		srcbAppends(&gr->srcGen, ";\n");
	}


	/* Workspace */
	if (reduxGenKernelRequiresWspace(gr)){
		srcbAppends(&gr->srcGen, "    \n");
		if (reduxGenKernelRequiresStateK0(gr)){
			srcbAppends(&gr->srcGen,
			"    TK0* restrict const W0      = (TK0*)(W     + W0Off);\n"
			"    TK0* restrict const W0L     = &W0[0];\n"
			"    TK0* restrict const W0R     = &W0[GDIM_0*D];\n"
			"    TK0* restrict const SHMEMK0 = (TK0*)(SHMEM + SHMEMK0Off);\n");
		}
		if (reduxGenKernelRequiresStateK1(gr)){
			srcbAppends(&gr->srcGen,
			"    TK1* restrict const W1      = (TK1*)(W     + W1Off);\n"
			"    TK1* restrict const W1L     = &W1[0];\n"
			"    TK1* restrict const W1R     = &W1[GDIM_0*D];\n"
			"    TK1* restrict const SHMEMK1 = (TK1*)(SHMEM + SHMEMK1Off);\n");
		}
		srcbAppends(&gr->srcGen,
		"    local_barrier();\n"
		"    INITREDUXSTATE(SHMEMK0[LID_0], SHMEMK1[LID_0]);\n"
		"    if(D<LDIM_0 && LID_0+LDIM_0<H){\n"
		"        INITREDUXSTATE(SHMEMK0[LID_0+LDIM_0], SHMEMK1[LID_0+LDIM_0]);\n"
		"    }\n"
		"    local_barrier();\n");
	}


	/* Fixup the division we did to one of the dimensions. */
	srcbAppendf(&gr->srcGen, "    \n");
	if (gr->nds>0){
		srcbAppendf(&gr->srcGen,
		"    _i%d                     *= (selector&2) ? 1 : LSlice;\n", gr->nds-1);
	}
	if (gr->ndd>0){
		srcbAppendf(&gr->srcGen,
		"    _i%d                     *= (selector&2) ? LSlice : 1;\n", gr->ndd-1);
	}


	/* Add a couple newlines before next section */
	srcbAppends(&gr->srcGen,
	"    \n"
	"    \n");
}
static void        reduxGenSrcAppendPhase0          (GpuReduction*        gr,
                                                     uint32_t             selector){
	int         i;
	const char* type;

	/**
	 * Convert index types depending on the template selected by the selector.
	 *
	 * If misaligned on the right, write partial reduction to right-half.
	 * If misaligned on the left,  write partial reduction to left-half.
	 *
	 * The Phase 1 collector blocks will take care of reading the partial
	 * reduction results and combining them.
	 */

	srcbAppends(&gr->srcGen, "            ");
	for (i=0;i<gr->nds;i++){
		type = reduxGenSrcAxisIsHuge(gr, selector, i) ? "TU64" : "TU32";
		srcbAppendf(&gr->srcGen, "%s i%d = _i%d;", type, i, i);
	}
	srcbAppends(&gr->srcGen, "\n"
	                         "            \n"
	                         "            if(misalignR){\n");
	reduxGenSrcAppendLoop(gr, selector, 1);
	srcbAppends(&gr->srcGen, "            }\n");
	reduxGenSrcAppendLoop(gr, selector, 0);
	srcbAppends(&gr->srcGen, "            if(misalignL){\n"
	                         "                HREDUX(SHMEMK0, SHMEMK1, perm, K0, K1);\n"
	                         "                if(LID_0 < D){\n"
	                         "                    SETREDUXSTATE(W0L[GID_0*D+LID_0],\n"
	                         "                                  W1L[GID_0*D+LID_0],\n"
	                         "                                  SHMEMK0[LID_0],\n"
	                         "                                  SHMEMK1[LID_0]);\n"
	                         "                }\n"
	                         "            }\n");
}
static void        reduxGenSrcAppendLoop            (GpuReduction*        gr,
                                                     uint32_t             selector,
                                                     int                  initial){
	int i;

	srcbAppends(&gr->srcGen, "            while(v > 0){v--;\n");
	reduxGenSrcAppendVertical (gr, selector);
	for (i=gr->nds-1;i >= gr->ndd;i--){
		reduxGenSrcAppendIncrement(gr, selector, initial, i);
	}
	srcbAppends(&gr->srcGen, "                HREDUX(SHMEMK0, SHMEMK1, perm, K0, K1);\n");
	reduxGenSrcAppendDstWrite(gr, selector, initial);
	srcbAppends(&gr->srcGen, "                INITREDUXSTATE(K0, K1);\n");
	for (i=gr->ndd-1;i >= 0;i--){
		reduxGenSrcAppendIncrement(gr, selector, initial, i);
	}
	srcbAppends(&gr->srcGen, "                break;\n"
	                         "            }\n");
}
static void        reduxGenSrcAppendVertical        (GpuReduction*        gr,
                                                     uint32_t             selector){
	int i = (selector&SELECTOR_SPLIT_FREE) ? gr->ndd-1 : gr->nds-1;

	if (i >= 0){
		srcbAppendf(&gr->srcGen, "                if(i%d+iSplit < L%d){\n"
		                         "                    LOADS0(tmpK0, S0);\n"
		                         "                    REDUX(K0, K1, tmpK0, I0);\n"
		                         "                }\n", i, i);
	}else{
		srcbAppends(&gr->srcGen, "                LOADS0(tmpK0, S0);\n"
		                         "                REDUX(K0, K1, tmpK0, I0);\n");
	}
}
static void        reduxGenSrcAppendIncrement       (GpuReduction*        gr,
                                                     uint32_t             selector,
                                                     int                  initial,
                                                     int                  axis){
	const char* cast        = reduxGenSrcAxisIsHuge(gr, selector, axis) ? "TS64" : "TS32";
	const char* breakOrCont = (initial) && (axis < gr->ndd) ? "break   " : "continue";

	/* Pointer bumps */
	srcbAppends(&gr->srcGen, "                ");
	if (reduxGenKernelRequiresLatticeS0(gr)){
		srcbAppendf(&gr->srcGen, "S0 -= S0J%d;", axis);
	}else{
		srcbAppends(&gr->srcGen, "           ");
	}
	if (reduxGenKernelRequiresLatticeD0(gr) && axis < gr->ndd){
		srcbAppendf(&gr->srcGen, "D0 -= D0J%d;", axis);
	}else{
		srcbAppends(&gr->srcGen, "           ");
	}
	if (reduxGenKernelRequiresLatticeD1(gr) && axis < gr->ndd){
		srcbAppendf(&gr->srcGen, "D1 -= D1J%d;", axis);
	}else{
		srcbAppends(&gr->srcGen, "           ");
	}
	if (reduxGenKernelRequiresLatticeI0(gr)){
		srcbAppendf(&gr->srcGen, "I0 -= I0J%d;", axis);
	}else{
		srcbAppends(&gr->srcGen, "           ");
	}

	/* Index Check */
	if (reduxGenSrcAxisIsSplit(gr, selector, axis)){
		srcbAppendf(&gr->srcGen, "i%d-=LSlice;if((%s)i%d >= 0){%s;}else{i%d+=LPadded;}\n",
		            axis, cast, axis, breakOrCont, axis);
	}else{
		srcbAppendf(&gr->srcGen, "i%d--;      if((%s)i%d >= 0){%s;}else{i%d+=L%d;}\n",
		            axis, cast, axis, breakOrCont, axis, axis);
	}
}
static void        reduxGenSrcAppendDstWrite        (GpuReduction*        gr,
                                                     uint32_t             selector,
                                                     int                  initial){
	srcbAppends(&gr->srcGen, "                local_barrier();\n");
	if (initial){
		srcbAppends(&gr->srcGen, "                if(LID_0 < D){\n"
		                         "                    SETREDUXSTATE(W0R[GID_0*D + LID_0],\n"
		                         "                                  W1R[GID_0*D + LID_0],\n"
		                         "                                  SHMEMK0[LID_0],\n"
		                         "                                  SHMEMK1[LID_0]);\n"
		                         "                }\n");
	}else{
		if (selector & SELECTOR_SPLIT_FREE){
			if (gr->ndd > 0){
				srcbAppendf(&gr->srcGen, "                if(LID_0 < ((L%d-i%d)<LSlice ? (L%d-i%d)*Dunit : D)){\n"
				                         "                    STORED0(D0, SHMEMK0[LID_0]);\n"
				                         "                    STORED1(D1, SHMEMK1[LID_0]);\n"
				                         "                }\n",
				            gr->ndd-1, gr->ndd-1, gr->ndd-1, gr->ndd-1);
			}else{
				srcbAppendf(&gr->srcGen, "                STORED0(D0, SHMEMK0[LID_0]);\n"
				                         "                STORED1(D1, SHMEMK1[LID_0]);\n");
			}
		}else{
			srcbAppends(&gr->srcGen, "                if(LID_0 < D){\n"
			                         "                    STORED0(D0, SHMEMK0[LID_0]);\n"
			                         "                    STORED1(D1, SHMEMK1[LID_0]);\n"
			                         "                }\n");
		}
	}
	srcbAppends(&gr->srcGen, "                local_barrier();\n");
}
static void        reduxGenSrcAppendPhase1          (GpuReduction*        gr){
	/**
	 * PHASE 1
	 *
	 * If we are a collector block, gather all partial results for the
	 * same points to the right of the current position in our workspace
	 * and accumulate them into our partial result, then write out to
	 * destination/destination argument.
	 *
	 * We perform a right-read of our workspace and a left-read of the
	 * other blocks' workspace.
	 */

	srcbAppends(&gr->srcGen,
	"        if(collector && LID_0 < D){\n"
	"            SETREDUXSTATE(K0, K1, W0R[(GID_0+0)*D+LID_0], W1R[(GID_0+0)*D+LID_0]);\n"
	"            \n"
	"            for(k=1,v=left+v-1,z=v+1; /* Starting with the first block to our right... */\n"
	"                v/B == z/B;           /* Is our write target the same as that of */\n"
	"                                      /* the target k blocks to our right? */\n"
	"                k++,z+=V){            /* Try moving one more to the right. */\n"
	"                REDUX(K0, K1, W0L[(GID_0+k)*D+LID_0], W1L[(GID_0+k)*D+LID_0]);\n"
	"            }\n"
	"            \n");
	if (gr->ndd > 0){
		srcbAppendf(&gr->srcGen,
		"            if(!(selector&2) || LID_0 < ((L%d-_i%d)<LSlice ? (L%d-_i%d)*Dunit : D)){\n"
		"                STORED0(D0, K0);\n"
		"                STORED1(D1, K1);\n"
		"            }\n"
		"        }\n",
		            gr->ndd-1, gr->ndd-1, gr->ndd-1, gr->ndd-1);
	}else{
		srcbAppends(&gr->srcGen,
		"            STORED0(D0, K0);\n"
		"            STORED1(D1, K1);\n"
		"        }\n");
	}
}
static int         reduxGenSrcAxisIsHuge            (GpuReduction*        gr,
                                                     uint32_t             selector,
                                                     int                  axis){
	int hugeType    = selector & SELECTOR_HUGE_AXIS;
	int isSplitFree = !!(selector & SELECTOR_SPLIT_FREE);
	int isAxisFree  = axis < gr->ndd;

	if       (hugeType == SELECTOR_HUGE_IS_SPLIT){
		return reduxGenSrcAxisIsSplit(gr, selector, axis);
	}else if (hugeType == SELECTOR_HUGE_SAME_TYPE){
		if (isSplitFree == isAxisFree){
			if (isAxisFree){
				return axis == gr->ndd-2;
			}else{
				return axis == gr->nds-2;
			}
		}else{
			return 0;
		}
	}else if (hugeType == SELECTOR_HUGE_OPPOSITE_TYPE){
		if (isSplitFree != isAxisFree){
			if (isAxisFree){
				return axis == gr->ndd-1;
			}else{
				return axis == gr->nds-1;
			}
		}else{
			return 0;
		}
	}else{
		return 0;
	}
}
static int         reduxGenSrcAxisIsSplit           (GpuReduction*        gr,
                                                     uint32_t             selector,
                                                     int                  axis){
	return  ( (selector & SELECTOR_SPLIT_FREE) && axis == gr->ndd-1) ||
	        (!(selector & SELECTOR_SPLIT_FREE) && axis == gr->nds-1);
}

/**
 * @brief Compile the generated kernel.
 */

static int         reduxGenCompile                  (GpuReduction*        gr){
	int ret, flags = 0;

	flags |= GA_USE_CLUDA;
	if (gr->TS0tc == GA_HALF || gr->TD0tc == GA_HALF){
		flags |= GA_USE_HALF|GA_USE_SMALL;
	}

	ret  = GpuKernel_init(&gr->k,
	                      gr->gpuCtx,
	                      1,
	                      (const char**)&gr->kSourceCode,
	                      &gr->kSourceCodeLen,
	                      gr->kName,
	                      gr->kNumArgs,
	                      gr->kArgTypeCodes,
	                      flags,
	                      &gr->kErrorString);

	if (ret != GA_NO_ERROR){
		return reduxGenCleanupMsg(gr, ret,
		       "Failed to compile reduction kernel \"%s\"!\n"
		       "Error code   is: %d\n"
		       "Error string is:\n"
		       "%s\n"
		       "Source code  is:\n"
		       "%s\n",
		       gr->kName, ret, gr->kErrorString, gr->kSourceCode);
	}

	return reduxGenComputeLaunchBounds(gr);
}

/**
 * @brief Compute the maximum number of threads this reduction operator will
 *        support launching.
 */

static int         reduxGenComputeLaunchBounds      (GpuReduction*        gr){
	int    ret;

	/**
	 * Compute the maximum number of threads this kernel will support,
	 * since this is critical to the scheduling and will not change now
	 * that the kernel is compiled.
	 */

	ret = gpukernel_property(gr->k.k, GA_KERNEL_PROP_MAXLSIZE, &gr->maxLK);
	if (ret != GA_NO_ERROR){
		return reduxGenCleanupMsg(gr, ret,
		       "Failed to read max local size for compiled kernel!\n");
	}
	gr->maxLK = gr->maxLK<gr->maxBS ? gr->maxLK : gr->maxBS;

	return reduxGenCleanup(gr, GA_NO_ERROR);
}

/**
 * @brief Cleanup generator context.
 */

static int         reduxGenCleanup                  (GpuReduction*        gr,  int ret){
	if (ret != GA_NO_ERROR){
		free(gr->kArgTypeCodes);
		free(gr->kSourceCode);
		free(gr->kErrorString);

		memset(gr, 0, sizeof(*gr));
		free(gr);
	}

	return ret;
}
static int         reduxGenCleanupMsg               (GpuReduction*        gr,  int ret,
                                                     const char*          fmt, ...){
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

static void        reduxGenCountArgs                (const GpuReduction*  gr,
                                                     int                  typecode,
                                                     const char*          typeName,
                                                     const char*          baseName,
                                                     int                  num,
                                                     void*                user){
	(void)gr;
	(void)typecode;
	(void)typeName;
	(void)baseName;
	(void)num;

	(*(int*)user)++;
}

/**
 * Record the typecodes in the arguments typecode array.
 */

static void        reduxGenSaveArgTypecodes         (const GpuReduction*  gr,
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

static void        reduxGenAppendArg                (const GpuReduction*  gr,
                                                     int                  typecode,
                                                     const char*          typeName,
                                                     const char*          baseName,
                                                     int                  num,
                                                     void*                user){
	(void)user;
	(void)typecode;

	if ((*(int*)user)++ > 0){
		srcbAppends(&((GpuReduction*)gr)->srcGen, ",\n                  ");
	}
	srcbAppendf(&((GpuReduction*)gr)->srcGen, "%-35s ", typeName);
	srcbAppendf(&((GpuReduction*)gr)->srcGen, baseName, num);
}

/**
 * Marshall argument declaration during invocation.
 */

static void        reduxInvMarshalArg               (const GpuReduction*  gr,
                                                     int                  typecode,
                                                     const char*          typeName,
                                                     const char*          baseName,
                                                     int                  num,
                                                     void*                user){
	redux_ctx* ctx;
	int*       i, k = num;

	(void)typecode;
	(void)typeName;

	ctx = (redux_ctx*)(((void**)user)[0]);
	i   = (int      *)(((void**)user)[1]);

	if       (strcmp(baseName, "selector") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->selector;
	}else if (strcmp(baseName, "U") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->U;
	}else if (strcmp(baseName, "V") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->V;
	}else if (strcmp(baseName, "B") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->B;
	}else if (strcmp(baseName, "D") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->D;
	}else if (strcmp(baseName, "Dunit") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->Dunit;
	}else if (strcmp(baseName, "H") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->H;
	}else if (strcmp(baseName, "LSlice") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->LSlice;
	}else if (strcmp(baseName, "LPadded") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->LPadded;
	}else if (strcmp(baseName, "L%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->L[k];
	}else if (strcmp(baseName, "L%di") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->Li[k];
	}else if (strcmp(baseName, "S0") == 0){
		ctx->kArgs[(*i)++] = (void*) ctx->S0Data;
	}else if (strcmp(baseName, "S0Off") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->S0Off;
	}else if (strcmp(baseName, "S0J%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->S0J[k];
	}else if (strcmp(baseName, "S0S%di") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->S0Si[k];
	}else if (strcmp(baseName, "D0") == 0){
		ctx->kArgs[(*i)++] = (void*) ctx->D0Data;
	}else if (strcmp(baseName, "D0Off") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->D0Off;
	}else if (strcmp(baseName, "D0J%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->D0J[k];
	}else if (strcmp(baseName, "D0S%di") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->D0Si[k];
	}else if (strcmp(baseName, "D1") == 0){
		ctx->kArgs[(*i)++] = (void*) ctx->D1Data;
	}else if (strcmp(baseName, "D1Off") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->D1Off;
	}else if (strcmp(baseName, "D1J%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->D1J[k];
	}else if (strcmp(baseName, "D1S%di") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->D1Si[k];
	}else if (strcmp(baseName, "I0J%d") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->I0J[k];
	}else if (strcmp(baseName, "I0S%di") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->I0Si[k];
	}else if (strcmp(baseName, "W") == 0){
		ctx->kArgs[(*i)++] = (void*) ctx->W;
	}else if (strcmp(baseName, "W0Off") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->W0Off;
	}else if (strcmp(baseName, "SHMEMK0Off") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->SHMEMK0Off;
	}else if (strcmp(baseName, "W1Off") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->W1Off;
	}else if (strcmp(baseName, "SHMEMK1Off") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->SHMEMK1Off;
	}else if (strcmp(baseName, "perm%di") == 0){
		ctx->kArgs[(*i)++] = (void*)&ctx->perm[k];
	}
}


/**
 * @brief Estimate the level of parallelism available in the GPU context of
 *        this reduction operator.
 *
 * This is a rough target number of threads.  It would definitely fill the
 * device, plus some substantial margin.
 */

static size_t      reduxGenEstimateParallelism      (const GpuReduction*  gr){
	/**
	 * An arbitrary margin factor ensuring there will be a few thread blocks
	 * per SMX.
	 *
	 * E.g. on Kepler, each SMX can handle up to two 1024-thread blocks
	 * simultaneously, so a margin of 16/SMX should ensure with very high
	 * likelyhood that all SMXes will be fed and kept busy.
	 */

	size_t marginFactor = 16;
	return marginFactor * gr->grAttr.numProcs * gr->grAttr.maxLg;
}

/**
 * @brief Return whether or not the reduction operator's interface or kernel
 *        require a specific argument, lattice or storage.
 *
 * Specifically, check if the reductions operator's:
 *   - Interface (reduxGenRequires*())              the passing of an s0/d0/d1    argument
 *   - Kernel    (reduxGenKernelRequiresLattice*()) the walking of an s0/d0/d1/i0 lattice
 *   - Kernel    (reduxGenKernelRequiresState*())   contains       a  k0/k1       state
 *   - Kernel    (reduxGenKernelRequiresWspace())   workspaces named  w*          for states k*.
 *
 * The reduction operator's interface, kernel and state are semantically
 * subtly different. The interface asks whether the GpuReduction_call(), and
 * therefore the generated kernel, must receive a specific argument:
 *
 *    - Argument s0 (Typically the source tensor)
 *    - Argument d0 (Typically the destination tensor)
 *    - Argument d1 (Typically the destination argument tensor)
 *
 * The kernel asks whether it must internally walk over a specific lattice, where:
 *
 *    - Lattice s0 is the lattice of pointers into the s0 tensor.
 *    - Lattice d0 is the lattice of pointers into the d0 tensor.
 *    - Lattice d1 is the lattice of pointers into the d1 tensor.
 *    - Lattice i0 is the lattice of flattened indices into the s0 tensor.
 *
 * The state asks whether it should contain:
 *
 *    - State k0 (Typically for accumulator states typed `TK` over the s0 lattice
 *                and written to the d0 lattice)
 *    - State k1 (Typically for indexes typed `TI` from the i0 lattice and written
 *                to the d1 lattice)
 *
 * The workspace asks whether it is required in order to save partial reduction
 * states k* computed during Phase 0.
 *
 *
 *
 * Currently:
 *
 *   - All GpuReductions require an s0 argument.
 *   - All GpuReductions except argmin/argmax require a d0 argument.
 *   - Only the argmin/argmax/minandargmin/maxandargmax GpuReductions require a d1 argument.
 *   - All and only the GpuReductions requiring a s0 argument require walking over the s0 lattice.
 *   - All and only the GpuReductions requiring a d0 argument require walking over the d0 lattice.
 *   - All and only the GpuReductions requiring a d1 argument require walking over the d1 lattice.
 *   - All and only the GpuReductions requiring a d1 argument require walking over the i0 lattice.
 *   - All and only the GpuReductions requiring a s0 lattice walk require a k0 state.
 *   - All and only the GpuReductions requiring a i0 lattice walk require a k1 state.
 *   - All GpuReductions potentially require a workspace for their states.
 *
 * However, if this reduction engine were generalized to multi-reduction, elemwise or
 * initialization operations, the above might not necessarily hold anymore.
 */

static int         reduxGenRequiresS0               (const GpuReduction*  gr){
	return GpuReductionAttr_requiresS0(&gr->grAttr);
}
static int         reduxGenRequiresD0               (const GpuReduction*  gr){
	return GpuReductionAttr_requiresD0(&gr->grAttr);
}
static int         reduxGenRequiresD1               (const GpuReduction*  gr){
	return GpuReductionAttr_requiresD1(&gr->grAttr);
}
static int         reduxGenKernelRequiresLatticeS0  (const GpuReduction*  gr){
	return reduxGenRequiresS0(gr);
}
static int         reduxGenKernelRequiresLatticeD0  (const GpuReduction*  gr){
	return reduxGenRequiresD0(gr);
}
static int         reduxGenKernelRequiresLatticeD1  (const GpuReduction*  gr){
	return reduxGenRequiresD1(gr);
}
static int         reduxGenKernelRequiresLatticeI0  (const GpuReduction*  gr){
	return reduxGenRequiresD1(gr);
}
static int         reduxGenKernelRequiresStateK0    (const GpuReduction*  gr){
	return reduxGenKernelRequiresLatticeS0(gr);
}
static int         reduxGenKernelRequiresStateK1    (const GpuReduction*  gr){
	return reduxGenKernelRequiresLatticeI0(gr);
}
static int         reduxGenKernelRequiresWspace     (const GpuReduction*  gr){
	(void)gr;
	return 1;
}


/**
 * Get size and alignment requirements of K0 and K1 states.
 */

static size_t      reduxGenGetK0Size                (const GpuReduction*  gr){
	return gr->TK0.size;
}
static size_t      reduxGenGetK0Align               (const GpuReduction*  gr){
	return gr->TK0.align;
}
static size_t      reduxGenGetK1Size                (const GpuReduction*  gr){
	return gr->TK1.size;
}
static size_t      reduxGenGetK1Align               (const GpuReduction*  gr){
	return gr->TK1.align;
}

/**
 * @brief Get the number of bytes of workspace per (partial) reduction per thread.
 */

static size_t      reduxGenGetReduxStateSize        (const GpuReduction*  gr){
	size_t total = 0, idxSize = gpuarray_get_elsize(gr->TS64tc);

	/* The accumulator and index types can be wider than dst/dstArg's types. */
	total += reduxGenKernelRequiresStateK0(gr) ? reduxGenGetK0Size(gr) : 0;
	total += reduxGenKernelRequiresStateK1(gr) ? reduxGenGetK1Size(gr) : 0;

	/* At minimum, there must be space for the offset permute. */
	total  = total < idxSize ? idxSize : total;

	/* Return the calculated amount of space. */
	return total;
}

/**
 * @brief Get the maximum number of threads this operator's kernel can handle.
 */

static size_t      reduxGenGetMaxLocalSize          (const GpuReduction*  gr){
	return gr->maxLK;
}

/**
 * @brief Get the shared memory consumption for a given block size.
 */

static size_t      reduxGenGetSHMEMSize             (const GpuReduction*  gr, size_t cells){
	size_t               total = 0, totalPermute;

	/* Compute size of SHMEM working space */
	total += reduxGenKernelRequiresStateK0(gr) ? cells*reduxGenGetK0Size(gr) : 0;
	total += reduxGenKernelRequiresStateK1(gr) ? cells*reduxGenGetK1Size(gr) : 0;

	/* But ensure space for pointer offset permute at beginning of kernel. */
	totalPermute = cells*gpuarray_get_type(gr->TS64tc)->size;
	total        = total < totalPermute ? totalPermute : total;

	return total;
}

/**
 * @brief Get the shared memory byte offset for the k0 and k1 states.
 */

static size_t      reduxGenGetSHMEMK0Off            (const GpuReduction*  gr, size_t cells){
	if (!reduxGenKernelRequiresWspace (gr)||
	   !reduxGenKernelRequiresStateK0(gr)||
	   !reduxGenKernelRequiresStateK1(gr)){
		return 0;
	}

	if (reduxGenGetK0Align(gr) > reduxGenGetK1Align(gr)){
		return 0;
	}else{
		return cells*reduxGenGetK1Size(gr);
	}
}
static size_t      reduxGenGetSHMEMK1Off            (const GpuReduction*  gr, size_t cells){
	if (!reduxGenKernelRequiresWspace (gr)||
	   !reduxGenKernelRequiresStateK0(gr)||
	   !reduxGenKernelRequiresStateK1(gr)){
		return 0;
	}

	if (reduxGenGetK0Align(gr) > reduxGenGetK1Align(gr)){
		return cells*reduxGenGetK0Size(gr);
	}else{
		return 0;
	}
}

/**
 * Get the amount of workspace memory required.
 *
 * NOT necessarily the same as amount of SHMEM! The workspace is NOT used for
 * intrablock offset permutes, for instance.
 */

static size_t      reduxGenGetWMEMSize              (const GpuReduction*  gr, size_t cells){
	size_t               total = 0;

	total += reduxGenKernelRequiresStateK0(gr) ? cells*reduxGenGetK0Size(gr) : 0;
	total += reduxGenKernelRequiresStateK1(gr) ? cells*reduxGenGetK1Size(gr) : 0;

	return total;
}

/**
 * @brief Get the workspace memory byte offset for the k0 and k1 states.
 */

static size_t      reduxGenGetWMEMK0Off             (const GpuReduction*  gr, size_t cells){
	return reduxGenGetSHMEMK0Off(gr, cells);
}
static size_t      reduxGenGetWMEMK1Off             (const GpuReduction*  gr, size_t cells){
	return reduxGenGetSHMEMK1Off(gr, cells);
}

/**
 * @brief Initialize the context.
 *
 * After this function, calling reduxInvCleanup*() becomes safe.
 */

static int         reduxInvInit                     (redux_ctx*           ctx){
	ctx->L           = NULL;
	ctx->Li          = NULL;
	ctx->S0J         = ctx->S0Si      = NULL;
	ctx->D0J         = ctx->D0Si      = NULL;
	ctx->D1J         = ctx->D1Si      = NULL;
	ctx->I0J         = ctx->I0Si      = NULL;
	ctx->perm        = NULL;
	ctx->kArgs       = NULL;
	ctx->xdSrc       = NULL;
	ctx->xdSrcPtrs   = NULL;
	ctx->xdSplit     = NULL;

	ctx->W           = NULL;

	ctx->prodAllAxes = ctx->prodRdxAxes   = ctx->prodFreeAxes  = 1;
	ctx->bs          = ctx->gs            = 1;

	return reduxInvInferProperties(ctx);
}

/**
 * @brief Begin inferring the properties of the reduction invocation.
 */

static int         reduxInvInferProperties          (redux_ctx*           ctx){
	axis_desc* a;
	int        i, j;
	size_t     d;


	/* Insane s0, reduxLen, d0 or d1? */
	if       (reduxInvRequiresS0(ctx) && !ctx->s0){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "s0 is NULL, but reduction requires it!\n");
	}
	if       (!ctx->reduxList){
		ctx->reduxLen = reduxInvRequiresS0(ctx) ? ctx->s0->nd : 0;
	}
	if       (reduxInvRequiresS0(ctx) && ctx->s0->nd  <= 0){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "s0 is a scalar, cannot reduce it further!\n");
	}else if (reduxInvRequiresS0(ctx) && ctx->reduxLen <  0){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "Length of list of axes to be reduced is less than 0!\n");
	}else if (reduxInvRequiresS0(ctx) && ctx->s0->nd  <  (unsigned)ctx->reduxLen){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "s0 has fewer axes than there are axes to reduce!\n");
	}else if (reduxInvRequiresD0(ctx) && !ctx->d0){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "d0 is NULL, but reduction requires it!\n");
	}else if (reduxInvRequiresD1(ctx) && !ctx->d1){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "d1 is NULL, but reduction requires it!\n");
	}else if (reduxInvRequiresD0(ctx) && reduxInvRequiresS0(ctx) && ctx->d0->nd+ctx->reduxLen != ctx->s0->nd){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "d0 is of incorrect rank for this reduction!\n");
	}else if (reduxInvRequiresD1(ctx) && reduxInvRequiresS0(ctx) && ctx->d1->nd+ctx->reduxLen != ctx->s0->nd){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "d1 is of incorrect rank for this reduction!\n");
	}
	ctx->nds0  = reduxInvRequiresS0(ctx) ? ctx->s0->nd : 0;
	ctx->nds0r = ctx->reduxLen;
	ctx->ndd0  = ctx->nds0   - ctx->nds0r;
	ctx->ndfs0 = ctx->nds0;


	/* Insane reduxList? */
	for (i=0;i<ctx->nds0r;i++){
		j = ctx->reduxList ? ctx->reduxList[i] : i;
		if (j < -ctx->nds0 || j >= ctx->nds0){
			return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
			       "Insane axis number %d! Should be [%d, %d)!\n",
			       j, -ctx->nds0, ctx->nds0);
		}
		j = j<0 ? ctx->nds0+j : j;
		d                 = ctx->s0->dimensions[j];
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
	 * empty destination tensor (whose axes are the free axes), because
	 * it has 0 space. The operation cannot then fulfill its contract.
	 *
	 * On the other hand, when some or all reduction axes of a tensor are of
	 * length 0, the reduction can be interpreted as initializing the
	 * destination tensor to the identity value of the operation. For lack of a
	 * better idea, the destination argument tensor can then be zeroed.
	 */

	for (i=0;i<ctx->nds0;i++){
		d                 =  ctx->s0->dimensions[i];
		ctx->zeroAllAxes += !d;
		ctx->prodAllAxes *=  d?d:1;
	}
	if (ctx->zeroAllAxes != ctx->zeroRdxAxes){
		return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
		       "Source tensor has length-0 axes that are not reduced!\n");
	}
	ctx->prodFreeAxes = ctx->prodAllAxes/ctx->prodRdxAxes;


	/**
	 * Allocate and construct source-tensor axis-description lists.
	 *
	 * While constructing the descriptions of each axis, verify that:
	 *
	 *   1. reduxLen has no duplicates.
	 *   2. d0 and/or d1's axes match s0's axes when stripped of
	 *      the reduction axes.
	 */

	ctx->xdSrc     = calloc(ctx->nds0,   sizeof(*ctx->xdSrc));
	ctx->xdSrcPtrs = calloc(ctx->nds0+1, sizeof(*ctx->xdSrcPtrs));
	if (!ctx->xdSrc || !ctx->xdSrcPtrs){
		return reduxInvCleanup(ctx, GA_MEMORY_ERROR);
	}
	for (i=0;i<ctx->nds0;i++){
		axisInit(&ctx->xdSrc[i],
		         ctx->s0->dimensions[i],
		         ctx->s0->strides[i]);
	}
	for (i=0;i<ctx->nds0r;i++){
		j = ctx->reduxList ? ctx->reduxList[i] : i;
		j = j<0 ? ctx->nds0+j : j;
		a = reduxInvGetSrcAxis(ctx, j);
		if (axisIsReduced(a)){
			return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
			       "Axis %d appears multiple times in the "
			       "reduction axis list!\n",
			       j);
		}
		axisMarkReduced(a, i);
	}
	for (i=j=0;i<ctx->nds0;i++){
		axis_desc* a     = reduxInvGetSrcAxis(ctx, i);
		size_t     s0Len = axisGetLen(a), d0Len, d1Len;

		if (axisIsReduced(a)){continue;}
		if (reduxInvRequiresD0(ctx)){
			d0Len = ctx->d0->dimensions[j];

			if (s0Len != d0Len){
				return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
				       "s0 axis %d has length %zu, but "
				       "corresponding d0 axis %d has length %zu!\n",
				       i, s0Len, j, d0Len);
			}

			a->d0S    = ctx->d0->strides[j];
		}
		if (reduxInvRequiresD1(ctx)){
			d1Len = ctx->d1->dimensions[j];

			if (s0Len != d1Len){
				return reduxInvCleanupMsg(ctx, GA_INVALID_ERROR,
				       "s0 axis %d has length %zu, but "
				       "corresponding d1 axis %d has length %zu!\n",
				       i, s0Len, j, d1Len);
			}

			a->d1S = ctx->d1->strides[j];
		}

		j++;
	}


	/**
	 * Grab gpudata buffers and byte offsets before we begin flattening the
	 * tensors. As we flatten the tensor, we may reverse some axes, leading to
	 * a bump of the byte offset.
	 */

	if (reduxInvRequiresS0(ctx)){
		ctx->S0Data = ctx->s0->data;
		ctx->S0Off  = ctx->s0->offset;
	}
	if (reduxInvRequiresD0(ctx)){
		ctx->D0Data = ctx->d0->data;
		ctx->D0Off  = ctx->d0->offset;
	}
	if (reduxInvRequiresD1(ctx)){
		ctx->D1Data = ctx->d1->data;
		ctx->D1Off  = ctx->d1->offset;
	}


	return reduxInvFlattenSource(ctx);
}

/**
 * @brief Flatten the source tensor as much as is practical.
 *
 * This makes the axis lengths as long as possible and the tensor itself as
 * contiguous as possible.
 */

static int         reduxInvFlattenSource            (redux_ctx*           ctx){
	axis_desc* axis, *flatAxis, *sortAxis;
	int        i, j, k, isSensitive;

	/**
	 * Pass 1: Flatten out 0- and 1-length axes. We already know that
	 *
	 *         a) There are no 0-length free axes, because that
	 *            constitutes an invalid input, and
	 *         b) How many 0-length reduction axes there are, because
	 *            we counted them in the error-checking code.
	 *
	 * So if there are any 0-length axes, we can delete all reduction axes and
	 * replace them with a single one.
	 *
	 * We can also delete 1-length axes outright, since they can always be
	 * ignored; They are always indexed at [0].
	 */

	for (i=j=0;i<ctx->ndfs0;i++){
		axis = reduxInvGetSrcAxis(ctx, i);
		if (!reduxTryFlattenOut(ctx, axis)){
			*reduxInvGetSrcAxis(ctx, j++) = *axis;
		}
	}
	if (ctx->zeroRdxAxes > 0){
		/* New reduction axis of 0 length. */
		axisInit       (reduxInvGetSrcAxis(ctx, j), 0, 0);
		axisMarkReduced(reduxInvGetSrcAxis(ctx, j), 0);
		j++;
	}
	ctx->ndfs0 = j;


	/**
	 * Pass 2: Flatten out continuous axes, where strides and sensitivity
	 *         allows it.
	 */

	k           = ctx->ndfs0;
	isSensitive = GpuReductionAttr_issensitive(&ctx->gr->grAttr);
	qsort(ctx->xdSrc, ctx->ndfs0, sizeof(*ctx->xdSrc),
	      isSensitive ? reduxSortFlatSensitive : reduxSortFlatInsensitive);
	for (i=j=1;i<ctx->ndfs0;i++){
		flatAxis = reduxInvGetSrcAxis(ctx, j-1);
		sortAxis = reduxInvGetSrcAxis(ctx, i);

		if (reduxTryFlattenInto(ctx, flatAxis, sortAxis)){
			k--;
		}else{
			*reduxInvGetSrcAxis(ctx, j++) = *sortAxis;
		}
	}
	ctx->ndfs0 = k;

	return reduxInvComputeKernelArgs(ctx);
}

/**
 * @brief Compute the arguments to the kernel.
 *
 * This is a multistep process and involves a lot of axis sorting on various
 * criteria.
 */

static int         reduxInvComputeKernelArgs        (redux_ctx*           ctx){
	axis_desc* axis, *prevAxis;
	size_t     target, aL, aLS, perm, i0S;
	int        i, j, haveSplitFreeAxis, haveSplitReducedAxis;


	/**
	 * STEP 0: Default Kernel Argument Values.
	 *
	 * They should be valid for a "scalar" job. In particular, for any
	 * non-existent axis, assume length 1.
	 */

	ctx->selector    = 0;
	ctx->U           = 1;
	ctx->V           = 1;
	ctx->B           = 1;
	ctx->D           = 1;
	ctx->H           = 1;
	ctx->LSlice      = 1;
	ctx->LPadded     = 1;
	ctx->L           = calloc(ctx->gr->nds,       sizeof(*ctx->L));
	ctx->Li          = calloc(ctx->gr->log2MaxBS, sizeof(*ctx->Li));
	ctx->S0J         = calloc(ctx->gr->nds,       sizeof(*ctx->S0J));
	ctx->S0Si        = calloc(ctx->gr->log2MaxBS, sizeof(*ctx->S0Si));
	ctx->D0J         = calloc(ctx->gr->ndd,       sizeof(*ctx->D0J));
	ctx->D0Si        = calloc(ctx->gr->log2MaxBS, sizeof(*ctx->D0Si));
	ctx->D1J         = calloc(ctx->gr->ndd,       sizeof(*ctx->D1J));
	ctx->D1Si        = calloc(ctx->gr->log2MaxBS, sizeof(*ctx->D1Si));
	ctx->I0J         = calloc(ctx->gr->nds,       sizeof(*ctx->I0J));
	ctx->I0Si        = calloc(ctx->gr->log2MaxBS, sizeof(*ctx->I0Si));
	ctx->W0Off       = 0;
	ctx->SHMEMK0Off  = 0;
	ctx->W1Off       = 0;
	ctx->SHMEMK1Off  = 0;
	ctx->perm        = calloc(ctx->gr->log2MaxBS, sizeof(*ctx->perm));
	ctx->bs          = 1;
	ctx->gs          = 1;
	ctx->kArgs       = calloc(ctx->gr->kNumArgs, sizeof(*ctx->kArgs));

	if (!ctx->L    || !ctx->Li   || !ctx->S0J  || !ctx->S0Si ||
	   !ctx->D0J  || !ctx->D0Si || !ctx->D1J  || !ctx->D1Si ||
	   !ctx->I0J  || !ctx->I0Si || !ctx->perm || !ctx->kArgs){
		return reduxInvCleanupMsg(ctx, GA_MEMORY_ERROR,
		       "Failed to allocate memory for kernel invocation arguments!\n");
	}

	for (i=0;i<ctx->gr->nds;i++){
		ctx->L[i]  = 1;
	}
	for (i=0;i<ctx->gr->log2MaxBS;i++){
		ctx->Li[i] = 1;
	}


	/**
	 * STEP 1: Select Intra-Block Axes.
	 *
	 * Sort the axes in the order likely to maximize contiguity of source
	 * memory accesses, then tag them to the kernel block size limit, possibly
	 * splitting an axis in the process.
	 */

	reduxSortAxisPtrsBy(ctx->xdSrcPtrs, ctx->xdSrc, ctx->ndfs0,
	                    reduxSortPtrS0AbsStride);
	target = reduxGenGetMaxLocalSize(ctx->gr);

	for (i=0;i<ctx->ndfs0 && i<ctx->gr->log2MaxBS;i++){
		axis = reduxInvGetSrcSortAxis(ctx, i);
		aL   = axisGetLen(axis);

		if (ctx->bs*aL <= target){
			ctx->bs     *= aL;
			axisMarkIntraBlock(axis, i, aL);
		}else{
			if (target/ctx->bs >= 2){
				aLS          = target/ctx->bs;
				ctx->bs     *= aLS;
				axisMarkIntraBlock(axis, i, aLS);
				ctx->xdSplit = axis;
				i++;
			}
			break;
		}
	}
	ctx->ndib   = i;
	ctx->LSlice = ctx->xdSplit ? axisGetIntraLen(ctx->xdSplit) : 1;


	/**
	 * STEP 2: Compute U, B, D, Dunit, H
	 */

	for (i=0;i<ctx->ndfs0;i++){
		axis    = reduxInvGetSrcAxis(ctx, i);
		ctx->U *= axisGetInterLen(axis);
		ctx->B *= axisIsReduced(axis) ? axisGetInterLen(axis) : 1;
		ctx->D *=!axisIsReduced(axis) ? axisGetIntraLen(axis) : 1;
	}
	ctx->H     = ctx->D<ctx->bs ? reduxNextPow2(ctx->bs) : ctx->bs;
	ctx->Dunit = ctx->D/ctx->LSlice;


	/**
	 * STEP 3: Compute shared memory parameters.
	 */

	ctx->shmemBytes  = reduxGenGetSHMEMSize (ctx->gr, ctx->H);
	ctx->SHMEMK0Off  = reduxGenGetSHMEMK0Off(ctx->gr, ctx->H);
	ctx->SHMEMK1Off  = reduxGenGetSHMEMK1Off(ctx->gr, ctx->H);


	/**
	 * STEP 4: Compute I0 stride values.
	 *
	 * This will be used for index calculation.
	 */

	reduxSortAxisPtrsBy(ctx->xdSrcPtrs, ctx->xdSrc, ctx->ndfs0,
	                    reduxSortPtrByReduxNum);
	for (i=0,i0S=1;i<ctx->ndfs0;i++){
		axis = reduxInvGetSrcSortAxis(ctx, i);

		if (axisIsReduced(axis)){
			axisSetI0Stride(axis, i0S);
			i0S *= axisGetLen(axis);
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

	reduxSortAxisPtrsBy(ctx->xdSrcPtrs, ctx->xdSrc, ctx->ndfs0,
	                    reduxInvRequiresD0(ctx)?
	                    reduxSortPtrD0WrSelect :
	                    reduxSortPtrD1WrSelect);
	for (i=0,perm=1;i<ctx->ndfs0;i++){
		axis = reduxInvGetSrcSortAxis(ctx, i);

		if (axisIsIntra(axis)){
			if (i>0 && axisIsReduced(axis)){
				prevAxis = reduxInvGetSrcSortAxis(ctx, i-1);
				if (!axisIsReduced(prevAxis)){
					/**
					 * The permute stride of the lowest-absolute-stride
					 * reduced axis must be a power of two to make horizontal
					 * reduction easier.
					 */

					perm = reduxNextPow2(perm);
				}
			}
			axisSetPerm(axis, perm);
			perm *= axisGetIntraLen(axis);
		}
	}


	/**
	 * STEP 6. Place the intra axis arguments
	 *
	 *              LN, perm, S0SNi, D0SNi, D1SNi, I0SNi
	 *
	 * For this we need the axes in final order of insertion.
	 */

	reduxSortAxisPtrsBy(ctx->xdSrcPtrs, ctx->xdSrc, ctx->ndfs0,
	                    reduxSortPtrInsertFinalOrder);
	for (i=0;i<ctx->ndib;i++){
		axis = reduxInvGetSrcSortAxis(ctx,  i);

		ctx->Li  [i] = axisGetIntraLen(axis);
		ctx->perm[i] = axisGetPerm    (axis);
		ctx->S0Si[i] = axisGetS0Stride(axis);
		ctx->D0Si[i] = axisGetD0Stride(axis);
		ctx->D1Si[i] = axisGetD1Stride(axis);
		ctx->I0Si[i] = axisGetI0Stride(axis);
	}


	/**
	 * STEP 7. Place the inter axis arguments
	 *
	 *              LN, S0JN, D0JN, D1JN, I0JN
	 *
	 * , where N in [0, ctx->gr->ndd) are free axes,
	 *         N in [ctx->gr->ndd, ctx->gr->nds) are reduced axes,
	 * and ctx->xdSrcPtr[...] are sorted in the reverse of that order for
	 * insertion, and excludes any intra axis (including the split one).
	 *
	 * How precisely the insertion is done depends closely on whether there is
	 * a split axis and if so whether it is free or reduced.
	 *
	 * - If there is a split axis and it is free, then it should be inserted as
	 *   the first free axis. Its jumps should be
	 *             S0JN = -S0SM*intrainterLenM + S0SN*splitFree
	 *             D0JN = -D0SM*intrainterLenM + D0SN*splitFree
	 *             D1JN = -D1SM*intrainterLenM + D1SN*splitFree
	 *             I0JN = -I0SM*intrainterLenM + I0SN*splitFree
	 * - If there is a split axis and it is reduced, then it should be inserted
	 *   as the first reduced axis. Its jump should be
	 *             S0JN = -S0SM*intrainterLenM + S0SN*splitReduced
	 *             I0JN = -I0SM*intrainterLenM + I0SN*splitReduced
	 * - If there is no split axis, proceed normally in filling the axes.
	 */

	haveSplitFreeAxis    = ctx->xdSplit && !axisIsReduced(ctx->xdSplit);
	haveSplitReducedAxis = ctx->xdSplit &&  axisIsReduced(ctx->xdSplit);
	j                    = ctx->gr->nds-1;

	/* If we have a reduced split axis, insert it before any other reduced axis. */
	if (haveSplitReducedAxis && j>=ctx->gr->ndd){
		ctx->L  [j]  =          axisGetLen     (ctx->xdSplit);
		ctx->S0J[j] += (ssize_t)axisGetS0Stride(ctx->xdSplit)*
		               (ssize_t)axisGetIntraLen(ctx->xdSplit);
		ctx->I0J[j] += (ssize_t)axisGetI0Stride(ctx->xdSplit)*
		               (ssize_t)axisGetIntraLen(ctx->xdSplit);
		if (j>0){
			ctx->S0J[j-1] -= (ssize_t)axisGetS0Stride     (ctx->xdSplit)*
			                 (ssize_t)axisGetIntraInterLen(ctx->xdSplit);
			ctx->I0J[j-1] -= (ssize_t)axisGetI0Stride     (ctx->xdSplit)*
			                 (ssize_t)axisGetIntraInterLen(ctx->xdSplit);
		}
		j--;
	}

	/* Insert rest of reduced axes. */
	for (;i<ctx->ndfs0 && j>=ctx->gr->ndd;i++,j--){
		axis = reduxInvGetSrcSortAxis(ctx, i);
		if (!axisIsReduced(axis)){
			break;
		}

		ctx->L  [j]  =          axisGetLen     (axis);
		ctx->S0J[j] += (ssize_t)axisGetS0Stride(axis)*
		               (ssize_t)axisGetIntraLen(axis);
		ctx->I0J[j] += (ssize_t)axisGetI0Stride(axis)*
		               (ssize_t)axisGetIntraLen(axis);
		if (j>0){
			ctx->S0J[j-1] -= (ssize_t)axisGetS0Stride     (axis)*
			                 (ssize_t)axisGetIntraInterLen(axis);
			ctx->I0J[j-1] -= (ssize_t)axisGetI0Stride     (axis)*
			                 (ssize_t)axisGetIntraInterLen(axis);
		}
	}

	/* If we have a free split axis, insert it before any other free axis. */
	j = ctx->gr->ndd-1;
	if (haveSplitFreeAxis && j>=0){
		ctx->L  [j]  =          axisGetLen     (ctx->xdSplit);
		ctx->S0J[j] += (ssize_t)axisGetS0Stride(ctx->xdSplit)*
		               (ssize_t)axisGetIntraLen(ctx->xdSplit);
		ctx->D0J[j] += (ssize_t)axisGetD0Stride(ctx->xdSplit)*
		               (ssize_t)axisGetIntraLen(ctx->xdSplit);
		ctx->D1J[j] += (ssize_t)axisGetD1Stride(ctx->xdSplit)*
		               (ssize_t)axisGetIntraLen(ctx->xdSplit);
		ctx->I0J[j] += (ssize_t)axisGetI0Stride(ctx->xdSplit)*
		               (ssize_t)axisGetIntraLen(ctx->xdSplit);
		if (j>0){
			ctx->S0J[j-1] -= (ssize_t)axisGetS0Stride     (ctx->xdSplit)*
			                 (ssize_t)axisGetIntraInterLen(ctx->xdSplit);
			ctx->D0J[j-1] -= (ssize_t)axisGetD0Stride     (ctx->xdSplit)*
			                 (ssize_t)axisGetIntraInterLen(ctx->xdSplit);
			ctx->D1J[j-1] -= (ssize_t)axisGetD1Stride     (ctx->xdSplit)*
			                 (ssize_t)axisGetIntraInterLen(ctx->xdSplit);
			ctx->I0J[j-1] -= (ssize_t)axisGetI0Stride     (ctx->xdSplit)*
			                 (ssize_t)axisGetIntraInterLen(ctx->xdSplit);
		}
		j--;
	}

	/* Insert rest of free axes. */
	for (;i<ctx->ndfs0 && j>=0;i++,j--){
		axis = reduxInvGetSrcSortAxis(ctx, i);
		if (axisIsReduced(axis)){
			break;
		}

		ctx->L  [j]  =          axisGetLen     (axis);
		ctx->S0J[j] += (ssize_t)axisGetS0Stride(axis)*
		               (ssize_t)axisGetIntraLen(axis);
		ctx->D0J[j] += (ssize_t)axisGetD0Stride(axis)*
		               (ssize_t)axisGetIntraLen(axis);
		ctx->D1J[j] += (ssize_t)axisGetD1Stride(axis)*
		               (ssize_t)axisGetIntraLen(axis);
		ctx->I0J[j] += (ssize_t)axisGetI0Stride(axis)*
		               (ssize_t)axisGetIntraLen(axis);
		if (j>0){
			ctx->S0J[j-1] -= (ssize_t)axisGetS0Stride     (axis)*
			                 (ssize_t)axisGetIntraInterLen(axis);
			ctx->D0J[j-1] -= (ssize_t)axisGetD0Stride     (axis)*
			                 (ssize_t)axisGetIntraInterLen(axis);
			ctx->D1J[j-1] -= (ssize_t)axisGetD1Stride     (axis)*
			                 (ssize_t)axisGetIntraInterLen(axis);
			ctx->I0J[j-1] -= (ssize_t)axisGetI0Stride     (axis)*
			                 (ssize_t)axisGetIntraInterLen(axis);
		}
	}


	/**
	 * STEP 8. Compute the template selector. Requires finding the huge axis,
	 *         if any. Then, compute LPadded, which depends on the selector
	 *         value we choose.
	 */

	if (ctx->xdSplit && !axisIsReduced(ctx->xdSplit)){
		ctx->selector |= SELECTOR_SPLIT_FREE;
	}
	for (i=0;i<ctx->ndfs0;i++){
		axis = reduxInvGetSrcAxis(ctx, i);

		if (axisGetLen(axis) >= ((uint64_t)1<<31)){
			if (axis == ctx->xdSplit){
				ctx->selector |= SELECTOR_HUGE_IS_SPLIT;
			}else if (axisIsReduced(axis) == axisIsReduced(ctx->xdSplit)){
				ctx->selector |= SELECTOR_HUGE_SAME_TYPE;
			}else{
				ctx->selector |= SELECTOR_HUGE_OPPOSITE_TYPE;
			}
		}
	}
	if (ctx->selector & SELECTOR_SPLIT_FREE){
		if (ctx->gr->ndd>0){
			ctx->LPadded = ctx->L[ctx->gr->ndd-1];
		}
	}else{
		if (ctx->gr->nds>0){
			ctx->LPadded = ctx->L[ctx->gr->nds-1];
		}
	}
	ctx->LPadded = DIVIDECEIL(ctx->LPadded, ctx->LSlice)*ctx->LSlice;


	/* Schedule. */
	return reduxInvSchedule(ctx);
}

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
 * To do this, we first choose gs to be the number of blocks that roughly fills
 * the available parallelism given the block size, but reduce it to at most U
 * (The universal amount of vertical reductions to be done).
 *
 * We then select V as the minimum number of vertical reductions per block
 * that will cover the universe U.
 *
 * Lastly, iff there exists a value V <= V' <= 2*V such that V' % B == 0, then
 * increase V to the smallest such V' and recompute ctx->gs.
 *
 * Once the scheduling is performed, the workspace can be allocated and
 * workspace offsets can be computed.
 */

static int        reduxInvSchedule              (redux_ctx*           ctx){
	const int flags = GA_BUFFER_READ_WRITE;
	size_t    WSPACESIZE;

	/**
	 * Scheduling
	 */

	ctx->gs = DIVIDECEIL(reduxInvEstimateParallelism(ctx),
	                     reduxGenGetMaxLocalSize(ctx->gr));
	ctx->gs = ctx->gs > ctx->U ? ctx->U : ctx->gs;
	ctx->V  = DIVIDECEIL(ctx->U, ctx->gs);
	if (ctx->V%ctx->B != 0 && ctx->V*2 >= ctx->B){
		ctx->V  = DIVIDECEIL(ctx->V, ctx->B)*ctx->B;
	}
	ctx->gs = DIVIDECEIL(ctx->U, ctx->V);

	/**
	 * Allocate required workspace.
	 */

	ctx->W0Off = reduxGenGetWMEMK0Off(ctx->gr, 2*ctx->gs*ctx->D);
	ctx->W1Off = reduxGenGetWMEMK1Off(ctx->gr, 2*ctx->gs*ctx->D);
	WSPACESIZE = reduxGenGetWMEMSize (ctx->gr, 2*ctx->gs*ctx->D);
	ctx->W     = gpudata_alloc(ctx->gr->gpuCtx, WSPACESIZE, 0, flags, 0);
	if (!ctx->W){
		return reduxInvCleanupMsg(ctx, GA_MEMORY_ERROR,
		       "Could not allocate %zu-byte workspace for reduction!\n",
		       WSPACESIZE);
	}

	return reduxInvoke(ctx);
}

/**
 * @brief Invoke the kernel.
 */

static int         reduxInvoke                      (redux_ctx*           ctx){
	int   ret, i=0;
	void* ptrs[2] = {ctx, &i};

	/**
	 * Argument Marshalling.
	 */

	reduxGenIterArgs(ctx->gr, reduxInvMarshalArg, ptrs);



	/**
	 * The kernel is now invoked once or twice, for phase 0 or 1.
	 *
	 * Phase 1 is optional iff V%B == 0.
	 */

	ret = GpuKernel_call((GpuKernel*)&ctx->gr->k, 1, &ctx->gs, &ctx->bs, ctx->shmemBytes, ctx->kArgs);
	if (ret != GA_NO_ERROR){
		return reduxInvCleanupMsg(ctx, ret,
		                          "Failure in kernel call, Phase 0!\n");
	}

	if (ctx->V % ctx->B != 0){
		ctx->selector |= SELECTOR_PHASE1;
		ret = GpuKernel_call((GpuKernel*)&ctx->gr->k, 1, &ctx->gs, &ctx->bs, ctx->shmemBytes, ctx->kArgs);
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

static int         reduxInvCleanup                  (redux_ctx*           ctx, int ret){
	ctx->gr                = NULL;
	ctx->s0                = NULL;
	ctx->d0                = NULL;
	ctx->d1                = NULL;
	ctx->reduxList         = NULL;

	free(ctx->xdSrc);
	free(ctx->xdSrcPtrs);
	free(ctx->L);
	free(ctx->Li);
	free(ctx->S0J);
	free(ctx->S0Si);
	free(ctx->D0J);
	free(ctx->D0Si);
	free(ctx->D1J);
	free(ctx->D1Si);
	free(ctx->I0J);
	free(ctx->I0Si);
	free(ctx->perm);
	free(ctx->kArgs);
	gpudata_release(ctx->W);

	ctx->xdSrc             = NULL;
	ctx->xdSrcPtrs         = NULL;
	ctx->L                 = NULL;
	ctx->Li                = NULL;
	ctx->S0J               = NULL;
	ctx->S0Si              = NULL;
	ctx->D0J               = NULL;
	ctx->D0Si              = NULL;
	ctx->D1J               = NULL;
	ctx->D1Si              = NULL;
	ctx->I0J               = NULL;
	ctx->I0Si              = NULL;
	ctx->perm              = NULL;
	ctx->kArgs             = NULL;
	ctx->W                 = NULL;

	return ret;
}
static int         reduxInvCleanupMsg               (redux_ctx*           ctx, int ret,
                                                     const char*          fmt, ...){
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

