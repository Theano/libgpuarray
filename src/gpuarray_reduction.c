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
#include "gpuarray/array.h"
#include "gpuarray/error.h"
#include "gpuarray/kernel.h"
#include "gpuarray/util.h"

#include "util/strb.h"
#include "util/integerfactoring.h"


/* Datatypes */
struct maxandargmax_ctx{
	/* Function Arguments. */
	GpuArray*       dstMax;
	GpuArray*       dstArgmax;
	const GpuArray* src;
	int             reduxLen;
	const int*      reduxList;
	
	/* General. */
	int             ret;
	int*            axisList;
	gpucontext*     gpuCtx;
	
	/* Source code Generator. */
	const char*     dstMaxType;
	const char*     dstArgmaxType;
	int             ndd;
	int             ndr;
	int             nds;
	int             ndh;
	strb            s;
	char*           sourceCode;
	GpuKernel       kernel;
	
	/* Scheduler */
	int             hwAxisList[3];
	size_t          blockSize [3];
	size_t          gridSize  [3];
	size_t          chunkSize [3];
	
	/* Invoker */
	gpudata*        srcStepsGD;
	gpudata*        srcSizeGD;
	gpudata*        chunkSizeGD;
	gpudata*        dstMaxStepsGD;
	gpudata*        dstArgmaxStepsGD;
};
typedef struct maxandargmax_ctx maxandargmax_ctx;



/* Function prototypes */
static int   axisInSet                          (int                v,
                                                 const int*         set,
                                                 size_t             setLen,
                                                 size_t*            where);
static void  appendIdxes                        (strb*              s,
                                                 const char*        prologue,
                                                 const char*        prefix,
                                                 int                startIdx,
                                                 int                endIdx,
                                                 const char*        suffix,
                                                 const char*        epilogue);
static int   maxandargmaxCheckargs              (maxandargmax_ctx*  ctx);
static int   maxandargmaxSelectHwAxes           (maxandargmax_ctx*  ctx);
static int   maxandargmaxGenSource              (maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendKernel           (maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendTypedefs         (maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendPrototype        (maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendOffsets          (maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendIndexDeclarations(maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendRangeCalculations(maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendLoops            (maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendLoopMacroDefs    (maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendLoopOuter        (maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendLoopInner        (maxandargmax_ctx*  ctx);
static void  maxandargmaxAppendLoopMacroUndefs  (maxandargmax_ctx*  ctx);
static void  maxandargmaxComputeAxisList        (maxandargmax_ctx*  ctx);
static int   maxandargmaxCompile                (maxandargmax_ctx*  ctx);
static int   maxandargmaxSchedule               (maxandargmax_ctx*  ctx);
static int   maxandargmaxInvoke                 (maxandargmax_ctx*  ctx);
static int   maxandargmaxCleanup                (maxandargmax_ctx*  ctx);


/* Function implementation */
GPUARRAY_PUBLIC int GpuArray_maxandargmax       (GpuArray*       dstMax,
                                                 GpuArray*       dstArgmax,
                                                 const GpuArray* src,
                                                 unsigned        reduxLen,
                                                 const unsigned* reduxList){
	maxandargmax_ctx  ctxSTACK = {dstMax, dstArgmax, src,
	                              (int)reduxLen, (const int*)reduxList},
	                 *ctx      = &ctxSTACK;
	
	if(maxandargmaxCheckargs   (ctx) == GA_NO_ERROR &&
	   maxandargmaxSelectHwAxes(ctx) == GA_NO_ERROR &&
	   maxandargmaxGenSource   (ctx) == GA_NO_ERROR &&
	   maxandargmaxCompile     (ctx) == GA_NO_ERROR &&
	   maxandargmaxSchedule    (ctx) == GA_NO_ERROR &&
	   maxandargmaxInvoke      (ctx) == GA_NO_ERROR){
		return maxandargmaxCleanup(ctx);
	}else{
		return maxandargmaxCleanup(ctx);
	}
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

static int   axisInSet                          (int                v,
                                                 const int*         set,
                                                 size_t             setLen,
                                                 size_t*            where){
	size_t i;
	
	for(i=0;i<setLen;i++){
		if(set[i] == v){
			if(where){*where = i;}
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

static void  appendIdxes                        (strb*              s,
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
	for(i=startIdx;i<endIdx;i++){
		strb_appendf(s, "%s%d%s%s", prefix, i, suffix, &","[i==endIdx-1]);
	}
	strb_appends(s, epilogue);
}

/**
 * @brief Check the sanity of the arguments, in agreement with the
 *        documentation for GpuArray_maxandargmax().
 *        
 *        Also initialize certain parts of the context.
 * 
 * @return GA_INVALID_ERROR if arguments invalid; GA_NO_ERROR otherwise.
 */

static int   maxandargmaxCheckargs              (maxandargmax_ctx*  ctx){
	int i;
	
	/**
	 * We initialize certain parts of the context.
	 */
	
	ctx->ret           = GA_NO_ERROR;
	ctx->axisList      = NULL;
	ctx->gpuCtx        = NULL;
	
	ctx->dstMaxType    = ctx->dstArgmaxType = NULL;
	ctx->ndh           = 0;
	ctx->s             = (strb)STRB_STATIC_INIT;
	ctx->sourceCode    = NULL;
	
	ctx->hwAxisList[0] = ctx->hwAxisList[1] = ctx->hwAxisList[2] = 0;
	ctx->blockSize [0] = ctx->blockSize [1] = ctx->blockSize [2] = 1;
	ctx->gridSize  [0] = ctx->gridSize  [1] = ctx->gridSize  [2] = 1;
	ctx->chunkSize [0] = ctx->chunkSize [1] = ctx->chunkSize [2] = 1;
	
	ctx->srcStepsGD    = ctx->srcSizeGD     = ctx->chunkSizeGD   =
	ctx->dstMaxStepsGD = ctx->dstArgmaxStepsGD = NULL;
	
	
	/* Insane src or reduxLen? */
	if(!ctx->dstMax || !ctx->dstArgmax || !ctx->src || ctx->src->nd == 0 ||
	    ctx->reduxLen == 0 || ctx->reduxLen > (int)ctx->src->nd){
		return ctx->ret=GA_INVALID_ERROR;
	}
	
	/* Insane or duplicate list entry? */
	for(i=0;i<ctx->reduxLen;i++){
		if(ctx->reduxList[i] <  0                            ||
		   ctx->reduxList[i] >= (int)ctx->src->nd            ||
		   axisInSet(ctx->reduxList[i], ctx->reduxList, i, 0)){
			return ctx->ret=GA_INVALID_ERROR;
		}
	}
	
	/* Unknown type? */
	ctx->dstMaxType    = gpuarray_get_type(ctx->src->typecode)->cluda_name;
	ctx->dstArgmaxType = gpuarray_get_type(GA_SSIZE)          ->cluda_name;
	if(!ctx->dstMaxType || !ctx->dstArgmaxType){
		return ctx->ret=GA_INVALID_ERROR;
	}
	
	/* GPU context non-existent? */
	ctx->gpuCtx        = GpuArray_context(ctx->src);
	if(!ctx->gpuCtx){
		return ctx->ret=GA_INVALID_ERROR;
	}
	
	
	/**
	 * We initialize some more parts of the context, using the guarantees
	 * we now have about the sanity of the arguments.
	 */
	
	ctx->nds = ctx->src->nd;
	ctx->ndr = ctx->reduxLen;
	ctx->ndd = ctx->nds - ctx->ndr;
	
	return ctx->ret;
}

/**
 * @brief Select which axes (up to 3) will be assigned to hardware
 *        dimensions.
 */

static int   maxandargmaxSelectHwAxes           (maxandargmax_ctx*  ctx){
	int    i, j, maxI = 0;
	size_t maxV;
	
	ctx->ndh = ctx->ndd<3 ? ctx->ndd : 3;
	
	/**
	 * The ctx->hwAxisLen largest axes are selected and assigned in
	 * descending order to X, Y, Z.
	 */
	
	for(i=0;i<ctx->ndh;i++){
		maxV = 0;
		
		for(j=0;j<ctx->nds;j++){
			if(!axisInSet(j, ctx->hwAxisList, i,        0) &&
			   !axisInSet(j, ctx->reduxList,  ctx->ndr, 0) &&
			   ctx->src->dimensions[j] >= maxV){
				maxV = ctx->src->dimensions[j];
				maxI = j;
			}
		}
		
		ctx->hwAxisList[i] = maxI;
	}
	
	return ctx->ret=GA_NO_ERROR;
}

/**
 * @brief Generate the kernel code for MaxAndArgmax.
 * 
 * @return GA_MEMORY_ERROR if not enough memory left; GA_NO_ERROR otherwise.
 */

static int   maxandargmaxGenSource              (maxandargmax_ctx*  ctx){
	/* Compute internal axis remapping. */
	ctx->axisList = malloc(ctx->nds * sizeof(unsigned));
	if(!ctx->axisList){
		return ctx->ret=GA_MEMORY_ERROR;
	}
	maxandargmaxComputeAxisList(ctx);
	
	/* Generate kernel proper. */
	strb_ensure(&ctx->s, 5*1024);
	maxandargmaxAppendKernel(ctx);
	free(ctx->axisList);
	ctx->axisList   = NULL;
	ctx->sourceCode = strb_cstr(&ctx->s);
	if(!ctx->sourceCode){
		return ctx->ret=GA_MEMORY_ERROR;
	}
	
	/* Return it. */
	return ctx->ret=GA_NO_ERROR;
}
static void  maxandargmaxAppendKernel           (maxandargmax_ctx*  ctx){
	maxandargmaxAppendTypedefs         (ctx);
	maxandargmaxAppendPrototype        (ctx);
	strb_appends           (&ctx->s, "{\n");
	maxandargmaxAppendOffsets          (ctx);
	maxandargmaxAppendIndexDeclarations(ctx);
	maxandargmaxAppendRangeCalculations(ctx);
	maxandargmaxAppendLoops            (ctx);
	strb_appends           (&ctx->s, "}\n");
}
static void  maxandargmaxAppendTypedefs         (maxandargmax_ctx*  ctx){
	strb_appends(&ctx->s, "/* Typedefs */\n");
	strb_appendf(&ctx->s, "typedef %s     T;/* The type of the array being processed. */\n", ctx->dstMaxType);
	strb_appendf(&ctx->s, "typedef %s     X;/* Index type: signed 32/64-bit. */\n",          ctx->dstArgmaxType);
	strb_appends(&ctx->s, "\n");
	strb_appends(&ctx->s, "\n");
	strb_appends(&ctx->s, "\n");
}
static void  maxandargmaxAppendPrototype        (maxandargmax_ctx*  ctx){
	strb_appends(&ctx->s, "KERNEL void maxandargmax(const GLOBAL_MEM T*        src,\n");
	strb_appends(&ctx->s, "                         const X         srcOff,\n");
	strb_appends(&ctx->s, "                         const GLOBAL_MEM X*        srcSteps,\n");
	strb_appends(&ctx->s, "                         const GLOBAL_MEM X*        srcSize,\n");
	strb_appends(&ctx->s, "                         const GLOBAL_MEM X*        chunkSize,\n");
	strb_appends(&ctx->s, "                         GLOBAL_MEM T*              dstMax,\n");
	strb_appends(&ctx->s, "                         const X         dstMaxOff,\n");
	strb_appends(&ctx->s, "                         const GLOBAL_MEM X*        dstMaxSteps,\n");
	strb_appends(&ctx->s, "                         GLOBAL_MEM X*              dstArgmax,\n");
	strb_appends(&ctx->s, "                         const X         dstArgmaxOff,\n");
	strb_appends(&ctx->s, "                         const GLOBAL_MEM X*        dstArgmaxSteps)");
}
static void  maxandargmaxAppendOffsets          (maxandargmax_ctx*  ctx){
	strb_appends(&ctx->s, "\t/* Add offsets */\n");
	strb_appends(&ctx->s, "\tsrc       = (const GLOBAL_MEM T*)((const GLOBAL_MEM char*)src       + srcOff);\n");
	strb_appends(&ctx->s, "\tdstMax    = (GLOBAL_MEM T*)      ((GLOBAL_MEM char*)      dstMax    + dstMaxOff);\n");
	strb_appends(&ctx->s, "\tdstArgmax = (GLOBAL_MEM X*)      ((GLOBAL_MEM char*)      dstArgmax + dstArgmaxOff);\n");
	strb_appends(&ctx->s, "\t\n");
	strb_appends(&ctx->s, "\t\n");
}
static void  maxandargmaxAppendIndexDeclarations(maxandargmax_ctx*  ctx){
	int i;
	strb_appends(&ctx->s, "\t/* GPU kernel coordinates. Always 3D. */\n");
	
	strb_appends(&ctx->s, "\tX bi0 = GID_0,        bi1 = GID_1,        bi2 = GID_2;\n");
	strb_appends(&ctx->s, "\tX bd0 = LDIM_0,       bd1 = LDIM_1,       bd2 = LDIM_2;\n");
	strb_appends(&ctx->s, "\tX ti0 = LID_0,        ti1 = LID_1,        ti2 = LID_2;\n");
	strb_appends(&ctx->s, "\tX gi0 = bi0*bd0+ti0,  gi1 = bi1*bd1+ti1,  gi2 = bi2*bd2+ti2;\n");
	if(ctx->ndh>0){
		strb_appends(&ctx->s, "\tX ");
		for(i=0;i<ctx->ndh;i++){
			strb_appendf(&ctx->s, "ci%u = chunkSize[%u]%s",
			             i, i, (i==ctx->ndh-1) ? ";\n" : ", ");
		}
	}
	
	strb_appends(&ctx->s, "\t\n");
	strb_appends(&ctx->s, "\t\n");
	strb_appends(&ctx->s, "\t/* Free indices & Reduction indices */\n");
	
	if(ctx->nds > 0){appendIdxes (&ctx->s, "\tX ", "i", 0,               ctx->nds, "",        ";\n");}
	if(ctx->nds > 0){appendIdxes (&ctx->s, "\tX ", "i", 0,               ctx->nds, "Dim",     ";\n");}
	if(ctx->nds > 0){appendIdxes (&ctx->s, "\tX ", "i", 0,               ctx->nds, "Start",   ";\n");}
	if(ctx->nds > 0){appendIdxes (&ctx->s, "\tX ", "i", 0,               ctx->nds, "End",     ";\n");}
	if(ctx->nds > 0){appendIdxes (&ctx->s, "\tX ", "i", 0,               ctx->nds, "SStep",   ";\n");}
	if(ctx->ndd > 0){appendIdxes (&ctx->s, "\tX ", "i", 0,               ctx->ndd, "MStep",   ";\n");}
	if(ctx->ndd > 0){appendIdxes (&ctx->s, "\tX ", "i", 0,               ctx->ndd, "AStep",   ";\n");}
	if(ctx->nds > ctx->ndd){appendIdxes (&ctx->s, "\tX ", "i", ctx->ndd, ctx->nds, "PDim",    ";\n");}
	
	strb_appends(&ctx->s, "\t\n");
	strb_appends(&ctx->s, "\t\n");
}
static void  maxandargmaxAppendRangeCalculations(maxandargmax_ctx*  ctx){
	size_t hwDim;
	int    i;
	
	/* Use internal remapping when computing the ranges for this thread. */
	strb_appends(&ctx->s, "\t/* Compute ranges for this thread. */\n");
	
	for(i=0;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "\ti%dDim     = srcSize[%d];\n", i, ctx->axisList[i]);
	}
	for(i=0;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "\ti%dSStep   = srcSteps[%d];\n", i, ctx->axisList[i]);
	}
	for(i=0;i<ctx->ndd;i++){
		strb_appendf(&ctx->s, "\ti%dMStep   = dstMaxSteps[%d];\n", i, i);
	}
	for(i=0;i<ctx->ndd;i++){
		strb_appendf(&ctx->s, "\ti%dAStep   = dstArgmaxSteps[%d];\n", i, i);
	}
	for(i=ctx->nds-1;i>=ctx->ndd;i--){
		/**
		 * If this is the last index, it's the first cumulative dimension
		 * product we generate, and thus we initialize to 1.
		 */
		
		if(i == ctx->nds-1){
			strb_appendf(&ctx->s, "\ti%dPDim    = 1;\n", i);
		}else{
			strb_appendf(&ctx->s, "\ti%dPDim    = i%dPDim * i%dDim;\n", i, i+1, i+1);
		}
	}
	for(i=0;i<ctx->nds;i++){
		/**
		 * Up to 3 dimensions get to rely on hardware loops.
		 * The others, if any, have to use software looping beginning at 0.
		 */
		
		if(axisInSet(ctx->axisList[i], ctx->hwAxisList, ctx->ndh, &hwDim)){
			strb_appendf(&ctx->s, "\ti%dStart   = gi%d * ci%d;\n", i, hwDim, hwDim);
		}else{
			strb_appendf(&ctx->s, "\ti%dStart   = 0;\n", i);
		}
	}
	for(i=0;i<ctx->nds;i++){
		/**
		 * Up to 3 dimensions get to rely on hardware loops.
		 * The others, if any, have to use software looping beginning at 0.
		 */
		
		if(axisInSet(ctx->axisList[i], ctx->hwAxisList, ctx->ndh, &hwDim)){
			strb_appendf(&ctx->s, "\ti%dEnd     = i%dStart + ci%d;\n", i, i, hwDim);
		}else{
			strb_appendf(&ctx->s, "\ti%dEnd     = i%dStart + i%dDim;\n", i, i, i);
		}
	}
	
	strb_appends(&ctx->s, "\t\n");
	strb_appends(&ctx->s, "\t\n");
}
static void  maxandargmaxAppendLoops            (maxandargmax_ctx*  ctx){
	strb_appends(&ctx->s, "\t/**\n");
	strb_appends(&ctx->s, "\t * FREE LOOPS.\n");
	strb_appends(&ctx->s, "\t */\n");
	strb_appends(&ctx->s, "\t\n");
	
	maxandargmaxAppendLoopMacroDefs  (ctx);
	maxandargmaxAppendLoopOuter      (ctx);
	maxandargmaxAppendLoopMacroUndefs(ctx);
}
static void  maxandargmaxAppendLoopMacroDefs    (maxandargmax_ctx*  ctx){
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
	 * SRCINDEXER Macro
	 */
	
	appendIdxes (&ctx->s, "#define SRCINDEXER(", "i", 0, ctx->nds, "", ")   (*(GLOBAL_MEM T*)((GLOBAL_MEM char*)src + ");
	for(i=0;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "i%d*i%dSStep + \\\n                                            ", i, i);
	}
	strb_appends(&ctx->s, "0))\n");
	
	/**
	 * RDXINDEXER Macro
	 */
	
	appendIdxes (&ctx->s, "#define RDXINDEXER(", "i", ctx->ndd, ctx->nds, "", ")              (");
	for(i=ctx->ndd;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "i%d*i%dPDim + \\\n                                        ", i, i);
	}
	strb_appends(&ctx->s, "0)\n");
	
	/**
	 * DSTMINDEXER Macro
	 */
	
	appendIdxes (&ctx->s, "#define DSTMINDEXER(", "i", 0, ctx->ndd, "", ")        (*(GLOBAL_MEM T*)((GLOBAL_MEM char*)dstMax + ");
	for(i=0;i<ctx->ndd;i++){
		strb_appendf(&ctx->s, "i%d*i%dMStep + \\\n                                                  ", i, i);
	}
	strb_appends(&ctx->s, "0))\n");
	
	/**
	 * DSTAINDEXER Macro
	 */
	
	appendIdxes (&ctx->s, "#define DSTAINDEXER(", "i", 0, ctx->ndd, "", ")        (*(GLOBAL_MEM X*)((GLOBAL_MEM char*)dstArgmax + ");
	for(i=0;i<ctx->ndd;i++){
		strb_appendf(&ctx->s, "i%d*i%dAStep + \\\n                                                     ", i, i);
	}
	strb_appends(&ctx->s, "0))\n");
}
static void  maxandargmaxAppendLoopOuter        (maxandargmax_ctx*  ctx){
	int i;
	
	/**
	 * Outer Loop Header Generation
	 */
	
	for(i=0;i<ctx->ndd;i++){
		strb_appendf(&ctx->s, "\tFOROVER(%d){ESCAPE(%d)\n", i, i);
	}
	
	/**
	 * Inner Loop Generation
	 */
	
	maxandargmaxAppendLoopInner(ctx);
	
	/**
	 * Outer Loop Trailer Generation
	 */
	
	for(i=0;i<ctx->ndd;i++){
		strb_appends(&ctx->s, "\t}\n");
	}
}
static void  maxandargmaxAppendLoopInner        (maxandargmax_ctx*  ctx){
	int i;
	
	/**
	 * Inner Loop Prologue
	 */
	
	strb_appends(&ctx->s, "\t/**\n");
	strb_appends(&ctx->s, "\t * Reduction initialization.\n");
	strb_appends(&ctx->s, "\t */\n");
	strb_appends(&ctx->s, "\t\n");
	
	appendIdxes (&ctx->s, "\tT maxV = SRCINDEXER(", "i", 0, ctx->ndd, "", "");
	if(ctx->ndd && ctx->ndr){strb_appends(&ctx->s, ",");}
	appendIdxes (&ctx->s, "", "i", ctx->ndd, ctx->nds, "Start", ");\n");
	
	appendIdxes (&ctx->s, "\tX maxI = RDXINDEXER(", "i", ctx->ndd, ctx->nds, "Start", ");\n");
	
	strb_appends(&ctx->s, "\t\n");
	strb_appends(&ctx->s, "\t/**\n");
	strb_appends(&ctx->s, "\t * REDUCTION LOOPS.\n");
	strb_appends(&ctx->s, "\t */\n");
	strb_appends(&ctx->s, "\t\n");
	
	/**
	 * Inner Loop Header Generation
	 */
	
	for(i=ctx->ndd;i<ctx->nds;i++){
		strb_appendf(&ctx->s, "\tFOROVER(%d){ESCAPE(%d)\n", i, i);
	}
	
	/**
	 * Inner Loop Body Generation
	 */
	
	appendIdxes (&ctx->s, "\tT V = SRCINDEXER(", "i", 0, ctx->nds, "", ");\n");
	strb_appends(&ctx->s, "\t\n");
	strb_appends(&ctx->s, "\tif(V > maxV){\n");
	strb_appends(&ctx->s, "\t\tmaxV = V;\n");
	appendIdxes (&ctx->s, "\t\tmaxI = RDXINDEXER(", "i", ctx->ndd, ctx->nds, "", ");\n");
	strb_appends(&ctx->s, "\t}\n");
	
	/**
	 * Inner Loop Trailer Generation
	 */
	
	for(i=ctx->ndd;i<ctx->nds;i++){
		strb_appends(&ctx->s, "\t}\n");
	}
	strb_appends(&ctx->s, "\t\n");
	
	/**
	 * Inner Loop Epilogue Generation
	 */
	
	strb_appends(&ctx->s, "\t/**\n");
	strb_appends(&ctx->s, "\t * Destination writeback.\n");
	strb_appends(&ctx->s, "\t */\n");
	strb_appends(&ctx->s, "\t\n");
	appendIdxes (&ctx->s, "\tDSTMINDEXER(", "i", 0, ctx->ndd, "", ") = maxV;\n");
	appendIdxes (&ctx->s, "\tDSTAINDEXER(", "i", 0, ctx->ndd, "", ") = maxI;\n");
}
static void  maxandargmaxAppendLoopMacroUndefs  (maxandargmax_ctx*  ctx){
	strb_appends(&ctx->s, "#undef FOROVER\n");
	strb_appends(&ctx->s, "#undef ESCAPE\n");
	strb_appends(&ctx->s, "#undef SRCINDEXER\n");
	strb_appends(&ctx->s, "#undef RDXINDEXER\n");
	strb_appends(&ctx->s, "#undef DSTMINDEXER\n");
	strb_appends(&ctx->s, "#undef DSTAINDEXER\n");
}
static void  maxandargmaxComputeAxisList        (maxandargmax_ctx*  ctx){
	int i, f=0;
	
	for(i=0;i<ctx->nds;i++){
		if(axisInSet(i, ctx->reduxList, ctx->ndr, 0)){
			continue;
		}
		ctx->axisList[f++] = i;
	}
	memcpy(&ctx->axisList[f], ctx->reduxList, ctx->ndr * sizeof(*ctx->reduxList));
}

/**
 * @brief Compile the kernel from source code.
 * 
 * @return
 */

static int   maxandargmaxCompile                (maxandargmax_ctx*  ctx){
	const int    ARG_TYPECODES[]   = {
		GA_BUFFER, /* src */
		GA_SIZE,   /* srcOff */
		GA_BUFFER, /* srcSteps */
		GA_BUFFER, /* srcSize */
		GA_BUFFER, /* chnkSize */
		GA_BUFFER, /* dstMax */
		GA_SIZE,   /* dstMaxOff */
		GA_BUFFER, /* dstMaxSteps */
		GA_BUFFER, /* dstArgmax */
		GA_SIZE,   /* dstArgmaxOff */
		GA_BUFFER  /* dstArgmaxSteps */
	};
	const size_t ARG_TYPECODES_LEN = sizeof(ARG_TYPECODES)/sizeof(*ARG_TYPECODES);
	const char*  SRCS[]            = {ctx->sourceCode};
	const size_t SRC_LENS[]        = {strlen(ctx->sourceCode)};
	const size_t SRCS_LEN          = sizeof(SRCS)/sizeof(*SRCS);
	
	ctx->ret = GpuKernel_init(&ctx->kernel,
	                          ctx->gpuCtx,
	                          SRCS_LEN,
	                          SRCS,
	                          SRC_LENS,
	                          "maxandargmax",
	                          ARG_TYPECODES_LEN,
	                          ARG_TYPECODES,
	                          GA_USE_CLUDA,
	                          (char**)0);
	free(ctx->sourceCode);
	ctx->sourceCode = NULL;
	
	return ctx->ret;
}

/**
 * Compute a good thread block size / grid size / software chunk size for Nvidia.
 */

static int   maxandargmaxSchedule               (maxandargmax_ctx*  ctx){
	int            i;
	size_t         warpMod;
	size_t         bestWarpMod  = 1;
	unsigned       bestWarpAxis = 0;
	uint64_t       maxLg;
	uint64_t       maxLs[3];
	uint64_t       maxGg;
	uint64_t       maxGs[3];
	uint64_t       dims [3];
	double         slack[3];
	ga_factor_list factBS[3];
	ga_factor_list factGS[3];
	ga_factor_list factCS[3];
	
	
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
	gpudata_property  (ctx->src->data, GA_CTX_PROP_MAXGSIZE,     &maxG);
	gpudata_property  (ctx->src->data, GA_CTX_PROP_MAXGSIZE0,    &maxG0);
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
	
	for(i=0;i<ctx->ndh;i++){
		dims[i] = ctx->src->dimensions[ctx->hwAxisList[i]];
		gaIFLInit(&factBS[i]);
		gaIFLInit(&factGS[i]);
		gaIFLInit(&factCS[i]);
		
		warpMod = dims[i]%warpSize;
		if(bestWarpMod>0 && (warpMod==0 || warpMod>=bestWarpMod)){
			bestWarpAxis = i;
			bestWarpMod  = warpMod;
		}
	}
	
	if(ctx->ndh > 0){
		dims[bestWarpAxis] = (dims[bestWarpAxis] + warpSize - 1)/warpSize;
		gaIFactorize(warpSize, 0, 0, &factBS[bestWarpAxis]);
	}
	
	/**
	 * Factorization job. We'll steadily increase the slack in case of failure
	 * in order to ensure we do get a factorization, which we place into
	 * chunkSize.
	 */
	
	for(i=0;i<ctx->ndh;i++){
		while(!gaIFactorize(dims[i], dims[i]*slack[i], maxLs[i], &factCS[i])){
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
	for(i=0;i<ctx->ndh;i++){
		ctx->blockSize[i] = gaIFLGetProduct(&factBS[i]);
		ctx->gridSize [i] = gaIFLGetProduct(&factGS[i]);
		ctx->chunkSize[i] = gaIFLGetProduct(&factCS[i]);
	}
	
	/* Return. */
	return ctx->ret=GA_NO_ERROR;
}

/**
 * Invoke the kernel.
 */

static int   maxandargmaxInvoke                 (maxandargmax_ctx*  ctx){
	void* args[11];
	
	/**
	 * Argument Marshalling. This the grossest gross thing in here.
	 */
	
	const int flags       = GA_BUFFER_READ_ONLY|GA_BUFFER_INIT;
	ctx->srcStepsGD       = gpudata_alloc(ctx->gpuCtx, ctx->nds    * sizeof(size_t),
	                                      ctx->src->strides,       flags, 0);
	ctx->srcSizeGD        = gpudata_alloc(ctx->gpuCtx, ctx->nds    * sizeof(size_t),
	                                      ctx->src->dimensions,    flags, 0);
	ctx->chunkSizeGD      = gpudata_alloc(ctx->gpuCtx, ctx->ndh * sizeof(size_t),
	                                      ctx->chunkSize,          flags, 0);
	ctx->dstMaxStepsGD    = gpudata_alloc(ctx->gpuCtx, ctx->ndd * sizeof(size_t),
	                                      ctx->dstMax->strides,    flags, 0);
	ctx->dstArgmaxStepsGD = gpudata_alloc(ctx->gpuCtx, ctx->ndd * sizeof(size_t),
	                                      ctx->dstArgmax->strides, flags, 0);
	args[ 0] = (void*) ctx->src->data;
	args[ 1] = (void*)&ctx->src->offset;
	args[ 2] = (void*) ctx->srcStepsGD;
	args[ 3] = (void*) ctx->srcSizeGD;
	args[ 4] = (void*) ctx->chunkSizeGD;
	args[ 5] = (void*) ctx->dstMax->data;
	args[ 6] = (void*)&ctx->dstMax->offset;
	args[ 7] = (void*) ctx->dstMaxStepsGD;
	args[ 8] = (void*) ctx->dstArgmax->data;
	args[ 9] = (void*)&ctx->dstArgmax->offset;
	args[10] = (void*) ctx->dstArgmaxStepsGD;
	
	if(ctx->srcStepsGD      &&
	   ctx->srcSizeGD       &&
	   ctx->chunkSizeGD     &&
	   ctx->dstMaxStepsGD   &&
	   ctx->dstArgmaxStepsGD){
		ctx->ret = GpuKernel_call(&ctx->kernel,
		                          ctx->ndh>0 ? ctx->ndh : 1,
		                          ctx->blockSize,
		                          ctx->gridSize,
		                          0,
		                          args);
	}else{
		ctx->ret = GA_MEMORY_ERROR;
	}
	
	gpudata_release(ctx->srcStepsGD);
	gpudata_release(ctx->srcSizeGD);
	gpudata_release(ctx->chunkSizeGD);
	gpudata_release(ctx->dstMaxStepsGD);
	gpudata_release(ctx->dstArgmaxStepsGD);
	
	return ctx->ret;
}

/**
 * Cleanup
 */

static int   maxandargmaxCleanup                (maxandargmax_ctx*  ctx){
	free(ctx->axisList);
	free(ctx->sourceCode);
	ctx->axisList       = NULL;
	ctx->sourceCode     = NULL;
	
	return ctx->ret;
}

