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
struct gen_kernel_ctx{
	unsigned        numIdx;
	unsigned        reduxLen;
	const unsigned* reduxList;
	unsigned        numFreeIdx;
	unsigned*       axisList;
	const char*     dstMaxType;
	const char*     dstArgmaxType;
};
typedef struct gen_kernel_ctx gen_kernel_ctx;



/* Function prototypes */
static int   checkargsMaxAndArgmax  (GpuArray*        dstMax,
                                     GpuArray*        dstArgmax,
                                     const GpuArray*  src,
                                     unsigned         reduxLen,
                                     const unsigned*  reduxList);
static char* genkernelMaxAndArgmax  (unsigned         numIdx,
                                     unsigned         reduxLen,
                                     const unsigned*  reduxList,
                                     const char*      dstMaxType,
                                     const char*      dstArgmaxType);
static void  appendKernel           (strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  appendTypedefs         (strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  appendPrototype        (strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  appendOffsets          (strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  appendIndexDeclarations(strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  appendIdxes            (strb*            s,
                                     const char*      prologue,
                                     const char*      prefix,
                                     int              startIdx,
                                     int              endIdx,
                                     const char*      suffix,
                                     const char*      epilogue);
static void  appendRangeCalculations(strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  appendLoops            (strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  appendLoopMacroDefs    (strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  appendLoopOuter        (strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  appendLoopInner        (strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  appendLoopMacroUndefs  (strb*            s,
                                     gen_kernel_ctx*  ctx);
static void  computeAxisList        (unsigned*        axisList,
                                     unsigned         numAxis,
                                     const unsigned*  reduxList,
                                     unsigned         reduxLen);
static void  scheduleMaxAndArgmax   (const GpuKernel* kernel,
                                     const GpuArray*  src,
                                     unsigned         reduxLen,
                                     const unsigned*  reduxList,
                                     size_t*          blockSize,
                                     size_t*          gridSize,
                                     size_t*          chunkSize);
static int   invokeMaxAndArgmax     (GpuKernel*       kernel,
                                     GpuArray*        dstMax,
                                     GpuArray*        dstArgmax,
                                     const GpuArray*  src,
                                     unsigned         reduxLen,
                                     const unsigned*  reduxList);


/* Function implementation */
GPUARRAY_PUBLIC int GpuArray_maxandargmax(GpuArray*       dstMax,
                                          GpuArray*       dstArgmax,
                                          const GpuArray* src,
                                          unsigned        reduxLen,
                                          const unsigned* reduxList){
	/**
	 * Sanity check on arguments
	 */
	
	if(!checkargsMaxAndArgmax(dstMax, dstArgmax, src, reduxLen, reduxList)){
		return GA_INVALID_ERROR;
	}
	
	
	/**
	 * Generate kernel source code.
	 */
	
	int          ret;
	const char*  dstMaxType      = gpuarray_get_type(src->typecode) -> cluda_name;
	const char*  dstArgmaxType   = gpuarray_get_type(GA_SIZE)       -> cluda_name;
	char*        s               = genkernelMaxAndArgmax(src->nd,
	                                                     reduxLen,
	                                                     reduxList,
	                                                     dstMaxType,
	                                                     dstArgmaxType);
	if(!s){return GA_MEMORY_ERROR;}
	
	
	/**
	 * Compile it.
	 */
	
	const int    ARG_TYPECODE[11] = {
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
	
	GpuKernel kernel;
	const size_t l = strlen(s);
	ret = GpuKernel_init(&kernel, gpudata_context(src->data),
	                     1, (const char**)&s, &l, "maxandargmax",
	                     11, ARG_TYPECODE, GA_USE_CLUDA, (char**)0);
	free(s);
	if(ret != GA_NO_ERROR){
		return ret;
	}
	
	
	/**
	 * Invoke it.
	 */
	
	return invokeMaxAndArgmax(&kernel, dstMax, dstArgmax, src, reduxLen, reduxList);
}

/**
 * @brief Check the sanity of the arguments, in agreement with the
 *        documentation for GpuArray_maxandargmax().
 * 
 * @param [in] dstMax
 * @param [in] dstArgmax
 * @param [in] src
 * @param [in] reduxLen
 * @param [in] reduxList
 * @return Zero if arguments invalid; Non-zero otherwise.
 */

static int   checkargsMaxAndArgmax  (GpuArray*         dstMax,
                                     GpuArray*         dstArgmax,
                                     const GpuArray*   src,
                                     unsigned          reduxLen,
                                     const unsigned*   reduxList){
	int i, j;
	
	/* Insane src or reduxLen? */
	if(!dstMax || !dstArgmax || !src || src->nd == 0 || reduxLen == 0 ||
	   reduxLen >= src->nd){
		return 0;
	}
	
	for(i=0;i<(int)reduxLen;i++){
		/* Insane list entry? */
		if(reduxList[i] >= src->nd){
			return 0;
		}
		
		for(j=i-1;j>=0;j--){
			/* Duplicate list entry? */
			if(reduxList[i] == reduxList[j]){
				return 0;
			}
		}
	}
	
	return 1;
}

/**
 * @brief Generate the kernel code for MaxAndArgmax.
 * 
 * @param [in]  numIdx
 * @param [in]  reduxLen
 * @param [in]  reduxList
 * @param [in]  dstMaxType
 * @param [in]  dstArgmaxType
 * @return A free()'able string containing source code implementing the
 *         kernel, or else NULL.
 */

static char* genkernelMaxAndArgmax (unsigned          numIdx,
                                    const unsigned    reduxLen,
                                    const unsigned*   reduxList,
                                    const char*       dstMaxType,
                                    const char*       dstArgmaxType){
	/* Save the parameters of the reduction in a generator context. */
	gen_kernel_ctx ctx;
	ctx.numIdx        = numIdx;
	ctx.reduxLen      = reduxLen;
	ctx.reduxList     = reduxList;
	ctx.numFreeIdx    = ctx.numIdx - ctx.reduxLen;
	ctx.axisList      = malloc(numIdx*sizeof(unsigned));
	ctx.dstMaxType    = dstMaxType;
	ctx.dstArgmaxType = dstArgmaxType;
	if(!ctx.axisList){
		return NULL;
	}
	
	/* Compute internal axis remapping. */
	computeAxisList(ctx.axisList, ctx.numIdx, ctx.reduxList, ctx.reduxLen);
	
	/* Generate kernel proper. */
	strb s = STRB_STATIC_INIT;
	strb_ensure(&s, 5*1024);
	appendKernel(&s, &ctx);
	free(ctx.axisList);
	
	/* Return it. */
	return strb_cstr(&s);
}
static void  appendKernel           (strb*            s,
                                     gen_kernel_ctx*  ctx){
	appendTypedefs         (s, ctx);
	appendPrototype        (s, ctx);
	strb_appends           (s, "{\n");
	appendOffsets          (s, ctx);
	appendIndexDeclarations(s, ctx);
	appendRangeCalculations(s, ctx);
	appendLoops            (s, ctx);
	strb_appends           (s, "}\n");
}
static void  appendTypedefs         (strb*            s,
                                     gen_kernel_ctx*  ctx){
	strb_appends(s, "/* Typedefs */\n");
	strb_appendf(s, "typedef %s     T;/* The type of the array being processed. */\n", ctx->dstMaxType);
	strb_appendf(s, "typedef %s     X;/* Index type: signed 32/64-bit. */\n",          ctx->dstArgmaxType);
	strb_appends(s, "\n");
	strb_appends(s, "\n");
	strb_appends(s, "\n");
}
static void  appendPrototype        (strb*            s,
                                     gen_kernel_ctx*  ctx){
	strb_appends(s, "KERNEL void maxandargmax(const T*        src,\n");
	strb_appends(s, "                         const X         srcOff,\n");
	strb_appends(s, "                         const X*        srcSteps,\n");
	strb_appends(s, "                         const X*        srcSize,\n");
	strb_appends(s, "                         const X*        chunkSize,\n");
	strb_appends(s, "                         T*              dstMax,\n");
	strb_appends(s, "                         const X         dstMaxOff,\n");
	strb_appends(s, "                         const X*        dstMaxSteps,\n");
	strb_appends(s, "                         X*              dstArgmax,\n");
	strb_appends(s, "                         const X         dstArgmaxOff,\n");
	strb_appends(s, "                         const X*        dstArgmaxSteps)");
}
static void  appendOffsets          (strb*            s,
                                     gen_kernel_ctx*  ctx){
	strb_appends(s, "/* Add offsets */\n");
	strb_appends(s, "src       = (const GLOBAL_MEM T*)((const GLOBAL_MEM char*)src       + srcOff);\n");
	strb_appends(s, "dstMax    = (GLOBAL_MEM T*)      ((GLOBAL_MEM char*)      dstMax    + dstMaxOff);\n");
	strb_appends(s, "dstArgmax = (GLOBAL_MEM X*)      ((GLOBAL_MEM char*)      dstArgmax + dstArgmaxOff);\n");
	strb_appends(s, "\n");
	strb_appends(s, "\n");
}
static void  appendIndexDeclarations(strb*            s,
                                     gen_kernel_ctx*  ctx){
	strb_appends(s, "\t/* GPU kernel coordinates. Always 3D. */\n");
	
	strb_appends(s, "\tX bi0 = GID_0,       bi1 = GID_1,       bi2 = GID_2;\n");
	strb_appends(s, "\tX bd0 = LDIM_0,      bd1 = LDIM_1,      bd2 = LDIM_2;\n");
	strb_appends(s, "\tX ti0 = LID_0,       ti1 = LID_1,       ti2 = LID_2;\n");
	strb_appends(s, "\tX gi0 = bi0*bd0+ti0, gi1 = bi1*bd1+ti1, gi2 = bi2*bd2+ti2;\n");
	
	strb_appends(s, "\t\n");
	strb_appends(s, "\t\n");
	strb_appends(s, "\t/* Free indices & Reduction indices */\n");
	
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "ChunkSz", ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "",        ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "Dim",     ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "Start",   ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "End",     ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "SStep",   ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numFreeIdx, "MStep",   ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numFreeIdx, "AStep",   ";\n");
	appendIdxes (s, "\tX ", "i", ctx->numFreeIdx, ctx->numIdx,     "PDim",    ";\n");
	
	strb_appends(s, "\t\n");
	strb_appends(s, "\t\n");
}
static void  appendIdxes            (strb*            s,
                                     const char*      prologue,
                                     const char*      prefix,
                                     int              startIdx,
                                     int              endIdx,
                                     const char*      suffix,
                                     const char*      epilogue){
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
static void  appendRangeCalculations(strb*            s,
                                     gen_kernel_ctx*  ctx){
	int i;
	
	/* Use internal remapping when computing the ranges for this thread. */
	strb_appends(s, "\t/* Compute ranges for this thread. */\n");
	
	for(i=0;i<ctx->numIdx    ;i++){/* i*Dim   = srcSize[*]; */
		strb_appendf(s, "\ti%dDim     = srcSize[%d];\n", i, ctx->axisList[i]);
	}
	for(i=0;i<ctx->numIdx    ;i++){/* i*SStep = srcSteps[*]; */
		strb_appendf(s, "\ti%dSStep   = srcSteps[%d];\n", i, ctx->axisList[i]);
	}
	for(i=0;i<ctx->numFreeIdx;i++){/* i*MStep = dstMaxSteps[*]; */
		strb_appendf(s, "\ti%dMStep   = dstMaxSteps[%d];\n", i, i);
	}
	for(i=0;i<ctx->numFreeIdx;i++){/* i*AStep = dstArgmaxSteps[*]; */
		strb_appendf(s, "\ti%dMStep   = dstArgmaxSteps[%d];\n", i, i);
	}
	for(i=0;i<ctx->numIdx    ;i++){/* i*ChunkSz = numBlk[*]; */
		strb_appendf(s, "\ti%dChunkSz = chunkSize[%d];\n", i, ctx->axisList[i]);
	}
	for(i=ctx->numIdx-1;i>=ctx->numFreeIdx;i--){/* i*PDim  = ...; */
		/**
		 * If this is the last index, it's the first cumulative dimension
		 * product we generate, and thus we initialize to 1.
		 */
		
		if(i == ctx->numIdx-1){
			strb_appendf(s, "\ti%dPDim  = 1;\n", i);
		}else{
			strb_appendf(s, "\ti%dPDim  = i%dPDim * i%dDim;\n", i, i+1, i);
		}
	}
	for(i=0;i<ctx->numIdx    ;i++){/* i*Start = ...; */
		/**
		 * The first 3 dimensions get to rely on hardware loops.
		 * The others, if any, have to use software looping beginning at 0.
		 */
		
		if(i < 3){
			strb_appendf(s, "\ti%dStart = gi%d * i%dChunkSz;\n", i, i, i);
		}else{
			strb_appendf(s, "\ti%dStart = 0;\n", i);
		}
	}
	for(i=0;i<ctx->numIdx    ;i++){/* i*End   = ...; */
		/**
		 * The first 3 dimensions get to rely on hardware loops.
		 * The others, if any, have to use software looping beginning at 0.
		 */
		
		if(i < 3){
			strb_appendf(s, "\ti%dEnd   = i%dStart + bd%d * i%dChunkSz;\n", i, i, i, i);
		}else{
			strb_appendf(s, "\ti%dEnd   = i%dStart + i%dDim;\n", i, i, i);
		}
	}
	
	strb_appends(s, "\t\n");
	strb_appends(s, "\t\n");
}
static void  appendLoops            (strb*            s,
                                     gen_kernel_ctx*  ctx){
	strb_appends(s, "\t/**\n");
	strb_appends(s, "\t * FREE LOOPS.\n");
	strb_appends(s, "\t */\n");
	strb_appends(s, "\t\n");
	
	appendLoopMacroDefs  (s, ctx);
	appendLoopOuter      (s, ctx);
	appendLoopMacroUndefs(s, ctx);
}
static void  appendLoopMacroDefs    (strb*            s,
                                     gen_kernel_ctx*  ctx){
	int i;
	
	/**
	 * FOROVER Macro
	 */
	
	strb_appends(s, "#define FOROVER(idx)    for(i##idx = i##idx##Start; i##idx < i##idx##End; i##idx++)\n");
	
	/**
	 * ESCAPE Macro
	 */
	
	strb_appends(s, "#define ESCAPE(idx)     if(i##idx >= i##idx##Dim){continue;}\n");
	
	/**
	 * SRCINDEXER Macro
	 */
	
	appendIdxes (s, "#define SRCINDEXER(", "i", 0, ctx->numIdx, "", ")   src[");
	for(i=0;i<ctx->numIdx;i++){
		strb_appendf(s, "i%d*i%dSStep + \\\n                                            ", i, i);
	}
	strb_appends(s, "0]\n");
	
	/**
	 * RDXINDEXER Macro
	 */
	
	appendIdxes (s, "#define RDXINDEXER(", "i", ctx->numFreeIdx, ctx->numIdx, "", ")              (");
	for(i=ctx->numFreeIdx;i<ctx->numIdx;i++){
		strb_appendf(s, "i%d*i%dPDim + \\\n                                        ", i, i);
	}
	strb_appends(s, "0)\n");
	
	/**
	 * DSTMINDEXER Macro
	 */
	
	appendIdxes (s, "#define DSTMINDEXER(", "i", 0, ctx->numFreeIdx, "", ")        dstMax[");
	for(i=0;i<ctx->numFreeIdx;i++){
		strb_appendf(s, "i%d*i%dMStep + \\\n                                                  ", i, i);
	}
	strb_appends(s, "0]\n");
	
	/**
	 * DSTAINDEXER Macro
	 */
	
	appendIdxes (s, "#define DSTAINDEXER(", "i", 0, ctx->numFreeIdx, "", ")        dstArgmax[");
	for(i=0;i<ctx->numFreeIdx;i++){
		strb_appendf(s, "i%d*i%dAStep + \\\n                                                     ", i, i);
	}
	strb_appends(s, "0]\n");
}
static void  appendLoopOuter        (strb*            s,
                                     gen_kernel_ctx*  ctx){
	int i;
	
	/**
	 * Outer Loop Header Generation
	 */
	
	for(i=0;i<ctx->numFreeIdx;i++){
		strb_appendf(s, "\tFOROVER(%d){ESCAPE(%d)\n", i, i);
	}
	
	/**
	 * Inner Loop Generation
	 */
	
	appendLoopInner(s, ctx);
	
	/**
	 * Outer Loop Trailer Generation
	 */
	
	for(i=0;i<ctx->numFreeIdx;i++){
		strb_appends(s, "\t}\n");
	}
}
static void  appendLoopInner        (strb*            s,
                                     gen_kernel_ctx*  ctx){
	int i;
	
	/**
	 * Inner Loop Prologue
	 */
	
	strb_appends(s, "\t/**\n");
	strb_appends(s, "\t * Reduction initialization.\n");
	strb_appends(s, "\t */\n");
	strb_appends(s, "\t\n");
	
	appendIdxes (s, "\tT maxV = SRCINDEXER(", "i", 0, ctx->numFreeIdx, "", "");
	if(ctx->numFreeIdx && ctx->reduxLen){strb_appends(s, ",");}
	appendIdxes (s, "", "i", ctx->numFreeIdx, ctx->numIdx, "Start", ");\n");
	
	appendIdxes (s, "\tX maxI = RDXINDEXER(", "i", ctx->numFreeIdx, ctx->numIdx, "Start", ");\n");
	
	strb_appends(s, "\t\n");
	strb_appends(s, "\t/**\n");
	strb_appends(s, "\t * REDUCTION LOOPS.\n");
	strb_appends(s, "\t */\n");
	strb_appends(s, "\t\n");
	
	/**
	 * Inner Loop Header Generation
	 */
	
	for(i=ctx->numFreeIdx;i<ctx->numIdx;i++){
		strb_appendf(s, "\tFOROVER(%d){ESCAPE(%d)\n", i, i);
	}
	
	/**
	 * Inner Loop Body Generation
	 */
	
	appendIdxes (s, "\tT V = SRCINDEXER(", "i", 0, ctx->numIdx, "", ");\n");
	strb_appends(s, "\t\n");
	strb_appends(s, "\tif(V > maxV){\n");
	strb_appends(s, "\t\tmaxV = V;\n");
	appendIdxes (s, "\t\tmaxI = RDXINDEXER(", "i", ctx->numFreeIdx, ctx->numIdx, "", ");\n");
	strb_appends(s, "\t}\n");
	
	/**
	 * Inner Loop Trailer Generation
	 */
	
	for(i=ctx->numFreeIdx;i<ctx->numIdx;i++){
		strb_appends(s, "\t}\n");
	}
	strb_appends(s, "\t\n");
	
	/**
	 * Inner Loop Epilogue Generation
	 */
	
	strb_appends(s, "\t/**\n");
	strb_appends(s, "\t * Destination writeback.\n");
	strb_appends(s, "\t */\n");
	strb_appends(s, "\t\n");
	appendIdxes (s, "\tDSTMINDEXER(", "i", 0, ctx->numFreeIdx, "", ") = maxV;\n");
	appendIdxes (s, "\tDSTAINDEXER(", "i", 0, ctx->numFreeIdx, "", ") = maxI;\n");
}
static void  appendLoopMacroUndefs  (strb*            s,
                                     gen_kernel_ctx*  ctx){
	strb_appends(s, "\t#undef FOROVER\n");
	strb_appends(s, "\t#undef ESCAPE\n");
	strb_appends(s, "\t#undef SRCINDEXER\n");
	strb_appends(s, "\t#undef RDXINDEXER\n");
	strb_appends(s, "\t#undef DSTMINDEXER\n");
	strb_appends(s, "\t#undef DSTAINDEXER\n");
}
static void  computeAxisList        (unsigned*        axisList,
                                     unsigned         numAxis,
                                     const unsigned*  reduxList,
                                     unsigned         reduxLen){
	unsigned i, j, f=0, r=numAxis-reduxLen;
	
	for(i=0;i<numAxis;i++){
		/**
		 * If axis i is in reduxList, add it to the end.
		 */
		
		for(j=0;j<reduxLen;j++){
			if(i==reduxList[j]){
				axisList[r++] = i;
				goto /* continue */ outerLoop;
			}
		}
		
		/**
		 * Otherwise, axis i is not in reduxList. Add it to the beginning.
		 */
		
		axisList[f++] = i;
		outerLoop:;
	}
}

/**
 * Compute a good thread block size / grid size / software chunk size for Nvidia.
 */

static void  scheduleMaxAndArgmax   (const GpuKernel* kernel,
                                     const GpuArray*  src,
                                     unsigned         reduxLen,
                                     const unsigned*  reduxList,
                                     size_t*          blockSize,
                                     size_t*          gridSize,
                                     size_t*          chunkSize){
	int i, j;
	
	/* Obtain the constraints of our problem. */
	size_t warpSize,
	       maxL, maxL0, maxL1, maxL2,  /* Maximum total and per-dimension thread/block sizes */
	       maxG, maxG0, maxG1, maxG2;  /* Maximum total and per-dimension block /grid  sizes */
	gpukernel_property(kernel->k, GA_KERNEL_PROP_PREFLSIZE, &warpSize);
	gpukernel_property(kernel->k, GA_KERNEL_PROP_MAXLSIZE,  &maxL);
	gpudata_property  (src->data, GA_CTX_PROP_MAXLSIZE0,    &maxL0);
	gpudata_property  (src->data, GA_CTX_PROP_MAXLSIZE1,    &maxL1);
	gpudata_property  (src->data, GA_CTX_PROP_MAXLSIZE2,    &maxL2);
	gpudata_property  (src->data, GA_CTX_PROP_MAXGSIZE,     &maxG);
	gpudata_property  (src->data, GA_CTX_PROP_MAXGSIZE0,    &maxG0);
	gpudata_property  (src->data, GA_CTX_PROP_MAXGSIZE1,    &maxG1);
	gpudata_property  (src->data, GA_CTX_PROP_MAXGSIZE2,    &maxG2);
	
	int numRdxIdx  = reduxLen;
	int numFreeIdx = src->nd - numRdxIdx;
	(void)numFreeIdx;
	
	/**
	 * Select which reduction dimensions will be associated with which hardware
	 * x, y and z dimensions.
	 */
	
	int            dims    [3];
	uint64_t       dimSize [3] = {  1,   1,   1};
	double         slack   [3] = {1.1, 1.1, 1.1};
	uint64_t       kSmooth [3];
	ga_factor_list factDims[3];
	ga_factor_list factTBS [3];
	uint64_t       tBS         =   1;
	uint64_t       minThrd     =  64;
	uint64_t       maxThrd     = 256;
	(void)dims;
	
	/************************************************************************
	 * FIXME: Need logic to select up to 3 dimensions and plug them in dimSize!
	 *        But what's the best dimension selection strategy to maximize
	 *        memory bandwidth?
	 *        Also need to fill out kSmooth[] based on all the GPU properties.
	 ************************************************************************/
	kSmooth[0] = maxL0;
	kSmooth[1] = maxL1;
	kSmooth[2] = maxL2;
	
	/**
	 * Factorization job. We'll steadily increase the slack in case of failure
	 * in order to ensure we do get a factorization.
	 */
	
	for(i=0;i<numRdxIdx;i++){
		while(!gaIFactorize(dimSize[i],
		                    dimSize[i]*slack[i],
		                    kSmooth[i],
		                    &factDims[i])){
			/**
			 * Error! Failed to factorize dimension "xyz"[i] with given slack
			 * and k-smoothness constraints! Increase slack. Once slack reaches
			 * 2.0 it will factorize guaranteed.
			 */
			
			slack[i] += 0.1;
		}
	}
	
	/**
	 * Use the factorization. We "withdraw" factors from the factor lists one
	 * at a time until we enter our target zone thread#. If the individual
	 * maxLn in dimension n is about to be breached, we move on to the next
	 * dimension.
	 * 
	 * The same process is then repeated with respect to grid size.
	 * 
	 * What's left after that is software blocking.
	 */
	
	gaIFLInit(&factTBS[0]);
	gaIFLInit(&factTBS[1]);
	gaIFLInit(&factTBS[2]);
	for(i=0;i<numRdxIdx;i++){
		for(j=0;j<15;j++){
			if(factDims[i].p[j] > 0){
				factDims[i].p[j]--;
				gaIFLAddFactors(&factTBS[i], factDims[i].f[j], 1);
				tBS *= factDims[i].f[j];
				
				if(tBS >= minThrd && tBS <= maxThrd){
					goto computeBS;
				}
			}
		}
	}
	
	computeBS:
	blockSize[0] = gaIFLGetProduct(&factTBS[0]);
	blockSize[1] = gaIFLGetProduct(&factTBS[1]);
	blockSize[2] = gaIFLGetProduct(&factTBS[2]);
	gridSize [0] = gaIFLGetProduct(&factDims[0]) / blockSize[0];
	gridSize [1] = gaIFLGetProduct(&factDims[1]) / blockSize[1];
	gridSize [2] = gaIFLGetProduct(&factDims[2]) / blockSize[2];
}

/**
 * Invoke the kernel.
 */

static int   invokeMaxAndArgmax     (GpuKernel*       k,
                                     GpuArray*        dstMax,
                                     GpuArray*        dstArgmax,
                                     const GpuArray*  src,
                                     unsigned         reduxLen,
                                     const unsigned*  reduxList){
	int         ret = 0;
	size_t      blockSize[3]   = {1,1,1};
	size_t      gridSize [3]   = {1,1,1};
	size_t      chunkSize[3]   = {1,1,1};
	gpudata*    srcStepsGD     = 0, *srcSizeGD        = 0, *chunkSizeGD = 0,
	            *dstMaxStepsGD = 0, *dstArgmaxStepsGD = 0;
	gpucontext* ctx            = GpuArray_context(src);
	
	
	/**
	 * Schedule the kernel.
	 * 
	 * This implies choosing the block, grid and chunk size appropriately.
	 */
	
	scheduleMaxAndArgmax(k, src, reduxLen, reduxList,
	                     blockSize, gridSize, chunkSize);
	
	
	/**
	 * Argument Marshalling. This the grossest gross thing in here.
	 */
	
	srcStepsGD       = gpudata_alloc(ctx, src->nd * sizeof(size_t),
	                                 src->strides,
	                                 GA_BUFFER_READ_ONLY|GA_BUFFER_INIT, &ret);
	if(ret){goto releaseGpudata;}
	srcSizeGD        = gpudata_alloc(ctx, src->nd * sizeof(size_t),
	                                 src->dimensions,
	                                 GA_BUFFER_READ_ONLY|GA_BUFFER_INIT, &ret);
	if(ret){goto releaseGpudata;}
	chunkSizeGD      = gpudata_alloc(ctx, src->nd * sizeof(size_t),
	                                 chunkSize,
	                                 GA_BUFFER_READ_ONLY|GA_BUFFER_INIT, &ret);
	if(ret){goto releaseGpudata;}
	dstMaxStepsGD    = gpudata_alloc(ctx, (src->nd - reduxLen) * sizeof(size_t),
	                                 dstMax->strides,
	                                 GA_BUFFER_READ_ONLY|GA_BUFFER_INIT, &ret);
	if(ret){goto releaseGpudata;}
	dstArgmaxStepsGD = gpudata_alloc(ctx, (src->nd - reduxLen) * sizeof(size_t),
	                                 dstArgmax->strides,
	                                 GA_BUFFER_READ_ONLY|GA_BUFFER_INIT, &ret);
	if(ret){goto releaseGpudata;}
	
	
	struct MaxAndArgmaxArgs{
		gpudata*  src;
		size_t    srcOff;
		gpudata*  srcSteps;
		gpudata*  srcSize;
		gpudata*  chunkSize;
		gpudata*  dstMax;
		size_t    dstMaxOff;
		gpudata*  dstMaxSteps;
		gpudata*  dstArgmax;
		size_t    dstArgmaxOff;
		gpudata*  dstArgmaxSteps;
	} argstr = {
		src->data,
		src->offset,
		srcStepsGD,
		srcSizeGD,
		chunkSizeGD,
		dstMax->data,
		dstMax->offset,
		dstMaxStepsGD,
		dstArgmax->data,
		dstArgmax->offset,
		dstArgmaxStepsGD
	};
	
	void* args[] = {
		(void*)&argstr.src,
		(void*)&argstr.srcOff,
		(void*)&argstr.srcSteps,
		(void*)&argstr.srcSize,
		(void*)&argstr.chunkSize,
		(void*)&argstr.dstMax,
		(void*)&argstr.dstMaxOff,
		(void*)&argstr.dstMaxSteps,
		(void*)&argstr.dstArgmax,
		(void*)&argstr.dstArgmaxOff,
		(void*)&argstr.dstArgmaxSteps
	};
	
	
	/**
	 * Call kernel, release arguments and return error code
	 */
	
	ret = GpuKernel_call(k, 3, blockSize, gridSize, 0, args);
	releaseGpudata:
	gpudata_release(srcStepsGD);
	gpudata_release(srcSizeGD);
	gpudata_release(chunkSizeGD);
	gpudata_release(dstMaxStepsGD);
	gpudata_release(dstArgmaxStepsGD);
	return ret;
}

