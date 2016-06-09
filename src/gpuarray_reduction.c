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


/* Datatypes */
struct gen_kernel_ctx{
	int         numIdx;
	const int*  isReduced;
	int         numRedIdx;
	int         numFreeIdx;
	const char* dstMaxType;
	const char* dstArgmaxType;
};
typedef struct gen_kernel_ctx gen_kernel_ctx;



/* Function prototypes */
static int   getRdxIdx              (const int       numIdx,
                                     const int*      isReduced);
static char* genkernel_maxandargmax (const int       numIdx,
                                     const int*      isReduced,
                                     const char*     dstMaxType,
                                     const char*     dstArgmaxType);
static void  appendKernel           (strb*           s,
                                     gen_kernel_ctx* ctx);
static void  appendTypedefs         (strb*           s,
                                     gen_kernel_ctx* ctx);
static void  appendPrototype        (strb*           s,
                                     gen_kernel_ctx* ctx);
static void  appendIndexDeclarations(strb*           s,
                                     gen_kernel_ctx* ctx);
static void  appendIdxes            (strb*           s,
                                     const char*     prologue,
                                     const char*     prefix,
                                     int             startIdx,
                                     int             endIdx,
                                     const char*     suffix,
                                     const char*     epilogue);
static void  appendRangeCalculations(strb*           s,
                                     gen_kernel_ctx* ctx);
static void  appendLoops            (strb*           s,
                                     gen_kernel_ctx* ctx);
static void  appendLoopMacroDefs    (strb*           s,
                                     gen_kernel_ctx* ctx);
static void  appendLoopOuter        (strb*           s,
                                     gen_kernel_ctx* ctx);
static void  appendLoopInner        (strb*           s,
                                     gen_kernel_ctx* ctx);
static void  appendLoopMacroUndefs  (strb*           s,
                                     gen_kernel_ctx* ctx);
static void  scheduleMaxAndArgmax   (size_t*         blockSize,
                                     size_t*         gridSize,
                                     const GpuArray* src,
                                     const int*      isReduced);
static void  invokeMaxAndArgmax     (GpuKernel*      kernel,
                                     const GpuArray* src,
                                     const int*      isReduced);


/* Function implementation */

/**
 * @brief Computes simultaneously the maxima and the arguments of maxima over
 * specified axes of the tensor.
 *
 * Returns two tensors of identical shape. Both tensors' axes are a subset of
 * the axes of the original tensor. The axes to be reduced are specified by
 * the caller, and the maxima and arguments of maxima are computed over them.
 *
 * @param [out] dstMax     The resulting tensor of maxima
 * @param [out] dstArgmax  the resulting tensor of arguments at maxima
 * @param [in]  src        The source tensor.
 * @param [in]  isReduced  Either NULL, or an array of booleans of the same
 *                         size as the dimensionality of the source tensor.
 *                         Axis k is reduced if isReduced[k] is non-zero,
 *                         and is preserved otherwise.
 * @return GA_NO_ERROR if the operation was successful, or a non-zero error
 *         code otherwise.
 */

GPUARRAY_PUBLIC int GpuArray_maxandargmax(GpuArray*       dstMax,
                                          GpuArray*       dstArgmax,
                                          const GpuArray* src,
                                          const int*      isReduced){
	/**
	 * Generate kernel source code
	 */
	
	const char*  dstMaxType      = gpuarray_get_type(src->typecode) -> cluda_name;
	const char*  dstArgmaxType   = gpuarray_get_type(GA_SIZE)       -> cluda_name;
	const char*  s               = genkernel_maxandargmax(src->nd,
	                                                      isReduced,
	                                                      dstMaxType,
	                                                      dstArgmaxType);
	if(!s){return GA_MEMORY_ERROR;}
	
	/* Compile it */
	const int    ARG_TYPECODE[8] = {
		GA_BUFFER, /* src */
		GA_BUFFER, /* srcSteps */
		GA_BUFFER, /* srcSize */
		GA_BUFFER, /* numBlk */
		GA_BUFFER, /* dstMax */
		GA_BUFFER, /* dstMaxSteps */
		GA_BUFFER, /* dstArgmax */
		GA_BUFFER  /* dstArgmaxSteps */
	};
	const size_t l = strlen(s);
	GpuKernel kernel;
	GpuKernel_init(&kernel, 0, 1, &s, &l, "maxandargmax",
	               8, ARG_TYPECODE, 0, (char**)0);
	
	/* Invoke it. */
	invokeMaxAndArgmax(&kernel, src, isReduced);

	/* Return error code */
	return GA_NO_ERROR;
}

/**
 * Count the number of dimensions to be reduced.
 */

static int  getRdxIdx(const int numIdx, const int* isReduced){
	int i, countReduced;
	for(i=0, countReduced = 0;i<numIdx;i++){
		countReduced += !!isReduced[i];
	}
	return countReduced;
}

/**
 * @brief Generate the kernel code for MaxAndArgmax.
 * 
 * @param [in]  numIdx
 * @param [in]  isReduced
 * @param [in]  dstMaxType
 * @param [in]  dstArgmaxType
 * @return A free()'able string containing source code implementing the
 *         kernel, or else NULL.
 */

static char* genkernel_maxandargmax(const int       numIdx,
                                    const int*      isReduced,
                                    const char*     dstMaxType,
                                    const char*     dstArgmaxType){
	/* Obtain the parameters of the reduction. */
	gen_kernel_ctx ctx;
	ctx.numIdx        = numIdx;
	ctx.isReduced     = isReduced;
	ctx.numRedIdx     = getRdxIdx(ctx.numIdx, ctx.isReduced);
	ctx.numFreeIdx    = ctx.numIdx - ctx.numRedIdx;
	ctx.dstMaxType    = dstMaxType;
	ctx.dstArgmaxType = dstArgmaxType;
	
	strb s = STRB_STATIC_INIT;
	strb_ensure(&s, 5*1024);
	appendKernel(&s, &ctx);
	return strb_cstr(&s);
}

static void  appendKernel           (strb*           s,
                                     gen_kernel_ctx* ctx){
	appendTypedefs         (s, ctx);
	appendPrototype        (s, ctx);
	strb_appends           (s, "{\n");
	appendIndexDeclarations(s, ctx);
	appendRangeCalculations(s, ctx);
	appendLoops            (s, ctx);
	strb_appends           (s, "}\n");
}
static void  appendTypedefs         (strb*           s,
                                     gen_kernel_ctx* ctx){
	strb_appends(s, "/* Typedefs */\n");
	strb_appendf(s, "typedef %s     T;/* The type of the array being processed. */\n", ctx->dstMaxType);
	strb_appendf(s, "typedef %s     X;/* Index type: signed 32/64-bit. */\n",          ctx->dstArgmaxType);
	strb_appends(s, "\n");
	strb_appends(s, "\n");
	strb_appends(s, "\n");
}
static void  appendPrototype        (strb*           s,
                                     gen_kernel_ctx* ctx){
	strb_appends(s, "KERNEL void maxandargmax(const T*        src,\n");
	strb_appends(s, "                         const X*        srcSteps,\n");
	strb_appends(s, "                         const X*        srcSize,\n");
	strb_appends(s, "                         const X*        blkNum,\n");
	strb_appends(s, "                         T*              dstMax,\n");
	strb_appends(s, "                         const X*        dstMaxSteps,\n");
	strb_appends(s, "                         X*              dstArgmax,\n");
	strb_appends(s, "                         const X*        dstArgmaxSteps)");
}
static void  appendIndexDeclarations(strb*           s,
                                     gen_kernel_ctx* ctx){
	strb_appends(s, "\t/* GPU kernel coordinates. Always 3D. */\n");
	
	strb_appends(s, "\tX bi0 = GID_0,   bi1 = GID_1,   bi2 = GID_2;\n");
	strb_appends(s, "\tX bd0 = LDIM_0,  bd1 = LDIM_1,  bd2 = LDIM_2;\n");
	strb_appends(s, "\tX ti0 = LID_0,   ti1 = LID_1,   ti2 = LID_2;\n");
	
	strb_appends(s, "\t\n");
	strb_appends(s, "\t\n");
	strb_appends(s, "\t/* Free indices & Reduction indices */\n");
	
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "Blk",   ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "",      ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "Dim",   ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "Start", ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "End",   ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numIdx,     "SStep", ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numFreeIdx, "MStep", ";\n");
	appendIdxes (s, "\tX ", "i", 0,               ctx->numFreeIdx, "AStep", ";\n");
	appendIdxes (s, "\tX ", "i", ctx->numFreeIdx, ctx->numIdx,     "PDim",  ";\n");
	
	strb_appends(s, "\t\n");
	strb_appends(s, "\t\n");
}
static void  appendIdxes            (strb*           s,
                                     const char*     prologue,
                                     const char*     prefix,
                                     int             startIdx,
                                     int             endIdx,
                                     const char*     suffix,
                                     const char*     epilogue){
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
static void  appendRangeCalculations(strb*           s,
                                     gen_kernel_ctx* ctx){
	int i;
	
	strb_appends(s, "\t/* Compute ranges for this thread. */\n");
	
	for(i=0;i<ctx->numIdx    ;i++){/* i*Dim   = srcSize[*]; */
		strb_appendf(s, "\ti%dDim   = srcSize[%d];\n", i, i);
	}
	for(i=0;i<ctx->numIdx    ;i++){/* i*SStep = srcSteps[*]; */
		strb_appendf(s, "\ti%dSStep = srcSteps[%d];\n", i, i);
	}
	for(i=0;i<ctx->numFreeIdx;i++){/* i*MStep = dstMaxSteps[*]; */
		strb_appendf(s, "\ti%dMStep = dstMaxSteps[%d];\n", i, i);
	}
	for(i=0;i<ctx->numFreeIdx;i++){/* i*AStep = dstArgmaxSteps[*]; */
		strb_appendf(s, "\ti%dMStep = dstArgmaxSteps[%d];\n", i, i);
	}
	for(i=0;i<ctx->numIdx    ;i++){/* i*Blk   = numBlk[*]; */
		strb_appendf(s, "\ti%dBlk   = numBlk[%d];\n", i, i);
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
			strb_appendf(s, "\ti%dStart = ((bi%d * bd%d) + ti%d) * i%dBlk;\n", i, i, i, i, i);
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
			strb_appendf(s, "\ti%dEnd   = i%dStart + bd%d * i%dBlk;\n", i, i, i, i);
		}else{
			strb_appendf(s, "\ti%dEnd   = i%dStart + i%dDim;\n", i, i, i);
		}
	}
	
	strb_appends(s, "\t\n");
	strb_appends(s, "\t\n");
}
static void  appendLoops            (strb*           s,
                                     gen_kernel_ctx* ctx){
	strb_appends(s, "\t/**\n");
	strb_appends(s, "\t * FREE LOOPS.\n");
	strb_appends(s, "\t */\n");
	strb_appends(s, "\t\n");
	
	appendLoopMacroDefs  (s, ctx);
	appendLoopOuter      (s, ctx);
	appendLoopMacroUndefs(s, ctx);
}
static void  appendLoopMacroDefs    (strb*           s,
                                     gen_kernel_ctx* ctx){
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
static void  appendLoopOuter        (strb*           s,
                                     gen_kernel_ctx* ctx){
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
static void  appendLoopInner        (strb*           s,
                                     gen_kernel_ctx* ctx){
	int i;
	
	/**
	 * Inner Loop Prologue
	 */
	
	strb_appends(s, "\t/**\n");
	strb_appends(s, "\t * Reduction initialization.\n");
	strb_appends(s, "\t */\n");
	strb_appends(s, "\t\n");
	
	appendIdxes (s, "\tT maxV = SRCINDEXER(", "i", 0, ctx->numFreeIdx, "", "");
	if(ctx->numFreeIdx && ctx->numRedIdx){strb_appends(s, ",");}
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
static void  appendLoopMacroUndefs  (strb*           s,
                                     gen_kernel_ctx* ctx){
	strb_appends(s, "\t#undef FOROVER\n");
	strb_appends(s, "\t#undef ESCAPE\n");
	strb_appends(s, "\t#undef SRCINDEXER\n");
	strb_appends(s, "\t#undef RDXINDEXER\n");
	strb_appends(s, "\t#undef DSTMINDEXER\n");
	strb_appends(s, "\t#undef DSTAINDEXER\n");
}


/**
 * FIXME: Implement a working scheduler and invoker.
 * 
 * To schedule effectively the work across several dimensions of possibly
 * ugly numbers, taking into account the limitations on thread & block
 * scheduling, will require some library capable of providing a "factoring"
 * of the tensor dimensions into "nice" prime numbers. Their product may
 * be allowed to be larger than the original number, provided it remains
 * within bounds.
 */

/**
 * Compute a good thread block size / grid size for Nvidia.
 */

static void  scheduleMaxAndArgmax   (size_t*         blockSize,
                                     size_t*         gridSize,
                                     const GpuArray* src,
                                     const int*      isReduced){
	//int maxThreadPerBlock = 1024;
	//int numFreeIdx = src->nd - getRdxIdx(src->nd, isReduced);
	
	/* Naive solution. Optimization is a tough problem. */
	blockSize[0] = blockSize[1] = blockSize[2] = 1;
	gridSize [0] = gridSize [1] = gridSize [2] = 1;
}

/**
 * Invoke the kernel.
 */

static void  invokeMaxAndArgmax     (GpuKernel*      kernel,
                                     const GpuArray* src,
                                     const int*      isReduced){
	size_t blockSize[3];
	size_t gridSize[3];
	
	scheduleMaxAndArgmax(blockSize, gridSize, src, isReduced);
	GpuKernel_call(kernel,
	               getRdxIdx(src->nd, isReduced),
	               blockSize,
	               gridSize,
	               0,
	               NULL);
}

