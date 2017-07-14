#ifndef GPUARRAY_REDUCTION_H
#define GPUARRAY_REDUCTION_H
/**
 * \file reduction.h
 * \brief Reduction functions.
 */

#include <gpuarray/array.h>

#ifdef _MSC_VER
#ifndef inline
#define inline __inline
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif


/* Data Structures */
struct         GpuReductionAttr;
struct         GpuReduction;
typedef struct GpuReductionAttr GpuReductionAttr;
typedef struct GpuReduction     GpuReduction;


/**
 * Supported array reduction operations.
 */

typedef enum _ga_reduce_op {
	                              /*     d0   ,    d1    */
	GA_ELEMWISE,
	GA_REDUCE_COPY=GA_ELEMWISE,   /*   (copy)            */
	GA_REDUCE_SUM,                /*     +               */
	GA_REDUCE_PROD,               /*     *               */
	GA_REDUCE_PRODNZ,             /*     * (!=0)         */
	GA_REDUCE_MIN,                /*   min()             */
	GA_REDUCE_MAX,                /*   max()             */
	GA_REDUCE_ARGMIN,             /*            argmin() */
	GA_REDUCE_ARGMAX,             /*            argmax() */
	GA_REDUCE_MINANDARGMIN,       /*   min()  , argmin() */
	GA_REDUCE_MAXANDARGMAX,       /*   max()  , argmax() */
	GA_REDUCE_AND,                /*     &               */
	GA_REDUCE_OR,                 /*     |               */
	GA_REDUCE_XOR,                /*     ^               */
	GA_REDUCE_ALL,                /*  &&/all()           */
	GA_REDUCE_ANY,                /*  ||/any()           */
	
	GA_REDUCE_ENDSUPPORTED        /* Must be last element in enum */
} ga_reduce_op;


/* External Functions */

/**
 * @brief Create, modify and free the attributes of a reduction operator.
 * 
 * @param [out] grAttr      The reduction operator attributes object.
 * @param [in]  op          The reduction operation.
 * @param [in]  maxSrcDims  The maximum number of supported source dimensions.
 * @param [in]  maxDstDims  The maximum number of supported destination dimensions.
 * @param [in]  s0Typecode  The typecode of the source tensor.
 * @param [in]  d0Typecode  The typecode of the first destination tensor.
 * @param [in]  d1Typecode  The typecode of the second destination tensor.
 * @param [in]  i0Typecode  The typecode of the indices.
 */

GPUARRAY_PUBLIC int   GpuReductionAttr_new          (GpuReductionAttr**         grAttr,
                                                     gpucontext*                gpuCtx);
GPUARRAY_PUBLIC int   GpuReductionAttr_setop        (GpuReductionAttr*          grAttr,
                                                     ga_reduce_op               op);
GPUARRAY_PUBLIC int   GpuReductionAttr_setdims      (GpuReductionAttr*          grAttr,
                                                     unsigned                   maxSrcDims,
                                                     unsigned                   maxDstDims);
GPUARRAY_PUBLIC int   GpuReductionAttr_sets0type    (GpuReductionAttr*          grAttr,
                                                     int                        s0Typecode);
GPUARRAY_PUBLIC int   GpuReductionAttr_setd0type    (GpuReductionAttr*          grAttr,
                                                     int                        d0Typecode);
GPUARRAY_PUBLIC int   GpuReductionAttr_setd1type    (GpuReductionAttr*          grAttr,
                                                     int                        d1Typecode);
GPUARRAY_PUBLIC int   GpuReductionAttr_seti0type    (GpuReductionAttr*          grAttr,
                                                     int                        i0Typecode);
GPUARRAY_PUBLIC int   GpuReductionAttr_appendopname (GpuReductionAttr*          grAttr,
                                                     size_t                     n,
                                                     char*                      name);
GPUARRAY_PUBLIC int   GpuReductionAttr_issensitive  (const GpuReductionAttr*    grAttr);
GPUARRAY_PUBLIC int   GpuReductionAttr_requiresS0   (const GpuReductionAttr*    grAttr);
GPUARRAY_PUBLIC int   GpuReductionAttr_requiresD0   (const GpuReductionAttr*    grAttr);
GPUARRAY_PUBLIC int   GpuReductionAttr_requiresD1   (const GpuReductionAttr*    grAttr);
GPUARRAY_PUBLIC void  GpuReductionAttr_free         (GpuReductionAttr*          grAttr);

/**
 * @brief Create a new GPU reduction operator with the given attributes.
 *
 * @param [out] gr           The reduction operator.
 * @param [in]  grAttr       The GPU context.
 *
 * @return GA_NO_ERROR      if the operator was created successfully
 *         GA_INVALID_ERROR if some argument was invalid
 *         GA_NO_MEMORY     if memory allocation failed anytime during creation
 *         or other non-zero error codes otherwise.
 */

GPUARRAY_PUBLIC int   GpuReduction_new              (GpuReduction**             gr,
                                                     const GpuReductionAttr*    grAttr);

/**
 * @brief Deallocate an operator allocated by GpuReduction_new().
 */

GPUARRAY_PUBLIC void  GpuReduction_free             (GpuReduction*              gr);

/**
 * @brief Invoke an operator allocated by GpuReduction_new() on a source tensor.
 *
 * Returns one (in the case of min-and-argmin/max-and-argmax, two) destination
 * tensors. The destination tensor(s)' axes are a strict subset of the axes of the
 * source tensor. The axes to be reduced are specified by the caller, and the
 * reduction is performed over these axes, which are then removed in the
 * destination.
 * 
 * @param [in]  gr         The reduction operator.
 * @param [out] d0         The destination tensor.
 * @param [out] d1         The second destination tensor, for argmin/argmax operations.
 * @param [in]  s0         The source tensor.
 * @param [in]  reduxLen   The number of axes reduced. Must be >= 1 and
 *                         <= s0->nd.
 * @param [in]  reduxList  A list of integers of length reduxLen, indicating
 *                         the axes to be reduced. The order of the axes
 *                         matters for dstArg index calculations (argmin, argmax,
 *                         minandargmin, maxandargmax). All entries in the list must be
 *                         unique, >= 0 and < src->nd.
 *                         
 *                         For example, if a 5D-tensor is maxandargmax-reduced with an
 *                         axis list of [3,4,1], then reduxLen shall be 3, and the
 *                         index calculation in every point shall take the form
 *                         
 *                             d1[i0,i2] = i3 * s0.shape[4] * s0.shape[1] +
 *                                         i4 * s0.shape[1]               +
 *                                         i1
 *                         
 *                         where (i3,i4,i1) are the coordinates of the maximum-
 *                         valued element within subtensor [i0,:,i2,:,:] of s0.
 * @param [in]  flags      Reduction operator invocation flags. Currently must be
 *                         set to 0.
 *
 * @return GA_NO_ERROR if the operator was invoked successfully, or a non-zero
 *         error code otherwise.
 */

GPUARRAY_PUBLIC int   GpuReduction_call            (const GpuReduction*        gr,
                                                    GpuArray*                  d0,
                                                    GpuArray*                  d1,
                                                    const GpuArray*            s0,
                                                    unsigned                   reduxLen,
                                                    const int*                 reduxList,
                                                    int                        flags);


#ifdef __cplusplus
}
#endif

#endif
