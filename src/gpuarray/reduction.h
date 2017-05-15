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


/**
 * Supported array reduction operations.
 */

typedef enum _ga_reduce_op {
	GA_REDUCE_SUM,             /*          +          */
	GA_REDUCE_PROD,            /*          *          */
	GA_REDUCE_PRODNZ,          /*          * (!=0)    */
	GA_REDUCE_MIN,             /*        min()        */
	GA_REDUCE_MAX,             /*        max()        */
	GA_REDUCE_ARGMIN,          /*       argmin()      */
	GA_REDUCE_ARGMAX,          /*       argmax()      */
	GA_REDUCE_MINANDARGMIN,    /*   min(), argmin()   */
	GA_REDUCE_MAXANDARGMAX,    /*   max(), argmax()   */
	GA_REDUCE_AND,             /*          &          */
	GA_REDUCE_OR,              /*          |          */
	GA_REDUCE_XOR,             /*          ^          */
	GA_REDUCE_ALL,             /*       &&/all()      */
	GA_REDUCE_ANY,             /*       ||/any()      */
} ga_reduce_op;



/**
 * @brief Compute a reduction over a list of axes to reduce.
 *
 * Returns one (in the case of min-and-argmin/max-and-argmax, two) destination
 * tensors. The destination tensor(s)' axes are a strict subset of the axes of the
 * source tensor. The axes to be reduced are specified by the caller, and the
 * reduction is performed over these axes, which are then removed in the
 * destination.
 *
 * @param [in]  op         The reduction operation to perform.
 * @param [out] dst        The destination tensor. Has the same type as the source.
 * @param [out] dstArg     For argument of minima/maxima operations. Has type int64.
 * @param [in]  src        The source tensor.
 * @param [in]  reduxLen   The number of axes reduced. Must be >= 1 and
 *                         <= src->nd.
 * @param [in]  reduxList  A list of integers of length reduxLen, indicating
 *                         the axes to be reduced. The order of the axes
 *                         matters for dstArg index calculations (GpuArray_argmin,
 *                         GpuArray_argmax, GpuArray_minandargmin,
 *                         GpuArray_maxandargmax). All entries in the list must be
 *                         unique, >= 0 and < src->nd.
 *                         
 *                         For example, if a 5D-tensor is max-reduced with an axis
 *                         list of [3,4,1], then reduxLen shall be 3, and the
 *                         index calculation in every point shall take the form
 *                         
 *                             dstArgmax[i0,i2] = i3 * src.shape[4] * src.shape[1] +
 *                                                i4 * src.shape[1]                +
 *                                                i1
 *                         
 *                         where (i3,i4,i1) are the coordinates of the maximum-
 *                         valued element within subtensor [i0,:,i2,:,:] of src.
 * @return GA_NO_ERROR if the operation was successful, or a non-zero error
 *         code otherwise.
 */

GPUARRAY_PUBLIC int GpuArray_reduction   (ga_reduce_op    op,
                                          GpuArray*       dst,
                                          GpuArray*       dstArg,
                                          const GpuArray* src,
                                          unsigned        reduxLen,
                                          const unsigned* reduxList);





#ifdef __cplusplus
}
#endif

#endif
