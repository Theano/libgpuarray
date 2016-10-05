#ifndef GPUARRAY_ARRAY_H
#define GPUARRAY_ARRAY_H
/**
 * \file array.h
 * \brief Array functions.
 */

#include <gpuarray/buffer.h>

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
 * Main array structure.
 */
typedef struct _GpuArray {
  /**
   * Device data buffer.
   */
  gpudata *data;
  /**
   * Size of each dimension.  The number of elements is #nd.
   */
  size_t *dimensions;
  /**
   * Stride for each dimension.  The number of elements is #nd.
   */
  ssize_t *strides;
  /**
   * Offset to the first array element into the device data buffer.
   */
  size_t offset;
  /**
   * Number of dimensions.
   */
  unsigned int nd;
  /**
   * Flags for this array (see \ref aflags).
   */
  int flags;
  /**
   * Type of the array elements.
   */
  int typecode;

/**
 * \defgroup aflags Array Flags
 * @{
 */
  /* Try to keep in sync with numpy values for now */
  /**
   * Array is C-contiguous.
   */
#define GA_C_CONTIGUOUS   0x0001
  /**
   * Array is Fortran-contiguous.
   */
#define GA_F_CONTIGUOUS   0x0002
  /**
   * Buffer data is properly aligned for the type.  This should always
   * be true for arrays allocated through this library.
   *
   * If this isn't true you can't use kernels on the data, since they
   * require aligned access.
   */
#define GA_ALIGNED        0x0100
  /**
   * Can write to the data buffer.  (This is always true for arrays
   * allocated through this library).
   */
#define GA_WRITEABLE      0x0400
  /**
   * Array data is behaved (properly aligned and writable).
   */
#define GA_BEHAVED        (GA_ALIGNED|GA_WRITEABLE)
  /**
   * Array layout is that of a C array.
   */
#define GA_CARRAY         (GA_C_CONTIGUOUS|GA_BEHAVED)
  /**
   * Array layout is that of a Fortran array.
   */
#define GA_FARRAY         (GA_F_CONTIGUOUS|GA_BEHAVED)
  /**
   * @}
   */
  /* Numpy flags that will not be supported at this level (and why):

     NPY_OWNDATA: data is refcounted
     NPY_NOTSWAPPED: data is alway native endian
     NPY_FORCECAST: no casts
     NPY_ENSUREARRAY: no inherited classes
     NPY_UPDATEIFCOPY: cannot support without refcount (or somesuch)

     Maybe will define other flags later */
} GpuArray;

/**
 * Type used to specify the desired order to some functions
 */
typedef enum _ga_order {
  /**
   * Any order is fine.
   */
  GA_ANY_ORDER=-1,
  /**
   * C order is desired.
   */
  GA_C_ORDER=0,
  /**
   * Fortran order is desired.
   */
  GA_F_ORDER=1
} ga_order;

/**
 * Checks if all the specified flags are set.
 *
 * \param a array
 * \param flags flags to check
 *
 * \returns true if all flags in `flags` are set and false otherwise.
 */
static inline int GpuArray_CHKFLAGS(const GpuArray *a, int flags) {
  return (a->flags & flags) == flags;
}
/* Add tests here when you need them */

/**
 * Checks if the array data is writable.
 *
 * \param a array
 *
 * \returns true if the data area of `a` is writable
 */
#define GpuArray_ISWRITEABLE(a) GpuArray_CHKFLAGS(a, GA_WRITEABLE)
/**
 * Checks if the array elements are aligned.
 *
 * \param a array
 *
 * \returns true if the elements of `a` are aligned.
 */
#define GpuArray_ISALIGNED(a) GpuArray_CHKFLAGS(a, GA_ALIGNED)
/**
 * Checks if the array elements are contiguous in memory.
 *
 * \param a array
 *
 * \returns true if the data area of `a` is contiguous
 */
#define GpuArray_ISONESEGMENT(a) ((a)->flags & (GA_C_CONTIGUOUS|GA_F_CONTIGUOUS))
/**
 * Checks if the array elements are c contiguous in memory.
 *
 * \param a array
 *
 * \returns true if the data area of `a` is contiguous
 */
#define GpuArray_IS_C_CONTIGUOUS(a) ((a)->flags & GA_C_CONTIGUOUS)
/**
 * Checks if the array elements are f contiguous in memory.
 *
 * \param a array
 *
 * \returns true if the data area of `a` is contiguous
 */
#define GpuArray_IS_F_CONTIGUOUS(a) ((a)->flags & GA_F_CONTIGUOUS)
/**
 * This is the same as GpuArray_IS_F_CONTIGUOUS, but not the same as PyArray_ISFORTRAN.
 *
 * PyArray_ISFORTRAN checks if the array elements are laid out if
 * Fortran order and NOT c order.
 *
 * \param a array
 *
 * \returns true if the data area of `a` is Fortran-contiguous
 */
#define GpuArray_ISFORTRAN(a) (GpuArray_CHKFLAGS(a, GA_F_CONTIGUOUS))
/**
 * Retrive the size of the elements in the array.
 *
 * \param a array
 *
 * \returns the size of the array elements.
 */
#define GpuArray_ITEMSIZE(a) gpuarray_get_elsize((a)->typecode)

/**
 * Initialize and allocate a new empty (uninitialized data) array.
 *
 * \param a the GpuArray structure to initialize.  Content will be
 * ignored so make sure to deallocate any previous array first.
 * \param ctx context in which to allocate array data. Must come from
 * the same backend as the operations vector.
 * \param typecode type of the elements in the array
 * \param nd desired order (number of dimensions)
 * \param dims size for each dimension.
 * \param ord desired layout of data.
 *
 * \returns A return of GA_NO_ERROR means that the structure is
 * properly initialized and that the memory requested is reserved on
 * the device.  Any other error code means that the structure is
 * left uninitialized.
 */
GPUARRAY_PUBLIC int GpuArray_empty(GpuArray *a, gpucontext *ctx, int typecode,
                                   unsigned int nd, const size_t *dims,
                                   ga_order ord);

/**
 * Initialize and allocate a new zero-initialized array.
 *
 * \param a the GpuArray structure to initialize.  Content will be
 * ignored so make sure to deallocate any previous array first.
 * \param ctx context in which to allocate array data. Must come from
 * the same backend as the operations vector.
 * \param typecode type of the elements in the array
 * \param nd desired order (number of dimensions)
 * \param dims size for each dimension.
 * \param ord desired layout of data.
 *
 * \returns A return of GA_NO_ERROR means that the structure is
 * properly initialized and that the memory requested is reserved on
 * the device.  Any other error code means that the structure is
 * left uninitialized.
 */
GPUARRAY_PUBLIC int GpuArray_zeros(GpuArray *a, gpucontext *ctx, int typecode,
                                   unsigned int nd, const size_t *dims,
                                   ga_order ord);

/**
 * Initialize and allocate a new array structure from a pre-existing buffer.
 *
 * The array will be considered to own the gpudata structure after the
 * call is made and will free it when deallocated.  An error return
 * from this function will deallocate `data`.
 * This increment the ref count of gpudata. This seem to contradict the above.
 *
 * \param a the GpuArray structure to initialize.  Content will be
 * ignored so make sure to deallocate any previous array first.
 * \param data buffer to user.
 * \param offset position of the first data element of the array in the buffer.
 * \param typecode type of the elements in the array
 * \param nd order of the data (number of dimensions).
 * \param dims size for each dimension.
 * \param strides stride for each dimension.
 * \param writeable true if the buffer is writable false otherwise.
 *
 * \returns A return of GA_NO_ERROR means that the structure is
 * properly initialized. Any other error code means that the structure
 * is left uninitialized and the provided buffer is deallocated.
 */
GPUARRAY_PUBLIC int GpuArray_fromdata(GpuArray *a,
                                      gpudata *data, size_t offset,
                                      int typecode, unsigned int nd,
                                      const size_t *dims,
                                      const ssize_t *strides, int writeable);

GPUARRAY_PUBLIC int GpuArray_copy_from_host(GpuArray *a,
                                            gpucontext *ctx, void *buf, int typecode,
                                            unsigned int nd, const size_t *dims,
                                            const ssize_t *strides);

/**
 * Initialize an array structure to provide a view of another.
 *
 * The new structure will point to the same data area and have the
 * same values of properties as the source one.  The data area is
 * shared and writes from one array will be reflected in the other.
 * The properties are copied and not shared and can be modified
 * independantly.
 *
 * \param v the result array
 * \param a the source array
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_view(GpuArray *v, const GpuArray *a);

/**
 * Blocks until all operations (kernels, copies) involving `a` are finished.
 *
 * \param a the array to synchronize
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_sync(GpuArray *a);

/**
 * Returns a sub-view of a source array.
 *
 * The indexing follows simple basic model where each dimension is
 * indexed separately.  For a single dimension the indexing selects
 * from the start index (included) to the end index (excluded) while
 * selecting one over step elements. As an example for the array `[ 0
 * 1 2 3 4 5 6 7 8 9 ]` indexed with start index 1 stop index 8 and
 * step 2 the result would be `[ 1 3 5 7 ]`.
 *
 * The special value 0 for step means that only one element
 * corresponding to the start index and the resulting array order will
 * be one smaller.
 *
 * \param r the result array
 * \param a the source array
 * \param starts the start of the subsection for each dimension (length must be a->nd)
 * \param stops the end of the subsection for each dimension (length must be a->nd)
 * \param steps the steps for the subsection for each dimension (length must be a->nd)
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_index(GpuArray *r, const GpuArray *a,
                                  const ssize_t *starts, const ssize_t *stops,
                                  const ssize_t *steps);

GPUARRAY_PUBLIC int GpuArray_index_inplace(GpuArray *a, const ssize_t *starts,
                                          const ssize_t *stops,
                                          const ssize_t *steps);

/**
 * Take a portion of an array along axis 0.
 *
 * This operation allows arbitrary indexing of an array along its
 * first axis. The indexed array `v` can be of any dimension or
 * strides. The result and index array (`a` and `i` respectively) need
 * to be C contiguous.
 *
 * The dimension 0 of `a` has to match dimension 0 of `i` and the
 * others have to match their equivalent on `v`. `i` has to have a
 * single dimension.
 *
 * If `check_error` is not 0, the function will check for indexing
 * errors in the kernel and will return GA_VALUE_ERROR in that
 * case. No other error will produce that error code. This is not
 * always done because it introduces a synchronization point which may
 * affect performance.
 *
 * \param a the result array (nd)
 * \param v the source array (nd)
 * \param i the index array (1d)
 * \param check_error whether to check for index errors or not
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_take1(GpuArray *a, const GpuArray *v,
                                   const GpuArray *i, int check_error);

/**
 * Sets the content of an array to the content of another array.
 *
 * The value array must be smaller or equal in number of dimensions to
 * the destination array.  Each of its dimensions' size must be either
 * exactly equal to the destination array's corresponding dimensions
 * or 1.  Dimensions of size 1 will be repeated to fill the full size
 * of the destination array. Extra size 1 dimensions will be added at
 * the end to make the two arrays shape-equivalent.
 *
 * \param a the destination array
 * \param v the value array
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_setarray(GpuArray *a, const GpuArray *v);

/**
 * Change the dimensions of an array.
 *
 * Return a new array with the desired dimensions. The new dimensions
 * must have the same total size as the old ones. A copy of the
 * underlying data may be performed if necessary, unless `nocopy` is
 * 0.
 *
 * \param res the result array
 * \param a the source array
 * \param nd new dimensions order
 * \param newdims new dimensions (length is nd)
 * \param ord the desired resulting order
 * \param nocopy if 0 error out if a data copy is required.
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_reshape(GpuArray *res, const GpuArray *a,
                                     unsigned int nd, const size_t *newdims,
                                     ga_order ord, int nocopy);

GPUARRAY_PUBLIC int GpuArray_reshape_inplace(GpuArray *a, unsigned int nd,
                                             const size_t *newdims,
                                             ga_order ord);

/**
 * Rearrange the axes of an array.
 *
 * Return a new array with its shape and strides swapped accordingly
 * to the `new_axes` parameter.  If `new_axes` is NULL then the order
 * is reversed.  The returned array is a view on the data of the old
 * one.
 *
 * \param res the result array
 * \param a the source array
 * \param new_axes either NULL or a list of a->nd elements
 *
 * \return GA_NO_ERROR if the operation was successful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_transpose(GpuArray *res, const GpuArray *a,
                                      const unsigned int *new_axes);

GPUARRAY_PUBLIC int GpuArray_transpose_inplace(GpuArray *a,
                                              const unsigned int *new_axes);

/**
 * Release all device and host memory associated with `a`.
 *
 * This function frees all host memory, and releases the device memory
 * if it is the owner. In case an array has views it is the
 * responsability of the caller to ensure a base array is not cleared
 * before its views.
 *
 * This function will also zero out the structure to prevent
 * accidental reuse.
 *
 * \param a the array to clear
 */
GPUARRAY_PUBLIC void GpuArray_clear(GpuArray *a);

/**
 * Checks if two arrays may share device memory.
 *
 * \param a an array
 * \param b an array
 *
 * \returns 1 if `a` and `b` may share a portion of their data.
 */
GPUARRAY_PUBLIC int GpuArray_share(const GpuArray *a, const GpuArray *b);

/**
 * Retursns the context of an array.
 *
 * \param a an array
 *
 * \returns the context in which `a` was allocated.
 */
GPUARRAY_PUBLIC gpucontext *GpuArray_context(const GpuArray *a);

/**
 * Copies all the elements of one array to another.
 *
 * The arrays `src` and `dst` must have the same size (total number of
 * elements) and be in the same context.
 *
 * \param dst destination array
 * \param src source array
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_move(GpuArray *dst, const GpuArray *src);

/**
 * Copy data from the host memory to the device memory.
 *
 * \param dst destination array (must be contiguous)
 * \param src source host memory (contiguous block)
 * \param src_sz size of data to copy (in bytes)
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_write(GpuArray *dst, const void *src,
                                   size_t src_sz);

/**
 * Copy data from the device memory to the host memory.
 *
 * \param dst destination host memory (contiguous block)
 * \param dst_sz size of data to copy (in bytes)
 * \param src source array (must be contiguous)
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_read(void *dst, size_t dst_sz,
                                  const GpuArray *src);

/**
 * Set all of an array's data to a byte pattern.
 *
 * \param a an array (must be contiguous)
 * \param data the byte to repeat
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_memset(GpuArray *a, int data);

/**
 * Make a copy of an array.
 *
 * This is analogue to GpuArray_view() except it copies the device
 * memory and no data is shared.
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_copy(GpuArray *res, const GpuArray *a,
                                 ga_order order);

/**
 * Copy between arrays in different contexts.
 *
 * This works like GpuArray_move() except it will work between arrays
 * that aren't in the same context.
 *
 * Source and target arrays must be contiguous.  This restriction may
 * be lifted in the future.
 *
 * \param r result array
 * \param a array to transfer
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_transfer(GpuArray *res, const GpuArray *a);

/**
 * Split an array into multiple views.
 *
 * The created arrays will be sub-portions of `a` where `axis` is
 * divided according to the values in `p`.  No checks are performed on
 * the values in `p` except to make sure that they don't reference
 * values outside of the bounds of the source array.
 *
 * If an error occurs partway during the operation, the created arrays
 * will be cleared before returning.
 *
 * \param rs list of array pointers to store results (must be of length n+1)
 * \param a array to split
 * \param n number of splits (length of p)
 * \param p list of split points
 * \param axis axis to split
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_split(GpuArray **rs, const GpuArray *a, size_t n,
                                   size_t *p, unsigned int axis);

/**
 * Concatenate the arrays in `as` along the axis `axis`.
 *
 * If an error occurs during the operation, the result array may be
 * cleared before returning.
 *
 * \param r the result array
 * \param as list of pointer to arrays to concatenate
 * \param n number of array in list `as`
 * \param axis the axis along which to concatenate
 * \param restype the typecode of the result array
 *
 * \return GA_NO_ERROR if the operation was succesful.
 * \return an error code otherwise
 */
GPUARRAY_PUBLIC int GpuArray_concatenate(GpuArray *r, const GpuArray **as,
                                         size_t n, unsigned int axis,
                                         int restype);

/**
 * Get a description of the last error in the context of `a`.
 *
 * The description may reflect operations with other arrays in the
 * same context if other operations were performed between the
 * occurence of the error and the call to this function.
 *
 * Operations in other contexts, however have no incidence on the
 * return value.
 *
 * \param a an array
 * \param err the error code returned
 *
 * \returns A user-readable string describing the nature of the error.
 */
GPUARRAY_PUBLIC const char *GpuArray_error(const GpuArray *a, int err);

/**
 * Print a textual description of `a` to the specified file
 * descriptor.
 *
 * \param fd a file descriptior open for writing
 * \param a an array
 */
GPUARRAY_PUBLIC void GpuArray_fprintf(FILE *fd, const GpuArray *a);

GPUARRAY_PUBLIC int GpuArray_fdump(FILE *fd, const GpuArray *a);

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
 * @param [in]  reduxLen   The number of axes reduced. Must be >= 1 and
 *                         <= src->nd.
 * @param [in]  reduxList  A list of integers of length reduxLen, indicating
 *                         the axes to be reduced. The order of the axes
 *                         matters for dstArgmax index calculations. All
 *                         entries in the list must be unique, >= 0 and
 *                         < src->nd.
 *                         
 *                         For example, if a 5D-tensor is reduced with an axis
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

GPUARRAY_PUBLIC int GpuArray_maxandargmax(GpuArray*       dstMax,
                                          GpuArray*       dstArgmax,
                                          const GpuArray* src,
                                          unsigned        reduxLen,
                                          const unsigned* reduxList);

#ifdef __cplusplus
}
#endif

#endif
