#ifndef COMPYTE_ARRAY_H
#define COMPYTE_ARRAY_H
/**
 * \file array.h
 * \brief Array functions.
 */

#include <compyte/buffer.h>

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
   * Backend operations vector.
   */
  compyte_buffer_ops *ops;
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
   * Array owns the GpuArray#data element (is responsible for freeing it).
   */
#define GA_OWNDATA        0x0004
  /**
   * Unused.
   */
#define GA_ENSURECOPY     0x0020
  /**
   * Buffer data is properly aligned for the type (currently this is
   * always assumed to be true).
   */
#define GA_ALIGNED        0x0100
  /**
   * Can write to the data buffer.  (This is always true for array
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
static inline int GpuArray_CHKFLAGS(GpuArray *a, int flags) {
  return (a->flags & flags) == flags;
}
/* Add tests here when you need them */
#define GpuArray_OWNSDATA(a) GpuArray_CHKFLAGS(a, GA_OWNDATA)
#define GpuArray_ISWRITEABLE(a) GpuArray_CHKFLAGS(a, GA_WRITEABLE)
#define GpuArray_ISALIGNED(a) GpuArray_CHKFLAGS(a, GA_ALIGNED)
#define GpuArray_ISONESEGMENT(a) ((a)->flags & (GA_C_CONTIGUOUS|GA_F_CONTIGUOUS))
#define GpuArray_ISFORTRAN(a) GpuArray_CHKFLAGS(a, GA_F_CONTIGUOUS)
#define GpuArray_ITEMSIZE(a) compyte_get_elsize((a)->typecode)

/**
 * Initialize and allocate a new empty (uninitialized data) array.
 *
 * \param a the GpuArray structure to initialize.  Content will be
 * ignored so make sure to deallocate any previous array first.
 * \param ops backend operations to use.
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
COMPYTE_PUBLIC int GpuArray_empty(GpuArray *a, compyte_buffer_ops *ops,
                                  void *ctx, int typecode, unsigned int nd,
                                  size_t *dims, ga_order ord);

/**
 * Initialize and allocate a new zero-initialized array.
 *
 * \param a the GpuArray structure to initialize.  Content will be
 * ignored so make sure to deallocate any previous array first.
 * \param ops backend operations to use.
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
COMPYTE_PUBLIC int GpuArray_zeros(GpuArray *a, compyte_buffer_ops *ops,
                                  void *ctx, int typecode, unsigned int nd,
                                  size_t *dims, ga_order ord);

/**
 * Initialize and allocate a new array structure from a pre-existing buffer.
 *
 * The array will be considered to own the gpudata structure after the
 * call is made and will free it when deallocated.  An error return
 * from this function will deallocate `data`.
 *
 * \param a the GpuArray structure to initialize.  Content will be
 * ignored so make sure to deallocate any previous array first.
 * \param ops backend that corresponds to the buffer.
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
COMPYTE_PUBLIC int GpuArray_fromdata(GpuArray *a, compyte_buffer_ops *ops,
                                     gpudata *data, size_t offset,
                                     int typecode, unsigned int nd,
                                     size_t *dims, ssize_t *strides,
                                     int writeable);

COMPYTE_PUBLIC int GpuArray_view(GpuArray *v, GpuArray *a);
COMPYTE_PUBLIC int GpuArray_sync(GpuArray *a);
COMPYTE_PUBLIC int GpuArray_index(GpuArray *r, GpuArray *a, ssize_t *starts, ssize_t *stops,
                   ssize_t *steps);

COMPYTE_PUBLIC int GpuArray_reshape(GpuArray *res, GpuArray *a, unsigned int nd, size_t *newdims, ga_order ord, int nocopy);
COMPYTE_PUBLIC void GpuArray_clear(GpuArray *a);

COMPYTE_PUBLIC int GpuArray_share(GpuArray *a, GpuArray *b);
COMPYTE_PUBLIC void *GpuArray_context(GpuArray *a);

COMPYTE_PUBLIC int GpuArray_move(GpuArray *dst, GpuArray *src);
COMPYTE_PUBLIC int GpuArray_write(GpuArray *dst, void *src, size_t src_sz);
COMPYTE_PUBLIC int GpuArray_read(void *dst, size_t dst_sz, GpuArray *src);

COMPYTE_PUBLIC int GpuArray_memset(GpuArray *a, int data);

COMPYTE_PUBLIC int GpuArray_copy(GpuArray *res, GpuArray *a, ga_order order);

COMPYTE_PUBLIC const char *GpuArray_error(GpuArray *a, int err);

COMPYTE_PUBLIC void GpuArray_fprintf(FILE *fd, const GpuArray *a);
COMPYTE_LOCAL int GpuArray_is_c_contiguous(const GpuArray *a);
COMPYTE_LOCAL int GpuArray_is_f_contiguous(const GpuArray *a);

#ifdef __cplusplus
}
#endif

#endif
