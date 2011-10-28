#ifndef _GPU_NDARRAY_H
#define _GPU_NDARRAY_H


typedef struct GpuNdArray{
  char* data; //pointer to data element [0,..,0].
  int offset;
  int nd; //the number of dimensions of the tensor

  /**
   * base:
   *  either NULL or a pointer to a fellow CudaNdarray into which this one is viewing.
   *  This pointer is never followed, except during Py_DECREF when we do not need it any longer.
   */
  void * base;
  ssize_t  * dimensions; //dim0, dim1, ... dim nd
  ssize_t * strides; //stride0, stride1, ... stride nd
  int flags; // Flags, see numpy flags
  //DTYPE dtype; // fine for numeric types
  //DtypeMeta * dtype_meta; // reserved for future use.
  //PyArray_Descr *descr;   /* Pointer to type structure */
} GpuNdArray;

#endif
/*
  Local Variables:
  mode:c++
  c-basic-offset:4
  c-file-style:"stroustrup"
  c-file-offsets:((innamespace . 0)(inline-open . 0))
  indent-tabs-mode:nil
  fill-column:79
  End:
*/
// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:textwidth=79 :
