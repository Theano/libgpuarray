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
  npy_intp * dimensions; //dim0, dim1, ... dim nd
  npy_intp * strides; //stride0, stride1, ... stride nd
  int flags; // Flags, see numpy flags
  //DTYPE dtype; // fine for numeric types
  //DtypeMeta * dtype_meta; // reserved for future use.
  //PyArray_Descr *descr;   /* Pointer to type structure */
} GpuNdArray;
