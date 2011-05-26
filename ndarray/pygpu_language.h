/**
 * This file contain the header for ALL code that depend on cuda or opencl.
 */
#ifndef _PYGPU_LANGUAGE_H
#define _PYGPU_LANGUAGE_H
#include <Python.h>
//#include <iostream>
//#include <pygpu_ndarray.cuh>

#ifdef __DEVICE_EMULATION__
#define NUM_VECTOR_OP_BLOCKS                4096
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     1  //This prevents printf from getting tangled up
#else
#define NUM_VECTOR_OP_BLOCKS                4096 //Max number of blocks to launch.  Should be read from device properties. (#10)
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     256  //Should be read from device properties. (#10)
#endif

#if 0
// Do not wait after every kernel & transfer.
#define CNDA_THREAD_SYNC
#else
// This is useful for using normal profiling tools
#define CNDA_THREAD_SYNC cudaThreadSynchronize();
#endif


#ifndef SHARED_SIZE 
#define SHARED_SIZE (16*1024)
#endif

/////////////////////////
// Alloc and Free
/////////////////////////
//If true, when there is a gpu malloc or free error, we print the size of allocated memory on the device.
#define COMPUTE_GPU_MEM_USED 0
#define VERBOSE_ALLOC_FREE 0
//If true, we fill with NAN allocated device memory.
#define ALLOC_MEMSET 0

static int _outstanding_mallocs[] = {0,0};

#if COMPUTE_GPU_MEM_USED
int _allocated_size = 0;
const int TABLE_SIZE = 10000;
struct table_struct{
  void* ptr;
  int size;
};
table_struct _alloc_size_table[TABLE_SIZE];
#endif

/**
 * Allocation and freeing of device memory should go through these functions so that the lib can track memory usage.
 *
 * device_malloc will set the Python error message before returning None.
 * device_free will return nonzero on failure (after setting the python error message)
 */
void * device_malloc(size_t size);
int device_free(void * ptr);
static PyObject *
outstanding_mallocs(PyObject* self, PyObject * args)
{
    return PyInt_FromLong(_outstanding_mallocs[0]);
}

int PyGpuNdArray_CopyFromPyGpuNdArray(PyGpuNdArrayObject * self, PyGpuNdArrayObject * other, bool unbroadcast = false);

#endif
