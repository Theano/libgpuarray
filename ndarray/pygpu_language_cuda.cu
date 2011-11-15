#include <pygpu_ndarray_object.h>
#include <pygpu_language.h>

#include <cublas.h>

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


char *
cublasGetErrorString(cublasStatus err)
{
    if (err == CUBLAS_STATUS_NOT_INITIALIZED) {
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    } else if (err == CUBLAS_STATUS_ALLOC_FAILED){
        return "CUBLAS_STATUS_ALLOC_FAILED";
    } else if (err == CUBLAS_STATUS_INVALID_VALUE){
        return "CUBLAS_STATUS_INVALID_VALUE";
    } else if (err == CUBLAS_STATUS_MAPPING_ERROR){
        return "CUBLAS_STATUS_MAPPING_ERROR";
    } else if (err == CUBLAS_STATUS_EXECUTION_FAILED){
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    } else if (err == CUBLAS_STATUS_INTERNAL_ERROR){
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    } else {
        return "UNKNOW ERROR";
    }

}

/////////////////////////
// Alloc and Free
/////////////////////////
void * device_malloc(size_t size)
{
    void * rval=NULL;
    cudaError_t err = cudaMalloc(&rval, size);
    if (cudaSuccess != err){
#if COMPUTE_GPU_MEM_USED
        fprintf(stderr, "Error allocating %li bytes of device memory (%s). %d already allocated\n",
		(long)size, cudaGetErrorString(err),_allocated_size);
#else
        fprintf(stderr, "Error allocating %li bytes of device memory (%s).\n",
		(long)size, cudaGetErrorString(err));
#endif
        PyErr_Format(PyExc_MemoryError, "Error allocating %li bytes of device memory (%s).",
		     (long)size, cudaGetErrorString(err));
        return NULL;
    }
    _outstanding_mallocs[0] += (rval != NULL);
#if COMPUTE_GPU_MEM_USED
    for(int i=0;i<TABLE_SIZE;i++){
      if(NULL==_alloc_size_table[i].ptr){
        _alloc_size_table[i].ptr=rval;
        _alloc_size_table[i].size=size;
        break;
      }
    }
    _allocated_size += size;
    DPRINTF("allocated %li bytes of device memory (%s). %d already allocated, ptr: %p\n",
            (long)size, cudaGetErrorString(err),_allocated_size,rval);
#else
    DPRINTF("allocated %li bytes of device memory (%s). ptr: %p\n",
            (long)size, cudaGetErrorString(err),rval);

#endif

    if(ALLOC_MEMSET){
      //We init them to nan to make sure we catch more debug case.
      cudaMemset(rval, 0xFF, size);
      //printf("MEMSET\n");
    }
    return rval;
}

int device_free(void *ptr)
{
    // if there is no gpu context, the call to cudaFree will fail; skip it entirely
    /*if(!g_gpu_context_active){
        return 0;
	}*/
    cudaError_t err =  cudaFree(ptr);
    if (cudaSuccess != err){
#if COMPUTE_GPU_MEM_USED
        fprintf(stderr, "Error freeing device pointer %p (%s).%d byte already allocated\n",
		ptr, cudaGetErrorString(err), _allocated_size);
#else
        fprintf(stderr, "Error freeing device pointer %p (%s).\n",
		ptr, cudaGetErrorString(err));
#endif
        PyErr_Format(PyExc_MemoryError, "error freeing device pointer %p (%s)",
		     ptr, cudaGetErrorString(err));
        return -1;
    }
    _outstanding_mallocs[0] -= (ptr != NULL);
#if COMPUTE_GPU_MEM_USED
    int i=0;
    size_t total_freed = 0;
    for(;i<TABLE_SIZE;i++)
      if(_alloc_size_table[i].ptr==ptr){
        _allocated_size -= _alloc_size_table[i].size;
        total_freed += _alloc_size_table[i].size;
        _alloc_size_table[i].ptr=0;
        _alloc_size_table[i].size=0;

        break;
      }
    if(i==TABLE_SIZE)
      printf("Unallocated unknow size!\n");
    DPRINTF("freed %li bytes of device memory (%s). %d already allocated, ptr=%p\n", (long)total_freed, cudaGetErrorString(err),_allocated_size,ptr);
#endif
    return 0;
}
//make the rightmost coords change fastest
//TODO: why does a downward for-loop not work????
//TODO: skip the last division (when d == 0)
#define decl_k_elemwise_unary_rowmajor(name, F, DTYPE)	\
__global__ void name (unsigned int numEls,  \
        unsigned int nd, \
        const ssize_t * dim,  \
        const DTYPE * a_data, const ssize_t * a_str, \
        DTYPE * z_data, const ssize_t * z_str) \
{ \
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    const unsigned int numThreads = blockDim.x * gridDim.x; \
\
    for (unsigned int i = idx; i < numEls; i += numThreads) {   \
        unsigned int ii = i; \
        const DTYPE * a_i = a_data; \
        DTYPE * z_i = z_data; \
        for (unsigned int _d = 0; _d < nd; ++_d) { \
            unsigned int d = nd - _d-1;  \
            /* i_d used to be unsigned, but their is a bug in nvcc 3.0. making it signed fix the bug.*/\
            int i_d = ii % dim[d]; /* i_d is our position in the d'th dimension   */ \
            ii = ii / dim[d]; \
            a_i += i_d * (a_str[d]/sizeof(DTYPE)); /* increment our a and z pointers by i_d elements */ \
            z_i += i_d * (z_str[d]/sizeof(DTYPE)); \
        } \
        z_i[0] = F(a_i[0]); \
    } \
}

template<typename T> __device__ T unary_copy(T a) { return a; }
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_float, unary_copy<float>, float)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_double, unary_copy<double>, double)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_uint8, unary_copy<uint8_t>, uint8_t)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_int8, unary_copy<int8_t>, int8_t)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_uint16, unary_copy<uint16_t>, uint16_t)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_int16, unary_copy<int16_t>, int16_t)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_uint32, unary_copy<uint32_t>, uint32_t)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_int32, unary_copy<int32_t>, int32_t)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_uint64, unary_copy<uint64_t>, uint64_t)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_int64, unary_copy<int64_t>, int64_t)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_complex64, unary_copy<npy_complex64>, npy_complex64)
decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_copy_complex128, unary_copy<npy_complex128>, npy_complex128)

//template<typename T> __device__ T unary_exp(T a) { return exp(a); }
//decl_k_elemwise_unary_rowmajor(k_elemwise_unary_rowmajor_exp, unary_exp<float>)

template<typename T>
static __global__ void k_copy_1d(const int N, const T * x, const ssize_t sx, T * y, const ssize_t sy)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += gridDim.x*blockDim.x) {
        y[i*sy] = x[i*sx];
    }
}

//copy from other into self
//don't allocated memory
int
PyGpuNdArray_CopyFromPyGpuNdArray(PyGpuNdArrayObject * self, PyGpuNdArrayObject * other, bool unbroadcast)
{
    DPRINTF("PyGpuNdArray_CopyFromPyGpuNdArray start nd=%d\n", PyGpuNdArray_NDIM(self));
    assert(PyGpuNdArray_TYPE(self) == PyGpuNdArray_TYPE(other));
    assert(PyGpuNdArray_ISWRITEABLE(self));
    //standard elemwise size checks
    if (PyGpuNdArray_NDIM(self) == -1) {
        PyErr_SetString(PyExc_TypeError, "can't copy into un-initialized PyGpuNdArrayObject");
        return -1;
    }
    if (PyGpuNdArray_NDIM(self) != PyGpuNdArray_NDIM(other)) {
        PyErr_Format(PyExc_NotImplementedError, "PyGpuNdArray_CopyFromPyGpuNdArray: need same number of dims. destination nd=%d, source nd=%d. No broadcasting implemented.", PyGpuNdArray_NDIM(self), PyGpuNdArray_NDIM(other));
        return -1;
    }
    //standard elemwise dim checks (also compute total size)
    unsigned int size = 1;
    unsigned int size_source = 1;
    for (int i = 0; i< PyGpuNdArray_NDIM(self); ++i) {
        if ((PyGpuNdArray_DIMS(self)[i] != PyGpuNdArray_DIMS(other)[i])
            && (1!=PyGpuNdArray_DIMS(other)[i] || !unbroadcast) ) {
          PyErr_Format(PyExc_ValueError, "need same dimensions for dim %d, destination=%ld, source=%ld",
                       i, PyGpuNdArray_DIMS(self)[i], PyGpuNdArray_DIMS(other)[i]);
            return -1;
        }
        size *= (unsigned int) PyGpuNdArray_DIMS(self)[i];
        size_source *= (unsigned int) PyGpuNdArray_DIMS(other)[i];
    }
    if (0 == size) {
        return 0; //nothing to copy, we're done.
    }

    //cublas don't support negative stride
    bool pos_stride = true;
    for (int i = 0; i < PyGpuNdArray_NDIM(other); ++i)
        if (PyGpuNdArray_STRIDE(other,i)<0)
            pos_stride = false;

    void * other_data = PyGpuNdArray_DATA(other) + PyGpuNdArray_OFFSET(other);
    void * self_data = PyGpuNdArray_DATA(self) + PyGpuNdArray_OFFSET(self);

    //Try to transfer with cublas(we suppose it is faster)
    if (PyGpuNdArray_ISCONTIGUOUS(self) && PyGpuNdArray_ISCONTIGUOUS(other) &&
	size == size_source && PyGpuNdArray_TYPE(self) == NPY_FLOAT32 &&
        pos_stride
        ) {
        cublasScopy(size, (float*) other_data, 1, (float*) self_data, 1);
        CNDA_THREAD_SYNC;
        if (CUBLAS_STATUS_SUCCESS != cublasGetError()) {
            PyErr_SetString(PyExc_RuntimeError, "Error copying memory");
            return -1;
        }

	DPRINTF("PyGpuNdArray_CopyFromPyGpuNdArray: cublasScopy end\n");
        return 0;
    }
    if (PyGpuNdArray_ISCONTIGUOUS(self) && PyGpuNdArray_ISCONTIGUOUS(other) &&
	size == size_source && PyGpuNdArray_TYPE(self) == NPY_FLOAT64 &&
        pos_stride) {
        cublasDcopy(size, (double*) other_data, 1, (double*) self_data, 1);
        CNDA_THREAD_SYNC;
        if (CUBLAS_STATUS_SUCCESS != cublasGetError()) {
            PyErr_SetString(PyExc_RuntimeError, "Error copying memory");
            return -1;
        }
	DPRINTF("PyGpuNdArray_CopyFromPyGpuNdArray cublasDcopy end\n");
        return 0;
    }

    //TODO: rewrite these copy operations to be more efficient
    //      See, for example the transpose example in the cuda_sdk.
    switch (PyGpuNdArray_NDIM(self)) {
        case 0: // scalar
            {
                // THIS CASE SHOULD NEVER HAPPEN BECAUSE SCALARS ARE ALWAYS C CONTIGUOUS
                assert(0);
            }; break;
        case 1: // vector
            {
                assert(PyGpuNdArray_ISALIGNED(self));
                assert(PyGpuNdArray_ISALIGNED(other));
                DPRINTF("PyGpuNdArray_CopyFromPyGpuNdArray: Copying non-contiguous vector\n");
                unsigned int n_blocks = min(size, (unsigned int)NUM_VECTOR_OP_BLOCKS);
                unsigned int n_threads = min(ceil_intdiv(size, n_blocks), (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);

		if (PyGpuNdArray_TYPE(self) == NPY_FLOAT32) {
                    const int elsize = sizeof(float);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (float*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (float*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_FLOAT64) {
                    const int elsize = sizeof(double);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (double*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (double*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_INT8) {
                    const int elsize = sizeof(int8_t);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (int8_t*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (int8_t*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_INT16) {
                    const int elsize = sizeof(int16_t);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (int16_t*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (int16_t*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_INT32) {
                    const int elsize = sizeof(int32_t);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (int32_t*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (int32_t*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_INT64) {
                    const int elsize = sizeof(int64_t);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (int64_t*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (int64_t*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_UINT8) {
                    const int elsize = sizeof(uint8_t);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (uint8_t*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (uint8_t*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_UINT16) {
                    const int elsize = sizeof(uint16_t);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (uint16_t*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (uint16_t*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_UINT32) {
                    const int elsize = sizeof(uint32_t);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (uint32_t*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (uint32_t*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_UINT64) {
                    const int elsize = sizeof(uint64_t);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (uint64_t*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (uint64_t*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_COMPLEX64) {
                    const int elsize = sizeof(npy_complex64);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (npy_complex64*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (npy_complex64*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else if (PyGpuNdArray_TYPE(self) == NPY_COMPLEX128) {
                    const int elsize = sizeof(npy_complex128);
                    k_copy_1d<<<n_blocks, n_threads>>>(size,
                                                       (npy_complex128*)other_data,
                                                       PyGpuNdArray_STRIDES(other)[0]/elsize,
                                                       (npy_complex128*)self_data,
                                                       PyGpuNdArray_STRIDES(self)[0]/elsize);
		} else {
		  PyErr_Format(PyExc_NotImplementedError, "PyGpuNdArray_CopyFromPyGpuNdArray: Don't implement copy for this dtype\n");
		  return -1;
		}

                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err) {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s. (n_blocks=%i, n_threads_per_block=%i)\n", "k_copy_1d", cudaGetErrorString(err), n_blocks, n_threads);
                    return -1;
                }
            }; break;
        default:
            {
                assert (cudaSuccess == cudaGetLastError());
                assert(PyGpuNdArray_ISALIGNED(self));
                assert(PyGpuNdArray_ISALIGNED(other));

                DPRINTF("PyGpuNdArray_CopyFromPyGpuNdArray: Copying with default version unbroadcast=%d\n", unbroadcast);
                // Identigy the dim of the output memory.
                PyGpuNdArrayObject * cuda_dims = other;
                if(unbroadcast)
                    cuda_dims = self;

                // Move the dim and strides information on the gpu memory
                int ndim = PyGpuNdArray_NDIM(other);
                void * strides_dev = device_malloc(sizeof(ssize_t)*ndim*3);
                ssize_t * strides_dev_p = (ssize_t *) strides_dev;
                cudaError_t err = cudaMemcpy(strides_dev, PyGpuNdArray_DIMS(cuda_dims), ndim*sizeof(ssize_t),cudaMemcpyHostToDevice);
                if (err != cudaSuccess){
                    PyErr_Format(PyExc_RuntimeError, "Cuda error when copying memory1: %s", cudaGetErrorString(err));
                    return -1;
                }
                err = cudaMemcpy((void*)(strides_dev_p+ndim), PyGpuNdArray_STRIDES(other), ndim*sizeof(ssize_t),cudaMemcpyHostToDevice);
                if (err != cudaSuccess){
                    PyErr_Format(PyExc_RuntimeError, "Cuda error when copying memory2: %s", cudaGetErrorString(err));
                    return -1;
                }
                err = cudaMemcpy((void*)(strides_dev_p+(ndim*2)), PyGpuNdArray_STRIDES(self), ndim*sizeof(ssize_t), cudaMemcpyHostToDevice);
                if (err != cudaSuccess){
                    PyErr_Format(PyExc_RuntimeError, "Cuda error when copying memory3: %s", cudaGetErrorString(err));
                    return -1;
                }
                void * strides_host = malloc(sizeof(ssize_t)*ndim*3);
                err = cudaMemcpy(strides_host, strides_dev, ndim*3*sizeof(ssize_t),cudaMemcpyDeviceToHost);
                if (err != cudaSuccess){
                    PyErr_Format(PyExc_RuntimeError, "Cuda error when copying memory4: %s", cudaGetErrorString(err));
                    return -1;
                }
#ifdef DEBUG
                for(int i=0;i<3*ndim;i++)
                    DPRINTF(" %ld", ((ssize_t *)strides_host)[i]);
                DPRINTF("\n");
#endif
                CNDA_THREAD_SYNC;
                if(cudaSuccess != cudaGetLastError()){
                    PyErr_Format(PyExc_NotImplementedError, "PyGpuNdArray_CopyFromPyGpuNdArray: error before copy\n");
		  return -1;
                }

                // call worker routine
                unsigned int n_blocks = min(size, (unsigned int)NUM_VECTOR_OP_BLOCKS);
                unsigned int threads_per_block = min(ceil_intdiv(size, n_blocks), (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);

		if ( PyGpuNdArray_TYPE(self) == NPY_FLOAT32) {
                    k_elemwise_unary_rowmajor_copy_float<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const float*)other_data,
                        strides_dev_p+ndim,
                        (float*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_FLOAT64) {
                    k_elemwise_unary_rowmajor_copy_double<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const double*)other_data,
                        strides_dev_p+ndim,
                        (double*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_INT8) {
                    k_elemwise_unary_rowmajor_copy_int8<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const int8_t*)other_data,
                        strides_dev_p+ndim,
                        (int8_t*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_INT16) {
                    k_elemwise_unary_rowmajor_copy_int16<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const int16_t*)other_data,
                        strides_dev_p+ndim,
                        (int16_t*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_INT32) {
                    k_elemwise_unary_rowmajor_copy_int32<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const int32_t*)other_data,
                        strides_dev_p+ndim,
                        (int32_t*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_INT64) {
                    k_elemwise_unary_rowmajor_copy_int64<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const int64_t*)other_data,
                        strides_dev_p+ndim,
                        (int64_t*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_UINT8) {
                    k_elemwise_unary_rowmajor_copy_uint8<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const uint8_t*)other_data,
                        strides_dev_p+ndim,
                        (uint8_t*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_UINT16) {
                    k_elemwise_unary_rowmajor_copy_uint16<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const uint16_t*)other_data,
                        strides_dev_p+ndim,
                        (uint16_t*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_UINT32) {
                    k_elemwise_unary_rowmajor_copy_uint32<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const uint32_t*)other_data,
                        strides_dev_p+ndim,
                        (uint32_t*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_UINT64) {
                    k_elemwise_unary_rowmajor_copy_uint64<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const uint64_t*)other_data,
                        strides_dev_p+ndim,
                        (uint64_t*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_COMPLEX64) {
                    k_elemwise_unary_rowmajor_copy_complex64<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const npy_complex64*)other_data,
                        strides_dev_p+ndim,
                        (npy_complex64*) self_data,
                        strides_dev_p+(ndim*2));
		} else if ( PyGpuNdArray_TYPE(self) == NPY_COMPLEX128) {
                    k_elemwise_unary_rowmajor_copy_complex128<<<n_blocks, threads_per_block>>>(
                        size,
                        (unsigned int)ndim,
                        strides_dev_p,
                        (const npy_complex128*)other_data,
                        strides_dev_p+ndim,
                        (npy_complex128*) self_data,
                        strides_dev_p+(ndim*2));
		} else {
		  PyErr_Format(PyExc_NotImplementedError, "PyGpuNdArray_CopyFromPyGpuNdArray: Don't implement copy for this dtype\n");
		  return -1;
		}
                CNDA_THREAD_SYNC;
                err = cudaGetLastError();
                if( cudaSuccess != err) {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s. (n_blocks=%i, n_threads_per_block=%i)\n", "k_elemwise_unary_rowmajor_copy", cudaGetErrorString(err), n_blocks, threads_per_block);
                    return -1;
                }
                device_free(strides_dev);
                free(strides_host);
            }
    };
    // Set flags
    if (false && PyGpuNdArray_NDIM(self) == 0) {
        //Numpy 1.4.1 is not consistent here
        //When we create a new numpy ndarray of 0 dim, it is not f contiguous
        //But when we take a subtensor that is of 0 dim, it is f contiguous!
        //We make as them for now...
        PyGpuNdArray_FLAGS(self) &= ~NPY_F_CONTIGUOUS;
        PyGpuNdArray_FLAGS(self) |= NPY_C_CONTIGUOUS;
    } else {
        if (PyGpuNdArray_is_c_contiguous(self)) {
            PyGpuNdArray_FLAGS(self) |= NPY_C_CONTIGUOUS;
        } else {
            PyGpuNdArray_FLAGS(self) &= ~NPY_C_CONTIGUOUS;
        }
        if (PyGpuNdArray_is_f_contiguous(self)) {
            PyGpuNdArray_FLAGS(self) |= NPY_F_CONTIGUOUS;
        } else {
            PyGpuNdArray_FLAGS(self) &= ~NPY_F_CONTIGUOUS;
        }
    }

    DPRINTF("PyGpuNdArray_CopyFromPyGpuNdArray end\n");
    return 0;
}

int PyGpuMemcpy(void * dst, const void * src, int dev_offset, size_t bytes, 
                PyGpuTransfert direction){
    DPRINTF("PyGpuMemcpy: start\n");
    cudaMemcpyKind dir;
    const char * ssrc;
    const char * ddst;
    if (direction == PyGpuDeviceToHost){
        dir = cudaMemcpyDeviceToHost;
        ssrc = (char*)src+dev_offset;
        ddst = (char*)dst;
    } else if (direction == PyGpuHostToDevice) {
        dir = cudaMemcpyHostToDevice;
        ssrc = (char*)src;
        ddst = (char*)dst + dev_offset;
    } else {
        PyErr_Format(PyExc_ValueError,
                     "PyGpuMemcpy: Received wrong direction %d!\n",
                     direction);
        return -1;
    }
    cudaError_t err = cudaMemcpy((void*)ddst, (void*)ssrc, bytes, dir);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        PyErr_Format(PyExc_RuntimeError, "PyGpuMemcpy: cudaMemcpy: error copying data to host (%s)",
                     cudaGetErrorString(err));
        return -1;
    }
    DPRINTF("PyGpuMemcpy: end\n");
    return 0;
}

int PyGpuMemset(void * dst, int data, size_t bytes){
    DPRINTF("PyGpuMemset: start\n");
    cudaError_t err = cudaMemset(dst, data, bytes);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        PyErr_Format(PyExc_MemoryError, "PyGpuMemset: Error memsetting %ld bytes of device memory(%s). %p",
                     bytes, cudaGetErrorString(err), PyGpuNdArray_DATA(dst));
    DPRINTF("PyGpuMemset: end error\n");
        return -1;
    }
    DPRINTF("PyGpuMemset: end\n");
    return 0;
}

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
