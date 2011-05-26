#include <pygpu_language.h>

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
    if(VERBOSE_ALLOC_FREE)
      fprintf(stderr, "allocated %li bytes of device memory (%s). %d already allocated, ptr: %p\n",
	      (long)size, cudaGetErrorString(err),_allocated_size,rval);
#else
    if(VERBOSE_ALLOC_FREE)
      fprintf(stderr, "allocated %li bytes of device memory (%s). ptr: %p\n",
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
    if(VERBOSE_ALLOC_FREE)
      fprintf(stderr, "freed %li bytes of device memory (%s). %d already allocated, ptr=%p\n", (long)total_freed, cudaGetErrorString(err),_allocated_size,ptr);
#endif
    return 0;
}



