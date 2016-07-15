#ifndef LIBGPUARRAY_GPUARRAY_EXTENSION_H
#define LIBGPUARRAY_GPUARRAY_EXTENSION_H
/** \file extension.h
 *  \brief Extensions access.
 */

#include <gpuarray/config.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

/* Keep in sync with the flags in private_cuda.h */
#define GPUARRAY_CUDA_CTX_NOFREE 0x10000000 /* DONTFREE */

#define GPUARRAY_CUDA_WAIT_READ  0x10000 /* CUDA_WAIT_READ */
#define GPUARRAY_CUDA_WAIT_WRITE 0x20000 /* CUDA_WAIT_WRITE */

/**
 * Obtain a function pointer for an extension.
 *
 * \returns A function pointer or NULL if the extension was not found.
 */
GPUARRAY_PUBLIC void * gpuarray_get_extension(const char *name);

#ifdef __cplusplus
}
#endif

#endif  // LIBGPUARRAY_GPUARRAY_EXTENSION_H
