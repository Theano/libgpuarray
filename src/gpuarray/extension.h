#ifndef GPUARRAY_EXTENSIONS_H
#define GPUARRAY_EXTENSIONS_H
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

/* Keep in sync with the flags in gpuarray_buffer_cuda.c */
#define GPUARRAY_CUDA_CTX_NOFREE 0x1

/**
 * Obtain a function pointer for an extension.
 *
 * \returns A function pointer or NULL if the extension was not found.
 */
GPUARRAY_PUBLIC void * gpuarray_get_extension(const char *name);

#ifdef __cplusplus
}
#endif

#endif
