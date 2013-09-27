#ifndef COMPYTE_ERROR_H
#define COMPYTE_ERROR_H
/** \file error.h
 *  \brief Error functions.
 */

#include <compyte/config.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

/**
 * List of all the possible error codes.
 */
enum ga_error {
  GA_NO_ERROR = 0,
  GA_MEMORY_ERROR,
  GA_VALUE_ERROR,
  GA_IMPL_ERROR, /* call buffer_error() for more details */
  GA_INVALID_ERROR,
  GA_UNSUPPORTED_ERROR,
  GA_SYS_ERROR, /* look at errno for more details */
  GA_RUN_ERROR,
  GA_DEVSUP_ERROR,
  GA_READONLY_ERROR,
  GA_WRITEONLY_ERROR,
  GA_BLAS_ERROR,
  GA_UNALIGNED_ERROR,
  /* Add more error types if needed, but at the end */
  /* Don't forget to sync with Gpu_error() */
};

/**
 * Returns a user-readable description for most error codes.
 *
 * Some errors only happen in a context and in those cases Gpu_error()
 * will provide more details as to the reason for the error.
 */
COMPYTE_PUBLIC const char *compyte_error_str(int err);

#ifdef __cplusplus
}
#endif

#endif
