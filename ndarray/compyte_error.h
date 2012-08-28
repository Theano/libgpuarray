#ifndef COMPYTE_ERROR_H
#define COMPYTE_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

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
  /* Add more error types if needed */
  /* Don't forget to sync with Gpu_error() */
};

const char *compyte_error_str(int err);

#ifdef __cplusplus
}
#endif

#endif
