#define _CRT_SECURE_NO_WARNINGS
#include "compyte/error.h"

#include <string.h>
#include <errno.h>

const char *compyte_error_str(int err) {
  switch (err) {
  case GA_NO_ERROR:          return "No error";
  case GA_MEMORY_ERROR:      return "Out of memory";
  case GA_VALUE_ERROR:       return "Value invalid or out of range";
  case GA_IMPL_ERROR:        return "Unknown device error";
  case GA_INVALID_ERROR:     return "Invalid value or operation";
  case GA_UNSUPPORTED_ERROR: return "Unsupported operation";
  case GA_SYS_ERROR:         return strerror(errno);
  case GA_RUN_ERROR:         return "Could not execute helper program";
  case GA_DEVSUP_ERROR:      return "Device does not support operation";
  case GA_READONLY_ERROR:    return "Buffer is read-only";
  case GA_WRITEONLY_ERROR:   return "Buffer is write-only";
  case GA_BLAS_ERROR:        return "Error in BLAS call";
  default: return "Unknown GA error";
  }
}
