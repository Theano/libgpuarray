#include <string.h>
#include <errno.h>

#include "compyte_buffer.h"
#include "compyte_error.h"

const char *Gpu_error(compyte_buffer_ops *o, int err) {
  if (err == GA_IMPL_ERROR)
    return o->buffer_error();
  else
    return compyte_error_str(err);
}
