#include <assert.h>

#include "compute_util.h"

compyte_type *compyte_get_type(int typecode) {
  if (typecode < GA_DELIM) {
    assert(scalar_types[typecode].typecode == typecode);
    return &scalar_types[typecode];
  } else {
    assert(vector_types[typecode-256].typecode = typecode);
    return &vector_types[typecode-256];
}

size_t compyte_get_elsize(int typecode) {
  return compyte_get_type(typecode)->size;
}
