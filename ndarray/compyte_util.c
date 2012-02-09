#include <assert.h>

#include <stdio.h>

#include "compyte_util.h"

compyte_type *compyte_get_type(int typecode) {
  if (typecode < GA_DELIM) {
    assert(scalar_types[typecode].typecode == typecode);
    return &scalar_types[typecode];
  } else {
    assert(vector_types[typecode-256].typecode = typecode);
    return &vector_types[typecode-256];
  }
}

size_t compyte_get_elsize(int typecode) {
  return compyte_get_type(typecode)->size;
}

int compyte_elem_perdim(char *strs[], unsigned int *count, unsigned int nd,
			const size_t *dims, const ssize_t *str,
			const char *id) {
  int i;
  
  if (nd > 0) {
    if (asprintf(&strs[*count], "int %si = i;", id) == -1)
      return -1;
    (*count)++;

    for (i = nd-1; i > 0; i--) {
      if (asprintf(&strs[*count], "%1$si = %1$si / %2$zu;"
		   "%1$s += (%1$si %% %2$zu) * %3$zd;",
		   id, dims[i], str[i]) == -1)
	return -1;
      (*count)++;
    }

    if (asprintf(&strs[*count], "%1$s += %1$si * %2$zd;", id, str[0]) == -1)
      return -1;
    (*count)++;
  }
  return 0;
}
