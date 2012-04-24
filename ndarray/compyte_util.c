#include <assert.h>

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

static inline ssize_t ssabs(ssize_t v) {
  return (v < 0 ? -v : v);
}

int compyte_elem_perdim(char *strs[], unsigned int *count, unsigned int nd,
			const size_t *dims, const ssize_t *str,
			const char *id, ssize_t elemsize) {
  int i;

  if (nd > 0) {
    if (asprintf(&strs[*count], "int %si = i;", id) == -1)
      return -1;
    (*count)++;

    for (i = nd-1; i > 0; i--) {
      assert(str[i]%elemsize == 0);
      if (asprintf(&strs[*count], "%s %c= ((%si %% %" SPREFIX "u) * %" SPREFIX
		   "d);%si = %si / %" SPREFIX "u;", id,
		   (str[i] < 0 ? '-' : '+'), id, dims[i],
		   ssabs(str[i]/elemsize), id, id, dims[i]) == -1)
	return -1;
      (*count)++;
    }
 
    assert(str[0]%elemsize == 0);
    if (asprintf(&strs[*count], "%s %c= (%si * %" SPREFIX "d);", id,
		 (str[0] < 0 ? '-' : '+'), id, ssabs(str[0]/elemsize)) == -1)
      return -1;
    (*count)++;
  }
  return 0;
}
