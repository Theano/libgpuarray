#include <assert.h>

#include "private.h"
#include "compyte/util.h"
#include "compyte/error.h"

static compyte_type **custom_types = NULL;
static int n_types = 0;
static compyte_type no_type = {NULL, 0, 0, -1};

int compyte_register_type(compyte_type *t, int *ret) {
  compyte_type **tmp;
  tmp = realloc(custom_types, (n_types+1)*sizeof(*tmp));
  if (tmp == NULL) {
    if (ret) *ret = GA_SYS_ERROR;
    return -1;
  }
  custom_types = tmp;
  t->typecode = 512 + n_types;
  custom_types[n_types++] = t;
  return t->typecode;
}

const compyte_type *compyte_get_type(int typecode) {
  if (typecode <= GA_DELIM) {
    if (typecode < GA_NBASE)
      return &scalar_types[typecode];
    else
      return &no_type;
  } else if (typecode < GA_ENDVEC) {
    if (typecode < GA_NVEC)
      return &vector_types[typecode - 256];
    else
      return &no_type;
  } else {
    if ((typecode - 512) < n_types)
      return custom_types[typecode - 512];
    else
      return &no_type;
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
			const char *id) {
  int i;

  if (nd > 0) {
    if (asprintf(&strs[*count], "int %si = i;", id) == -1)
      return -1;
    (*count)++;

    for (i = nd-1; i > 0; i--) {
      if (asprintf(&strs[*count], "%s %c= ((%si %% %" SPREFIX "u) * "
                   "%" SPREFIX "d);%si = %si / %" SPREFIX "u;", id,
		   (str[i] < 0 ? '-' : '+'), id, dims[i],
		   ssabs(str[i]), id, id, dims[i]) == -1)
	return -1;
      (*count)++;
    }
 
    if (asprintf(&strs[*count], "%s %c= (%si * %" SPREFIX "d);", id,
		 (str[0] < 0 ? '-' : '+'), id, ssabs(str[0])) == -1)
      return -1;
    (*count)++;
  }
  return 0;
}
