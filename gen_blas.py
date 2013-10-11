import sys

from mako import exceptions
from mako.template import Template

TYPE_TO_CHARS = dict(float='s', double='d')

class Argument(object):
    def __init__(self, name, const=True, pydef=None):
        self.name = name
        self.const = const
        self.pydef = pydef

    def ismatrix(self):
        return False

    def isarray(self):
        return False

class ScalarArg(Argument):
    def format_as_arg(self, ctype, _=None):
        const = ""
        if self.const:
            const += "const "
        return const + self.ctype + ' ' + self.name

    format_simple_arg = format_as_arg

    def format_simple_call(self, _):
        return self.name

    def format_as_call(self):
        return self.tf_macro + '(' + self.name + ')'

class scalar(ScalarArg):
    tf_macro = 'SCAL'
    def format_as_arg(self, ctype, _=None):
        const = ""
        if self.const:
            const += "const "
        return const + ctype + ' ' + self.name

    format_simple_arg = format_as_arg

    def format_simple_call(self, _):
        return self.name

    def format_as_callarg(self, ctype):
        return '(' + ctype + ')' + self.name

    def format_pyarg(self):
        res = "double "+self.name
        if self.pydef is not None:
            res += '='+self.pydef
        return res
    
class size(ScalarArg):
    tf_macro = 'SZ'
    ctype = 'size_t'

    def format_as_callarg(self, ctype):
        return self.name.lower()

class inc(ScalarArg):
    tf_macro = ''
    ctype = 'int'

    def format_as_callarg(self, ctype):
        return self.name[-1] + 'p->strides[0] / elsize'

class trans(ScalarArg):
    tf_macro = 'TRANS'
    ctype = 'cb_transpose'

    def format_as_callarg(self, ctype):
        return self.name

class ArrayArg(Argument):
    def __init__(self, name, output=False):
        pydef = None
        if output:
            pydef = 'None'
        Argument.__init__(self, name, pydef=pydef)
        self.isoutput = output

    def isarray(self):
        return True

    def format_simple_arg(self, ctype, arraytype):
        return arraytype + self.name

    def format_simple_call(self, arraypat):
        return arraypat % (self.name,)

    def format_as_arg(self, ctype):
        return 'gpudata *' + self.name + ', const size_t off' + self.name

    def format_as_call(self):
        return 'ARRAY(' + self.name + ', dtype)'

    def format_as_callarg(self, ctype):
        return self.name + 'p->data, ' + self.name + 'p->offset / elsize'

    def format_pyarg(self):
        res = 'GpuArray '+self.name
        if self.pydef is not None:
            res += '='+self.pydef
        return res

class matrix(ArrayArg):
    def ismatrix(self):
        return True

class vector(ArrayArg):
    pass

class BlasOp(object):
    def __init__(self, name, types, arguments, check_dims, py_decls,
                 py_ensure_output):
        self.name = name
        self.types = types
        self.arguments = arguments
        self.check_dims = check_dims
        self.py_decls = py_decls
        self.py_ensure_output = py_ensure_output
        self.has_order = any(arg.ismatrix() for arg in self.arguments)

    def matrix_args(self):
        return [arg for arg in self.arguments if arg.ismatrix()]

    def array_args(self):
        return [arg for arg in self.arguments if arg.isarray()]

    def simple_args(self):
        return [arg for arg in self.arguments if (arg.isarray() or type(arg) is scalar or type(arg) is trans)]

    def py_args(self):
        return [arg for arg in self.arguments if (arg.isarray() or type(arg) is scalar)]

    def args_per_class(self, cls):
        return [arg for arg in self.arguments if type(arg) is cls]

    def format_arguments(self, ctype):
        order = ''
        if self.has_order:
            order = 'const cb_order order, '
        return order + ', '.join(arg.format_as_arg(ctype) for arg in self.arguments)
    def format_blas_args(self, ctype):
        return ', '.join(arg.format_as_callarg(ctype) for arg in self.arguments)

    def format_call_args(self):
        order = ''
        if self.has_order:
            order = 'ORDER '
        return order + ', '.join(arg.format_as_call() for arg in self.arguments)

    def format_simple_args(self, ctype, arraytype):
        return ', '.join(arg.format_simple_arg(ctype, arraytype) for arg in self.simple_args())

    def format_simple_call(self, arraypat):
        return ', '.join(arg.format_simple_call(arraypat) for arg in self.simple_args())

    def format_pyargs(self):
        l = [arg.format_pyarg() for arg in self.py_args()]
        l.extend('trans_'+t.name[-1].lower()+'=False' for t in self.args_per_class(trans))
        l.extend('overwrite_'+a.name.lower()+'=False' for a in self.array_args() if a.isoutput)
        return ', '.join(l)

class Dtype(object):
    def __init__(self, name, c):
        self.name = name
        self.c = c
float = Dtype('float', 's')
double = Dtype('double', 'd')

check_dims_gemv = """
  if (transA == cb_no_trans) {
    m = A->dimensions[0];
    n = A->dimensions[1];
  } else {
    m = A->dimensions[1];
    n = A->dimensions[0];
  }

  if (Y->dimensions[0] != m || X->dimensions[0] != n)
    return GA_VALUE_ERROR;

  m = A->dimensions[0];
  n = A->dimensions[1];
"""

py_decls_gemv = "cdef size_t Yshp"

py_ensure_output_gemv = """
    if A.ga.nd != 2:
        raise TypeError, "A is not a matrix"
    if transA == cb_no_trans:
        Yshp = A.ga.dimensions[0]
    else:
        Yshp = A.ga.dimensions[1]
    if Y is None:
        if beta != 0.0:
            raise ValueError, "Y not provided and beta != 0"
        Y = pygpu_empty(1, &Yshp, A.ga.typecode, GA_ANY_ORDER, A.context, None)
        overwrite_y = True
"""

check_dims_gemm = """
  if (transA == cb_no_trans) {
    m = A->dimensions[0];
    k = A->dimensions[1];
  } else {
    m = A->dimensions[1];
    k = A->dimensions[0];
  }

  if (transB == cb_no_trans) {
    n = B->dimensions[1];
    if (B->dimensions[0] != k)
      return GA_VALUE_ERROR;
  } else {
    n = B->dimensions[0];
    if (B->dimensions[1] != k)
      return GA_VALUE_ERROR;
  }

  if (C->dimensions[0] != m || C->dimensions[1] != n)
    return GA_VALUE_ERROR;
"""

py_decls_gemm = "cdef size_t[2] Cshp"

py_ensure_output_gemm = """
    if A.ga.nd != 2:
        raise TypeError, "A is not a matrix"
    if B.ga.nd != 2:
        raise TypeError, "B is not a matrix"
    if transA == cb_no_trans:
        Cshp[0] = A.ga.dimensions[0]
    else:
        Cshp[0] = A.ga.dimensions[1]
    if transB == cb_no_trans:
        Cshp[1] = B.ga.dimensions[1]
    else:
        Cshp[0] = B.ga.dimensions[0]
    if C is None:
        if beta != 0.0:
            raise ValueError, "C not provided and beta != 0"
        C = pygpu_empty(2, Cshp, A.ga.typecode, GA_ANY_ORDER, A.context, None)
        overwrite_c = True
"""

OPS = [
    BlasOp('gemv', (float, double),
           [trans('transA'), size('M'), size('N'), scalar('alpha'),
            matrix('A'), size('lda'), vector('X'), inc('incX'),
            scalar('beta', pydef='0.0'), vector('Y', output=True),
            inc('incY')],
           check_dims=check_dims_gemv, py_decls=py_decls_gemv,
           py_ensure_output=py_ensure_output_gemv)
]

# having two (or three) layers of backslash-interpreting can be pretty
# confusing if you want to output a backslash.  Add to that mako's
# parsers bugs around backslahes and the 'pass a parameter that is a
# backslash string' approach seems the most likely to work on a range
# of versions.
BS = '\\'

GENERIC_TMPL = Template("""
/* This file is generated by gen_blas.py in the root of the distribution */
#if !defined(FETCH_CONTEXT) || !defined(PREFIX) || !defined(ARRAY) || !defined(POST_CALL)
#error "required macros not defined"
#endif

#ifdef ORDER
% for op in ops:
#ifndef PREP_ORDER_${op.name.upper()}
#define PREP_ORDER_${op.name.upper()}
#endif
#ifndef HANDLE_ORDER_${op.name.upper()}
#define HANDLE_ORDER_${op.name.upper()}
#endif
% endfor
#else
#define ORDER
#endif

#ifndef INIT_ARGS
#define INIT_ARGS
#endif

#ifndef TRAIL_ARGS
#define TRAIL_ARGS
#endif

#ifndef SZ
#define SZ(a) a
#endif

#ifndef TRANS
#define TRANS(t) t
#endif

#ifndef SCAL
#define SCAL(s) s
#endif

#ifndef FUNC_INIT
#define FUNC_INIT
#endif

#ifndef FUNC_FINI
#define FUNC_FINI
#endif

#define __GLUE(part1, part2) __GLUE_INT(part1, part2)
#define __GLUE_INT(part1, part2) part1 ## part2

% for op in ops:
#define ${op.name.upper()}(dtype, typec, TYPEC)			    ${bs}
  static int typec ## ${op.name}(${op.format_arguments('dtype')}) { ${bs}
    FETCH_CONTEXT(${op.array_args()[0].name});			    ${bs}
    FUNC_DECLS;							    ${bs}
    PREP_ORDER_${op.name.upper()};		                    ${bs}
								    ${bs}
    HANDLE_ORDER_${op.name.upper()};	                            ${bs}
    FUNC_INIT;							    ${bs}
								    ${bs}
% for a in op.array_args():
    ARRAY_INIT(${a.name});					    ${bs}
% endfor
								    ${bs}
    PRE_CALL __GLUE(PREFIX(typec, TYPEC), ${op.name})(INIT_ARGS ${op.format_call_args()} TRAIL_ARGS); ${bs}
    POST_CALL;							    ${bs}
								    ${bs}
% for a in op.array_args():
    ARRAY_FINI(${a.name});					    ${bs}
% endfor
    FUNC_FINI;							    ${bs}
								    ${bs}
    return GA_NO_ERROR;						    ${bs}
  }

% for type in op.types:
${op.name.upper()}(${type.name}, ${type.c}, ${type.c.upper()})
% endfor
% endfor

COMPYTE_LOCAL compyte_blas_ops __GLUE(NAME, _ops) = {
  setup,
  teardown,
% for op in ops:
 % for type in op.types:
  ${type.c}${op.name},
 % endfor
% endfor
};
""")

BUFFERBLAS_TMPL = Template("""
/* This file is generated by gen_blas.py in the root of the distribution */
#ifndef COMPYTE_BUFFER_BLAS_H
#define COMPYTE_BUFFER_BLAS_H

#include <compyte/buffer.h>
#include <compyte/config.h>

typedef enum _cb_order {
  cb_row,
  cb_column
} cb_order;

#define cb_c cb_row
#define cb_fortran cb_column

typedef enum _cb_side {
  cb_left,
  cb_right
} cb_side;

typedef enum _cb_transpose {
  cb_no_trans,
  cb_trans,
  cb_conj_trans
} cb_transpose;

typedef enum _cb_uplo {
  cb_upper,
  cb_lower
} cb_uplo;

typedef struct _compyte_blas_ops {
  int (*setup)(void *ctx);
  void (*teardown)(void *ctx);
% for op in ops:
 % for type in op.types:
  int (*${type.c}${op.name})(${op.format_arguments(type.name)});
 % endfor
% endfor
} compyte_blas_ops;

#endif
""")

BLAS_TMPL = Template("""
/* This file is generated by gen_blas.py in the root of the distribution */
#ifndef COMPYTE_BLAS_H
#define COMPYTE_BLAS_H

#include <compyte/buffer_blas.h>
#include <compyte/array.h>

% for op in ops:
COMPYTE_PUBLIC int GpuArray_r${op.name}(${op.format_simple_args('double', 'GpuArray *')},
                                        int nocopy);
 % for type in op.types:
#define GpuArray_${type.c}${op.name} GpuArray_r${op.name}
 % endfor
% endfor
#endif
""")

ARRAYBLAS_TMPL = Template("""
/* This file is generated by gen_blas.py in the root of the distribution */
#include "compyte/blas.h"
#include "compyte/buffer_blas.h"
#include "compyte/types.h"
#include "compyte/util.h"
#include "compyte/error.h"

% for op in ops:
int GpuArray_r${op.name}(${op.format_simple_args('double', 'GpuArray *')},
                         int nocopy) {
 % for a in op.array_args():
  GpuArray *${a.name}p = ${a.name};
  % if not a.isoutput:
  GpuArray copy${a.name};
  % endif
 % endfor
  compyte_blas_ops *blas;
  void *ctx;
  size_t elsize;
  size_t m, n, k;
 % for m in op.matrix_args():
  size_t ld${m.name.lower()};
 % endfor
  cb_order o;
  int err;
<% firsta = op.array_args()[0].name %>

  if (${firsta}->typecode != GA_FLOAT && ${firsta}->typecode != GA_DOUBLE)
    return GA_INVALID_ERROR;

<%
def ndcond(ary):
    if ary.ismatrix(): 
        return ary.name + "->nd != 2"
    else:
        return ary.name + "->nd != 1"

def typecond(first, ary):
    return ary.name + "->typecode != " + first + "->typecode"

def aligncond(a):
    return "!(" + a.name + "->flags & GA_ALIGNED)"
%>
  if (${'||'.join(ndcond(a) for a in op.array_args())} ||
      ${'||'.join(typecond(firsta, a) for a in op.array_args())})
    return GA_VALUE_ERROR;

  if (${'||'.join(aligncond(a) for a in op.array_args())})
    return GA_UNALIGNED_ERROR;

  ${op.check_dims}

  elsize = compyte_get_elsize(${firsta}->typecode);

% for a in op.array_args():
 % if a.ismatrix():
  if (!GpuArray_ISONESEGMENT(${a.name})) {
   % if a.isoutput:
    err = GA_VALUE_ERROR;
    goto cleanup;
   % else:
    if (nocopy)
      return GA_COPY_ERROR;
    else {
      err = GpuArray_copy(&copy${a.name}, ${a.name}, GA_F_ORDER);
      if (err != GA_NO_ERROR)
	goto cleanup;
      ${a.name}p = &copy${a.name};
    }
   % endif
  }
 % else:
  if (${a.name}->strides[0] < 0) {
  % if a.isoutput:
    err = GA_VALUE_ERROR;
    goto cleanup;
  % else:
    if (nocopy)
      return GA_COPY_ERROR;
    else {
      err = GpuArray_copy(&copy${a.name}, ${a.name}, GA_ANY_ORDER);
      if (err != GA_NO_ERROR)
	goto cleanup;
      ${a.name}p = &copy${a.name};
    }
   % endif
  }
 % endif
% endfor

% for m in op.matrix_args():
  if (${m.name}p->flags & GA_F_CONTIGUOUS) {
    o = cb_fortran;
    ld${m.name.lower()} = ${m.name}p->dimensions[0];
  } else if (${m.name}p->flags & GA_C_CONTIGUOUS) {
    o = cb_c;
    ld${m.name.lower()} = ${m.name}p->dimensions[1];
  } else {
    /* Might be worth looking at making degenerate matrices (1xn) work here. */
    err = GA_VALUE_ERROR;
    goto cleanup;
  }
% endfor

  err = ${firsta}p->ops->property(NULL, ${firsta}p->data, NULL, GA_BUFFER_PROP_CTX, &ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;
  err = ${firsta}p->ops->property(ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS, &blas);
  if (err != GA_NO_ERROR)
    goto cleanup;

  err = blas->setup(ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;

  if (${firsta}p->typecode == GA_FLOAT)
    err = blas->s${op.name}(o, ${op.format_blas_args('float')});
  else
    err = blas->d${op.name}(o, ${op.format_blas_args('double')});

 cleanup:
% for a in op.array_args():
 % if not a.isoutput:
  if (${a.name}p == &copy${a.name})
    GpuArray_clear(&copy${a.name});
 % endif
% endfor
  return err;
}
% endfor
""")

BLAS_PYX_TMPL = Template("""
# This file is generated by gen_blas.py in the root of the distribution
from pygpu.gpuarray import GpuArrayException
from pygpu.gpuarray cimport (_GpuArray, GpuArray, GA_NO_ERROR, GpuArray_error,
                             pygpu_copy, pygpu_empty, GA_ANY_ORDER, GA_F_ORDER,
                             GpuArray_ISONESEGMENT)

cdef extern from "compyte/buffer_blas.h":
    ctypedef enum cb_transpose:
        cb_no_trans,
        cb_trans,
        cb_conj_trans

cdef extern from "compyte/blas.h":
% for op in ops:
    int GpuArray_r${op.name}(${op.format_simple_args('double', '_GpuArray *')},
                             int nocopy)
% endfor

% for op in ops:
cdef blas_r${op.name}(${op.format_simple_args('double', 'GpuArray ')},
                      bint nocopy):
    cdef int err
    err = GpuArray_r${op.name}(${op.format_simple_call('&%s.ga')}, nocopy);
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&${op.array_args()[0].name}.ga, err), err)

cdef api GpuArray pygpu_blas_r${op.name}(${op.format_simple_args('double', 'GpuArray ')}):
    blas_r${op.name}(${op.format_simple_call('%s')}, 0)
<%
outas = []
for a in op.array_args():
    if a.isoutput:
        outas.append(a.name)
assert len(outas) is not 0
outa = ', '.join(outas)
%>
    return ${outa}
% endfor

% for op in ops:
def ${op.name}(${op.format_pyargs()}):
 % for m in op.matrix_args():
  % if not m.isoutput:
    cdef cb_transpose trans${m.name}
  % endif
 % endfor
    ${op.py_decls}

 % for m in op.matrix_args():
  % if not m.isoutput:
    if trans_${m.name.lower()}:
        trans${m.name} = cb_trans
    else:
        trans${m.name} = cb_no_trans
  % endif
 % endfor

    ${op.py_ensure_output}

    if not overwrite_y:
        Y = pygpu_copy(Y, GA_ANY_ORDER)
    return pygpu_blas_r${op.name}(${op.format_simple_call('%s')})
% endfor
""")

try:
    generic = GENERIC_TMPL.render(ops=OPS, bs=BS)
    bufferblas = BUFFERBLAS_TMPL.render(ops=OPS)
    blas = BLAS_TMPL.render(ops=OPS)
    arrayblas = ARRAYBLAS_TMPL.render(ops=OPS)
    blas_pyx = BLAS_PYX_TMPL.render(ops=OPS)
except Exception:
    print exceptions.text_error_template().render()
    sys.exit(1)

with open('src/generic_blas.inc.c', 'w') as f:
    f.write(generic)

with open('src/compyte/buffer_blas.h', 'w') as f:
    f.write(bufferblas)

with open('src/compyte/blas.h', 'w') as f:
    f.write(blas)

with open('src/compyte_array_blas.c', 'w') as f:
    f.write(arrayblas)

with open('pygpu/blas.pyx', 'w') as f:
    f.write(blas_pyx)
