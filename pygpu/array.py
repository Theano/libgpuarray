from __future__ import division
import numpy as np

from elemwise import elemwise1, elemwise2, ielemwise2, compare, ElemwiseKernel
from dtypes import dtype_to_ctype, get_np_obj, get_common_dtype
from tools import as_argument, ArrayArg
import gpuarray as array

class gpuarray(array.GpuArray):
    """
    Extension class for gpuarray.GpuArray to add numpy mathematical
    operations between arrays.  These operations are all performed on
    the GPU but this is not the most efficient way since it will
    involve the creation of temporaries (just like numpy) for all
    intermediate results.

    This class may help transition code from numpy to pygpu by acting
    more like a drop-in replacement for numpy.ndarray than the raw
    GpuArray class.
    """
    ### add
    def __add__(self, other):
        return elemwise2(self, '+', other, self)

    def __radd__(self, other):
        return elemwise2(other, '+', self, self)

    def __iadd__(self, other):
        return ielemwise2(self, '+', other)

    ### sub
    def __sub__(self, other):
        return elemwise2(self, '-', other, self)

    def __rsub__(self, other):
        return elemwise2(other, '-', self, self)

    def __isub__(self, other):
        return ielemwise2(self, '-', other)

    ### mul
    def __mul__(self, other):
        return elemwise2(self, '*', other, self)

    def __rmul__(self, other):
        return elemwise2(other, '*', self, self)

    def __imul__(self, other):
        return ielemwise2(self, '*', other)

    ### div
    def __div__(self, other):
        return elemwise2(self, '/', other, self)

    def __rdiv__(self, other):
        return elemwise2(other, '/', self, self)

    def __idiv__(self, other):
        return ielemwise2(self, '/', other)

    ### truediv
    def __truediv__(self, other):
        np1 = get_np_obj(self)
        np2 = get_np_obj(other)
        res = (np1.__truediv__(np2)).dtype
        return elemwise2(self, '/', other, self, odtype=res)

    def __rtruediv__(self, other):
        np1 = get_np_obj(self)
        np2 = get_np_obj(other)
        res = (np2.__truediv__(np1)).dtype
        return elemwise2(other, '/', self, self, odtype=res)

    def __itruediv__(self, other):
        np2 = get_np_obj(other)
        kw = {}
        if self.dtype == np.float32 or np2.dtype == np.float32:
            kw['op_tmpl'] = "a[i] = (float)a[i] / (float)%(b)s"
        if self.dtype == np.float64 or np2.dtype == np.float64:
            kw['op_tmpl'] = "a[i] = (double)a[i] / (double)%(b)s"
        return ielemwise2(self, '/', other, **kw)

    ### floordiv
    def __floordiv__(self, other):
        out_dtype = get_common_dtype(self, other, True)
        kw = {}
        if out_dtype.kind == 'f':
            kw['op_tmpl'] = "res[i] = floor((%(out_t)s)%(a)s / (%(out_t)s)%(b)s)"
        return elemwise2(self, '/', other, self, odtype=out_dtype, **kw)

    def __rfloordiv__(self, other):
        out_dtype = get_common_dtype(other, self, True)
        kw = {}
        if out_dtype.kind == 'f':
            kw['op_tmpl'] = "res[i] = floor((%(out_t)s)%(a)s / (%(out_t)s)%(b)s)"
        return elemwise2(other, '/', self, self, odtype=out_dtype, **kw)

    def __ifloordiv__(self, other):
        out_dtype = self.dtype
        kw = {}
        if out_dtype == np.float32:
            kw['op_tmpl'] = "a[i] = floor((float)a[i] / (float)%(b)s)"
        if out_dtype == np.float64:
            kw['op_tmpl'] = "a[i] = floor((double)a[i] / (double)%(b)s)"
        return ielemwise2(self, '/', other, **kw)

    ### mod
    def __mod__(self, other):
        out_dtype = get_common_dtype(self, other, True)
        kw = {}
        if out_dtype.kind == 'f':
            kw['op_tmpl'] = "res[i] = fmod((%(out_t)s)%(a)s, (%(out_t)s)%(b)s)"
        return elemwise2(self, '%', other, self, odtype=out_dtype, **kw)

    def __rmod__(self, other):
        out_dtype = get_common_dtype(other, self, True)
        kw = {}
        if out_dtype.kind == 'f':
            kw['op_tmpl'] = "res[i] = fmod((%(out_t)s)%(a)s, (%(out_t)s)%(b)s)"
        return elemwise2(other, '%', self, self, odtype=out_dtype, **kw)

    def __imod__(self, other):
        out_dtype = get_common_dtype(self, other, self.dtype == np.float64)
        kw = {}
        if out_dtype == np.float32:
            kw['op_tmpl'] = "a[i] = fmod((float)a[i], (float)%(b)s)"
        if out_dtype == np.float64:
            kw['op_tmpl'] = "a[i] = fmod((double)a[i], (double)%(b)s)"
        return ielemwise2(self, '%', other, **kw)

    ### divmod
    def __divmod__(self, other):
        if not isinstance(other, array.GpuArray):
            other = np.asarray(other)
        odtype = get_common_dtype(self, other, True)

        a_arg = as_argument(self, 'a')
        b_arg = as_argument(other, 'b')
        args = [ArrayArg(odtype, 'div'), ArrayArg(odtype, 'mod'), a_arg, b_arg]

        div = self._empty_like_me(dtype=odtype)
        mod = self._empty_like_me(dtype=odtype)

        if odtype.kind == 'f':
            tmpl = "div[i] = floor((%(out_t)s)%(a)s / (%(out_t)s)%(b)s)," \
                "mod[i] = fmod((%(out_t)s)%(a)s, (%(out_t)s)%(b)s)"
        else:
            tmpl = "div[i] = (%(out_t)s)%(a)s / (%(out_t)s)%(b)s," \
                "mod[i] = %(a)s %% %(b)s"

        ksrc = tmpl % {'a': a_arg.expr(), 'b': b_arg.expr(),
                       'out_t': dtype_to_ctype(odtype)}

        k = ElemwiseKernel(self.context, args, ksrc)
        k(div, mod, self, other)
        return (div, mod)

    def __rdivmod__(self, other):
        if not isinstance(other, array.GpuArray):
            other = np.asarray(other)
        odtype = get_common_dtype(other, self, True)

        a_arg = as_argument(other, 'a')
        b_arg = as_argument(self, 'b')
        args = [ArrayArg(odtype, 'div'), ArrayArg(odtype, 'mod'), a_arg, b_arg]

        div = self._empty_like_me(dtype=odtype)
        mod = self._empty_like_me(dtype=odtype)

        if odtype.kind == 'f':
            tmpl = "div[i] = floor((%(out_t)s)%(a)s / (%(out_t)s)%(b)s)," \
                "mod[i] = fmod((%(out_t)s)%(a)s, (%(out_t)s)%(b)s)"
        else:
            tmpl = "div[i] = (%(out_t)s)%(a)s / (%(out_t)s)%(b)s," \
                "mod[i] = %(a)s %% %(b)s"

        ksrc = tmpl % {'a': a_arg.expr(), 'b': b_arg.expr(),
                       'out_t': dtype_to_ctype(odtype)}

        k = ElemwiseKernel(self.context, args, ksrc)
        k(div, mod, other, self)
        return (div, mod)

    def __neg__(self):
        return elemwise1(self, '-')

    def __pos__(self):
        return elemwise1(self, '+')

    def __abs__(self):
        if self.dtype.kind == 'u':
            return self.copy()
        if self.dtype.kind == 'f':
            oper = "res[i] = fabs(a[i])"
        else:
            oper = "res[i] = abs(a[i])"
        return elemwise1(self, None, oper=oper)

    ### richcmp
    def __lt__(self, other):
        return compare(self, '<', other)

    def __le__(self, other):
        return compare(self, '<=', other)

    def __eq__(self, other):
        return compare(self, '==', other)

    def __ne__(self, other):
        return compare(self, '!=', other)

    def __ge__(self, other):
        return compare(self, '>=', other)

    def __gt__(self, other):
        return compare(self, '>', other)
