from __future__ import division
import numpy as np

from .elemwise import elemwise1, elemwise2, ielemwise2, compare, arg, GpuElemwise, as_argument
from .reduction import reduce1
from .dtypes import dtype_to_ctype, get_np_obj, get_common_dtype
from . import gpuarray


class ndgpuarray(gpuarray.GpuArray):
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
    # add
    def __add__(self, other):
        return elemwise2(self, '+', other, self, broadcast=True)

    def __radd__(self, other):
        return elemwise2(other, '+', self, self, broadcast=True)

    def __iadd__(self, other):
        return ielemwise2(self, '+', other, broadcast=True)

    # sub
    def __sub__(self, other):
        return elemwise2(self, '-', other, self, broadcast=True)

    def __rsub__(self, other):
        return elemwise2(other, '-', self, self, broadcast=True)

    def __isub__(self, other):
        return ielemwise2(self, '-', other, broadcast=True)

    # mul
    def __mul__(self, other):
        return elemwise2(self, '*', other, self, broadcast=True)

    def __rmul__(self, other):
        return elemwise2(other, '*', self, self, broadcast=True)

    def __imul__(self, other):
        return ielemwise2(self, '*', other, broadcast=True)

    # div
    def __div__(self, other):
        return elemwise2(self, '/', other, self, broadcast=True)

    def __rdiv__(self, other):
        return elemwise2(other, '/', self, self, broadcast=True)

    def __idiv__(self, other):
        return ielemwise2(self, '/', other, broadcast=True)

    # truediv
    def __truediv__(self, other):
        np1 = get_np_obj(self)
        np2 = get_np_obj(other)
        res = (np1.__truediv__(np2)).dtype
        return elemwise2(self, '/', other, self, odtype=res, broadcast=True)

    def __rtruediv__(self, other):
        np1 = get_np_obj(self)
        np2 = get_np_obj(other)
        res = (np2.__truediv__(np1)).dtype
        return elemwise2(other, '/', self, self, odtype=res, broadcast=True)

    def __itruediv__(self, other):
        np2 = get_np_obj(other)
        kw = {'broadcast': True}
        if self.dtype == np.float32 or np2.dtype == np.float32:
            kw['op_tmpl'] = "a = (float)a / (float)b"
        if self.dtype == np.float64 or np2.dtype == np.float64:
            kw['op_tmpl'] = "a = (double)a / (double)b"
        return ielemwise2(self, '/', other, **kw)

    # floordiv
    def __floordiv__(self, other):
        out_dtype = get_common_dtype(self, other, True)
        kw = {'broadcast': True}
        if out_dtype.kind == 'f':
            kw['op_tmpl'] = "res = floor((%(out_t)s)a / (%(out_t)s)b)"
        return elemwise2(self, '/', other, self, odtype=out_dtype, **kw)

    def __rfloordiv__(self, other):
        out_dtype = get_common_dtype(other, self, True)
        kw = {'broadcast': True}
        if out_dtype.kind == 'f':
            kw['op_tmpl'] = "res = floor((%(out_t)s)a / (%(out_t)s)b)"
        return elemwise2(other, '/', self, self, odtype=out_dtype, **kw)

    def __ifloordiv__(self, other):
        out_dtype = self.dtype
        kw = {'broadcast': True}
        if out_dtype == np.float32:
            kw['op_tmpl'] = "a = floor((float)a / (float)b)"
        if out_dtype == np.float64:
            kw['op_tmpl'] = "a = floor((double)a / (double)b)"
        return ielemwise2(self, '/', other, **kw)

    # mod
    def __mod__(self, other):
        out_dtype = get_common_dtype(self, other, True)
        kw = {'broadcast': True}
        if out_dtype.kind == 'f':
            kw['op_tmpl'] = "res = fmod((%(out_t)s)a, (%(out_t)s)b)"
        return elemwise2(self, '%', other, self, odtype=out_dtype, **kw)

    def __rmod__(self, other):
        out_dtype = get_common_dtype(other, self, True)
        kw = {'broadcast': True}
        if out_dtype.kind == 'f':
            kw['op_tmpl'] = "res = fmod((%(out_t)s)a, (%(out_t)s)b)"
        return elemwise2(other, '%', self, self, odtype=out_dtype, **kw)

    def __imod__(self, other):
        out_dtype = get_common_dtype(self, other, self.dtype == np.float64)
        kw = {'broadcast': True}
        if out_dtype == np.float32:
            kw['op_tmpl'] = "a = fmod((float)a, (float)b)"
        if out_dtype == np.float64:
            kw['op_tmpl'] = "a = fmod((double)a, (double)b)"
        return ielemwise2(self, '%', other, **kw)

    # divmod
    def __divmod__(self, other):
        if not isinstance(other, gpuarray.GpuArray):
            other = np.asarray(other)
        odtype = get_common_dtype(self, other, True)

        a_arg = as_argument(self, 'a', read=True)
        b_arg = as_argument(other, 'b', read=True)
        args = [arg('div', odtype, write=True), arg('mod', odtype, write=True), a_arg, b_arg]

        div = self._empty_like_me(dtype=odtype)
        mod = self._empty_like_me(dtype=odtype)

        if odtype.kind == 'f':
            tmpl = "div = floor((%(out_t)s)a / (%(out_t)s)b)," \
                "mod = fmod((%(out_t)s)a, (%(out_t)s)b)"
        else:
            tmpl = "div = (%(out_t)s)a / (%(out_t)s)b," \
                "mod = a %% b"

        ksrc = tmpl % {'out_t': dtype_to_ctype(odtype)}

        k = GpuElemwise(self.context, ksrc, args)
        k(div, mod, self, other, broadcast=True)
        return (div, mod)

    def __rdivmod__(self, other):
        if not isinstance(other, gpuarray.GpuArray):
            other = np.asarray(other)
        odtype = get_common_dtype(other, self, True)

        a_arg = as_argument(other, 'a', read=True)
        b_arg = as_argument(self, 'b', read=True)
        args = [arg('div', odtype, write=True), arg('mod', odtype, write=True), a_arg, b_arg]

        div = self._empty_like_me(dtype=odtype)
        mod = self._empty_like_me(dtype=odtype)

        if odtype.kind == 'f':
            tmpl = "div = floor((%(out_t)s)a / (%(out_t)s)b)," \
                "mod = fmod((%(out_t)s)a, (%(out_t)s)b)"
        else:
            tmpl = "div = (%(out_t)s)a / (%(out_t)s)b," \
                "mod = a %% b"

        ksrc = tmpl % {'out_t': dtype_to_ctype(odtype)}

        k = GpuElemwise(self.context, ksrc, args)
        k(div, mod, other, self, broadcast=True)
        return (div, mod)

    def __neg__(self):
        return elemwise1(self, '-')

    def __pos__(self):
        return elemwise1(self, '+')

    def __abs__(self):
        if self.dtype.kind == 'u':
            return self.copy()
        if self.dtype.kind == 'f':
            oper = "res = fabs(a)"
        elif self.dtype.itemsize < 4:
            # cuda 5.5 finds the c++ stdlib definition if we don't cast here.
            oper = "res = abs((int)a)"
        else:
            oper = "res = abs(a)"
        return elemwise1(self, None, oper=oper)

    # richcmp
    def __lt__(self, other):
        return compare(self, '<', other, broadcast=True)

    def __le__(self, other):
        return compare(self, '<=', other, broadcast=True)

    def __eq__(self, other):
        return compare(self, '==', other, broadcast=True)

    def __ne__(self, other):
        return compare(self, '!=', other, broadcast=True)

    def __ge__(self, other):
        return compare(self, '>=', other, broadcast=True)

    def __gt__(self, other):
        return compare(self, '>', other, broadcast=True)

    # misc other things
    @property
    def T(self):
        if self.ndim < 2:
            return self
        return self.transpose()

    """
Since these functions are untested (thus probably wrong), we disable them.
    def clip(self, a_min, a_max, out=None):
        oper=('res = a > %(max)s ? %(max)s : '
              '(a < %(min)s ? %(min)s : a)' % dict(min=a_min, max=a_max))
        return elemwise1(self, '', oper=oper, out=out)

    def fill(self, value):
        self[...] = value
"""
    # reductions
    def all(self, axis=None, out=None):
        if self.ndim == 0:
            return self.copy()
        return reduce1(self, '&&', '1', np.dtype('bool'),
                       axis=axis, out=out)

    def any(self, axis=None, out=None):
        if self.ndim == 0:
            return self.copy()
        return reduce1(self, '||', '0', np.dtype('bool'),
                       axis=axis, out=out)

    def prod(self, axis=None, dtype=None, out=None):
        if dtype is None:
            dtype = self.dtype
            # we only upcast integers that are smaller than the plaform default
            if dtype.kind == 'i':
                di = np.dtype('int')
                if di.itemsize > dtype.itemsize:
                    dtype = di
            if dtype.kind == 'u':
                di = np.dtype('uint')
                if di.itemsize > dtype.itemsize:
                    dtype = di
        return reduce1(self, '*', '1', dtype, axis=axis, out=out)

#    def max(self, axis=None, out=None);
#        nd = self.ndim
#        if nd == 0:
#            return self.copy()
#        idx = (0,) * nd
#        n = str(self.__getitem__(idx).__array__())
#        return reduce1(self, '', n, self.dtype, axis=axis, out=out,
#                       oper='max(a, b)')

#    def min(self, axis=None, out=None):
#        nd = self.ndim
#        if nd == 0:
#            return self.copy()
#        idx = (0,) * nd
#        n = str(self.__getitem__(idx).__array__())
#        return reduce1(self, '', n, self.dtype, axis=axis, out=out,
#                       oper='min(a, b)')

    def sum(self, axis=None, dtype=None, out=None):
        if dtype is None:
            dtype = self.dtype
            # we only upcast integers that are smaller than the plaform default
            if dtype.kind == 'i':
                di = np.dtype('int')
                if di.itemsize > dtype.itemsize:
                    dtype = di
            if dtype.kind == 'u':
                di = np.dtype('uint')
                if di.itemsize > dtype.itemsize:
                    dtype = di
        return reduce1(self, '+', '0', dtype, axis=axis, out=out)
