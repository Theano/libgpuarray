from __future__ import division
from pytools import memoize_method
import numpy as np

from elemwise import elemwise1, elemwise2, ielemwise2, compare, ElemwiseKernel
from dtypes import dtype_to_ctype, get_np_obj, get_common_dtype
from tools import as_argument, ArrayArg
import gpuarray as array

class gpuarray(array.GpuArray):
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


def f_contiguous_strides(itemsize, shape):
    if shape:
        strides = [itemsize]
        for s in shape[:-1]:
            strides.append(strides[-1]*s)
        return tuple(strides)
    else:
        return ()


def c_contiguous_strides(itemsize, shape):
    if shape:
        strides = [itemsize]
        for s in shape[:0:-1]:
            strides.append(strides[-1]*s)
        return tuple(strides[::-1])
    else:
        return ()


class ArrayFlags:
    def __init__(self, ary):
        self.array = ary

    @property
    @memoize_method
    def f_contiguous(self):
        return self.array.strides == f_contiguous_strides(
                self.array.dtype.itemsize, self.array.shape)

    @property
    @memoize_method
    def c_contiguous(self):
        return self.array.strides == c_contiguous_strides(
                self.array.dtype.itemsize, self.array.shape)

    @property
    @memoize_method
    def forc(self):
        return self.f_contiguous or self.c_contiguous


def bound(a):
    high = a.bytes
    low = a.bytes

    for stri, shp in zip(a.strides, a.shape):
        if stri<0:
            low += (stri)*(shp-1)
        else:
            high += (stri)*(shp-1)
    return low, high


def may_share_memory(a, b):
    # When this is called with a an ndarray and b
    # a sparse matrix, np.may_share_memory fails.
    if a is b:
        return True
    if a.__class__ is b.__class__:
        a_l, a_h = bound(a)
        b_l, b_h = bound(b)
        if b_l >= a_h or a_l >= b_h:
            return False
        return True
    else:
        return False


# {{{ as_strided implementation

# stolen from numpy to be compatible with older versions of numpy

class _DummyArray(object):
    """ Dummy object that just exists to hang __array_interface__ dictionaries
    and possibly keep alive a reference to a base array.
    """
    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base


def as_strided(x, shape=None, strides=None):
    """ Make an ndarray from the given array with the given shape and strides.
    """
    # work around Numpy bug 1873 (reported by Irwin Zaid)
    # Since this is stolen from numpy, this implementation has the same bug.
    # http://projects.scipy.org/numpy/ticket/1873

    if not x.dtype.isbuiltin:
        if (shape is not None and x.shape != shape) \
                or (strides is not None and x.strides != strides):
            raise NotImplementedError(
                    "as_strided won't work on non-native arrays for now."
                    "See http://projects.scipy.org/numpy/ticket/1873")
        else:
            return x

    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)
    return np.asarray(_DummyArray(interface, base=x))

# }}}
