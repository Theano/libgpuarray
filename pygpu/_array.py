from __future__ import division
import numpy

from .elemwise import elemwise1, arg, GpuElemwise, as_argument
from .dtypes import dtype_to_ctype, get_common_dtype
from . import ufuncs
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
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Unwrap out if necessary
        out_tuple = kwargs.pop('out', ())
        if not all(isinstance(out, gpuarray.GpuArray) for out in out_tuple):
            return NotImplemented

        if len(out_tuple) == 1:
            kwargs['out'] = out_tuple[0]
        elif len(out_tuple) == 2:
            kwargs['out1'] = out_tuple[0]
            kwargs['out2'] = out_tuple[1]

        native_ufunc = getattr(ufuncs, ufunc.__name__, None)
        if native_ufunc is None:
            return NotImplemented

        if method == '__call__':
            return native_ufunc(*inputs, **kwargs)
        else:
            function = getattr(native_ufunc, method, None)
            return function(*inputs, **kwargs)

    # The special methods call explicitly into __array_ufunc__ to make
    # sure that the native ufuncs are used also for Numpy 1.12 and below.
    # The only difference will be the behavior when Numpy ufuncs are
    # used: for 1.12 and below, numpy.add(x_gpu, y) will cast x_gpu to
    # a Numpy array, 1.13 and above calls into the native GPU ufunc.

    # add
    def __add__(self, other):
        return self.__array_ufunc__(numpy.add, '__call__', self, other)

    def __radd__(self, other):
        return self.__array_ufunc__(numpy.add, '__call__', other, self)

    def __iadd__(self, other):
        return self.__array_ufunc__(numpy.add, '__call__', self, other,
                                    out=self)

    # sub
    def __sub__(self, other):
        return self.__array_ufunc__(numpy.subtract, '__call__', self, other)

    def __rsub__(self, other):
        return self.__array_ufunc__(numpy.subtract, '__call__', other, self)

    def __isub__(self, other):
        return self.__array_ufunc__(numpy.subtract, '__call__', self, other,
                                    out=self)

    # mul
    def __mul__(self, other):
        return self.__array_ufunc__(numpy.multiply, '__call__', self, other)

    def __rmul__(self, other):
        return self.__array_ufunc__(numpy.multiply, '__call__', other, self)

    def __imul__(self, other):
        return self.__array_ufunc__(numpy.multiply, '__call__', self, other,
                                    out=self)

    # div
    def __div__(self, other):
        return self.__array_ufunc__(numpy.divide, '__call__', self, other)

    def __rdiv__(self, other):
        return self.__array_ufunc__(numpy.divide, '__call__', other, self)

    def __idiv__(self, other):
        return self.__array_ufunc__(numpy.divide, '__call__', self, other,
                                    out=self)

    # truediv
    def __truediv__(self, other):
        return self.__array_ufunc__(numpy.true_divide, '__call__',
                                    self, other)

    def __rtruediv__(self, other):
        return self.__array_ufunc__(numpy.true_divide, '__call__',
                                    other, self)

    def __itruediv__(self, other):
        return self.__array_ufunc__(numpy.true_divide, '__call__',
                                    self, other, out=self)

    # floordiv
    def __floordiv__(self, other):
        return self.__array_ufunc__(numpy.floor_divide, '__call__',
                                    self, other)

    def __rfloordiv__(self, other):
        return self.__array_ufunc__(numpy.floor_divide, '__call__',
                                    other, self)

    def __ifloordiv__(self, other):
        return self.__array_ufunc__(numpy.floor_divide, '__call__',
                                    self, other, out=self)

    # mod
    def __mod__(self, other):
        return self.__array_ufunc__(numpy.mod, '__call__', self, other)

    def __rmod__(self, other):
        return self.__array_ufunc__(numpy.mod, '__call__', other, self)

    def __imod__(self, other):
        return self.__array_ufunc__(numpy.mod, '__call__', self, other,
                                    out=self)

    # divmod
    def __divmod__(self, other):
        if not isinstance(other, gpuarray.GpuArray):
            other = numpy.asarray(other)
        odtype = get_common_dtype(self, other, True)

        a_arg = as_argument(self, 'a', read=True)
        b_arg = as_argument(other, 'b', read=True)
        args = [arg('div', odtype, write=True),
                arg('mod', odtype, write=True),
                a_arg, b_arg]

        div = self._empty_like_me(dtype=odtype)
        mod = self._empty_like_me(dtype=odtype)

        if odtype.kind == 'f':
            tmpl = ("div = floor((%(out_t)s)a / (%(out_t)s)b),"
                    "mod = fmod((%(out_t)s)a, (%(out_t)s)b)")
        else:
            tmpl = ("div = (%(out_t)s)a / (%(out_t)s)b,"
                    "mod = a %% b")

        ksrc = tmpl % {'out_t': dtype_to_ctype(odtype)}

        k = GpuElemwise(self.context, ksrc, args)
        k(div, mod, self, other, broadcast=True)
        return (div, mod)

    def __rdivmod__(self, other):
        if not isinstance(other, gpuarray.GpuArray):
            other = numpy.asarray(other)
        odtype = get_common_dtype(other, self, True)

        a_arg = as_argument(other, 'a', read=True)
        b_arg = as_argument(self, 'b', read=True)
        args = [arg('div', odtype, write=True),
                arg('mod', odtype, write=True),
                a_arg, b_arg]

        div = self._empty_like_me(dtype=odtype)
        mod = self._empty_like_me(dtype=odtype)

        if odtype.kind == 'f':
            tmpl = ("div = floor((%(out_t)s)a / (%(out_t)s)b),"
                    "mod = fmod((%(out_t)s)a, (%(out_t)s)b)")
        else:
            tmpl = ("div = (%(out_t)s)a / (%(out_t)s)b,"
                    "mod = a %% b")

        ksrc = tmpl % {'out_t': dtype_to_ctype(odtype)}

        k = GpuElemwise(self.context, ksrc, args)
        k(div, mod, other, self, broadcast=True)
        return (div, mod)

    # unary ops
    def __neg__(self):
        return self.__array_ufunc__(numpy.negative, '__call__', self)

    def __pos__(self):
        return elemwise1(self, '+')

    def __abs__(self):
        return self.__array_ufunc__(numpy.abs, '__call__', self)

    def __invert__(self):
        return self.__array_ufunc__(numpy.invert, '__call__', self)

    # richcmp
    def __lt__(self, other):
        return self.__array_ufunc__(numpy.less, '__call__', self, other)

    def __le__(self, other):
        return self.__array_ufunc__(numpy.less_equal, '__call__', self, other)

    def __eq__(self, other):
        return self.__array_ufunc__(numpy.equal, '__call__', self, other)

    def __ne__(self, other):
        return self.__array_ufunc__(numpy.not_equal, '__call__', self, other)

    def __gt__(self, other):
        return self.__array_ufunc__(numpy.greater, '__call__', self, other)

    def __ge__(self, other):
        return self.__array_ufunc__(numpy.greater_equal, '__call__', self,
                                    other)

    # pow
    # TODO: pow can take a third modulo argument
    def __pow__(self, other):
        return self.__array_ufunc__(numpy.power, '__call__', self, other)

    def __rpow__(self, other):
        return self.__array_ufunc__(numpy.power, '__call__', other, self)

    def __ipow__(self, other):
        return self.__array_ufunc__(numpy.power, '__call__', self, other,
                                    out=self)

    # shifts
    def __lshift__(self, other):
        return self.__array_ufunc__(numpy.left_shift, '__call__',
                                    self, other)

    def __rlshift__(self, other):
        return self.__array_ufunc__(numpy.left_shift, '__call__',
                                    other, self)

    def __ilshift__(self, other):
        return self.__array_ufunc__(numpy.left_shift, '__call__',
                                    self, other, out=self)

    def __rshift__(self, other):
        return self.__array_ufunc__(numpy.right_shift, '__call__',
                                    self, other)

    def __rrshift__(self, other):
        return self.__array_ufunc__(numpy.right_shift, '__call__',
                                    other, self)

    def __irshift__(self, other):
        return self.__array_ufunc__(numpy.right_shift, '__call__',
                                    self, other, out=self)

    # logical ops
    def __and__(self, other):
        return self.__array_ufunc__(numpy.logical_and, '__call__',
                                    self, other)

    def __rand__(self, other):
        return self.__array_ufunc__(numpy.logical_and, '__call__',
                                    other, self)

    def __iand__(self, other):
        return self.__array_ufunc__(numpy.logical_and, '__call__',
                                    self, other, out=self)

    def __or__(self, other):
        return self.__array_ufunc__(numpy.logical_or, '__call__',
                                    self, other)

    def __ror__(self, other):
        return self.__array_ufunc__(numpy.logical_or, '__call__',
                                    other, self)

    def __ior__(self, other):
        return self.__array_ufunc__(numpy.logical_or, '__call__',
                                    self, other, out=self)

    def __xor__(self, other):
        return self.__array_ufunc__(numpy.logical_xor, '__call__',
                                    self, other)

    def __rxor__(self, other):
        return self.__array_ufunc__(numpy.logical_xor, '__call__',
                                    other, self)

    def __ixor__(self, other):
        return self.__array_ufunc__(numpy.logical_xor, '__call__',
                                    self, other, out=self)

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
    def all(self, axis=None, out=None, keepdims=False):
        return ufuncs.all(self, axis, out, keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        return ufuncs.any(self, axis, out, keepdims)

    def prod(self, axis=None, out=None, keepdims=False):
        return ufuncs.prod(self, axis, out, keepdims)

    def sum(self, axis=None, out=None, keepdims=False):
        return ufuncs.sum(self, axis, out, keepdims)

    def min(self, axis=None, out=None, keepdims=False):
        return ufuncs.amin(self, axis, out, keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        return ufuncs.amax(self, axis, out, keepdims)
