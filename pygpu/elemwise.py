import numpy

from .dtypes import dtype_to_ctype, get_common_dtype
from . import gpuarray
from ._elemwise import GpuElemwise, arg

__all__ = ['GpuElemwise', 'elemwise1', 'elemwise2', 'ielemwise2', 'compare']


def _dtype(o):
    if hasattr(o, 'dtype'):
        return o.dtype
    return numpy.asarray(o).dtype


def as_argument(o, name, read=False, write=False):
    if not read and not write:
        raise ValueError('argument is neither read not write')
    return arg(name, _dtype(o), scalar=not isinstance(o, gpuarray.GpuArray),
               read=read, write=write)


def elemwise1(a, op, oper=None, op_tmpl="res = %(op)sa", out=None,
              convert_f16=True):
    args = (as_argument(a, 'res', write=True), as_argument(a, 'a', read=True))
    if out is None:
        res = a._empty_like_me()
    else:
        res = out

    if oper is None:
        oper = op_tmpl % {'op': op}

    k = GpuElemwise(a.context, oper, args, convert_f16=convert_f16)
    k(res, a)
    return res


def elemwise2(a, op, b, ary, odtype=None, oper=None,
              op_tmpl="res = (%(out_t)s)a %(op)s (%(out_t)s)b",
              broadcast=False, convert_f16=True):
    ndim_extend = True
    if not isinstance(a, gpuarray.GpuArray):
        a = numpy.asarray(a)
        ndim_extend = False
    if not isinstance(b, gpuarray.GpuArray):
        b = numpy.asarray(b)
        ndim_extend = False
    if odtype is None:
        odtype = get_common_dtype(a, b, True)

    a_arg = as_argument(a, 'a', read=True)
    b_arg = as_argument(b, 'b', read=True)

    args = [arg('res', odtype, write=True), a_arg, b_arg]

    if ndim_extend:
        if a.ndim != b.ndim:
            nd = max(a.ndim, b.ndim)
            if a.ndim < nd:
                a = a.reshape(((1,) * (nd - a.ndim)) + a.shape)
            if b.ndim < nd:
                b = b.reshape(((1,) * (nd - b.ndim)) + b.shape)
        out_shape = tuple(max(sa, sb) for sa, sb in zip(a.shape, b.shape))
        res = gpuarray.empty(out_shape, dtype=odtype, context=ary.context,
                             cls=ary.__class__)
    else:
        res = ary._empty_like_me(dtype=odtype)

    if oper is None:
        if convert_f16 and odtype == 'float16':
            odtype = numpy.dtype('float32')
        oper = op_tmpl % {'op': op, 'out_t': dtype_to_ctype(odtype)}

    k = GpuElemwise(ary.context, oper, args, convert_f16=convert_f16)
    k(res, a, b, broadcast=broadcast)
    return res


def ielemwise2(a, op, b, oper=None, op_tmpl="a = a %(op)s b",
               broadcast=False, convert_f16=True):
    if not isinstance(b, gpuarray.GpuArray):
        b = numpy.asarray(b)

    a_arg = as_argument(a, 'a', read=True, write=True)
    b_arg = as_argument(b, 'b', read=True)

    args = [a_arg, b_arg]

    if oper is None:
        oper = op_tmpl % {'op': op}

    k = GpuElemwise(a.context, oper, args, convert_f16=convert_f16)
    k(a, b, broadcast=broadcast)
    return a


def compare(a, op, b, broadcast=False, convert_f16=True):
    return elemwise2(a, op, b, a, odtype=numpy.dtype('bool'),
                     op_tmpl="res = (a %(op)s b)",
                     broadcast=broadcast, convert_f16=convert_f16)
