import operator
import numpy

from ..ndarray import pygpu_ndarray as gpuarray
from ..elemwise import ElemwiseKernel

from .support import (guard_devsup, rand, check_flags, check_meta, check_all,
                      kind, context, gen_gpuarray, dtypes_no_complex)


def test_elemwise_ops_array():
    for op in [operator.add, operator.floordiv, operator.mod,
               operator.mul, operator.truediv]:
        for dtype1 in dtypes_no_complex:
            for dtype2 in dtypes_no_complex:
                for shape in [(500,), (50, 5), (5, 6, 7)]:
                    yield elemwise_ops_array, op, dtype1, dtype2, shape


@guard_devsup
def elemwise_ops_array(op, dtype1, dtype2, shape):
    ac, ag = gen_gpuarray(shape, dtype1, kind=kind, ctx=context)
    bc, bg = gen_gpuarray(shape, dtype2, nozeros=True, kind=kind, ctx=context)

    out_c = op(ac, bc)
    out_g = op(ag, bg)

    assert out_c.shape == out_g.shape
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))


def test_elemwise_ops_mixed():
    for op in [operator.add, operator.floordiv, operator.mod,
               operator.mul, operator.truediv]:
        for dtype in dtypes_no_complex:
            for shape in [(500,), (50, 5), (5, 6, 7)]:
                for elem in [2, 0.3, numpy.asarray(3, dtype='uint8'),
                             numpy.asarray(7, dtype='uint32'),
                             numpy.asarray(2.45, dtype='float32')]:
                    yield elemwise_ops_mixed, op, dtype, shape, elem


@guard_devsup
def elemwise_ops_mixed(op, dtype, shape, elem):
    c, g = gen_gpuarray(shape, dtype, nozeros=True, kind=kind, ctx=context)

    out_c = op(c, elem)
    out_g = op(g, elem)

    assert out_c.shape == out_g.shape
    print out_c.dtype, out_g.dtype
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))

    out_c = op(elem, c)
    out_g = op(elem, g)

    assert out_c.shape == out_g.shape
    print out_c.dtype, out_g.dtype
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))
