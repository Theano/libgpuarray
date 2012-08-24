import operator
import numpy

from ..ndarray import pygpu_ndarray as gpuarray
from ..elemwise import ElemwiseKernel

from .support import (guard_devsup, rand, check_flags, check_meta, check_all,
                      kind, context, gen_gpuarray, dtypes_no_complex)


operators2 = [operator.add, operator.sub, operator.div, operator.floordiv,
              operator.mod, operator.mul, operator.truediv]
ioperators2 = [operator.iadd, operator.isub, operator.idiv, operator.ifloordiv,
               operator.imod, operator.imul, operator.itruediv]
elems = [2, 0.3, numpy.asarray(3, dtype='int8'),
         numpy.asarray(7, dtype='uint32'),
         numpy.asarray(2.45, dtype='float32')]


def test_elemwise2_ops_array():
    for op in operators2:
        for dtype1 in dtypes_no_complex:
            for dtype2 in dtypes_no_complex:
                    yield elemwise2_ops_array, op, dtype1, dtype2, (50,)


def test_ielemwise2_ops_array():
    for op in ioperators2:
        for dtype1 in dtypes_no_complex:
            for dtype2 in dtypes_no_complex:
                    yield ielemwise2_ops_array, op, dtype1, dtype2, (50,)


@guard_devsup
def elemwise2_ops_array(op, dtype1, dtype2, shape):
    ac, ag = gen_gpuarray(shape, dtype1, kind=kind, ctx=context)
    bc, bg = gen_gpuarray(shape, dtype2, nozeros=True, kind=kind, ctx=context)

    out_c = op(ac, bc)
    out_g = op(ag, bg)

    assert out_c.shape == out_g.shape
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))


@guard_devsup
def ielemwise2_ops_array(op, dtype1, dtype2, shape):
    incr = 0
    if op == operator.isub and dtype1[0] == 'u':
        # array elements are smaller than 10 by default, so we avoid underflow
        incr = 10
    ac, ag = gen_gpuarray(shape, dtype1, incr=incr, kind=kind, ctx=context)
    bc, bg = gen_gpuarray(shape, dtype2, nozeros=True, kind=kind, ctx=context)

    out_c = op(ac, bc)
    out_g = op(ag, bg)

    assert out_g is ag
    assert out_c.shape == out_g.shape
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))


def test_elemwise_layouts():
    for shape in [(20, 30), (50, 8, 9)]:
        for offseted_outer in [True, False]:
            for offseted_inner in [True, False]:
                for sliced in [1, 2]:
                    for order in ['c', 'f']:
                        yield elemwise_layouts, shape, offseted_outer, \
                            offseted_inner, sliced, order


@guard_devsup
def elemwise_layouts(shape, offseted_outer, offseted_inner, sliced, order):
    ac, ag = gen_gpuarray(shape, dtype='float32', sliced=sliced, order=order,
                          offseted_outer=offseted_outer,
                          offseted_inner=offseted_inner)
    bc, bg = gen_gpuarray(shape, dtype='float32')

    outc = ac + bc
    outg = ag + bg

    assert outc.shape == outg.shape
    assert outc.dtype == outg.dtype
    assert numpy.allclose(outc, numpy.asarray(outg))


def test_elemwise2_ops_mixed():
    for op in operators2:
        for dtype in dtypes_no_complex:
            for elem in elems:
                yield elemwise2_ops_mixed, op, dtype, (50,), elem


def test_ielemwise2_ops_mixed():
    for op in ioperators2:
        for dtype in dtypes_no_complex:
            for elem in elems:
                yield ielemwise2_ops_mixed, op, dtype, (50,), elem


@guard_devsup
def elemwise2_ops_mixed(op, dtype, shape, elem):
    c, g = gen_gpuarray(shape, dtype, kind=kind, ctx=context)

    out_c = op(c, elem)
    out_g = op(g, elem)

    assert out_c.shape == out_g.shape
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))

    c, g = gen_gpuarray(shape, dtype, nozeros=True, kind=kind, ctx=context)
    out_c = op(elem, c)
    out_g = op(elem, g)

    assert out_c.shape == out_g.shape
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))


@guard_devsup
def ielemwise2_ops_mixed(op, dtype, shape, elem):
    incr = 0
    if op == operator.isub and dtype[0] == 'u':
        # array elements are smaller than 10 by default, so we avoid underflow
        incr = 10
    c, g = gen_gpuarray(shape, dtype, incr=incr, kind=kind, ctx=context)

    out_c = op(c, elem)
    out_g = op(g, elem)

    assert out_g is g
    assert out_c.shape == out_g.shape
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))


def test_divmod():
    for dtype1 in dtypes_no_complex:
        for dtype2 in dtypes_no_complex:
            yield divmod_array, dtype1, dtype2, (50,)
    for dtype in dtypes_no_complex:
        for elem in elems:
            yield divmod_mixed, dtype, (50,), elem


@guard_devsup
def divmod_array(dtype1, dtype2, shape):
    ac, ag = gen_gpuarray(shape, dtype1, kind=kind, ctx=context)
    bc, bg = gen_gpuarray(shape, dtype2, nozeros=True, kind=kind, ctx=context)

    out_c = divmod(ac, bc)
    out_g = divmod(ag, bg)

    assert out_c[0].shape == out_g[0].shape
    assert out_c[1].shape == out_g[1].shape
    assert out_c[0].dtype == out_g[0].dtype
    assert out_c[1].dtype == out_g[1].dtype
    assert numpy.allclose(out_c[0], numpy.asarray(out_g[0]))
    assert numpy.allclose(out_c[1], numpy.asarray(out_g[1]))


@guard_devsup
def divmod_mixed(dtype, shape, elem):
    c, g = gen_gpuarray(shape, dtype, nozeros=True, kind=kind, ctx=context)

    out_c = divmod(c, elem)
    out_g = divmod(g, elem)

    assert out_c[0].shape == out_g[0].shape
    assert out_c[1].shape == out_g[1].shape
    assert out_c[0].dtype == out_g[0].dtype
    assert out_c[1].dtype == out_g[1].dtype
    assert numpy.allclose(out_c[0], numpy.asarray(out_g[0]))
    assert numpy.allclose(out_c[1], numpy.asarray(out_g[1]))

    out_c = divmod(elem, c)
    out_g = divmod(elem, g)

    assert out_c[0].shape == out_g[0].shape
    assert out_c[1].shape == out_g[1].shape
    assert out_c[0].dtype == out_g[0].dtype
    assert out_c[1].dtype == out_g[1].dtype
    assert numpy.allclose(out_c[0], numpy.asarray(out_g[0]))
    assert numpy.allclose(out_c[1], numpy.asarray(out_g[1]))
