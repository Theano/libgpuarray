import operator
import numpy

from pygpu import gpuarray, ndgpuarray as elemary

from six import PY2

from .support import (guard_devsup, context, gen_gpuarray, check_meta_content)

dtypes_test = ['float32', 'int8', 'uint64']

operators1 = [operator.neg, operator.pos, operator.abs]
operators2 = [operator.add, operator.sub, operator.floordiv,
              operator.mod, operator.mul, operator.truediv,
              operator.eq, operator.ne, operator.lt, operator.le,
              operator.gt, operator.ge]
if PY2:
    operators2.append(operator.div)

ioperators2 = [operator.iadd, operator.isub, operator.ifloordiv,
               operator.imod, operator.imul, operator.itruediv]
if PY2:
    ioperators2.append(operator.idiv)

elems = [2, 0.3, numpy.asarray(3, dtype='int8'),
         numpy.asarray(7, dtype='uint32'),
         numpy.asarray(2.45, dtype='float32')]


def test_elemwise1_ops_array():
    for op in operators1:
        for dtype in dtypes_test:
            yield elemwise1_ops_array, op, dtype


@guard_devsup
def elemwise1_ops_array(op, dtype):
    c, g = gen_gpuarray((50,), dtype, ctx=context, cls=elemary)

    out_c = op(c)
    out_g = op(g)

    assert out_c.shape == out_g.shape
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))


def test_elemwise2_ops_array():
    for op in operators2:
        for dtype1 in dtypes_test:
            for dtype2 in dtypes_test:
                yield elemwise2_ops_array, op, dtype1, dtype2, (50,)


def test_ielemwise2_ops_array():
    for op in ioperators2:
        for dtype1 in dtypes_test:
            for dtype2 in dtypes_test:
                yield ielemwise2_ops_array, op, dtype1, dtype2, (50,)


@guard_devsup
def elemwise2_ops_array(op, dtype1, dtype2, shape):
    ac, ag = gen_gpuarray(shape, dtype1, ctx=context, cls=elemary)
    bc, bg = gen_gpuarray(shape, dtype2, nozeros=True, ctx=context, cls=elemary)

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
    ac, ag = gen_gpuarray(shape, dtype1, incr=incr, ctx=context,
                          cls=elemary)
    bc, bg = gen_gpuarray(shape, dtype2, nozeros=True, ctx=context,
                          cls=elemary)

    try:
        out_c = op(ac, bc)
    except TypeError:
        # TODO: currently, we use old Numpy semantic and tolerate more case.
        # So we can't test that we raise the same error
        return
    out_g = op(ag, bg)

    assert out_g is ag
    assert numpy.allclose(out_c, numpy.asarray(out_g), atol=1e-6)


def test_elemwise_f16():
    yield elemwise1_ops_array, operator.neg, 'float16'
    yield elemwise2_ops_array, operator.add, 'float16', 'float16', (50,)
    yield ielemwise2_ops_array, operator.iadd, 'float16', 'float16', (50,)


def test_elemwise2_ops_mixed():
    for op in operators2:
        for dtype in dtypes_test:
            for elem in elems:
                yield elemwise2_ops_mixed, op, dtype, (50,), elem


def test_ielemwise2_ops_mixed():
    for op in ioperators2:
        for dtype in dtypes_test:
            for elem in elems:
                yield ielemwise2_ops_mixed, op, dtype, (50,), elem


@guard_devsup
def elemwise2_ops_mixed(op, dtype, shape, elem):
    c, g = gen_gpuarray(shape, dtype, ctx=context, cls=elemary)

    out_c = op(c, elem)
    out_g = op(g, elem)

    assert out_c.shape == out_g.shape
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))

    c, g = gen_gpuarray(shape, dtype, nozeros=True, ctx=context,
                        cls=elemary)
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
    c, g = gen_gpuarray(shape, dtype, incr=incr, ctx=context,
                        cls=elemary)

    try:
        out_c = op(c, elem)
    except TypeError:
        # TODO: currently, we use old Numpy semantic and tolerate more case.
        # So we can't test that we raise the same error
        return
    out_g = op(g, elem)

    assert out_g is g
    assert out_c.shape == out_g.shape
    assert out_c.dtype == out_g.dtype
    assert numpy.allclose(out_c, numpy.asarray(out_g))


def test_divmod():
    for dtype1 in dtypes_test:
        for dtype2 in dtypes_test:
            yield divmod_array, dtype1, dtype2, (50,)
    for dtype in dtypes_test:
        for elem in elems:
            yield divmod_mixed, dtype, (50,), elem


@guard_devsup
def divmod_array(dtype1, dtype2, shape):
    ac, ag = gen_gpuarray(shape, dtype1, ctx=context, cls=elemary)
    bc, bg = gen_gpuarray(shape, dtype2, nozeros=True, ctx=context,
                          cls=elemary)

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
    c, g = gen_gpuarray(shape, dtype, nozeros=True, ctx=context,
                        cls=elemary)

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


def test_elemwise_bool():
    a = gpuarray.empty((2,), context=context)
    exc = None
    try:
        bool(a)
    except ValueError as e:
        exc = e
    assert exc is not None
    a = gpuarray.zeros((1,), context=context)
    assert not bool(a)
    a = gpuarray.zeros((), context=context)
    assert not bool(a)


def test_broadcast():
    for shapea, shapeb in [((3, 5), (3, 5)),
                           ((1, 5), (3, 5)),
                           ((3, 5), (3, 1)),
                           ((1, 5), (3, 1)),
                           ((3, 1), (3, 5)),
                           ((3, 5), (3, 1)),
                           ((1, 1), (1, 1)),
                           ((3, 4, 5), (4, 5)),
                           ((4, 5), (3, 4, 5)),
                           ((), ())]:
        yield broadcast, shapea, shapeb


def broadcast(shapea, shapeb):
    ac, ag = gen_gpuarray(shapea, 'float32', ctx=context, cls=elemary)
    bc, bg = gen_gpuarray(shapeb, 'float32', ctx=context, cls=elemary)

    rc = ac + bc
    rg = ag + bg

    check_meta_content(rg, rc)
