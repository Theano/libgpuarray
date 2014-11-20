import operator
import numpy

from pygpu import gpuarray, PyGpuArray as elemary
from pygpu.elemwise import ElemwiseKernel
from pygpu.tools import check_args, ArrayArg, ScalarArg

from .support import (guard_devsup, rand, check_flags, check_meta, check_all,
                      context, gen_gpuarray, dtypes_no_complex,
                      check_meta_content)


operators1 = [operator.neg, operator.pos, operator.abs]
operators2 = [operator.add, operator.sub, operator.div, operator.floordiv,
              operator.mod, operator.mul, operator.truediv,
              operator.eq, operator.ne, operator.lt, operator.le,
              operator.gt, operator.ge]
ioperators2 = [operator.iadd, operator.isub, operator.idiv, operator.ifloordiv,
               operator.imod, operator.imul, operator.itruediv]
elems = [2, 0.3, numpy.asarray(3, dtype='int8'),
         numpy.asarray(7, dtype='uint32'),
         numpy.asarray(2.45, dtype='float32')]


def test_elemwise1_ops_array():
    for op in operators1:
        for dtype in dtypes_no_complex:
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

    out_c = op(ac, bc)
    out_g = op(ag, bg)

    assert out_g is ag
    assert numpy.allclose(out_c, numpy.asarray(out_g))


def test_elemwise_layouts():
    for shape in [(), (20, 30), (50, 8, 9)]:
        for offseted_outer in [True, False]:
            for offseted_inner in [True, False]:
                for sliced in [1, 2]:
                    for order in ['c', 'f']:
                        yield elemwise_layouts, shape, offseted_outer, \
                            offseted_inner, sliced, order
                        yield elemwise_layouts_mixed, shape, offseted_outer, \
                            offseted_inner, sliced, order


def test_elemwise_0():
    elemwise_layouts((0,), False, False, 1, 'c')


@guard_devsup
def elemwise_layouts(shape, offseted_outer, offseted_inner, sliced, order):
    ac, ag = gen_gpuarray(shape, dtype='float32', sliced=sliced, order=order,
                          offseted_outer=offseted_outer,
                          offseted_inner=offseted_inner, ctx=context)
    bc, bg = gen_gpuarray(shape, dtype='float32', ctx=context)

    outg = gpuarray.empty(shape, dtype='float32', context=context)

    k = ElemwiseKernel(context, "float *a, float *b, float *c",
                       "c[i] = a[i] + b[i]")
    # will use contig or basic
    k(ag, bg, outg)
    outc = ac + bc
    assert numpy.allclose(numpy.asarray(outg), outc)

    # test basic
    outg = gpuarray.empty(shape, dtype='float32', context=context)
    k.call_basic(ag, bg, outg)
    assert numpy.allclose(numpy.asarray(outg), outc)

    # test dimspec
    outg = gpuarray.empty(shape, dtype='float32', context=context)
    k.call_dimspec(ag, bg, outg)
    assert numpy.allclose(numpy.asarray(outg), outc)

    # test specialized
    outg = gpuarray.empty(shape, dtype='float32', context=context)
    k.call_specialized(ag, bg, outg)
    assert numpy.allclose(numpy.asarray(outg), outc)


@guard_devsup
def elemwise_layouts_mixed(shape, offseted_outer, offseted_inner, sliced,
                           order):
    ac, ag = gen_gpuarray(shape, dtype='float32', sliced=sliced, order=order,
                          offseted_outer=offseted_outer,
                          offseted_inner=offseted_inner, ctx=context)
    b = numpy.asarray(2.0, dtype='float32')

    outg = gpuarray.empty(shape, dtype='float32', context=context)

    k = ElemwiseKernel(context, "float *a, float b, float *c",
                       "c[i] = a[i] + b")
    # will use contig or basic
    k(ag, b, outg)
    outc = ac + b
    assert numpy.allclose(numpy.asarray(outg), outc)

    # test basic
    outg = gpuarray.empty(shape, dtype='float32', context=context)
    k.call_basic(ag, b, outg)
    assert numpy.allclose(numpy.asarray(outg), outc)

    # test dimspec
    outg = gpuarray.empty(shape, dtype='float32', context=context)
    k.call_dimspec(ag, b, outg)
    assert numpy.allclose(numpy.asarray(outg), outc)

    # test specialized
    outg = gpuarray.empty(shape, dtype='float32', context=context)
    k.call_specialized(ag, b, outg)
    assert numpy.allclose(numpy.asarray(outg), outc)


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
    except ValueError, e:
        exc = e
    assert e is not None
    a = gpuarray.zeros((1,), context=context)
    assert bool(a) == False
    a = gpuarray.zeros((), context=context)
    assert bool(a) == False


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


def test_elemwise_collapse():
    for dtype1 in dtypes_no_complex:
        for dtype2 in dtypes_no_complex:
            for shape1, shape2, expected in [
                # 1d to test this special case
                ((40,), (40,), 1),
                ((40,), (1,), 1),
                # No broadcastable dimensions
                ((4, 5, 6, 9), (4, 5, 6, 9), 1),
                # All inputs have one(and the same) broadcastable dimension
                ((1, 4, 5, 9), (1, 4, 5, 9), 1),
                ((4, 1, 5, 9), (4, 1, 5, 9), 1),
                ((4, 5, 1, 9), (4, 5, 1, 9), 1),
                ((4, 5, 9, 1), (4, 5, 9, 1), 1),
                # One inputs have one broadcastable dimension
                ((1, 5, 6, 9), (4, 5, 6, 9), 2),
                ((4, 1, 6, 9), (4, 5, 6, 9), 3),
                ((4, 5, 1, 9), (4, 5, 6, 9), 3),
                ((4, 5, 6, 1), (4, 5, 6, 9), 2),
                # One inputs have two broadcastable dimension
                ((1, 1, 6, 9), (4, 5, 6, 9), 2),
                ((1, 5, 1, 9), (4, 5, 6, 9), 4),
                ((1, 5, 6, 1), (4, 5, 6, 9), 3),
                ((4, 1, 1, 9), (4, 5, 6, 9), 3),
                ((4, 1, 6, 1), (4, 5, 6, 9), 4),
                ((4, 5, 1, 1), (4, 5, 6, 9), 2),
                # One inputs have tree broadcastable dimension
                ((1, 1, 1, 9), (4, 5, 6, 9), 2),
                ((1, 1, 6, 1), (4, 5, 6, 9), 3),
                ((1, 5, 1, 1), (4, 5, 6, 9), 3),
                ((4, 1, 1, 1), (4, 5, 6, 9), 2),
                # One scalar
                ((1, 1, 1, 1), (4, 5, 6, 9), 1),
                # One scalar, the other 1 broadcast dims
                ((1, 1, 1, 1), (4, 5, 6, 1), 1),
                ]:
                yield elemwise_collapse, dtype1, dtype2, shape1, shape2, \
                    expected


def elemwise_collapse(dtype1, dtype2, shape1, shape2, expected):
    assert len(shape1) == len(shape2)

    # int8 does not cause problematic upcasts
    scalar = numpy.asarray(1, dtype='int8')

    a_cpu, a_gpu = gen_gpuarray(shape1, dtype1, ctx=context)
    b_cpu, b_gpu = gen_gpuarray(shape2, dtype2, ctx=context)

    o_shape = []
    for i in range(len(shape1)):
        o_shape.append(max(shape1[i], shape2[i]))

    o = gpuarray.empty(o_shape, dtype=(a_cpu + b_cpu).dtype, context=context)

    n, nd, dims, strs, offsets, contig = check_args((a_gpu, b_gpu),
                                                    collapse=True,
                                                    broadcast=True)

    assert nd == expected, (shape1, shape2, dims, nd, expected)

    k = ElemwiseKernel(context, [ArrayArg(numpy.dtype(dtype1), 'a'),
                                 ArrayArg(numpy.dtype(dtype2), 'b'),
                                 ArrayArg(o.dtype, 'o')], "o[i] = a[i] + b[i]")
    out_cpu = a_cpu + b_cpu
    k(a_gpu, b_gpu, o, collapse=True, broadcast=True)

    assert numpy.allclose(numpy.asarray(o), out_cpu)

    k(a_gpu, b_gpu, o, collapse=False, broadcast=True)

    assert numpy.allclose(numpy.asarray(o), out_cpu)

    broadcast = any([True for i in shape1 + shape2
                     if i == 1])

    n, nd, dims, strs, offsets, contig = check_args((a_gpu, b_gpu, scalar),
                                                    collapse=True,
                                                    broadcast=True)
    assert nd == expected

    k = ElemwiseKernel(context, [ArrayArg(numpy.dtype(dtype1), 'a'),
                                 ArrayArg(numpy.dtype(dtype2), 'b'),
                                 ScalarArg(scalar.dtype, 's'),
                                 ArrayArg(o.dtype, 'o')],
                       "o[i] = a[i] + b[i] + s")
    out_cpu = a_cpu + b_cpu + scalar
    k(a_gpu, b_gpu, scalar, o, collapse=True, broadcast=True)

    assert numpy.allclose(numpy.asarray(o), out_cpu)

    k(a_gpu, b_gpu, scalar, o, collapse=False, broadcast=True)

    assert numpy.allclose(numpy.asarray(o), out_cpu)

    if expected == 1:
        expected2 = 2
    else:
        expected2 = expected

    if len(shape1) != 4:
        return

    if shape1[0] != 1:
        c_cpu, c_gpu = gen_gpuarray(shape1, dtype=dtype1, sliced=2, ctx=context)
        n, nd, dims, strs, offsets,contig = check_args((c_gpu, b_gpu),
                                                       collapse=True,
                                                       broadcast=True)
        if broadcast:
            assert nd >= expected
        else:
            assert nd == expected2
