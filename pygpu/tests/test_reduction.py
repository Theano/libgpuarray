import numpy

from pygpu import gpuarray, ndgpuarray as elemary
from pygpu.reduction import ReductionKernel

from .support import (guard_devsup, check_meta_content, context, gen_gpuarray,
                      dtypes_no_complex_big, dtypes_no_complex)


def test_red_array_basic():
    for dtype in dtypes_no_complex_big:
        for shape, redux in [((10,), [True]),
                             ((20, 30), [True, True]),
                             ((20, 30), [True, False]),
                             ((20, 30), [False, True]),
                             ((8, 5, 10), [True, True, True]),
                             ((8, 5, 10), [True, True, False]),
                             ((8, 5, 10), [True, False, True]),
                             ((8, 5, 10), [False, True, True]),
                             ((8, 5, 10), [True, False, False]),
                             ((8, 5, 10), [False, True, False]),
                             ((8, 5, 10), [False, False, True]),
                             ]:
            yield red_array_sum, dtype, shape, redux


@guard_devsup
def red_array_sum(dtype, shape, redux):
    c, g = gen_gpuarray(shape, dtype, ctx=context)

    axes = [i for i in range(len(redux)) if redux[i]]
    axes.reverse()
    out_c = c
    # numpy.sum doesn't support multiple axis before 1.7.0
    for ax in axes:
        out_c = numpy.apply_along_axis(sum, ax, out_c).astype(dtype)
    out_g = ReductionKernel(context, dtype, "0", "a + b", redux)(g)

    assert out_c.shape == out_g.shape
    assert out_g.dtype == numpy.dtype(dtype)
    # since we do not use the same summing algorithm,
    # there will be differences
    assert numpy.allclose(out_c, numpy.asarray(out_g), rtol=2e-5)


def test_red_big_array():
    for redux in [[True, False, False],
                  [True, False, True],
                  [False, True, True],
                  [False, True, False]]:
        yield red_array_sum, 'float32', (2000, 30, 100), redux


def test_red_broadcast():
    from pygpu.tools import as_argument

    dtype = float
    xshape = (5, 10, 15)
    yshape = (1, 10, 15)
    redux = [False, True, False]

    nx, gx = gen_gpuarray(xshape, dtype, ctx=context)
    ny, gy = gen_gpuarray(yshape, dtype, ctx=context)

    nz = nx*ny
    axes = [i for i in range(len(redux)) if redux[i]]
    axes.reverse()
    # numpy.sum doesn't support multiple axis before 1.7.0
    for ax in axes:
        nz = numpy.apply_along_axis(sum, ax, nz).astype(dtype)

    args = [as_argument(gx, 'a'), as_argument(gy, 'b')]
    gz = ReductionKernel(context, dtype, "0", "a+b", redux, map_expr="a[i]*b[i]", arguments=args)(gx, gy, broadcast=True)

    assert numpy.allclose(nz, numpy.asarray(gz))

def test_reduction_ops():
    for axis in [None, 0, 1]:
        for op in ['all', 'any']:
            yield reduction_op, op, 'bool', axis
        for op in ['prod', 'sum']:  # 'min', 'max']:
            for dtype in dtypes_no_complex:
                yield reduction_op, op, dtype, axis


def reduction_op(op, dtype, axis):
    c, g = gen_gpuarray((2, 3), dtype=dtype, ctx=context, cls=elemary)

    rc = getattr(c, op)(axis=axis)
    rg = getattr(g, op)(axis=axis)

    check_meta_content(rg, rc)

    outc = numpy.empty(rc.shape, dtype=rc.dtype)
    outg = gpuarray.empty(rg.shape, dtype=rg.dtype, context=context)

    rc = getattr(c, op)(axis=axis, out=outc)
    rg = getattr(g, op)(axis=axis, out=outg)

    check_meta_content(outg, outc)


def test_reduction_wrong_type():
    c, g = gen_gpuarray((2, 3), dtype='float32', ctx=context, cls=elemary)
    out1 = gpuarray.empty((2, 3), dtype='int32', context=context)
    out2 = gpuarray.empty((3, 2), dtype='float32', context=context)

    try:
        g.sum(out=out1)
        assert False, "Expected a TypeError out of the sum"
    except TypeError:
        pass

    try:
        g.sum(out=out2)
        assert False, "Expected a TypeError out of the sum"
    except TypeError:
        pass


def test_reduction_0d():
    c, g = gen_gpuarray((), dtype='bool', ctx=context, cls=elemary)

    rc = c.any()
    rg = g.any()

    assert numpy.all(rc == numpy.asarray(rg))

    rc = c.all()
    rg = g.all()

    assert numpy.all(rc == numpy.asarray(rg))
