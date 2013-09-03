import operator
import numpy

from pygpu import gpuarray
from pygpu.reduction import ReductionKernel

from .support import (guard_devsup, rand, check_flags, check_meta, check_all,
                      context, gen_gpuarray, dtypes_no_complex_big)

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
