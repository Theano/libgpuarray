import operator
import numpy

from pygpu import gpuarray
from pygpu.reduction import ReductionKernel

from .support import (guard_devsup, rand, check_flags, check_meta, check_all,
                      context, gen_gpuarray, dtypes_no_complex)

def test_red1_array():
    for dtype in dtypes_no_complex:
        yield red1_array, dtype


@guard_devsup
def red1_array(dtype):
    c, g = gen_gpuarray((10,), dtype, ctx=context)
    
    out_c = c.sum()
    out_g = ReductionKernel(context, dtype, "0", "a + b")(g)

    assert out_c.shape == out_g.shape
    assert out_g.dtype == numpy.dtype(dtype)
    assert numpy.allclose(out_c, numpy.asarray(out_g))
