from __future__ import print_function

import os, sys
import numpy
from nose.plugins.skip import SkipTest

from pygpu import gpuarray


if numpy.__version__ < '1.6.0':
    skip_single_f = True
else:
    skip_single_f = False

dtypes_all = ["float32", "float64",
              "int8", "int16", "uint8", "uint16",
              "int32", "int64", "uint32", "uint64"]

dtypes_no_complex = dtypes_all

# Sometimes int8 is just a source of trouble (like with overflows)
dtypes_no_complex_big = ["float32", "float64", "int16", "uint16",
                         "int32", "int64", "uint32", "uint64"]

def get_env_dev():
    for name in ['COMPYTE_TEST_DEVICE', 'DEVICE']:
        if name in os.environ:
            return os.environ[name]
    return "opencl0:0"


context = gpuarray.init(get_env_dev())
print("*** Testing for", context.devname, file=sys.stderr)


def guard_devsup(func):
    def f(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except gpuarray.UnsupportedException as e:
            raise SkipTest("operation not supported")
    return f


def rand(shape, dtype):
    r = numpy.random.randn(*shape) * 10
    if r.dtype.startswith('u'):
        r = numpy.absolute(r)
    return r.astype(dtype)


def check_flags(x, y):
    assert isinstance(x, gpuarray.GpuArray)
    if y.size == 0 and y.flags["C_CONTIGUOUS"] and y.flags["F_CONTIGUOUS"]:
        # Different numpy version have different value for
        # C_CONTIGUOUS in that case.
        pass
    elif x.flags["C_CONTIGUOUS"] != y.flags["C_CONTIGUOUS"]:
        # Numpy 1.10 can set c/f contiguous more frequently by
        # ignoring strides on dimensions of size 1.
        assert x.flags["C_CONTIGUOUS"] is True, (x.flags, y.flags)
        assert x.flags["F_CONTIGUOUS"] is False, (x.flags, y.flags)
        assert y.flags["C_CONTIGUOUS"] is False, (x.flags, y.flags)
        # That depend of numpy version.
        # assert y.flags["F_CONTIGUOUS"] is True, (x.flags, y.flags)
    else:
        if not (skip_single_f and x.shape == ()):
            # Numpy below 1.6.0 does not have a consistent handling of
            # f-contiguous for 0-d arrays
            if not any([s == 1 for s in x.shape]):
                # Numpy 1.10 can set f contiguous more frequently by
                # ignoring strides on dimensions of size 1.
                assert x.flags["F_CONTIGUOUS"] == y.flags["F_CONTIGUOUS"], (
                    x.flags, y.flags)
        else:
            assert x.flags["F_CONTIGUOUS"]
    assert x.flags["WRITEABLE"] == y.flags["WRITEABLE"], (x.flags, y.flags)
    # Don't check for OWNDATA since it is always true for a GpuArray
    assert x.flags["ALIGNED"] == y.flags["ALIGNED"], (x.flags, y.flags)
    assert x.flags["UPDATEIFCOPY"] == y.flags["UPDATEIFCOPY"], (x.flags,
                                                                y.flags)


def check_meta_only(x, y):
    assert isinstance(x, gpuarray.GpuArray)
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    if y.size != 0:
        assert x.strides == y.strides


def check_content(x, y):
    assert isinstance(x, gpuarray.GpuArray)
    assert numpy.allclose(numpy.asarray(x), numpy.asarray(y))


def check_meta(x, y):
    check_meta_only(x, y)
    check_flags(x, y)


def check_all(x, y):
    check_meta(x, y)
    check_content(x, y)


def check_meta_content(x, y):
    check_meta_only(x, y)
    check_content(x, y)


def gen_gpuarray(shape_orig, dtype='float32', offseted_outer=False,
                 offseted_inner=False, sliced=1, order='c', nozeros=False,
                 incr=0, ctx=None, cls=None):
    if sliced is True:
        sliced = 2
    elif sliced is False:
        sliced = 1
    shape = numpy.asarray(shape_orig).copy()
    if sliced != 1 and len(shape) > 0:
        shape[0] *= numpy.absolute(sliced)
    if offseted_outer and len(shape) > 0:
        shape[0] += 1
    if offseted_inner and len(shape) > 0:
        shape[-1] += 1

    low = 0.0
    if nozeros:
        low = 1.0

    a = numpy.random.uniform(low, 10.0, shape)
    a += incr

    a = numpy.asarray(a, dtype=dtype)
    assert order in ['c', 'f']
    if order == 'f' and len(shape) > 0:
        a = numpy.asfortranarray(a)
    b = gpuarray.array(a, context=ctx, cls=cls)
    if order == 'f' and len(shape) > 0 and b.size > 1:
        assert b.flags['F_CONTIGUOUS']

    if offseted_outer and len(shape) > 0:
        b = b[1:]
        a = a[1:]
    if offseted_inner and len(shape) > 0:
        # The b[..., 1:] act as the test for this subtensor case.
        b = b[..., 1:]
        a = a[..., 1:]
    if sliced != 1 and len(shape) > 0:
        a = a[::sliced]
        b = b[::sliced]

    if False and shape_orig == ():
        assert a.shape == (1,)
        assert b.shape == (1,)
    else:
        assert a.shape == shape_orig, (a.shape, shape_orig)
        assert b.shape == shape_orig, (b.shape, shape_orig)

    assert numpy.allclose(a, numpy.asarray(b)), (a, numpy.asarray(b))

    return a, b
