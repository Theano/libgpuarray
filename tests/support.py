import numpy
from nose.plugins.skip import SkipTest
from nose.plugins import Plugin

from ..ndarray import pygpu_ndarray as gpuarray


if numpy.__version__ < '1.6.0':
    skip_single_f = True
else:
    skip_single_f = False

dtypes_all = ["float32", "float64", "complex64", "complex128",
              "int8", "int16", "uint8", "uint16",
              "int32", "int64", "uint32", "uint64"]

dtypes_no_complex = ["float32", "float64",
                     "int8", "int16", "uint8", "uint16",
                     "int32", "int64", "uint32", "uint64"]

kind = "opencl"
#kind = "cuda"
context = gpuarray.init(kind, 0)


def guard_devsup(func):
    def f(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except gpuarray.GpuArrayException as e:
            if e.args[0] == "Device does not support operation":
                raise SkipTest("operation not supported")
            raise
    return f


def rand(shape, dtype):
    r = numpy.random.randn(*shape) * 10
    if r.dtype.startswith('u'):
        r = numpy.absolute(r)
    return r.astype(dtype)


def check_flags(x, y):
    assert isinstance(x, gpuarray.GpuArray)
    assert x.flags["C_CONTIGUOUS"] == y.flags["C_CONTIGUOUS"]
    if not (skip_single_f and x.shape == ()):
        # Numpy below 1.6.0 does not have a consistent handling of
        # f-contiguous for 0-d arrays
        assert x.flags["F_CONTIGUOUS"] == y.flags["F_CONTIGUOUS"]
    else:
        assert x.flags["F_CONTIGUOUS"]
    assert x.flags["WRITEABLE"] == y.flags["WRITEABLE"]
    # Don't check for OWNDATA after indexing since GpuArray do own it
    # and ndarrays don't.  It's an implementation detail anyway.
    if y.base is None:
        assert x.flags["OWNDATA"] == y.flags["OWNDATA"]
    assert x.flags["ALIGNED"] == y.flags["ALIGNED"]
    assert x.flags["UPDATEIFCOPY"] == y.flags["UPDATEIFCOPY"]


def check_meta(x, y):
    assert isinstance(x, gpuarray.GpuArray)
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    if y.size != 0:
        assert x.strides == y.strides
    check_flags(x, y)


def check_all(x, y):
    assert isinstance(x, gpuarray.GpuArray)
    check_meta(x, y)
    assert numpy.allclose(numpy.asarray(x), numpy.asarray(y))


def gen_gpuarray(shape_orig, dtype='float32', offseted_outer=False,
                 offseted_inner=False, sliced=1, order='c', nozeros=False,
                 incr=0, kind=None, ctx=None):
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
    b = gpuarray.array(a, context=ctx, kind=kind)
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

    assert numpy.allclose(a, numpy.asarray(b))

    return a, b
