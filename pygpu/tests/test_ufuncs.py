from nose.tools import assert_raises
import numpy

import pygpu
from pygpu.dtypes import NAME_TO_DTYPE
from pygpu import ufuncs
from pygpu.ufuncs import UNARY_UFUNCS, BINARY_UFUNCS
from pygpu.tests.support import context


pygpu.set_default_context(context)
numpy.seterr(invalid='ignore', divide='ignore')  # avoid stdout warnings


# --- Helper functions and global definitions --- #

def npy_and_gpuary_arrays(shape, dtype, positive=False):
    """Return Numpy and GPU arrays of given shape and dtype."""
    if dtype == numpy.bool:
        npy_arr = numpy.random.randint(0, 1, size=shape, dtype=dtype)
    elif numpy.issubsctype(dtype, numpy.unsignedinteger):
        npy_arr = numpy.random.randint(1, 10, size=shape, dtype=dtype)
    elif numpy.issubsctype(dtype, numpy.integer):
        low = 1 if positive else -10
        npy_arr = numpy.random.randint(low, 10, size=shape, dtype=dtype)
    elif numpy.issubsctype(dtype, numpy.floating):
        npy_arr = numpy.random.normal(size=shape).astype(dtype)
        if positive:
            npy_arr = numpy.abs(npy_arr) + 0.1
    elif numpy.issubsctype(dtype, numpy.complexfloating):
        npy_arr_re = numpy.random.normal(size=shape).astype(dtype)
        npy_arr_im = numpy.random.normal(size=shape).astype(dtype)
        npy_arr = npy_arr_re + 1j * npy_arr_im
        if positive:
            npy_arr = (numpy.abs(npy_arr) + 0.1).astype(dtype)
    else:
        assert False

    gpuary_arr = pygpu.array(npy_arr, dtype=dtype)
    return npy_arr, gpuary_arr


reductions = ['sum', 'prod', 'amin', 'amax']
axis_params = [None, 0, 2, (1, 2), (0,), (0, 1), (0, 1, 2)]
dtypes = set(NAME_TO_DTYPE.values())

# --- Ufuncs & Reductions --- #


def test_reduction():
    for reduction in reductions:
        for axis in axis_params:
            for keepdims in [True, False]:
                for dtype in dtypes:
                    yield check_reduction, reduction, dtype, axis, keepdims


def check_reduction(reduction, dtype, axis, keepdims):
    """Test GpuArray reduction against equivalent Numpy result."""
    gpuary_reduction = getattr(ufuncs, reduction)
    npy_reduction = getattr(numpy, reduction)

    npy_arr, gpuary_arr = npy_and_gpuary_arrays(shape=(2, 3, 4),
                                                dtype=dtype)
    # Determine relative tolerance from dtype
    try:
        res = numpy.finfo(dtype).resolution
    except ValueError:
        res = 0
    rtol = 2 * npy_arr.size * res

    if (numpy.issubsctype(dtype, numpy.complexfloating) and
            reduction in ('amin', 'amax')):
        with assert_raises(ValueError):
            gpuary_reduction(gpuary_arr, axis=axis, keepdims=keepdims)
        return

    # No explicit out dtype
    npy_result = npy_reduction(npy_arr, axis=axis, keepdims=keepdims)
    gpuary_result = gpuary_reduction(gpuary_arr, axis=axis,
                                     keepdims=keepdims)
    if numpy.isscalar(npy_result):
        assert numpy.isscalar(gpuary_result)
        assert numpy.isclose(gpuary_result, npy_result, rtol=rtol,
                             atol=rtol)
    else:
        assert npy_result.shape == gpuary_result.shape
        assert npy_result.dtype == gpuary_result.dtype
        assert numpy.allclose(npy_result, gpuary_result, rtol=rtol,
                              atol=rtol)

    # With out array
    out = pygpu.empty(npy_result.shape, npy_result.dtype)
    gpuary_result = gpuary_reduction(gpuary_arr, axis=axis, out=out,
                                     keepdims=keepdims)
    assert numpy.allclose(npy_result, out, rtol=rtol, atol=rtol)
    assert out is gpuary_result

    # Explicit out dtype, supported by some reductions only
    if reduction in ('sum', 'prod'):
        out_dtype = numpy.promote_types(dtype, numpy.float32)
        gpuary_result = gpuary_reduction(gpuary_arr, axis=axis,
                                         dtype=out_dtype,
                                         keepdims=keepdims)
        assert gpuary_result.dtype == out_dtype
        assert numpy.allclose(npy_result, out, rtol=rtol, atol=rtol)

    # Using numpy arrays as input
    gpuary_result = gpuary_reduction(npy_arr, axis=axis, keepdims=keepdims)
    if numpy.isscalar(npy_result):
        assert numpy.isscalar(gpuary_result)
        assert numpy.isclose(gpuary_result, npy_result, rtol=rtol,
                             atol=rtol)
    else:
        assert npy_result.shape == gpuary_result.shape
        assert npy_result.dtype == gpuary_result.dtype
        assert numpy.allclose(npy_result, gpuary_result, rtol=rtol,
                              atol=rtol)


fail_unary = [('reciprocal', bool),  # Numpy maps to logical_not.. why?
              ('sign', bool),  # no valid signature
              ('fmod', bool)  # wrong result where second array is 0
              ]


def test_unary_ufuncs():
    for ufunc in UNARY_UFUNCS:
        for dtype in dtypes:
            if (ufunc, dtype) not in fail_unary:
                yield check_unary_ufunc, ufunc, dtype


def check_unary_ufunc(ufunc, dtype):
    """Test GpuArray unary ufuncs against equivalent Numpy results."""
    gpuary_ufunc = getattr(ufuncs, ufunc)
    npy_ufunc = getattr(numpy, ufunc)

    npy_arr, gpuary_arr = npy_and_gpuary_arrays(shape=(2, 3), dtype=dtype,
                                                positive=True)
    try:
        res = numpy.finfo(dtype).resolution
    except ValueError:
        res = numpy.finfo(numpy.promote_types(dtype,
                                              numpy.float16)).resolution
    rtol = 10 * res

    try:
        npy_result = npy_ufunc(npy_arr)
    except TypeError:
        # Make sure we raise the same error as Numpy
        with assert_raises(TypeError):
            gpuary_ufunc(gpuary_arr)
    else:
        if npy_result.dtype == numpy.dtype('float16'):
            # We use float32 as minimum for GPU arrays, do the same here
            npy_result = npy_result.astype('float32')

        gpuary_result = gpuary_ufunc(gpuary_arr)
        assert npy_result.shape == gpuary_result.shape
        assert npy_result.dtype == gpuary_result.dtype
        assert numpy.allclose(npy_result, gpuary_result, rtol=rtol,
                              atol=rtol, equal_nan=True)

        # In-place
        out = pygpu.empty(npy_result.shape, npy_result.dtype)
        gpuary_ufunc(gpuary_arr, out)
        assert numpy.allclose(npy_result, gpuary_result, rtol=rtol,
                              equal_nan=True)


sint_dtypes = [dt for dt in NAME_TO_DTYPE.values()
               if numpy.issubsctype(dt, numpy.signedinteger)]
uint_dtypes = [dt for dt in NAME_TO_DTYPE.values()
               if numpy.issubsctype(dt, numpy.unsignedinteger)]
int_dtypes = sint_dtypes + uint_dtypes
float_dtypes = [dt for dt in NAME_TO_DTYPE.values()
                if numpy.issubsctype(dt, numpy.floating)]
# List of failing combinations.
# If a dtype key is not present, it is equivalent to "all dtypes".
# Reasons for failure:
# - fmod: division by zero handled differently
# - nextafter: for bool, numpy upcasts to float16 only, we cast to float32
# - ldexp: numpy upcasts to float32, we to float64
# - left_shift: numpy wraps around (True << -3 is huge), our code doesn't
# - floor_divide: division by zero handled differently
# - logical_xor: our code yields True for True ^ 0.1 due to rounding
# - remainder: division by zero handled differently
fail_binary = {'fmod': {'dtype2': [bool]},  # wrong where array2 is 0
               'nextafter': {'dtype1': [bool], 'dtype2': int_dtypes},
               'ldexp': {'dtype2': [numpy.uint16]},
               'left_shift': {'dtype2': sint_dtypes},
               'floor_divide': {'dtype2': [bool]},
               'logical_xor': {'dtype1': [bool], 'dtype2': float_dtypes},
               'remainder': {'dtype2': [bool]},
               }
# ufuncs where negative scalars trigger upcasting that differs from Numpy
upcast_wrong_neg = ('copysign', 'ldexp', 'hypot', 'arctan2', 'nextafter')


def test_binary_ufuncs():
    for ufunc in BINARY_UFUNCS:
        for dtype1 in dtypes:
            for dtype2 in dtypes:
                if (ufunc in fail_binary and
                    dtype1 in fail_binary[ufunc].get('dtype1', [dtype1]) and
                        dtype2 in fail_binary[ufunc].get('dtype2', [dtype2])):
                    pass
                else:
                    yield check_binary_ufunc, ufunc, dtype1, dtype2

                scalars_left = [-2, -1, -0.5, 0, 1, 2.5, 4]
                scalars_right = list(scalars_left)

                # For scalars, we need to exclude some cases

                # Obvious things first: no invalid scalars for the given dtype
                if numpy.issubsctype(dtype1, numpy.integer):
                    scalars_left = [x for x in scalars_left if int(x) == x]
                if numpy.issubsctype(dtype2, numpy.integer):
                    scalars_right = [x for x in scalars_right if int(x) == x]
                if numpy.issubsctype(dtype1, numpy.unsignedinteger):
                    scalars_left = [x for x in scalars_left if x >= 0]
                if numpy.issubsctype(dtype2, numpy.unsignedinteger):
                    scalars_right = [x for x in scalars_right if x >= 0]

                # Special treatment for some ufuncs
                if ufunc == 'power':
                    if (numpy.issubsctype(dtype1, numpy.integer) or
                            dtype1 == bool):
                        # Negative integer power of integer is invalid
                        scalars_right = [x for x in scalars_right if x >= 0]
                    if numpy.issubsctype(dtype2, numpy.signedinteger):
                        # Negative integer power of integer is invalid
                        scalars_left = [x for x in scalars_right if x >= 0]
                if ufunc in ('fmod', 'remainder', 'floor_divide'):
                    # These ufuncs divide by the right operand, so we remove 0
                    scalars_left = [x for x in scalars_left if x >= 0]
                if ufunc == 'copysign':
                    if numpy.issubsctype(dtype2, numpy.unsignedinteger):
                        # Don't try to copy negative sign to unsigned array
                        scalars_left = [x for x in scalars_left if x >= 0]
                if ufunc == 'ldexp':
                    # Only integers as second argument
                    scalars_right = [x for x in scalars_right if int(x) == x]
                    if numpy.issubsctype(dtype2, numpy.unsignedinteger):
                        # This is not allowed due to invalid type coercion
                        scalars_left = []

                # Remove negative scalars for ufuncs with differing upcasting
                if ufunc in upcast_wrong_neg:
                    scalars_left = [x for x in scalars_left if x >= 0]
                    scalars_right = [x for x in scalars_left if x >= 0]

                for scalar in scalars_left:
                    if (ufunc in fail_binary and
                            fail_binary[ufunc].get('dtype2', [dtype2])):
                        pass
                    else:
                        yield (check_binary_ufunc_scalar_left,
                               ufunc, scalar, dtype2)

                for scalar in scalars_right:
                    if (ufunc in fail_binary and
                            fail_binary[ufunc].get('dtype1', [dtype1])):
                        pass
                    else:
                        yield (check_binary_ufunc_scalar_right,
                               ufunc, scalar, dtype1)


def check_binary_ufunc(ufunc, dtype1, dtype2):
    """Test GpuArray binary ufunc against equivalent Numpy result."""
    gpuary_ufunc = getattr(ufuncs, ufunc)
    npy_ufunc = getattr(numpy, ufunc)

    npy_arr, gpuary_arr = npy_and_gpuary_arrays(shape=(2, 3),
                                                dtype=dtype1,
                                                positive=True)
    npy_arr2, gpuary_arr2 = npy_and_gpuary_arrays(shape=(2, 3),
                                                  dtype=dtype2,
                                                  positive=True)
    try:
        res = numpy.finfo(numpy.result_type(dtype1, dtype2)).resolution
    except ValueError:
        res = numpy.finfo(numpy.result_type(numpy.float16,
                                            dtype1, dtype2)).resolution
    rtol = 10 * res

    try:
        npy_result = npy_ufunc(npy_arr, npy_arr2)
    except TypeError:
        # Make sure we raise the same error as Numpy
        with assert_raises(TypeError):
            gpuary_ufunc(gpuary_arr, gpuary_arr2)
    else:
        if npy_result.dtype == numpy.dtype('float16'):
            # We use float32 as minimum for GPU arrays, do the same here
            npy_result = npy_result.astype('float32')

        gpuary_result = gpuary_ufunc(gpuary_arr, gpuary_arr2)
        assert npy_result.shape == gpuary_result.shape
        assert npy_result.dtype == gpuary_result.dtype
        assert numpy.allclose(npy_result, gpuary_result, rtol=rtol,
                              atol=rtol, equal_nan=True)

        # In-place
        out = pygpu.empty(npy_result.shape, npy_result.dtype)
        gpuary_ufunc(gpuary_arr, gpuary_arr2, out)
        assert numpy.allclose(npy_result, gpuary_result, rtol=rtol,
                              equal_nan=True)


def check_binary_ufunc_scalar_left(ufunc, scalar, dtype):
    """Test GpuArray binary ufunc with scalar first operand."""
    gpuary_ufunc = getattr(ufuncs, ufunc)
    npy_ufunc = getattr(numpy, ufunc)

    npy_arr, gpuary_arr = npy_and_gpuary_arrays(shape=(2, 3),
                                                dtype=dtype,
                                                positive=True)
    try:
        res = numpy.finfo(dtype).resolution
    except ValueError:
        res = numpy.finfo(numpy.promote_types(dtype,
                                              numpy.float16)).resolution
    rtol = 10 * res

    try:
        npy_result = npy_ufunc(scalar, npy_arr)
    except TypeError:
        # Make sure we raise the same error as Numpy
        with assert_raises(TypeError):
            gpuary_ufunc(gpuary_arr, scalar)
    else:
        if npy_result.dtype == numpy.dtype('float16'):
            # We use float32 as minimum for GPU arrays, do the same here
            npy_result = npy_result.astype('float32')

        gpuary_result = gpuary_ufunc(scalar, gpuary_arr)
        assert npy_result.shape == gpuary_result.shape
        assert npy_result.dtype == gpuary_result.dtype
        assert numpy.allclose(npy_result, gpuary_result, rtol=rtol,
                              atol=rtol, equal_nan=True)


def check_binary_ufunc_scalar_right(ufunc, scalar, dtype):
    """Test GpuArray binary ufunc with scalar second operand."""
    gpuary_ufunc = getattr(ufuncs, ufunc)
    npy_ufunc = getattr(numpy, ufunc)

    npy_arr, gpuary_arr = npy_and_gpuary_arrays(shape=(2, 3),
                                                dtype=dtype,
                                                positive=True)
    try:
        res = numpy.finfo(dtype).resolution
    except ValueError:
        res = numpy.finfo(numpy.promote_types(dtype,
                                              numpy.float16)).resolution
    rtol = 10 * res

    try:
        npy_result = npy_ufunc(npy_arr, scalar)
    except TypeError:
        # Make sure we raise the same error as Numpy
        with assert_raises(TypeError):
            gpuary_ufunc(gpuary_arr, scalar)
    else:
        if npy_result.dtype == numpy.dtype('float16'):
            # We use float32 as minimum for GPU arrays, do the same here
            npy_result = npy_result.astype('float32')

        gpuary_result = gpuary_ufunc(gpuary_arr, scalar)
        assert npy_result.shape == gpuary_result.shape
        assert npy_result.dtype == gpuary_result.dtype
        assert numpy.allclose(npy_result, gpuary_result, rtol=rtol,
                              atol=rtol, equal_nan=True)
