from itertools import chain
import numpy
from unittest import TestCase

import pygpu
from pygpu import ufuncs
from pygpu.ufuncs import UNARY_UFUNCS, BINARY_UFUNCS
from .support import context


pygpu.set_default_context(context)
numpy.seterr(invalid='ignore', divide='ignore')  # avoid stdout warnings


# --- Helper functions and global definitions --- #

def npy_and_gpuary_arrays(shape, dtype, positive=False):
    """Return Numpy and GPU arrays of given shape and dtype."""
    if numpy.issubsctype(dtype, numpy.unsignedinteger):
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
dtypes = [numpy.dtype(dt)
          for dt in chain(numpy.sctypes['int'], numpy.sctypes['uint'],
                          numpy.sctypes['float'])
          if numpy.dtype(dt) not in [numpy.dtype('float16'),
                                     numpy.dtype('float128'),
                                     numpy.dtype('complex256')]]

# --- Ufuncs & Reductions --- #


class test_reduction(TestCase):

    def test_all(self):
        for reduction in reductions:
            for axis in axis_params:
                for keepdims in [True, False]:
                    for dtype in dtypes:
                        self.check_reduction(reduction, dtype, axis, keepdims)

    def check_reduction(self, reduction, dtype, axis, keepdims):
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
            with self.assertRaises(ValueError):
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


class test_unary_ufuncs(TestCase):

    def test_all(self):
        for ufunc in UNARY_UFUNCS:
            for dtype in dtypes:
                self.check_unary_ufunc(ufunc, dtype)

    def check_unary_ufunc(self, ufunc, dtype):
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
            with self.assertRaises(TypeError):
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


class test_binary_ufuncs(TestCase):

    def test_all(self):
        for ufunc in BINARY_UFUNCS:
            for dtype1 in dtypes:
                for dtype2 in dtypes:
                    self.check_binary_ufunc(ufunc, dtype1, dtype2)

    def check_binary_ufunc(self, ufunc, dtype1, dtype2):
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
            with self.assertRaises(TypeError):
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
