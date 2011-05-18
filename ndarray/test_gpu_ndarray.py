import numpy

import pygpu_ndarray as gpu_ndarray


def test_transfer():
    for shp in [(5,),(6,7),(4,8,9),(1,8,9)]:
        for dtype in ["int8", "int16", "int32", "int64",
                      "uint8", "uint16", "uint32", "uint64",
                      "float32", "float64",
                      "complex64", "complex128"
                      ]:
            a = numpy.random.rand(*shp) * 10
            b = gpu_ndarray.GpuNdArrayObject(a)
            c = numpy.asarray(b)

            assert numpy.allclose(c,a)
            assert a.shape == b.shape == c.shape
            assert a.strides == b.strides == c.strides
            assert a.dtype == c.dtype
            assert c.flags.c_contiguous

def test_transfer_not_contiguous():
    for shp in [(5,),(6,7),(4,8,9),(1,8,9)]:
        for dtype in ["int8", "int16", "int32", "int64",
                      "uint8", "uint16", "uint32", "uint64",
                      "float32", "float64",
                      "complex64", "complex128"
                      ]:
            a = numpy.random.rand(*shp) * 10
            a = a[::-1]
            b = gpu_ndarray.GpuNdArrayObject(a)
            c = numpy.asarray(b)

            assert numpy.allclose(c,a)
            assert a.shape == b.shape == c.shape
            # We copy a to a c contiguous array before the transfer
            assert (-a.strides[0],)+a.strides[1:] == b.strides == c.strides
            assert a.dtype == c.dtype
            assert c.flags.c_contiguous

def test_transfer_fortran():
    for shp in [(5,),(6,7),(4,8,9),(1,8,9)]:
        for dtype in ["int8", "int16", "int32", "int64",
                      "uint8", "uint16", "uint32", "uint64",
                      "float32", "float64",
                      "complex64", "complex128"
                      ]:
            a = numpy.random.rand(*shp) * 10
            a_ = numpy.asfortranarray(a)
            if len(shp)>1:
                assert a_.strides != a.strides
            a = a_
            b = gpu_ndarray.GpuNdArrayObject(a)
            c = numpy.asarray(b)

            assert a.shape == b.shape == c.shape
            assert a.dtype == c.dtype
            #assert c.flags.f_contiguous
            #print a.strides, b.strides, c.strides//TODO: fix why the strides are not the same...
            assert a.strides == b.strides #== c.strides
            assert numpy.allclose(c,a)
