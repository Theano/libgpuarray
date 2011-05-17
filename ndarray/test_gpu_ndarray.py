import numpy

import pygpu_ndarray as gpu_ndarray


def test_transfer():
    for dtype in ["int8", "int16", "int32", "int64",
                  "uint8", "uint16", "uint32", "uint64",
                  "float32", "float64",
                  "complex64", "complex128"
                  ]:
        print dir(gpu_ndarray)
        a = numpy.random.rand(5,6,7) * 10
        b = gpu_ndarray.GpuNdArrayObject(a)


        c = numpy.asarray(b)
        assert numpy.allclose(c,a)
        assert a.shape == b.shape == c.shape
