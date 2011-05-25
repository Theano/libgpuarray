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
            assert a.dtype == b.dtype == c.dtype
            assert c.flags.c_contiguous

def test_transfer_not_contiguous():
    """
    Test transfer when the input on the CPU is not contiguous
    TODO: test when the input on the gpu is not contiguous
    """
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
            assert a.dtype == b.dtype == c.dtype
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
            assert a.dtype == b.dtype == c.dtype
            assert a.flags.f_contiguous
            assert c.flags.f_contiguous
            assert a.strides == b.strides == c.strides
            assert numpy.allclose(c,a)

def test_mapping_getitem_ellipsis():
    for shp in [(5,),(6,7),(4,8,9),(1,8,9)]:
        for dtype in ["int8", "int16", "int32", "int64",
                      "uint8", "uint16", "uint32", "uint64",
                      "float32", "float64",
                      "complex64", "complex128"
                      ]:
            a = numpy.asarray(numpy.random.rand(*shp), dtype=dtype)
            a_gpu = gpu_ndarray.GpuNdArrayObject(a)

            b = a_gpu[...]
            assert b.bytes == a_gpu.bytes
            assert b.strides == a.strides
            assert b.shape == a.shape
            b_cpu = numpy.asarray(b)
            assert numpy.allclose(a, b_cpu)

def test_len():
    for shp in [(5,),(6,7),(4,8,9),(1,8,9)]:
        for dtype in ["int8", "int16", "int32", "int64",
                      "uint8", "uint16", "uint32", "uint64",
                      "float32", "float64",
                      "complex64", "complex128"
                      ]:
            a = numpy.asarray(numpy.random.rand(*shp), dtype=dtype)
            a_gpu = gpu_ndarray.GpuNdArrayObject(a)
            assert len(a_gpu) == shp[0]

def test_mapping_getitem_w_int():
    def _cmpb(x,y):
        assert x.base
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        #x_ = numpy.asarray(x)
        #assert x_.shape == y.shape
        #assert x_.dtype == y.dtype
        #if not numpy.all(x_ == y):
        #    print x
        #    print y
        #assert numpy.all(x_ == y)

    def _cmp(x,y):
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        assert x.strides == y.strides
        x = numpy.asarray(x)
        if not numpy.all(x == y):
            print x
            print y
        assert numpy.all(x == y),(x, y)

    def _cmpf(x,*y):
        try:
            x.__getitem__(y)
        except IndexError:
            pass
        else:
            raise Exception("Did not generate out or bound error")

    def _cmpfV(x,*y):
        try:
            if len(y)==1:
                x.__getitem__(*y)
            else:
                x.__getitem__(y)
        except ValueError:
            pass
        else:
            raise Exception("Did not generate value error")

    def _cmpfI(x,*y):
        try:
            if len(y)==1:
                x.__getitem__(*y)
            else:
                x.__getitem__(y)
        except NotImplementedError:
            pass
        else:
            raise Exception("Did not generate not implemented error.")

    for dtype in ["int8", "int16", "int32", "int64",
                  "uint8", "uint16", "uint32", "uint64",
                  "float32", "float64",
                  "complex64", "complex128"
                  ]:

        # test vector
        dim =(2,)
        a = numpy.asarray(numpy.random.rand(*dim)*10, dtype=dtype)
        _a = gpu_ndarray.GpuNdArrayObject(a)
        import sys
        init_ref_count = sys.getrefcount(_a)
        _cmp(_a[...], a[...])
        _cmp(_a[...], a[...])
        _cmp(_a[...], a[...])
        _cmp(_a[...], a[...])
        _cmp(_a[...], a[...])


        #import pdb;pdb.set_trace()
        _cmpb(_a[-1], a[-1])
        _cmpb(_a[1], a[1])
        _cmpb(_a[0], a[0])
        _cmp(_a[::1], a[::1])
        #_cmp(_a[::-1], a[::-1])#TODO implement and enable
        _cmp(_a[...], a[...])
        _cmpf(_a,2)

        # test scalar
        dim =()
        a = numpy.asarray(numpy.random.rand(*dim)*10, dtype=dtype)
        _a = gpu_ndarray.GpuNdArrayObject(a)
        _cmp(numpy.asarray(_a[...]), a[...])
        _cmpf(_a,0)
        _cmpfV(_a,slice(1))

        # test 4d-tensor
        dim =(5,4,3,2)
        a = numpy.asarray(numpy.random.rand(*dim)*10, dtype=dtype)
        _a = gpu_ndarray.GpuNdArrayObject(a)

        _cmpf(_a,slice(-1),slice(-1),10,-10)
        _cmpf(_a,slice(-1),slice(-1),-10,slice(-1))
        _cmpf(_a,0,slice(0,-1,-20),-10)
        _cmpf(_a,10)
        _cmpf(_a,(10,0,0,0))
        _cmpf(_a,-10)

        #test with integer
        _cmpb(_a[1], a[1])
        _cmpb(_a[-1], a[-1])
        _cmpb(_a[numpy.int64(1)], a[numpy.int64(1)])
        _cmpb(_a[numpy.int64(-1)], a[numpy.int64(-1)])

        #test with slice
        _cmp(numpy.asarray(_a[1:]), a[1:])
        _cmp(numpy.asarray(_a[1:2]), a[1:2])
        _cmp(numpy.asarray(_a[-1:1]), a[-1:1])


        #test with tuple (mix slice, integer, numpy.int64)
        if False:
            _cmp(numpy.asarray(_a[:,:,::numpy.int64(-1), ::-1]), a[:,:,::-1,::-1])
            _cmp(numpy.asarray(_a[:,:,numpy.int64(1),-1]), a[:,:,1,-1])
            _cmp(numpy.asarray(_a[:,:,::-1, ::-1]), a[:,:,::-1,::-1])
            _cmp(numpy.asarray(_a[:,:,::-10, ::-10]), a[:,:,::-10,::-10])
            _cmp(numpy.asarray(_a[:,:,1,-1]), a[:,:,1,-1])
            _cmp(numpy.asarray(_a[:,:,-1,:]), a[:,:,-1,:])
            _cmp(numpy.asarray(_a[:,::-2,-1,:]), a[:,::-2,-1,:])
            _cmp(numpy.asarray(_a[:,::-20,-1,:]), a[:,::-20,-1,:])
            _cmp(numpy.asarray(_a[:,::-2,-1]), a[:,::-2,-1])
            _cmp(numpy.asarray(_a[0,::-2,-1]), a[0,::-2,-1])

        _cmpf(numpy.asarray(_a[-1,-1,-1,-2]), a[-1,-1,-1,-2])
        _cmp(numpy.asarray(_a[...]), a[...])
