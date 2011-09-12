# TODO: test other dtype
import time

import numpy
import theano

import pygpu_ndarray as gpu_ndarray
from gen_elemwise import MyGpuNdArray, elemwise_collapses

def speed_elemwise_collapse():
    """ used to time if the collapse of ccontiguous dims are useful """

    shape = (30,40,50,600)
    a = gpu_ndarray.GpuNdArrayObject(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2[:,::2,:,:]
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3+b * tensor.exp(1 + b**a3)
    f = pfunc([b], [c])


    v = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    v = v[:,::2,:,:]
    v=gpu_ndarray.GpuNdArrayObject(v)
    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    t1=time.time()
    for i in range(100):
        #let debugmode catch errors
        f(v)
    t2=time.time()

def speed_elemwise_collapse2():
    """ used to test the speed up of the generalised collapse of ccontiguous dims"""

    shape = (30,40,50,600)
    a = gpu_ndarray.GpuNdArrayObject(numpy.asarray(numpy.random.rand(*shape),dtype='float32'))
    a = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2[:,:,:,::2]
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3+b * tensor.exp(1 + b**a3)
    f = pfunc([b], [c])


    v = numpy.asarray(numpy.random.rand(*shape),dtype='float32')
    v = v[:,:,:,::2]
    v=gpu_ndarray.GpuNdArrayObject(v)
    for id,n in enumerate(f.maker.env.toposort()):
        print id, n
    t1=time.time()
    for i in range(100):
        #let debugmode catch errors
        f(v)
    t2=time.time()

def test_elemwise_collapse():
    """ Test collapsing under many broadcast and strided pattern """

    dtype = 'float32'

    scalar_cpu = numpy.asarray(numpy.random.rand(1,1,1,1),dtype=dtype)
    scalar_gpu = gpu_ndarray.GpuNdArrayObject(scalar_cpu)
    scalar_gpu1 = MyGpuNdArray(scalar_gpu)
    for shape1_, shape2_, expected in [
        # No broadcastable dimensions
        ((4,5,6,9),(4,5,6,9),0),
        # All inputs have one(and the same) broadcastable dimension
        ((1,4,5,9),(1,4,5,9),0),
        ((4,1,5,9),(4,1,5,9),0),
        ((4,5,1,9),(4,5,1,9),0),
        ((4,5,9,1),(4,5,9,1),0),
        # One inputs have one broadcastable dimension
        ((1,5,6,9),(4,5,6,9),2),
        ((4,1,6,9),(4,5,6,9),3),
        ((4,5,1,9),(4,5,6,9),3),
        ((4,5,6,1),(4,5,6,9),2),
        # One inputs have two broadcastable dimension
        ((1,1,6,9),(4,5,6,9),2),
        ((1,5,1,9),(4,5,6,9),4),
        ((1,5,6,1),(4,5,6,9),3),
        ((4,1,1,9),(4,5,6,9),3),
        ((4,1,6,1),(4,5,6,9),4),
        ((4,5,1,1),(4,5,6,9),2),
        # One inputs have tree broadcastable dimension
        ((1,1,1,9),(4,5,6,9),2),
        ((1,1,6,1),(4,5,6,9),3),
        ((1,5,1,1),(4,5,6,9),3),
        ((4,1,1,1),(4,5,6,9),2),
        # One scalar
        ((1,1,1,1),(4,5,6,9),1),
        # One scalar, the other 1 broadcast dims
        ((1,1,1,1),(4,5,6,1),1),
        ]:
        for shape1, shape2 in [(shape1_,shape2_),(shape2_,shape1_)]:
            a_cpu = numpy.asarray(numpy.random.rand(*shape1),
                                  dtype=dtype)
            a = gpu_ndarray.GpuNdArrayObject(a_cpu)
            a1 = MyGpuNdArray(a)

            b_cpu = numpy.asarray(numpy.random.rand(*shape2),dtype=dtype)
            b = gpu_ndarray.GpuNdArrayObject(b_cpu)
            b1 = MyGpuNdArray(b)

            assert len(shape1) == len(shape2)
            o_shape = []
            for i in range(len(shape1)):
                o_shape.append(max(shape1[i],shape2[i]))
            o = gpu_ndarray.empty(o_shape, dtype = dtype)


            # 1.1 Check direct collapse
            nd_collaps, info = elemwise_collapses([a, b],[o])
            assert nd_collaps == expected, (shape1, shape2, nd_collaps, expected, info)
            # 1.2 Check computation are still valid
            f = MyGpuNdArray.gen_fct(theano.tensor.add, [a1, b1],
                                     len(shape1))
            assert numpy.allclose(numpy.asarray(f([a1,b1])), a_cpu+b_cpu)
            assert numpy.allclose(numpy.asarray(MyGpuNdArray.adds(a1,b1)), a_cpu+b_cpu)
            

            # 2.1 What if we add a scalar?
            nd_collaps, info = elemwise_collapses([a, b, scalar_gpu],[o])
            if expected == 0:
                expected2 = 1
            else: expected2 = expected
            assert nd_collaps == expected2, (shape1, shape2, nd_collaps, expected, info)
            # 2.2 Check computation
            assert numpy.allclose(numpy.asarray(MyGpuNdArray.adds(a1,b1, scalar_gpu1)), a_cpu+b_cpu+scalar_cpu)

            # 3.1 What if one of the dimensions is strided?
            broadcast = any([True for i in a.shape+b.shape if i==1])
            if expected == 0:
                expected2 = 2
            else: 
                expected2 = expected

            if a.shape[0] != 1:
                shape = list(shape1)
                shape[0]*=2
                c_cpu = numpy.asarray(numpy.random.rand(*shape),
                                      dtype='float32')
                c = gpu_ndarray.GpuNdArrayObject(c_cpu)[::2]
                c1 = MyGpuNdArray(c)
                
                err = ("strided", c.shape, shape2, nd_collaps, expected, info)
                nd_collaps, info = elemwise_collapses([c, b],[o])
                if broadcast:
                    assert nd_collaps >= expected, err
                else:
                    assert nd_collaps == expected2, err
                assert numpy.allclose(numpy.asarray(MyGpuNdArray.adds(c1,b1)), numpy.asarray(c)+b_cpu)
                    
            if a.shape[1] != 1:
                shape = list(shape1)
                shape[1]*=2
                c_cpu = numpy.asarray(numpy.random.rand(*shape),
                                      dtype='float32')
                c = gpu_ndarray.GpuNdArrayObject(c_cpu)[::,::2]
                c1 = MyGpuNdArray(c)

                err = ("strided", c.shape, shape2, nd_collaps, expected, info)
                nd_collaps, info = elemwise_collapses([c, b],[o])
                if broadcast:
                    assert nd_collaps >= expected, err
                else:
                    assert nd_collaps == expected2, err
                    pass
                assert numpy.allclose(numpy.asarray(MyGpuNdArray.adds(c1,b1)), numpy.asarray(c)+b_cpu)

            if a.shape[2] != 1:
                shape = list(shape1)
                shape[2]*=2
                c_cpu = numpy.asarray(numpy.random.rand(*shape),
                                      dtype='float32')
                c = gpu_ndarray.GpuNdArrayObject(c_cpu)[::,::,::2]
                c1 = MyGpuNdArray(c)

                err = ("strided", c.shape, shape2, nd_collaps, expected, info)
                nd_collaps, info = elemwise_collapses([c, b],[o])
                if broadcast:
                    assert nd_collaps >= expected, err
                else:
                    assert nd_collaps == expected2, err
                    pass
                assert numpy.allclose(numpy.asarray(MyGpuNdArray.adds(c1,b1)), numpy.asarray(c)+b_cpu)

            if a.shape[3] != 1:
                shape = list(shape1)
                shape[3]*=2
                c_cpu = numpy.asarray(numpy.random.rand(*shape),
                                      dtype='float32')
                c = gpu_ndarray.GpuNdArrayObject(c_cpu)[::,::,::,::2]
                c1 = MyGpuNdArray(c)

                err = ("strided", c.shape, shape2, nd_collaps, expected, info)
                nd_collaps, info = elemwise_collapses([c, b],[o])
                if broadcast:
                    assert nd_collaps >= expected, err
                else:
                    assert nd_collaps == 1,err
                    pass
                assert numpy.allclose(numpy.asarray(MyGpuNdArray.adds(c1,b1)), numpy.asarray(c)+b_cpu)


