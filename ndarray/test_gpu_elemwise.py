# TODO: test other dtype
import numpy
import theano

import pygpu_ndarray as gpu_ndarray
from gen_elemwise import MyGpuNdArray, elemwise_collapses
from test_gpu_ndarray import (dtypes_all, enable_double,
                              gen_gpu_nd_array, product)


def rand(shape, dtype):
    r = numpy.random.randn(*shape) * 10
    if dtype.startswith("u"):
        r = numpy.absolute(r)
    return r.astype(dtype)


# numpy.allclose seam to have problem with int8...
def all_close(x, y):
    return (numpy.allclose(x, y) or
            numpy.absolute(x - y).max() == 0)


def test_elemwise_collapse():
    """ Test collapsing under many broadcast and strided pattern """

    for dtype1 in ["int16", "float32", "int8"]:
        for dtype2 in ["int16", "float32", "int8"]:

            for shape1_, shape2_, expected in [
                # 1d to test this special case
                ((40,), (40,), 0),
                ((40,), (1,), 1),
                # No broadcastable dimensions
                ((4, 5, 6, 9), (4, 5, 6, 9), 0),
                # All inputs have one(and the same) broadcastable dimension
                ((1, 4, 5, 9), (1, 4, 5, 9), 0),
                ((4, 1, 5, 9), (4, 1, 5, 9), 0),
                ((4, 5, 1, 9), (4, 5, 1, 9), 0),
                ((4, 5, 9, 1), (4, 5, 9, 1), 0),
                # One inputs have one broadcastable dimension
                ((1, 5, 6, 9), (4, 5, 6, 9), 2),
                ((4, 1, 6, 9), (4, 5, 6, 9), 3),
                ((4, 5, 1, 9), (4, 5, 6, 9), 3),
                ((4, 5, 6, 1), (4, 5, 6, 9), 2),
                # One inputs have two broadcastable dimension
                ((1, 1, 6, 9), (4, 5, 6, 9), 2),
                ((1, 5, 1, 9), (4, 5, 6, 9), 4),
                ((1, 5, 6, 1), (4, 5, 6, 9), 3),
                ((4, 1, 1, 9), (4, 5, 6, 9), 3),
                ((4, 1, 6, 1), (4, 5, 6, 9), 4),
                ((4, 5, 1, 1), (4, 5, 6, 9), 2),
                # One inputs have tree broadcastable dimension
                ((1, 1, 1, 9), (4, 5, 6, 9), 2),
                ((1, 1, 6, 1), (4, 5, 6, 9), 3),
                ((1, 5, 1, 1), (4, 5, 6, 9), 3),
                ((4, 1, 1, 1), (4, 5, 6, 9), 2),
                # One scalar
                ((1, 1, 1, 1), (4, 5, 6, 9), 1),
                # One scalar, the other 1 broadcast dims
                ((1, 1, 1, 1), (4, 5, 6, 1), 1),
                ]:
                scalar_cpu = rand((1,) * len(shape1_), dtype=dtype1)
                scalar_gpu = gpu_ndarray.GpuNdArrayObject(scalar_cpu)
                scalar_gpu1 = MyGpuNdArray(scalar_gpu)
                for shape1, shape2 in [(shape1_, shape2_), (shape2_, shape1_)]:
                    a_cpu = rand(shape1, dtype=dtype1)
                    a = gpu_ndarray.GpuNdArrayObject(a_cpu)
                    a1 = MyGpuNdArray(a)

                    b_cpu = rand(shape2, dtype=dtype2)
                    b = gpu_ndarray.GpuNdArrayObject(b_cpu)
                    b1 = MyGpuNdArray(b)

                    assert len(shape1) == len(shape2)
                    o_shape = []
                    for i in range(len(shape1)):
                        o_shape.append(max(shape1[i], shape2[i]))
                    o = gpu_ndarray.empty(o_shape, dtype=(a_cpu + b_cpu).dtype)

                    # 1.1 Check direct collapse
                    nd_collaps, info = elemwise_collapses([a, b], [o])
                    assert nd_collaps == expected, (shape1, shape2,
                                                    nd_collaps, expected, info)

                    # 1.2 Check computation are still valid
                    f = MyGpuNdArray.gen_fct(theano.tensor.add, [a1, b1],
                                             len(shape1))
                    out = f([a1, b1])
                    out2 = f([a1, b1], out=out)
                    assert out is out2
                    assert numpy.allclose(numpy.asarray(f([a1, b1])),
                                          a_cpu + b_cpu)
                    assert numpy.allclose(numpy.asarray(
                            MyGpuNdArray.adds(a1, b1)), a_cpu + b_cpu)
                    assert numpy.allclose(numpy.asarray(
                            MyGpuNdArray.add(a1, b1)), a_cpu + b_cpu)
                    assert MyGpuNdArray.add(a1, b1, out=out2) is out2

                    # 1.3 Check work without collaping
                    f = MyGpuNdArray.gen_fct(theano.tensor.add, [a1, b1],
                                             len(shape1), collapse=False)
                    out = f([a1, b1])
                    out2 = f([a1, b1], out=out)
                    assert out is out2
                    assert numpy.allclose(numpy.asarray(f([a1, b1])),
                                          a_cpu + b_cpu)
                    assert numpy.allclose(numpy.asarray(MyGpuNdArray.adds(
                                a1, b1)), a_cpu + b_cpu)
                    assert numpy.allclose(numpy.asarray(MyGpuNdArray.add(
                                a1, b1)), a_cpu + b_cpu)
                    assert MyGpuNdArray.add(a1, b1, out=out2) is out2

                    # 2.1 What if we add a scalar?
                    nd_collaps, info = elemwise_collapses(
                        [a, b, scalar_gpu], [o])
                    if expected == 0:
                        expected2 = 1
                    else:
                        expected2 = expected
                    assert nd_collaps == expected2, (shape1, shape2,
                                                     nd_collaps, expected,
                                                     info)
                    # 2.2 Check computation
                    assert numpy.allclose(numpy.asarray(MyGpuNdArray.adds(
                                a1, b1, scalar_gpu1)),
                                          a_cpu + b_cpu + scalar_cpu)

                    # 3.1 What if one of the dimensions is strided?
                    broadcast = any([True for i in a.shape + b.shape
                                     if i == 1])
                    if expected == 0:
                        expected2 = 2
                    else:
                        expected2 = expected

                    if len(shape1_) != 4:
                        continue

                    if a.shape[0] != 1:
                        shape = list(shape1)
                        shape[0] *= 2
                        c_cpu = rand(shape, dtype='float32')
                        c = gpu_ndarray.GpuNdArrayObject(c_cpu)[::2]
                        c1 = MyGpuNdArray(c)

                        err = ("strided", c.shape, shape2,
                               nd_collaps, expected, info)
                        nd_collaps, info = elemwise_collapses([c, b], [o])
                        if broadcast:
                            assert nd_collaps >= expected, err
                        else:
                            assert nd_collaps == expected2, err
                        assert numpy.allclose(numpy.asarray(
                                MyGpuNdArray.adds(c1, b1)),
                                              numpy.asarray(c) + b_cpu)

                    if a.shape[1] != 1:
                        shape = list(shape1)
                        shape[1] *= 2
                        c_cpu = rand(shape, dtype='float32')
                        c = gpu_ndarray.GpuNdArrayObject(c_cpu)[::, ::2]
                        c1 = MyGpuNdArray(c)

                        err = ("strided", c.shape, shape2,
                               nd_collaps, expected, info)
                        nd_collaps, info = elemwise_collapses([c, b], [o])
                        if broadcast:
                            assert nd_collaps >= expected, err
                        else:
                            assert nd_collaps == expected2, err
                            pass
                        assert numpy.allclose(numpy.asarray(
                                MyGpuNdArray.adds(c1, b1)),
                                              numpy.asarray(c) + b_cpu)

                    if a.shape[2] != 1:
                        shape = list(shape1)
                        shape[2] *= 2
                        c_cpu = rand(shape, dtype='float32')
                        c = gpu_ndarray.GpuNdArrayObject(c_cpu)[::, ::, ::2]
                        c1 = MyGpuNdArray(c)

                        err = ("strided", c.shape, shape2,
                               nd_collaps, expected, info)
                        nd_collaps, info = elemwise_collapses([c, b], [o])
                        if broadcast:
                            assert nd_collaps >= expected, err
                        else:
                            assert nd_collaps == expected2, err
                            pass
                        assert numpy.allclose(numpy.asarray(
                                MyGpuNdArray.adds(c1, b1)),
                                              numpy.asarray(c) + b_cpu)

                    if a.shape[3] != 1:
                        shape = list(shape1)
                        shape[3] *= 2
                        c_cpu = rand(shape, dtype='float32')
                        c = gpu_ndarray.GpuNdArrayObject(c_cpu)[::, ::,
                            ::, ::2]
                        c1 = MyGpuNdArray(c)

                        err = ("strided", c.shape, shape2,
                               nd_collaps, expected, info)
                        nd_collaps, info = elemwise_collapses([c, b], [o])
                        if broadcast:
                            assert nd_collaps >= expected, err
                        else:
                            assert nd_collaps == 1, err
                            pass
                        assert numpy.allclose(numpy.asarray(
                                MyGpuNdArray.adds(c1, b1)),
                                              numpy.asarray(c) + b_cpu)


def test_elemwise_mixed_dtype():
    to_cpu = numpy.asarray

    for dtype1 in ["int16", "float32", "int8"]:
        for dtype2 in ["int16", "float32", "int8"]:
            dtypeo = str((numpy.zeros(1, dtype=dtype1) +
                          numpy.zeros(1, dtype=dtype2)).dtype)
            #print "dtypes", dtype1, dtype2, "o dtype", dtypeo

            #print "    Test inside a wrapping python object 2 inputs"
            for shape in [(500,), (50, 5), (5, 6, 7)]:
                input_vals = [rand(shape, dtype) for dtype in [dtype1, dtype2]]
                del dtype
                gpu_vals = [gpu_ndarray.GpuNdArrayObject(i)
                            for i in input_vals]
                assert all([numpy.allclose(to_cpu(ig), i)
                            for ig, i in zip(gpu_vals, input_vals)])

                gpu_vals = [MyGpuNdArray(x) for x in gpu_vals]
                out = gpu_vals[0] + gpu_vals[1]
                assert numpy.allclose(to_cpu(out),
                                      input_vals[0] + input_vals[1])
                out = gpu_vals[0] - gpu_vals[1]
                assert numpy.allclose(to_cpu(out),
                                      input_vals[0] - input_vals[1])
                out = gpu_vals[0] * gpu_vals[1]
                assert all_close(to_cpu(out),
                                 input_vals[0] * input_vals[1])
                if dtypeo.startswith("float"):
                    # TODO: execute for all dtype
                    out = gpu_vals[0] / gpu_vals[1]
                    assert numpy.allclose(to_cpu(out),
                                          input_vals[0] / input_vals[1])

            nb_in = 4
            #print "    Test inside a wrapping python object %d inputs"%nb_in
            for shape in [(500,), (50, 5), (5, 6, 7)]:
                input_vals = [rand(shape, dtype)
                              for dtype in [dtype1, dtype2, dtype1, dtype2]]
                gpu_vals = [gpu_ndarray.GpuNdArrayObject(i)
                            for i in input_vals]
                assert all([numpy.allclose(to_cpu(ig), i)
                            for ig, i in zip(gpu_vals, input_vals)])

                gpu_vals = [MyGpuNdArray(x) for x in gpu_vals]
                out = MyGpuNdArray.adds(*gpu_vals)
                assert numpy.allclose(to_cpu(out),
                                      reduce(numpy.add, input_vals))

                out = MyGpuNdArray.multiplys(*gpu_vals)
                assert all_close(to_cpu(out),
                                 reduce(numpy.multiply, input_vals))

            #print "    Test broadcasting"
            for shapes in [((1, 5), (4, 5)), ((33, 10), (33, 1)),
                           ((33, 1, 5), (33, 10, 1)),
                           ((33, 1, 5), (33, 10, 1), ((1, 10, 5))),
                           ]:
                input_vals = [rand(shape, dtype) for shape, dtype
                              in zip(shapes, [dtype1, dtype2])]
                gpu_vals = [gpu_ndarray.GpuNdArrayObject(i)
                            for i in input_vals]
                assert all([numpy.allclose(to_cpu(ig), i)
                            for ig, i in zip(gpu_vals, input_vals)])

                gpu_vals = [MyGpuNdArray(x) for x in gpu_vals]
                out = MyGpuNdArray.adds(*gpu_vals)
                assert numpy.allclose(to_cpu(out),
                                      reduce(numpy.add, input_vals))

                out = MyGpuNdArray.multiplys(*gpu_vals)
                assert all_close(to_cpu(out),
                                 reduce(numpy.multiply, input_vals))


def test_sum():
    to_cpu = numpy.asarray
    dtypes = list(dtypes_all)
    # I remove *int8 as currently the output have the same dtype
    # And this cause overflow
    dtypes.remove("int8")
    dtypes.remove("uint8")
    # I need to find how pycuda handle complexe in c.
    # I probably just need to add an header.
    dtypes.remove("complex64")
    if  enable_double:
        dtypes.remove("complex128")
    for shape in [
        # need something bigger then 32, 1024 or 4096.
        # Those are corner case.

        # 1d, take only a few seconds on a GTX470
        (0,), (5,), (31,), (32,), (33,),
        (1023,), (1024,), (1025,),
        (4095,), (4096,), (4097,),
        (32 * 1024 - 1,), (32 * 1024,), (32 * 1024 + 1,),

        # 2d, take 2 minutes on a GTX 470
        (0, 0), (1, 0), (0, 1,), (5, 4),
        (31, 31), (31, 32), (31, 33),
        (32, 31), (32, 32), (32, 33),
        (33, 31), (33, 32), (33, 33),
        (1024, 32), (1025, 32),
        (1024, 33), (1025, 33),
        (4096, 32), (32, 4096), (4096, 33), (33, 4096),
        (4097, 32), (32, 4097), (4097, 33), (33, 4097),

        # 3d, take 2 minutes on a GTX 470
        (0, 0, 0), (0, 1, 0), (0, 0, 1),
        (5, 4, 3), (5, 4, 3), (5, 4, 3),
        (4096, 2, 33), (2, 4096, 33), (33, 2, 4096),
        (4097, 2, 33), (2, 4097, 33), (33, 2, 4097),
        (4096, 33, 2), (33, 4096, 2), (2, 33, 4096),
        (4097, 33, 2), (33, 4097, 2), (2, 33, 4097),

        # 4d, take 1 minutes on a GTX 470
        (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0),
        (0, 0, 1, 0), (0, 0, 0, 1),
        (5, 4, 3, 2),
        (1024, 32, 2, 3), (3, 1024, 32, 2), (2, 3, 1024, 32),
        (1024, 2, 32, 3), (3, 1024, 2, 32), (1024, 3, 2, 32),
        (1025, 33, 2, 3), (3, 1025, 33, 2), (2, 3, 1025, 33),
        (1025, 2, 33, 3), (3, 1025, 2, 33), (1025, 3, 2, 33),
        (4100, 4, 3, 2), (4, 4100, 3, 2),
        (4, 3, 4100, 2), (4, 3, 2, 4100),

        # 5d, work only if c contiguous
        (5, 4, 3, 10, 11),
        ]:

        for dtype, off_o, off_i, sliced, order in product(
            *([dtypes] +
              [[False, True]] +
              [[False, True]] +
              [[-1, 2, -2, 1]] +
              [['f', 'c']])):

            cpu_val, gpu_val = gen_gpu_nd_array(shape, dtype, off_o,
                                                off_i, sliced, order)

            if len(shape) > 4 and not (gpu_val.flags["C_CONTIGUOUS"] or
                                       gpu_val.flags["F_CONTIGUOUS"]):
                continue
            gpu_val = MyGpuNdArray(gpu_val)
            cpu_sum = cpu_val.sum()
#            print dtype, shape, off_o, off_i, sliced, order
#            print (cpu_val.strides,
#                   cpu_val.flags["C_CONTIGUOUS"],
#                   cpu_val.flags["F_CONTIGUOUS"])
#            print (gpu_val.strides,
#                   gpu_val.flags["C_CONTIGUOUS"],
#                   gpu_val.flags["F_CONTIGUOUS"])
            gpu_sum = to_cpu(gpu_val.sum())

            def get_rtol(orig, after_reduction):
                if after_reduction.size == 0:
                    return 0
                if orig.size // after_reduction.size > 500000:
                    rtols = {"float32": 4.3e-5}
                elif orig.size // after_reduction.size > 100000:
                    rtols = {"float32": 3e-5}
                elif orig.size // after_reduction.size > 50000:
                    rtols = {"float32": 2e-5}
                else:
                    rtols = {"float32": 1e-5}
                if dtype in rtols:
                    rtol = rtols[dtype]
                else:
                    rtol = 1e-8
                return rtol
            rtol = get_rtol(gpu_val, gpu_sum)
            cpu_sum = cpu_sum.astype(dtype)
            if not (dtype.endswith("int16") and numpy.prod(shape) > 20000):
                assert (numpy.allclose(cpu_sum, gpu_sum, rtol=rtol) or
                        cpu_sum == gpu_sum), (
                    dtype, shape, cpu_sum, gpu_sum,
                    (cpu_sum - gpu_sum) / cpu_sum)

            # Test pattern 10 and 01
            # Test pattern 100, 010 and 001
            if len(shape) in [2, 3]:
                for axis in range(len(shape)):
                    gpu_sum = to_cpu(gpu_val.sum(axis=[axis]))
                    cpu_sum = cpu_val.sum(axis=axis)
                    rtol = get_rtol(gpu_val, gpu_sum)
                    if cpu_sum.size > 0:
                        argmax = numpy.absolute(cpu_sum - gpu_sum).argmax()
                        cpu_max = cpu_sum.flatten()[argmax]
                        gpu_max = gpu_sum.flatten()[argmax]
                    assert numpy.allclose(cpu_sum, gpu_sum), (
                        "axis=%d" % axis, dtype, shape, cpu_sum.shape,
                        cpu_sum, gpu_sum,
                        cpu_max, gpu_max, (cpu_max - gpu_max) / cpu_max)
