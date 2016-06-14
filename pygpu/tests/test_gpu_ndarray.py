from __future__ import print_function

import unittest
import copy
from six.moves import range
from six import PY3
import pickle

import numpy

import pygpu
from pygpu.gpuarray import GpuArray, GpuContext, GpuKernel

from .support import (guard_devsup, check_meta, check_flags, check_all,
                      check_content, gen_gpuarray, context as ctx, dtypes_all,
                      dtypes_no_complex, skip_single_f)


def product(*args, **kwds):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def permutations(elements):
    if len(elements) <= 1:
        yield elements
    else:
        for perm in permutations(elements[1:]):
            for i in range(len(elements)):
                yield perm[:i] + elements[:1] + perm[i:]


def test_hash():
    g = pygpu.empty((2, 3), context=ctx)
    exc = None
    try:
        h = hash(g)
    except TypeError as e:
        exc = e
    assert exc is not None


def test_transfer():
    for shp in [(), (5,), (6, 7), (4, 8, 9), (1, 8, 9)]:
        for dtype in dtypes_all:
            for offseted in [True, False]:
                yield transfer, shp, dtype, offseted


def transfer(shp, dtype, offseted):
    a, b = gen_gpuarray(shp, dtype, offseted, ctx=ctx)
    c = numpy.asarray(b)

    assert numpy.allclose(c, a)
    assert a.shape == b.shape == c.shape
    assert a.strides == b.strides == c.strides
    assert a.dtype == b.dtype == c.dtype == dtype
    assert c.flags.c_contiguous

def test_cast():
    for shp in [(), (5,), (6, 7), (4, 8, 9), (1, 8, 9)]:
        for dtype1 in dtypes_no_complex:
            for dtype2 in dtypes_no_complex:
                    yield cast, shp, dtype1, dtype2

@guard_devsup
def cast(shp, dtype1, dtype2):
    a, b = gen_gpuarray(shp, dtype1, False, ctx=ctx)
    ac = a.astype(dtype2)
    bc = b.astype(dtype2)

    assert ac.dtype == bc.dtype
    assert ac.shape == bc.shape
    assert numpy.allclose(a, numpy.asarray(b))


def test_transfer_not_contiguous():
    for shp in [(5,), (6, 7), (4, 8, 9), (1, 8, 9)]:
        for dtype in dtypes_all:
            yield transfer_not_contiguous, shp, dtype


@guard_devsup
def transfer_not_contiguous(shp, dtype):
    a = numpy.random.rand(*shp) * 10
    a = a[::-1]
    b = pygpu.array(a, context=ctx)
    c = numpy.asarray(b)

    assert numpy.allclose(c, a)
    assert a.shape == b.shape == c.shape
    # the result array (c) is C contiguous
    assert a.strides == b.strides == (-c.strides[0],) + c.strides[1:]
    assert a.dtype == b.dtype == c.dtype
    assert c.flags.c_contiguous


def test_transfer_fortran():
    for shp in [(5,), (6, 7), (4, 8, 9), (1, 8, 9)]:
        for dtype in dtypes_all:
            yield transfer_fortran, shp, dtype


@guard_devsup
def transfer_fortran(shp, dtype):
    a = numpy.random.rand(*shp) * 10
    a_ = numpy.asfortranarray(a)
    if len(shp) > 1:
        assert a_.strides != a.strides
    a = a_
    b = pygpu.array(a, context=ctx)
    c = numpy.asarray(b)

    assert a.shape == b.shape == c.shape
    assert a.dtype == b.dtype == c.dtype
    assert a.flags.f_contiguous
    assert c.flags.f_contiguous
    assert a.strides == b.strides == c.strides
    assert numpy.allclose(c, a)


def test_ascontiguousarray():
    for shp in [(), (5,), (6, 7), (4, 8, 9), (1, 8, 9)]:
        for dtype in dtypes_all:
            for offseted_o in [True, False]:
                for offseted_i in [True, True]:
                    for sliced in [1, 2, -1, -2]:
                        for order in ['f', 'c']:
                            yield ascontiguousarray, shp, dtype, offseted_o, \
                                offseted_i, sliced, order


@guard_devsup
def ascontiguousarray(shp, dtype, offseted_o, offseted_i, sliced, order):
    cpu, gpu = gen_gpuarray(shp, dtype, offseted_o, offseted_i, sliced, order,
                            ctx=ctx)

    a = numpy.ascontiguousarray(cpu)
    b = pygpu.ascontiguousarray(gpu)

    # numpy upcast with a view to 1d scalar.
    if (sliced != 1 or shp == () or
        (offseted_i and len(shp) > 1)):
        assert b is not gpu
        if sliced == 1 and not offseted_i:
            assert (a.data is cpu.data) == (b.bytes is gpu.bytes)
    else:
        assert b is gpu

    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.flags.c_contiguous
    assert b.flags['C_CONTIGUOUS']
    assert a.strides == b.strides
    assert numpy.allclose(cpu, a)
    assert numpy.allclose(cpu, b)


def test_asfortranarray():
    for shp in [(), (5,), (6, 7), (4, 8, 9), (1, 8, 9)]:
        for dtype in dtypes_all:
            for offseted_outer in [True, False]:
                for offseted_inner in [True, False]:
                    for sliced in [1, 2, -1, -2]:
                        for order in ['f', 'c']:
                            yield asfortranarray, shp, dtype, offseted_outer, \
                                offseted_inner, sliced, order


@guard_devsup
def asfortranarray(shp, dtype, offseted_outer, offseted_inner, sliced, order):
    cpu, gpu = gen_gpuarray(shp, dtype, offseted_outer, offseted_inner, sliced,
                            order, ctx=ctx)

    a = numpy.asfortranarray(cpu)
    b = pygpu.asfortranarray(gpu)

    # numpy upcast with a view to 1d scalar.
    if gpu.flags['F_CONTIGUOUS']:
        assert b.gpudata == gpu.gpudata
    elif (sliced != 1 or shp == () or (offseted_outer and len(shp) > 1) or
        (order != 'f' and len(shp) > 1)):
        assert b is not gpu
    else:
        assert b is gpu

    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.flags.f_contiguous
    assert b.flags['F_CONTIGUOUS']
    if not any([s == 1 for s in cpu.shape]):
        # Older version then Numpy 1.10 do not set c/f contiguous more
        # frequently as we do. This cause extra copy.
        assert a.strides == b.strides
    assert numpy.allclose(cpu, a)
    assert numpy.allclose(cpu, b)


def test_zeros():
    for shp in [(), (0,), (5,),
                (0, 0), (1, 0), (0, 1), (6, 7),
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (4, 8, 9), (1, 8, 9)]:
        for order in ["C", "F"]:
            for dtype in dtypes_all:
                yield zeros, shp, order, dtype


@guard_devsup
def zeros(shp, order, dtype):
    x = pygpu.zeros(shp, dtype, order, context=ctx)
    y = numpy.zeros(shp, dtype, order)
    check_all(x, y)


def test_zeros_no_dtype():
    # no dtype and order param
    x = pygpu.zeros((), context=ctx)
    y = numpy.zeros(())
    check_meta(x, y)


def test_zero_noparam():
    try:
        pygpu.zeros()
        assert False
    except TypeError:
        pass


def test_empty():
    for shp in [(), (0,), (5,),
                (0, 0), (1, 0), (0, 1), (6, 7),
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (4, 8, 9), (1, 8, 9)]:
        for order in ["C", "F"]:
            for dtype in dtypes_all:
                yield empty, shp, order, dtype


def empty(shp, order, dtype):
    x = pygpu.empty(shp, dtype, order, context=ctx)
    y = numpy.empty(shp, dtype, order)
    check_meta(x, y)


def test_empty_no_dtype():
    x = pygpu.empty((), context=ctx)# no dtype and order param
    y = numpy.empty(())
    check_meta(x, y)


def test_empty_no_params():
    try:
        pygpu.empty()
        assert False
    except TypeError:
        pass


def test_mapping_getitem_ellipsis():
    for shp in [(5,), (6, 7), (4, 8, 9), (1, 8, 9)]:
        for dtype in dtypes_all:
            for offseted in [True, False]:
                yield mapping_getitem_ellipsis, shp, dtype, offseted


def mapping_getitem_ellipsis(shp, dtype, offseted):
    a, a_gpu = gen_gpuarray(shp, dtype, offseted, ctx=ctx)
    b = a_gpu[...]
    assert b.gpudata == a_gpu.gpudata
    assert b.strides == a.strides
    assert b.shape == a.shape
    b_cpu = numpy.asarray(b)
    assert numpy.allclose(a, b_cpu)


def test_mapping_setitem_ellipsis():
    for shp in [(9,), (8, 9), (4, 8, 9), (1, 8, 9)]:
        for dtype in dtypes_all:
            for offseted in [True, False]:
                yield mapping_setitem_ellipsis, shp, dtype, offseted
                yield mapping_setitem_ellipsis2, shp, dtype, offseted

@guard_devsup
def mapping_setitem_ellipsis(shp, dtype, offseted):
    a, a_gpu = gen_gpuarray(shp, dtype, offseted, ctx=ctx)
    a[...] = 2
    a_gpu[...] = 2
    assert numpy.allclose(a, numpy.asarray(a_gpu))

@guard_devsup
def mapping_setitem_ellipsis2(shp, dtype, offseted):
    a, a_gpu = gen_gpuarray(shp, dtype, offseted, ctx=ctx)
    b, b_gpu = gen_gpuarray(shp[1:], dtype, False, ctx=ctx)
    a[:] = b
    a_gpu[:] = b_gpu
    assert numpy.allclose(a, numpy.asarray(b_gpu))


def test_copy_view():
    for shp in [(5,), (6, 7), (4, 8, 9), (1, 8, 9)]:
        for dtype in dtypes_all:
            for offseted in [False, True]:
                # order1 is the order of the original data
                for order1 in ['c', 'f']:
                    # order2 is the order wanted after copy
                    for order2 in ['c', 'f']:
                        yield copy_view, shp, dtype, offseted, order1, order2


def check_memory_region(a, a_op, b, b_op):
    assert numpy.may_share_memory(a, a_op) == \
        pygpu.gpuarray.may_share_memory(b, b_op)

@guard_devsup
def copy_view(shp, dtype, offseted, order1, order2):
    #TODO test copy unbroadcast!
    a, b = gen_gpuarray(shp, dtype, offseted, order=order1, ctx=ctx)

    assert numpy.allclose(a, numpy.asarray(b))
    check_flags(b, a)

    c = b.copy(order2)
    assert numpy.allclose(a, numpy.asarray(c))
    check_flags(c, a.copy(order2))
    check_memory_region(a, a.copy(order2), b, c)

    d = copy.copy(b)
    assert numpy.allclose(a, numpy.asarray(d))
    check_flags(d, copy.copy(a))
    check_memory_region(a, copy.copy(a), b, d)

    e = b.view()
    assert numpy.allclose(a, numpy.asarray(e))
    check_flags(e, a.view())
    check_memory_region(a, a.view(), b, e)

    f = copy.deepcopy(b)
    assert numpy.allclose(a, numpy.asarray(f))
    check_flags(f, copy.deepcopy(a))
    check_memory_region(a, copy.deepcopy(a), b, f)

    g = copy.copy(b.view())
    assert numpy.allclose(a, numpy.asarray(g))
    check_memory_region(a, copy.copy(a.view()), b, g)
    check_flags(g, copy.copy(a.view()))


def test_shape():
    for shps in [((), (1,)), ((5,), (1, 5)), ((5,), (5, 1)), ((2, 3), (6,)),
                 ((6,), (2, 3)), ((1,), ()),
                 ((4,), (-1,)), ((4, 3), (-1,)),
                 ((4, 3), (-1, 3)), ((4, 3), (4, -1)), ((4, 3), (3, -1)),
                 ((4, 3), (12, -1)), ((4, 3), (-1, 12)),
                 ((5, 4, 3, 2), (2, -1, 12)), ((4, 2), (2, 2, -1)),
                 # ((4, 3), (13, -1)),
    ]:
        for offseted in [True, False]:
            for order1 in ['c', 'f']:
                if not -1 in shps[1]:
                    yield shape_, shps, offseted, order1
                for order2 in ['a', 'c', 'f']:
                    yield reshape, shps, offseted, order1, order2


def shape_(shps, offseted, order):
    ac, ag = gen_gpuarray(shps[0], 'float32', offseted, order=order, ctx=ctx)
    try:
        ac.shape = shps[1]
    except AttributeError:
        # If numpy says it can't be done, we don't try to test it
        return
    ag.shape = shps[1]
    assert ac.strides == ag.strides, (ac.strides, ag.strides)
    # np.allclose don't test shapes
    assert ac.shape == ag.shape, (ac.shape, ag.shape)
    assert numpy.allclose(ac, numpy.asarray(ag))


def reshape(shps, offseted, order1, order2):
    ac, ag = gen_gpuarray(shps[0], 'float32', offseted, order=order1, ctx=ctx)
    outc = ac.reshape(shps[1], order=order2)
    outg = ag.reshape(shps[1], order=order2)
    assert outc.shape == outg.shape
    assert outc.strides == outg.strides
    assert numpy.allclose(outc, numpy.asarray(outg))


def test_transpose():
    for shp in [(2, 3), (4, 8, 9), (1, 2, 3, 4)]:
        for offseted in [True, False]:
            for order in ['c', 'f']:
                for sliced in [1, 2, -2, -1]:
                    yield transpose, shp, offseted, sliced, order
                    for perm in permutations(list(range(len(shp)))):
                        yield transpose_perm, shp, perm, offseted, sliced, order


def transpose(shp, offseted, sliced, order):
    ac, ag = gen_gpuarray(shp, 'float32', offseted, sliced=sliced,
                          order=order, ctx=ctx)
    rc = ac.transpose()
    rg = ag.transpose()

    check_all(rg, rc)
    # also check that we are exactly equal since this only a copy op
    assert numpy.all(rc == numpy.asarray(rg))

    # Test NumPy interface
    rg = numpy.transpose(ag)
    check_all(rg, rc)
    # also check that we are exactly equal since this only a copy op
    assert numpy.all(rc == numpy.asarray(rg))


def transpose_perm(shp, perm, offseted, sliced, order):
    ac, ag = gen_gpuarray(shp, 'float32', offseted, sliced=sliced,
                          order=order, ctx=ctx)
    rc = ac.transpose(perm)
    rg = ag.transpose(perm)

    check_all(rg, rc)
    # also check that we are exactly equal since this only a copy op
    assert numpy.all(rc == numpy.asarray(rg))

    # Test NumPy interface
    rg = numpy.transpose(ag, perm)
    check_all(rg, rc)
    # also check that we are exactly equal since this only a copy op
    assert numpy.all(rc == numpy.asarray(rg))


def test_transpose_args():
    ac, ag = gen_gpuarray((4, 3, 2), 'float32', ctx=ctx)

    rc = ac.transpose(0, 2, 1)
    rg = ag.transpose(0, 2, 1)

    check_all(rg, rc)
    # also check that we are exactly equal since this only a copy op
    assert numpy.all(rc == numpy.asarray(rg))


def test_len():
    for shp in [(5,), (6, 7), (4, 8, 9), (1, 8, 9)]:
        for dtype in dtypes_all:
            for offseted in [True, False]:
                yield len_, shp, dtype, offseted


def len_(shp, dtype, offseted):
    a, a_gpu = gen_gpuarray(shp, dtype, offseted, ctx=ctx)
    assert len(a_gpu) == shp[0]


def test_mapping_getitem_w_int():
    for dtype in dtypes_all:
        for offseted in [True, False]:
            yield mapping_getitem_w_int, dtype, offseted


@guard_devsup
def mapping_getitem_w_int(dtype, offseted):
    # test vector
    dim = (2,)
    a, _a = gen_gpuarray(dim, dtype, offseted, ctx=ctx)

    import sys
    init_ref_count = sys.getrefcount(_a)
    _cmp(_a[...], a[...])
    _cmp(_a[...], a[...])
    _cmp(_a[...], a[...])
    _cmp(_a[...], a[...])
    _cmp(_a[...], a[...])

    _cmp(_a[-1], a[-1])
    _cmp(_a[1], a[1])
    _cmp(_a[0], a[0])
    _cmp(_a[::1], a[::1])
    _cmpNs(_a[::-1], a[::-1])
    _cmp(_a[...], a[...])
    _cmpf(_a, 2)

    # test scalar
    dim = ()
    a, _a = gen_gpuarray(dim, dtype, offseted, ctx=ctx)
    _cmp(_a[...], a[...])
    _cmpf(_a, 0)
    _cmpf(_a, slice(1))

    # test 4d-tensor
    dim = (5, 4, 3, 2)
    a, _a = gen_gpuarray(dim, dtype, offseted, ctx=ctx)
    _cmpf(_a, slice(-1), slice(-1), 10, -10)
    _cmpf(_a, slice(-1), slice(-1), -10, slice(-1))
    _cmpf(_a, 0, slice(0, -1, -20), -10)
    _cmpf(_a, 10)
    _cmpf(_a, (10, 0, 0, 0))
    _cmpf(_a, -10)

    #test with integer
    _cmp(_a[1], a[1])
    _cmp(_a[-1], a[-1])
    _cmp(_a[numpy.int64(1)], a[numpy.int64(1)])
    _cmp(_a[numpy.int64(-1)], a[numpy.int64(-1)])

    #test with slice
    _cmp(_a[1:], a[1:])
    _cmp(_a[1:2], a[1:2])
    _cmp(_a[-1:1], a[-1:1])
    _cmp(_a[6:7:], a[6:7:])

    #test with tuple (mix slice, integer, numpy.int64)
    _cmpNs(_a[0, 0, ::numpy.int64(-1), ::-1], a[0, 0, ::-1, ::-1])
    _cmpNs(_a[:, :, ::numpy.int64(-1), ::-1], a[:, :, ::-1, ::-1])
    _cmpNs(_a[:, :, numpy.int64(1), -1], a[:, :, 1, -1])
    _cmpNs(_a[:, :, ::-1, ::-1], a[:, :, ::-1, ::-1])
    _cmpNs(_a[:, :, ::-10, ::-10], a[:, :, ::-10, ::-10])
    _cmpNs(_a[:, :, 1, -1], a[:, :, 1, -1])
    _cmpNs(_a[:, :, -1, :], a[:, :, -1, :])
    _cmpNs(_a[:, ::-2, -1, :], a[:, ::-2, -1, :])
    _cmpNs(_a[:, ::-20, -1, :], a[:, ::-20, -1, :])
    _cmpNs(_a[:, ::-2, -1], a[:, ::-2, -1])
    _cmpNs(_a[0, ::-2, -1], a[0, ::-2, -1])
    _cmp(_a[-1, -1, -1, -2], a[-1, -1, -1, -2])

    #test ellipse
    _cmp(_a[...], a[...])


def _cmp(x,y):
    assert isinstance(x, GpuArray)
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert x.strides == y.strides
    assert x.flags["C_CONTIGUOUS"] == y.flags["C_CONTIGUOUS"], (x.flags,
                                                                y.flags)
    if y.size == 0:
        # F_CONTIGUOUS flags change definition with different numpy version
        # TODO: ideally, we should be F_CONTIGUOUS in that case.
        pass
    elif not (skip_single_f and y.shape == ()):
        assert x.flags["F_CONTIGUOUS"] == y.flags["F_CONTIGUOUS"], (x.flags,
                                                                    y.flags)
    else:
        assert x.flags["F_CONTIGUOUS"]
    # GpuArrays always own their data so don't check that flag.
    if x.flags["WRITEABLE"] != y.flags["WRITEABLE"]:
        assert x.ndim == 0
    assert x.flags["ALIGNED"] == y.flags["ALIGNED"], (x.flags, y.flags)
    assert x.flags["UPDATEIFCOPY"] == y.flags["UPDATEIFCOPY"], (x.flags,
                                                                y.flags)
    x = numpy.asarray(x)
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert x.strides == y.strides
    if not numpy.all(x == y):
        print(x)
        print(y)
    assert numpy.all(x == y), (x, y)


def _cmpNs(x, y):
    """
    Don't compare the stride after the transfer
    There is a copy that have been made on the gpu before the transfer
    """
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert x.strides == y.strides
    assert x.flags["C_CONTIGUOUS"] == y.flags["C_CONTIGUOUS"]
    assert x.flags["F_CONTIGUOUS"] == y.flags["F_CONTIGUOUS"]
    assert x.flags["WRITEABLE"] == y.flags["WRITEABLE"]
    assert x.flags["ALIGNED"] == y.flags["ALIGNED"]
    # we don't check owndata since it is always true for GpuArrays
    assert x.flags["UPDATEIFCOPY"] == y.flags["UPDATEIFCOPY"]
    x_ = numpy.asarray(x)
    assert x_.shape == y.shape
    assert x_.dtype == y.dtype
    assert numpy.all(x_ == y), (x_, y)


def _cmpf(x, *y):
    try:
        x.__getitem__(y)
    except IndexError:
        pass
    else:
        raise Exception("Did not generate out or bound error")


def _cmpfV(x, *y):
    try:
        if len(y) == 1:
            x.__getitem__(*y)
        else:
            x.__getitem__(y)
    except ValueError:
        pass
    else:
        raise Exception("Did not generate value error")


def test_take1():
    yield do_take1, (4, 3), [2, 0], False
    yield do_take1, (4, 3), [2, 0], True
    yield do_take1, (12, 4, 3), [1, 1, 1, 1, 1, 2, 2, 3, 3, 0, 0, 9], False


def do_take1(shp, idx, offseted):
    c, g = gen_gpuarray(shp, dtype='float32', ctx=ctx, order='c')
    ci = numpy.asarray(idx)
    gi = pygpu.asarray(ci, context=ctx)

    rc = c.take(ci, axis=0)
    rg = g.take1(gi)

    check_content(rg, rc)


def test_flags():
    for fl in ['C', 'F', 'W', 'B', 'O', 'A', 'U', 'CA', 'FA', 'FNC', 'FORC',
               'CARRAY', 'FARRAY', 'FORTRAN', 'BEHAVED', 'OWNDATA', 'ALIGNED',
               'WRITEABLE', 'CONTIGUOUS', 'UPDATEIFCOPY', 'C_CONTIGUOUS',
               'F_CONTIGUOUS']:
        yield flag_dict, fl
    for p in ['c_contiguous', 'f_contiguous', 'contiguous', 'fortran',
              'updateifcopy', 'owndata', 'aligned', 'writeable', 'behaved',
              'carray', 'forc', 'fnc', 'farray']:
        yield flag_prop, p

def flag_dict(fl):
    c2, g2 = gen_gpuarray((2, 3), dtype='float32', ctx=ctx, order='c')
    c3, g3 = gen_gpuarray((2, 3), dtype='float32', ctx=ctx, order='f')

    assert c2.flags[fl] == g2.flags[fl]
    assert c3.flags[fl] == g3.flags[fl]

def flag_prop(p):
    c2, g2 = gen_gpuarray((2, 3), dtype='float32', ctx=ctx, order='c')
    c3, g3 = gen_gpuarray((2, 3), dtype='float32', ctx=ctx, order='f')

    assert getattr(c2.flags, p) == getattr(g2.flags, p)
    assert getattr(c3.flags, p) == getattr(g3.flags, p)


class TestPickle(unittest.TestCase):
    def test_GpuArray(self):
        with self.assertRaises(RuntimeError):
            pickle.dumps(pygpu.zeros((32,), context=ctx))
        with self.assertRaises(RuntimeError):
            pickle.dumps(pygpu.zeros((32,), context=ctx), protocol=0)
        with self.assertRaises(RuntimeError):
            pickle.dumps(pygpu.zeros((32,), context=ctx), protocol=1)
        with self.assertRaises(RuntimeError):
            pickle.dumps(pygpu.zeros((32,), context=ctx), protocol=2)
        if PY3:
            with self.assertRaises(RuntimeError):
                pickle.dumps(pygpu.zeros((32,), context=ctx), protocol=3)
        with self.assertRaises(RuntimeError):
            pickle.dumps(pygpu.zeros((32,), context=ctx), protocol=-1)

    def test_GpuContext(self):
        with self.assertRaises(RuntimeError):
            pickle.dumps(ctx)
        with self.assertRaises(RuntimeError):
            pickle.dumps(ctx, protocol=0)
        with self.assertRaises(RuntimeError):
            pickle.dumps(ctx, protocol=1)
        with self.assertRaises(RuntimeError):
            pickle.dumps(ctx, protocol=2)
        if PY3:
            with self.assertRaises(RuntimeError):
                pickle.dumps(ctx, protocol=3)
        with self.assertRaises(RuntimeError):
            pickle.dumps(ctx, protocol=-1)

    def test_GpuKernel(self):
        k = GpuKernel("KERNEL void nothing(GLOBAL_MEM ga_float *in) {in[0] = 0;}", "nothing", [], context=ctx)
        with self.assertRaises(RuntimeError):
            pickle.dumps(k)
        with self.assertRaises(RuntimeError):
            pickle.dumps(k, protocol=0)
        with self.assertRaises(RuntimeError):
            pickle.dumps(k, protocol=1)
        with self.assertRaises(RuntimeError):
            pickle.dumps(k, protocol=2)
        if PY3:
            with self.assertRaises(RuntimeError):
                pickle.dumps(k, protocol=3)
        with self.assertRaises(RuntimeError):
            pickle.dumps(k, protocol=-1)
