import numpy
from nose.plugins.skip import SkipTest

from .support import (guard_devsup, gen_gpuarray, context)

try:
    import scipy.linalg.blas
    try:
        fblas = scipy.linalg.blas.fblas
    except AttributeError:
        fblas = scipy.linalg.blas
except ImportError as e:
    raise SkipTest("no scipy blas to compare against")

import pygpu.blas as gblas

def test_dot():
    # TODO [WIP]
    raise NotImplementedError()

def test_gemv():
    for shape in [(100, 128), (128, 50)]:
        for order in ['f', 'c']:
            for trans in [False, True]:
                for offseted_i in [True, False]:
                    for sliced in [1, 2, -1, -2]:
                        yield gemv, shape, 'float32', order, trans, \
                            offseted_i, sliced, True, False
    for overwrite in [True, False]:
        for init_y in [True, False]:
            yield gemv, (4, 3), 'float32', 'f', False, False, 1, \
                overwrite, init_y
    yield gemv, (32, 32), 'float64', 'f', False, False, 1, True, False
    for alpha in [0, 1, -1, 0.6]:
        for beta in [0, 1, -1, 0.6]:
            for overwite in [True, False]:
                yield gemv, (32, 32), 'float32', 'f', False, False, 1, \
                    overwrite, True, alpha, beta


@guard_devsup
def gemv(shp, dtype, order, trans, offseted_i, sliced,
          overwrite, init_y, alpha=1.0, beta=0.0):
    cA, gA = gen_gpuarray(shp, dtype, order=order, offseted_inner=offseted_i,
                          sliced=sliced, ctx=context)
    if trans:
        shpX = (shp[0],)
        shpY = (shp[1],)
    else:
        shpX = (shp[1],)
        shpY = (shp[0],)
    cX, gX = gen_gpuarray(shpX, dtype, offseted_inner=offseted_i,
                          sliced=sliced, ctx=context)
    if init_y:
        cY, gY = gen_gpuarray(shpY, dtype, ctx=context)
    else:
        cY, gY = None, None

    if dtype == 'float32':
        cr = fblas.sgemv(alpha, cA, cX, beta, cY, trans=trans,
                         overwrite_y=overwrite)
    else:
        cr = fblas.dgemv(alpha, cA, cX, beta, cY, trans=trans,
                         overwrite_y=overwrite)
    gr = gblas.gemv(alpha, gA, gX, beta, gY, trans_a=trans,
                    overwrite_y=overwrite)

    numpy.testing.assert_allclose(cr, numpy.asarray(gr), rtol=1e-6)


def test_gemm():
    for m, n, k in [(48, 15, 32), (15, 32, 48)]:
        for order in [('f', 'f', 'f'), ('c', 'c', 'c'),
                      ('f', 'f', 'c'), ('f', 'c', 'f'),
                      ('f', 'c', 'c'), ('c', 'f', 'f'),
                      ('c', 'f', 'c'), ('c', 'c', 'f')]:
            for trans in [(False, False), (True, True),
                          (False, True), (True, False)]:
                for offseted_o in [False, True]:
                    yield gemm, m, n, k, 'float32', order, trans, \
                        offseted_o, 1, False, False
    for sliced in [1, 2, -1, -2]:
        for overwrite in [True, False]:
            for init_res in [True, False]:
                yield gemm, 4, 3, 2, 'float32', ('f', 'f', 'f'), \
                    (False, False), False, sliced, overwrite, init_res
    yield gemm, 32, 32, 32, 'float64', ('f', 'f', 'f'), (False, False), \
        False, 1, False, False
    for alpha in [0, 1, -1, 0.6]:
        for beta in [0, 1, -1, 0.6]:
            for overwrite in [True, False]:
                yield gemm, 32, 23, 32, 'float32', ('f', 'f', 'f'), \
                    (False, False), False, 1, overwrite, True, alpha, beta

@guard_devsup
def gemm(m, n, k, dtype, order, trans, offseted_o, sliced, overwrite,
         init_res, alpha=1.0, beta=0.0):
    if trans[0]:
        shpA = (k,m)
    else:
        shpA = (m,k)
    if trans[1]:
        shpB = (n,k)
    else:
        shpB = (k,n)

    cA, gA = gen_gpuarray(shpA, dtype, order=order[0],
                          offseted_outer=offseted_o,
                          sliced=sliced, ctx=context)
    cB, gB = gen_gpuarray(shpB, dtype, order=order[1],
                          offseted_outer=offseted_o,
                          sliced=sliced, ctx=context)
    if init_res:
        cC, gC = gen_gpuarray((m,n), dtype, order=order[2], ctx=context)
    else:
        cC, gC = None, None

    if dtype == 'float32':
        cr = fblas.sgemm(alpha, cA, cB, beta, cC, trans_a=trans[0],
                         trans_b=trans[1], overwrite_c=overwrite)
    else:
        cr = fblas.dgemm(alpha, cA, cB, beta, cC, trans_a=trans[0],
                         trans_b=trans[1], overwrite_c=overwrite)
    gr = gblas.gemm(alpha, gA, gB, beta, gC, trans_a=trans[0],
                    trans_b=trans[1], overwrite_c=overwrite)

    numpy.testing.assert_allclose(cr, numpy.asarray(gr), rtol=1e-6)


def test_ger():
    for m, n in [(4, 5)]:
        for order in ['f', 'c']:
            for sliced_x in [1, 2, -2, -1]:
                for sliced_y in [1, 2, -2, -1]:
                    yield ger, m, n, 'float32', order, sliced_x, sliced_y, \
                        False

    yield ger, 4, 5, 'float64', 'f', 1, 1, False

    for init_res in [True, False]:
        for overwrite in [True, False]:
            yield ger, 4, 5, 'float32', 'f', 1, 1, init_res, overwrite


def ger(m, n, dtype, order, sliced_x, sliced_y, init_res, overwrite=False):
    cX, gX = gen_gpuarray((m,), dtype, order, sliced=sliced_x, ctx=context)
    cY, gY = gen_gpuarray((n,), dtype, order, sliced=sliced_y, ctx=context)

    if init_res:
        cA, gA = gen_gpuarray((m, n), dtype, order, ctx=context)
    else:
        cA, gA = None, None

    if dtype == 'float32':
        cr = fblas.sger(1.0, cX, cY, a=cA, overwrite_a=overwrite)
    else:
        cr = fblas.dger(1.0, cX, cY, a=cA, overwrite_a=overwrite)

    gr = gblas.ger(1.0, gX, gY, gA, overwrite_a=overwrite)

    numpy.testing.assert_allclose(cr, numpy.asarray(gr), rtol=1e-6)
