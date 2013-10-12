import numpy

from .support import (guard_devsup, gen_gpuarray, context)

try:
    import scipy.linalg.blas
    try:
        fblas = scipy.linalg.blas.fblas
    except AttributeError:
        fblas = scipy.linalg.blas
except ImportError, e:
    raise SkipTest("no scipy blas to compare against")

import pygpu.blas as gblas

def test_gemv():
    for shape in [(100, 128), (128, 100)]:
        for dtype in ['float32', 'float64']:
            for order in ['f', 'c']:
                for trans in [False, True]:
                    for offseted_o in [True, False]:
                        for offseted_i in [True, False]:
                            for sliced in [1, 2, -1, -2]:
                                for overwrite in [True, False]:
                                    for init_y in [True, False]:
                                        yield gemv, shape, dtype, order, \
                                            trans, offseted_i, offseted_o, \
                                            sliced, overwrite, init_y

@guard_devsup
def gemv(shp, dtype, order, trans, offseted_i, offseted_o, sliced,
          overwrite, init_y):
    cA, gA = gen_gpuarray(shp, dtype, order=order, offseted_outer=offseted_o,
                          offseted_inner=offseted_i, sliced=sliced,
                          ctx=context)
    if trans:
        shpX = (shp[0],)
        shpY = (shp[1],)
    else:
        shpX = (shp[1],)
        shpY = (shp[0],)
    cX, gX = gen_gpuarray(shpX, dtype, offseted_outer=offseted_o,
                          offseted_inner=offseted_i, sliced=sliced,
                          ctx=context)
    if init_y:
        cY, gY = gen_gpuarray(shpY, dtype, ctx=context)
    else:
        cY, gY = None, None

    if dtype == 'float32':
        cr = fblas.sgemv(1, cA, cX, 0, cY, trans=trans, overwrite_y=overwrite)
    else:
        cr = fblas.dgemv(1, cA, cX, 0, cY, trans=trans, overwrite_y=overwrite)
    gr = gblas.gemv(1, gA, gX, 0, gY, trans_a=trans, overwrite_y=overwrite)

    numpy.testing.assert_allclose(cr, numpy.asarray(gr), rtol=1e-6)

def test_gemm():
    for m, n, k in [(48, 15, 32)]:#, (48, 32, 15), (15, 48, 32), (15, 32, 48),
#                    (32, 48, 15), (32, 15, 48)]:
        for dtype in ['float32']:#, 'float64']:
            for order in [('f', 'f', 'f'), ('c', 'c', 'c'), ('f', 'f', 'c'),
                          ('f', 'c', 'f'), ('f', 'c', 'c'), ('c', 'f', 'f')]:
#                          ('c', 'f', 'c'), ('c', 'c', 'f')]:
                for trans in [(False, False)]:#, (False, True), (True, False),
#                              (True, True)]:
                    for offseted_o in [False]:#, True]:
                        for offseted_i in [False]:#, True]:
                            for sliced in [1]:#, 2, -1, -2]:
                                for overwrite in [True]:#, False]:
                                    for init_res in [True]:#, False]:
                                        yield gemm, m, n, k, dtype, order, \
                                            trans, offseted_i, offseted_o, \
                                            sliced, overwrite, init_res

@guard_devsup
def gemm(m, n, k, dtype, order, trans, offseted_i, offseted_o,
         sliced, overwrite, init_res):
    if trans[0]:
        shpA = (k,m)
    else:
        shpA = (m,k)
    if trans[1]:
        shpB = (n,k)
    else:
        shpB = (k,n)

    cA, gA = gen_gpuarray(shpA, dtype, order=order[0], offseted_outer=offseted_o,
                          offseted_inner=offseted_i, sliced=sliced,
                          ctx=context)
    cB, gB = gen_gpuarray(shpB, dtype, order=order[1], offseted_outer=offseted_o,
                          offseted_inner=offseted_i, sliced=sliced,
                          ctx=context)
    if init_res:
        cC, gC = gen_gpuarray((m,n), dtype, order=order[2], ctx=context)
    else:
        cC, gC = None, None

    if dtype == 'float32':
        cr = fblas.sgemm(1, cA, cB, 0, cC, trans_a=trans[0], trans_b=trans[1],
                         overwrite_c=overwrite)
    else:
        cr = fblas.dgemm(1, cA, cB, 0, cC, trans_a=trans[0], trans_b=trans[1],
                         overwrite_c=overwrite)
    gr = gblas.gemm(1, gA, gB, 0, gC, trans_a=trans[0], trans_b=trans[1],
                    overwrite_c=overwrite)

    numpy.testing.assert_allclose(cr, numpy.asarray(gr), rtol=1e-6)
