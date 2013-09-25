import numpy

from .support import (gen_gpuarray, context)

try:
    import scipy.linalg.blas
    try:
        fblas = scipy.linalg.blas.fblas
    except AttributeError:
        fblas = scipy.linalg.blas
except ImportError, e:
    raise SkipTest("no scipy blas to compare against")

import pygpu.blas as gblas

def test_sgemv():
    for shape in [(128, 128)]:
        for order in ['f', 'c']:
            yield sgemv_notrans, shape, order
            yield sgemv_trans, shape, order

def sgemv_notrans(shp, order):
    cA, gA = gen_gpuarray(shp, 'float32', order=order, ctx=context)
    cX, gX = gen_gpuarray((shp[0],), 'float32', ctx=context)
    cY, gY = gen_gpuarray((shp[1],), 'float32', ctx=context)

    fblas.sgemv(1, cA, cX, 0, cY, trans=False, overwrite_y=True)
    gblas.sgemv(1, gA, gX, 0, gY, overwrite_y=True)

    assert numpy.allclose(cY, numpy.asarray(gY))

def sgemv_trans(shp, order):
    cA, gA = gen_gpuarray(shp, 'float32', order=order, ctx=context)
    cX, gX = gen_gpuarray((shp[1],), 'float32', ctx=context)
    cY, gY = gen_gpuarray((shp[0],), 'float32', ctx=context)

    fblas.sgemv(1, cA, cX, 0, cY, trans=False, overwrite_y=True)
    gblas.sgemv(1, gA, gX, 0, gY, overwrite_y=True)

    assert numpy.allclose(cY, numpy.asarray(gY))
