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

type_to_prefix = {
    'float32': 's',
    'float64': 'd',
}

def test_sgemv():
    for shape in [(128, 128), (400, 784)]:
        for order in ['f', 'c']:
            for trans in [False, True]:
                for offseted in [True, False]:
                    yield sgemv, shape, order, trans, offseted

def sgemv(shp, order, trans, offseted_o):
    cA, gA = gen_gpuarray(shp, 'float32', order=order, ctx=context)
    cX, gX = gen_gpuarray((shp[1],), 'float32', offseted_o, ctx=context)
    cY, gY = gen_gpuarray((shp[0],), 'float32', offseted_o, ctx=context)

    fblas.sgemv(1, cA, cX, 0, cY, trans=False, overwrite_y=True)
    gblas.sgemv(1, gA, gX, 0, gY, trans=False, overwrite_y=True)

    assert numpy.allclose(cY, numpy.asarray(gY))
