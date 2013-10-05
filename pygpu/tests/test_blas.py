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
    for shape in [(100, 128)]:
        for dtype in ['float32', 'float64']:
            for order in ['f', 'c']:
                for trans in [False, True]:
                    for offseted_o in [True, False]:
                        for offseted_i in [True, False]:
                            for sliced in [1, 2, -1, -2]:
                                for overwrite in [True, False]:
                                    for init_y in [True, False]:
                                        yield sgemv, shape, dtype, order, \
                                            trans, offseted_i, offseted_o, \
                                            sliced, overwrite, init_y

#@guard_devsup
def sgemv(shp, dtype, order, trans, offseted_i, offseted_o, sliced,
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
    gr = gblas.gemv(1, gA, gX, 0, gY, trans=trans, overwrite_y=overwrite)

    assert numpy.allclose(cr, numpy.asarray(gr))
