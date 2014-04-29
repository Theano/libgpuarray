import numpy
import pygpu

from .support import (gen_gpuarray, context)


def test_array_split():
    xc, xg = gen_gpuarray((8,), 'float32', ctx=context)
    rc = numpy.array_split(xc, 3)
    rg = pygpu.array_split(xg, 3)

    assert len(rc) == len(rg)
    for pc, pg in zip(rc, rg):
        numpy.testing.assert_allclose(pc, numpy.asarray(pg))

def test_split():
    for spl in (3, [3, 5, 6, 10]):
        yield xsplit, '', (9,), spl

def test_xsplit():
    for l in ('h', 'v'):
        for spl in (2, [3, 6]):
            yield xsplit, l, (4, 4), spl
        yield xsplit, l, (2, 2, 2), 2
    for spl in (2, [3, 6]):
        yield xsplit, 'd', (2, 2, 4), spl
        
def xsplit(l, shp, spl):
    xc, xg = gen_gpuarray(shp, 'float32', ctx=context)
    n = l + 'split'
    rc = getattr(numpy, n)(xc, spl)
    rg = getattr(pygpu, n)(xg, spl)

    assert len(rc) == len(rg)
    for pc, pg in zip(rc, rg):
        numpy.testing.assert_allclose(pc, numpy.asarray(pg))
