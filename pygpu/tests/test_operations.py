import numpy
import pygpu

from .support import (gen_gpuarray, context, SkipTest)


def test_array_split():
    xc, xg = gen_gpuarray((8,), 'float32', ctx=context)
    rc = numpy.array_split(xc, 3)
    rg = pygpu.array_split(xg, 3)

    assert len(rc) == len(rg)
    for pc, pg in zip(rc, rg):
        numpy.testing.assert_allclose(pc, numpy.asarray(pg))

    xc, xg = gen_gpuarray((8,), 'float32', ctx=context)
    rc = numpy.array_split(xc, 3, axis=-1)
    rg = pygpu.array_split(xg, 3, axis=-1)

    assert len(rc) == len(rg)
    for pc, pg in zip(rc, rg):
        numpy.testing.assert_allclose(pc, numpy.asarray(pg))


def test_split():
    for spl in (3, [3, 5, 6, 10]):
        yield xsplit, '', (9,), spl


def test_xsplit():
    if tuple(int(v) for v in numpy.version.version.split('.')[:2]) < (1, 11):
        raise SkipTest("Numpy version too old")
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


def test_concatenate():
    ac, ag = gen_gpuarray((2, 2), 'float32', ctx=context)
    bc, bg = gen_gpuarray((1, 2), 'float32', ctx=context)

    rc = numpy.concatenate((ac, bc), axis=0)
    rg = pygpu.concatenate((ag, bg), axis=0)

    numpy.testing.assert_allclose(rc, numpy.asarray(rg))

    rc = numpy.concatenate((ac, bc.T), axis=1)
    rg = pygpu.concatenate((ag, bg.T), axis=1)

    numpy.testing.assert_allclose(rc, numpy.asarray(rg))

    rc = numpy.concatenate((ac, bc.T), axis=-1)
    rg = pygpu.concatenate((ag, bg.T), axis=-1)

    numpy.testing.assert_allclose(rc, numpy.asarray(rg))


def test_hstack():
    for shp in [(3,), (3, 1)]:
        yield xstack, 'h', (shp, shp), (), context


def test_vstack():
    for shp in [(3,), (3, 1)]:
        yield xstack, 'v', (shp, shp), (), context


def test_dstack():
    for shp in [(3,), (3, 1)]:
        yield xstack, 'd', (shp, shp), (), context


def xstack(l, shps, tup, ctx):
    tupc = list(tup)
    tupg = list(tup)
    for shp in shps:
        tc, tg = gen_gpuarray(shp, 'float32', ctx=context)
        tupc.append(tc)
        tupg.append(tg)
    n = l + 'stack'
    rc = getattr(numpy, n)(tuple(tupc))
    rg = getattr(pygpu, n)(tuple(tupg), ctx)

    numpy.testing.assert_allclose(rc, numpy.asarray(rg))
