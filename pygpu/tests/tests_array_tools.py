from pygpu.array_tools import (tril, triu)
from .support import (gen_gpuarray, context)
import numpy


def test_triu_inplace():
    ac, ag = gen_gpuarray((10, 10), 'float32', ctx=context)
    result = triu(ag, context, inplace=True)
    assert numpy.all(numpy.triu(ac) == result)
    assert numpy.all(numpy.triu(ac) == ag)


def test_triu_inplace_order_f():
    ac, ag = gen_gpuarray((10, 10), 'float32', order='f', ctx=context)
    result = triu(ag, context, inplace=True)
    assert numpy.all(numpy.triu(ac) == result)
    assert numpy.all(numpy.triu(ac) == ag)


def test_triu_no_inplace():
    ac, ag = gen_gpuarray((10, 10), 'float32', ctx=context)
    result = triu(ag, context, inplace=False)
    assert numpy.all(numpy.triu(ac) == result)
    assert numpy.all(ac == ag)


def test_tril_inplace():
    ac, ag = gen_gpuarray((10, 10), 'float32', ctx=context)
    result = tril(ag, context, inplace=True)
    assert numpy.all(numpy.tril(ac) == result)
    assert numpy.all(numpy.tril(ac) == ag)


def test_tril_inplace_order_f():
    ac, ag = gen_gpuarray((10, 10), 'float32', order='f', ctx=context)
    result = tril(ag, context, inplace=True)
    assert numpy.all(numpy.tril(ac) == result)
    assert numpy.all(numpy.tril(ac) == ag)


def test_tril_no_inplace():
    ac, ag = gen_gpuarray((10, 10), 'float32', ctx=context)
    result = tril(ag, context, inplace=False)
    assert numpy.all(numpy.tril(ac) == result)
    assert numpy.all(ac == ag)
