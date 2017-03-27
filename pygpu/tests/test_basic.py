from pygpu.basic import (tril, triu)
from .support import (gen_gpuarray, context)
import numpy


def test_tril():
    for shape in [(10, 5), (5, 10), (10, 10)]:
        for order in ['c', 'f']:
            for inplace in [True, False]:
                ac, ag = gen_gpuarray(shape, 'float32',
                                      order=order, ctx=context)
                result = tril(ag, inplace=inplace)
                assert numpy.all(numpy.tril(ac) == result)
                if inplace:
                    assert numpy.all(numpy.tril(ac) == ag)
                else:
                    assert numpy.all(ac == ag)


def test_triu():
    for shape in [(10, 5), (5, 10), (10, 10)]:
        for order in ['c', 'f']:
            for inplace in [True, False]:
                ac, ag = gen_gpuarray(shape, 'float32',
                                      order=order, ctx=context)
                result = triu(ag, inplace=inplace)
                assert numpy.all(numpy.triu(ac) == result)
                if inplace:
                    assert numpy.all(numpy.triu(ac) == ag)
                else:
                    assert numpy.all(ac == ag)
