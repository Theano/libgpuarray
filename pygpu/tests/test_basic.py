import pygpu

from pygpu.basic import (tril, triu)
from unittest import TestCase
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


class test_errors(TestCase):

    def runTest(self):
        self.assertRaises(ValueError, self.run_1d_triu)
        self.assertRaises(ValueError, self.run_3d_triu)
        self.assertRaises(ValueError, self.run_1d_tril)
        self.assertRaises(ValueError, self.run_3d_tril)

        self.assertRaises(ValueError, self.run_noncontiguous_tril)
        self.assertRaises(ValueError, self.run_noncontiguous_triu)

    def run_1d_triu(self):
        ac, ag = gen_gpuarray((10, ), 'float32', ctx=context)
        triu(ag)

    def run_3d_triu(self):
        ac, ag = gen_gpuarray((10, 10, 10), 'float32', ctx=context)
        triu(ag)

    def run_1d_tril(self):
        ac, ag = gen_gpuarray((10, ), 'float32', ctx=context)
        tril(ag)

    def run_3d_tril(self):
        ac, ag = gen_gpuarray((10, 10, 10), 'float32', ctx=context)
        tril(ag)

    def run_noncontiguous_tril(self):
        a = numpy.random.rand(5, 5)
        a = a[::-1]
        b = pygpu.array(a, context=context)
        assert b.flags.c_contiguous is b.flags.f_contiguous is False
        tril(b)

    def run_noncontiguous_triu(self):
        a = numpy.random.rand(5, 5)
        a = a[::-1]
        b = pygpu.array(a, context=context)
        assert b.flags.c_contiguous is b.flags.f_contiguous is False
        triu(b)
