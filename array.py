from __future__ import division
from pytools import memoize_method
import numpy as np




def f_contiguous_strides(itemsize, shape):
    if shape:
        strides = [itemsize]
        for s in shape[:-1]:
            strides.append(strides[-1]*s)
        return tuple(strides)
    else:
        return ()

def c_contiguous_strides(itemsize, shape):
    if shape:
        strides = [itemsize]
        for s in shape[:0:-1]:
            strides.append(strides[-1]*s)
        return tuple(strides[::-1])
    else:
        return ()




class ArrayFlags:
    def __init__(self, ary):
        self.array = ary

    @property
    @memoize_method
    def f_contiguous(self):
        return self.array.strides == f_contiguous_strides(
                self.array.dtype.itemsize, self.array.shape)

    @property
    @memoize_method
    def c_contiguous(self):
        return self.array.strides == c_contiguous_strides(
                self.array.dtype.itemsize, self.array.shape)

    @property
    @memoize_method
    def forc(self):
        return self.f_contiguous or self.c_contiguous




def get_common_dtype(obj1, obj2):
    return (obj1.dtype.type(0) + obj2.dtype.type(0)).dtype




# {{{ as_strided implementation

# stolen from numpy to be compatible with older versions of numpy

class _DummyArray(object):
    """ Dummy object that just exists to hang __array_interface__ dictionaries
    and possibly keep alive a reference to a base array.
    """
    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base

def as_strided(x, shape=None, strides=None):
    """ Make an ndarray from the given array with the given shape and strides.
    """
    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)
    return np.asarray(_DummyArray(interface, base=x))

# }}}
