from __future__ import division
from pytools import memoize_method




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

