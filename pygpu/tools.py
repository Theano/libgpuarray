import numpy
from dtypes import dtype_to_ctype, _fill_dtype_registry
from gpuarray import GpuArray

_fill_dtype_registry(respect_windows=False)


def as_argument(obj, name):
    if isinstance(obj, GpuArray):
        return ArrayArg(obj.dtype, name)
    else:
        return ScalarArg(numpy.asarray(obj).dtype, name)


class Argument(object):
    def __init__(self, dtype, name):
        self.dtype = dtype
        self.name = name


class ArrayArg(Argument):
    def decltype(self):
        return "GLOBAL_MEM %s *" % (dtype_to_ctype(self.dtype))

    def expr(self):
        return "%s[i]" % (self.name,)

    def isarray(self):
        return True


class ScalarArg(Argument):
    def decltype(self):
        return dtype_to_ctype(self.dtype)

    def expr(self):
        return self.name

    def isarray(self):
        return False
