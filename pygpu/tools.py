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

    def ctype(self):
        return dtype_to_ctype(self.dtype)


class ArrayArg(Argument):
    def decltype(self):
        return "GLOBAL_MEM %s *" % (self.ctype(),)

    def expr(self):
        return "%s[i]" % (self.name,)

    def isarray(self):
        return True


class ScalarArg(Argument):
    def decltype(self):
        return self.ctype()

    def expr(self):
        return self.name

    def isarray(self):
        return False


def check_args(args, collapse=True):
    arrays = []
    strs = []
    offsets = []
    for arg in args:
        if isinstance(arg, GpuArray):
            strs.append(arg.strides)
            offsets.append(arg.offset)
            arrays.append(arg)
        else:
            strs.append(None)
            offsets.append(None)

        if len(arrays) < 1:
            raise ArugmentError("No arrays in kernel arguments, "
                                "something is wrong")
    n = arrays[0].size
    nd = arrays[0].ndim
    dims = arrays[0].shape
    c_contig = True
    f_contig = True
    for ary in arrays:
        if dims != ary.shape:
            raise ValueError("Some array differs from the others in shape")
        c_contig = c_contig and ary.flags['C_CONTIGUOUS']
        f_contig = f_contig and ary.flags['F_CONTIGUOUS']

    contig = c_contig or f_contig

    if not contig and collapse and nd > 1:
        # make the strides and dims editable
        dims = list(dims)
        strs = [list(str) for str in strs]
        # remove dimensions that are of size 1
        for i in range(nd-1, -1, -1):
            if dims[i] == 1:
                del dims[i]
                for str in strs:
                    del str[i]
                nd -= 1

        # collapse contiguous dimensions
        for i in range(nd-1, 0, -1):
            if all(str[i] * dims[i] == str[i-1] for str in strs):
                dims[i-1] *= dims[i]
                del dims[i]
                for str in strs:
                    str[i-1] = str[i]
                    del str[i]
                nd -= 1
        # re-wrap dims and tuples
        dims = tuple(dims)
        strs = [tuple(str) for str in strs]

    return n, nd, dims, strs, offsets, contig
