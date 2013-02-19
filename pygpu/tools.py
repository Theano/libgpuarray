import collections
import functools
from itertools import ifilterfalse
from heapq import nsmallest
from operator import itemgetter

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

    def __hash__(self):
        return hash(type(self)) ^ hash(self.dtype) ^ hash(self.name)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.dtype == other.dtype and
                self.name == other.name)


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


def check_args(args, collapse=False, broadcast=False):
    """
    Returns the properties of arguments and checks if they all match
    (are all the same shape)

    If `collapse` is True dimension collapsing will be performed.
    If `collapse` is None dimension collapsing will be perform if some
    arguments are non-contiguous.
    If `collapse` is False dimension collapsing will not be performed.

    `broadcast` is untested and should be left to the default value (False).
    """
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
            raise TypeError("No arrays in kernel arguments, "
                            "something is wrong")
    n = arrays[0].size
    nd = arrays[0].ndim
    dims = arrays[0].shape
    tdims = dims
    c_contig = True
    f_contig = True

    if broadcast or collapse:
        # make the strides and dims editable
        dims = list(dims)
        strs = [list(str) if str is not None else str for str in strs]

    if broadcast and 1 in dims:
        # Get the full shape in dims (no ones unless all arrays have it).
        for ary in arrays:
            shp = ary.shape
            if len(dims) != len(shp):
                raise ValueError("Array order differs")
            for i, d in enumerate(shp):
                if dims[i] != d and dims[i] == 1:
                    dims[i] = d
        tdims = tuple(dims)

    for i, ary in enumerate(arrays):
        fl = ary.flags
        shp = ary.shape
        c_contig = c_contig and fl['C_CONTIGUOUS']
        f_contig = f_contig and fl['F_CONTIGUOUS']
        if tdims != shp:
            if broadcast:
                # we checked for matching order in the
                # broadcast preparation loop above
                for j, d in enumerate(shp):
                    if dims[j] != d:
                        # Might want to add a per-dimension enable mechanism
                        if d == 1:
                            strs[i][j] = 0
                            c_contig = False
                            f_contig = False
                        else:
                            raise ValueError("Array shape differs")
            else:
                raise ValueError("Array shape differs")

    contig = c_contig or f_contig

    if not contig and collapse is None:
        # make the strides and dims editable
        dims = list(dims)
        strs = [list(str) if str is not None else str for str in strs]
        collapse = True

    if nd > 1 and collapse:
        # remove dimensions that are of size 1
        for i in range(nd-1, -1, -1):
            if dims[i] == 1:
                del dims[i]
                for str in strs:
                    if str is not None:
                        del str[i]
                nd -= 1

        # collapse contiguous dimensions
        for i in range(nd-1, 0, -1):
            if all(str is None or str[i] * dims[i] == str[i-1] for str in strs):
                dims[i-1] *= dims[i]
                del dims[i]
                for str in strs:
                    if str is not None:
                        str[i-1] = str[i]
                        del str[i]
                nd -= 1
    if broadcast or collapse:
        # re-wrap dims and tuples
        dims = tuple(dims)
        strs = [tuple(str) if str is not None else None for str in strs]

    return n, nd, dims, tuple(strs), tuple(offsets), contig

class Counter(dict):
    'Mapping where default values are zero'
    def __missing__(self, key):
        return 0

def lfu_cache(maxsize=20):
    def decorating_function(user_function):
        cache = {}
        use_count = Counter()

        @functools.wraps(user_function)
        def wrapper(*key):
            use_count[key] += 1

            try:
                result = cache[key]
                wrapper.hits += 1
            except KeyError:
                result = user_function(*key)
                cache[key] = result
                wrapper.misses += 1

                # purge least frequently used cache entry
                if len(cache) > wrapper.maxsize:
                    for key, _ in nsmallest(maxsize // 10,
                                            use_count.iteritems(),
                                            key=itemgetter(1)):
                        del cache[key], use_count[key]

            return result

        def clear():
            cache.clear()
            use_count.clear()
            wrapper.hits = wrapper.misses = 0

        @functools.wraps(user_function)
        def get(*key):
            result = cache[key]
            use_count[key] += 1
            wrapper.hits += 1
            return result

        wrapper.hits = wrapper.misses = 0
        wrapper.maxsize = maxsize
        wrapper.clear = clear
        wrapper.get = get
        return wrapper
    return decorating_function
