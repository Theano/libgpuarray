import functools
import six
from six.moves import reduce

from heapq import nsmallest
from operator import itemgetter, mul

import numpy

from .dtypes import dtype_to_ctype, _fill_dtype_registry
from .gpuarray import GpuArray

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
        return "GLOBAL_MEM {} *".format(self.ctype())

    def expr(self):
        return "{}[i]".format(self.name)

    def isarray(self):
        return True

    def spec(self):
        return GpuArray


class ScalarArg(Argument):
    def decltype(self):
        return self.ctype()

    def expr(self):
        return self.name

    def isarray(self):
        return False

    def spec(self):
        return self.dtype


def check_contig(args):
    dims = None
    c_contig = f_contig = True
    offsets = []
    for arg in args:
        if not isinstance(arg, GpuArray):
            offsets.append(None)
            continue

        if dims is None:
            dims = arg.shape
            n = arg.size
        elif arg.shape != dims:
            return None, None, False
        offsets.append(arg.offset)
        fl = arg.flags
        c_contig = c_contig and fl['C_CONTIGUOUS']
        f_contig = f_contig and fl['F_CONTIGUOUS']
    if not (c_contig or f_contig):
        return None, None, False
    return n, tuple(offsets), True


def check_args(args, collapse=False, broadcast=False):
    """
    Returns the properties of arguments and checks if they all match
    (are all the same shape)

    If `collapse` is True dimension collapsing will be performed.
    If `collapse` is False dimension collapsing will not be performed.

    If `broadcast` is True array broadcasting will be performed which
    means that dimensions which are of size 1 in some arrays but not
    others will be repeated to match the size of the other arrays.
    If `broadcast` is False no broadcasting takes place.
    """

    # For compatibility with old collapse=None option
    if collapse is None:
        collapse = True

    strs = []
    offsets = []
    dims = None
    for arg in args:
        if isinstance(arg, GpuArray):
            strs.append(arg.strides)
            offsets.append(arg.offset)
            if dims is None:
                n, nd, dims = arg.size, arg.ndim, arg.shape
            else:
                if arg.ndim != nd:
                    raise ValueError("Array order differs")
                if not broadcast and arg.shape != dims:
                    raise ValueError("Array shape differs")
        else:
            strs.append(None)
            offsets.append(None)

    if dims is None:
        raise TypeError("No arrays in kernel arguments, "
                        "something is wrong")
    tdims = dims

    if broadcast or collapse:
        # make the strides and dims editable
        dims = list(dims)
        strs = [list(str) if str is not None else str for str in strs]

    if broadcast:
        # Set strides to 0s when needed.
        # Get the full shape in dims (no ones unless all arrays have it).
        if 1 in dims:
            for i, ary in enumerate(args):
                if strs[i] is None:
                    continue
                shp = ary.shape
                for i, d in enumerate(shp):
                    if dims[i] != d and dims[i] == 1:
                        dims[i] = d
                        n *= d
            tdims = tuple(dims)

        for i, ary in enumerate(args):
            if strs[i] is None:
                continue
            shp = ary.shape
            if tdims != shp:
                for j, d in enumerate(shp):
                    if dims[j] != d:
                        # Might want to add a per-dimension enable mechanism
                        if d == 1:
                            strs[i][j] = 0
                        else:
                            raise ValueError("Array shape differs")

    if collapse and nd > 1:
        # remove dimensions that are of size 1
        for i in range(nd - 1, -1, -1):
            if nd > 1 and dims[i] == 1:
                del dims[i]
                for str in strs:
                    if str is not None:
                        del str[i]
                nd -= 1

        # collapse contiguous dimensions
        for i in range(nd - 1, 0, -1):
            if all(str is None or str[i] * dims[i] == str[i - 1]
                   for str in strs):
                dims[i - 1] *= dims[i]
                del dims[i]
                for str in strs:
                    if str is not None:
                        str[i - 1] = str[i]
                        del str[i]
                nd -= 1

    if broadcast or collapse:
        # re-wrap dims and tuples
        dims = tuple(dims)
        strs = [tuple(str) if str is not None else None for str in strs]

    return n, nd, dims, tuple(strs), tuple(offsets)


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
                    for key, _ in nsmallest(wrapper.maxsize // 10,
                                            six.iteritems(use_count),
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


def lru_cache(maxsize=20):
    def decorating_function(user_function):
        cache = {}
        last_use = {}
        time = [0]  # workaround for Python 2, which doesn't have nonlocal

        @functools.wraps(user_function)
        def wrapper(*key):
            time[0] += 1
            last_use[key] = time[0]

            try:
                result = cache[key]
                wrapper.hits += 1
            except KeyError:
                result = user_function(*key)
                cache[key] = result
                wrapper.misses += 1

                # purge least recently used cache entries
                if len(cache) > wrapper.maxsize:
                    for key, _ in nsmallest(wrapper.maxsize // 10,
                                            six.iteritems(last_use),
                                            key=itemgetter(1)):
                        del cache[key], last_use[key]

            return result

        def clear():
            cache.clear()
            last_use.clear()
            wrapper.hits = wrapper.misses = 0
            time[0] = 0

        @functools.wraps(user_function)
        def get(*key):
            result = cache[key]
            time[0] += 1
            last_use[key] = time[0]
            wrapper.hits += 1
            return result

        wrapper.hits = wrapper.misses = 0
        wrapper.maxsize = maxsize
        wrapper.clear = clear
        wrapper.get = get
        return wrapper
    return decorating_function


def prod(iterable):
    return reduce(mul, iterable, 1)
