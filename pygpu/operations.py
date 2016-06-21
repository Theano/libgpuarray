from six.moves import range

from .gpuarray import _split, _concatenate, dtype_to_typecode
from .dtypes import upcast
from . import array, asarray


def _replace_0_with_empty(aryl, m):
    for i in range(len(aryl)):
        if (len(aryl[i].shape) == 0 or
                any(s == 0 for s in aryl[i].shape)):
            aryl[i] = array([], dtype=m.dtype, cls=type(m), context=m.context)
    return aryl


def atleast_1d(*arys):
    res = []
    for ary in arys:
        ary = asarray(ary)
        if len(ary.shape) == 0:
            result = ary.reshape((1,))
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def atleast_2d(*arys):
    res = []
    for ary in arys:
        ary = asarray(ary)
        if len(ary.shape) == 0:
            result = ary.reshape((1, 1))
        elif len(ary.shape) == 1:
            result = ary.reshape((1, ary.shape[0]))
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def atleast_3d(*arys):
    res = []
    for ary in arys:
        ary = asarray(ary)
        if len(ary.shape) == 0:
            result = ary.reshape((1, 1, 1))
        elif len(ary.shape) == 1:
            result = ary.reshape((1, ary.shape[0], 1))
        elif len(ary.shape) == 2:
            result = ary.reshape(ary.shape + (1,))
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def split(ary, indices_or_sections, axis=0):
    try:
        len(indices_or_sections)
    except TypeError:
        if ary.shape[axis] % indices_or_sections != 0:
            raise ValueError("array split does not result in an "
                             "equal division")
    return array_split(ary, indices_or_sections, axis)


def array_split(ary, indices_or_sections, axis=0):
    try:
        indices = list(indices_or_sections)
        res = _split(ary, indices, axis)
    except TypeError:
        if axis < 0:
            axis += ary.ndim
        if axis < 0:
            raise ValueError('axis out of bounds')
        nsec = int(indices_or_sections)
        if nsec <= 0:
            raise ValueError('number of sections must be larger than 0.')
        neach, extra = divmod(ary.shape[axis], nsec)
        # this madness is to support the numpy interface
        # it is supported by tests, but little else
        divs = (list(range(neach + 1, (neach + 1) * extra + 1, neach + 1)) +
                list(range((neach + 1) * extra + neach, ary.shape[axis], neach)))
        res = _split(ary, divs, axis)
    return _replace_0_with_empty(res, ary)


def hsplit(ary, indices_or_sections):
    if len(ary.shape) == 0:
        raise ValueError('hsplit only works on arrays of 1 or more dimensions')
    if len(ary.shape) > 1:
        axis = 1
    else:
        axis = 0
    return split(ary, indices_or_sections, axis=axis)


def vsplit(ary, indices_or_sections):
    if len(ary.shape) < 2:
        raise ValueError('vsplit only works on arrays of 2 or more dimensions')
    return split(ary, indices_or_sections, axis=0)


def dsplit(ary, indices_or_sections):
    if len(ary.shape) < 3:
        raise ValueError('vsplit only works on arrays of 3 or more dimensions')
    return split(ary, indices_or_sections, axis=2)


def concatenate(arys, axis=0, context=None):
    if len(arys) == 0:
        raise ValueError("concatenation of zero-length sequences is "
                         "impossible")
    if axis < 0:
        axis += arys[0].ndim
    if axis < 0:
        raise ValueError('axis out of bounds')
    al = [asarray(a, context=context) for a in arys]
    if context is None:
        context = al[0].context
    outtype = upcast(*[a.dtype for a in arys])
    return _concatenate(al, axis, dtype_to_typecode(outtype), type(al[0]),
                        context)


def vstack(tup, context=None):
    return concatenate([atleast_2d(a) for a in tup], 0, context)


def hstack(tup, context=None):
    tup = [atleast_1d(a) for a in tup]
    if tup[0].ndim == 1:
        return concatenate(tup, 0, context)
    else:
        return concatenate(tup, 1, context)


def dstack(tup, context=None):
    return concatenate([atleast_3d(a) for a in tup], 2, context)
