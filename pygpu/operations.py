from .gpuarray import _split
from . import array

def _replace_0_with_empty(aryl, m):
    for i in range(len(aryl)):
        if (len(aryl[i].shape) == 0 or
            any(s == 0 for s in aryl[i].shape)):
            aryl[i] = array([], dtype=m.dtype, cls=type(m), context=m.context)
    return aryl


def split(ary, indices_or_sections, axis=0):
    try: len(indices_or_sections)
    except TypeError:
        if ary.shape[axis] % indices_or_sections != 0:
            raise ValueError, ("array split does not result in an "
                               "equal division")
    return array_split(ary, indices_or_sections, axis)

def array_split(ary, indices_or_sections, axis=0):
    try:
        indices = list(indices_or_sections)
        res = _split(ary, indices, axis)
    except TypeError:
        nsec = int(indices_or_sections)
        if nsec <= 0:
            raise ValueError, 'number of sections must be larger than 0.'
        neach, extra = divmod(ary.shape[axis], nsec)
        # this madness is to support the numpy interface
        # it is supported by tests, but little else
        divs = list(range(neach + 1, (neach + 1) * extra + 1, neach + 1) +
                    range((neach + 1) * extra + neach, ary.shape[axis], neach))
        res = _split(ary, divs, axis)
    return _replace_0_with_empty(res, ary)


def hsplit(ary, indices_or_sections):
    if len(ary.shape) == 0:
        raise ValueError, 'hsplit only works on arrays of 1 or more dimensions'
    if len(ary.shape) > 1:
        axis = 1
    else:
        axis = 0
    return split(ary, indices_or_sections, axis=axis)


def vsplit(ary, indices_or_sections):
    if len(ary.shape) < 2:
        raise ValueError, 'vsplit only works on arrays of 2 or more dimensions'
    return split(ary, indices_or_sections, axis=0)


def dsplit(ary, indices_or_sections):
    if len(ary.shape) < 3:
        raise ValueError, 'vsplit only works on arrays of 3 or more dimensions'
    return split(ary, indices_or_sections, axis=2)
