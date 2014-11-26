"""Type mapping helpers."""

from __future__ import division

__copyright__ = "Copyright (C) 2011 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import gpuarray
import numpy as np


# {{{ registry

NAME_TO_DTYPE = {}

def register_dtype(dtype, c_names):
    """
    Associate a numpy dtype with its C equivalents.

    :param dtype: type to associate
    :type dtype: numpy.dtype or string
    :param c_names: list of C type names
    :type c_names: str or list

    Will register `dtype` for use with the gpuarray module.  If the
    c_names argument is a list then the first element of that list is
    taken as the primary association and will be used for generated C
    code.  The other types will be mapped to the provided dtype when
    going in the other direction.
    """
    if isinstance(c_names, str):
        c_names = [c_names]

    dtype = np.dtype(dtype)

    # register if not already there
    try:
        gpuarray.dtype_to_ctype(dtype)
    except ValueError:
        gpuarray.register_dtype(dtype, c_names[0])

    for nm in c_names:
        if nm in NAME_TO_DTYPE and NAME_TO_DTYPE[nm] != dtype:
            raise RuntimeError("name '%s' already registered" % nm)
        NAME_TO_DTYPE[nm] = dtype

def _fill_dtype_registry(respect_windows):
    from sys import platform

    register_dtype(np.bool, ["ga_bool", "bool"])
    register_dtype(np.int8, ["ga_byte", "char", "signed char"])
    register_dtype(np.uint8, ["ga_ubyte", "unsigned char"])
    register_dtype(np.int16, ["ga_short", "short", "signed short", "signed short int", "short signed int"])
    register_dtype(np.uint16, ["ga_ushort", "unsigned short", "unsigned short int", "short unsigned int"])
    register_dtype(np.int32, ["ga_int", "int", "signed int"])
    register_dtype(np.uint32, ["ga_uint", "unsigned", "unsigned int"])

    register_dtype(np.int64, ["ga_long"])
    register_dtype(np.uint64, ["ga_ulong"])
    is_64_bit = tuple.__itemsize__ * 8 == 64
    if is_64_bit:
        if 'win32' in platform and respect_windows:
            i64_name = "long long"
        else:
            i64_name = "long"
        register_dtype(np.int64, [i64_name, "%s int" % i64_name,
                                  "signed %s int" % i64_name,
                                  "%s signed int" % i64_name])
        register_dtype(np.uint64, ["unsigned %s" % i64_name,
                                   "unsigned %s int" % i64_name,
                                   "%s unsigned int" % i64_name])

    # According to this uintp may not have the same hash as uint32:
    # http://projects.scipy.org/numpy/ticket/2017
    # Failing tests tell me this is the case for intp too.
    if is_64_bit:
        register_dtype(np.intp, ["ga_long"])
        register_dtype(np.uintp, ["ga_ulong"])
    else:
        register_dtype(np.intp, ["ga_int"])
        register_dtype(np.uintp, ["ga_uint"])

    register_dtype(np.float32, ["ga_float", "float"])
    register_dtype(np.float64, ["ga_double", "double"])

# }}}

# {{{ dtype -> ctype

def dtype_to_ctype(dtype, with_fp_tex_hack=False):
    """
    Return the C type that corresponds to `dtype`.

    :param dtype: a numpy dtype
    """
    if dtype is None:
        raise ValueError("dtype may not be None")

    dtype = np.dtype(dtype)
    if with_fp_tex_hack:
        if dtype == np.float32:
            return "fp_tex_float"
        elif dtype == np.float64:
            return "fp_tex_double"

    return gpuarray.dtype_to_ctype(dtype)

# }}}

# {{{ c declarator parsing

def parse_c_arg_backend(c_arg, scalar_arg_class, vec_arg_class):
    c_arg = c_arg.replace("const", "").replace("volatile", "")

    # process and remove declarator
    import re
    decl_re = re.compile(r"(\**)\s*([_a-zA-Z0-9]+)(\s*\[[ 0-9]*\])*\s*$")
    decl_match = decl_re.search(c_arg)

    if decl_match is None:
        raise ValueError("couldn't parse C declarator '%s'" % c_arg)

    name = decl_match.group(2)

    if decl_match.group(1) or decl_match.group(3) is not None:
        arg_class = vec_arg_class
    else:
        arg_class = scalar_arg_class

    tp = c_arg[:decl_match.start()]
    tp = " ".join(tp.split())

    try:
        dtype = NAME_TO_DTYPE[tp]
    except KeyError:
        raise ValueError("unknown type '%s'" % tp)

    return arg_class(dtype, name)

# }}}


def get_np_obj(obj):
    """
    Returns a numpy object of the same dtype and comportement as the
    source suitable for output dtype determination.

    This is used since the casting rules of numpy are rather obscure
    and the best way to imitate them is to try an operation ans see
    what it does.
    """
    if isinstance(obj, np.ndarray) and obj.shape == ():
        return obj
    try:
        return np.ones(1, dtype=obj.dtype)
    except AttributeError:
        return np.asarray(obj)


def get_common_dtype(obj1, obj2, allow_double):
    """
    Returns the proper output type for a numpy operation involving the
    two provided objects.  This may not be suitable for certain
    obscure numpy operations.

    If `allow_double` is False, a return type of float64 will be
    forced to float32 and complex128 will be forced to complex64.
    """
    # Yes, numpy behaves differently depending on whether
    # we're dealing with arrays or scalars.

    np1 = get_np_obj(obj1)
    np2 = get_np_obj(obj2)

    result = (np1 + np2).dtype

    if not allow_double:
        if result == np.float64:
            result = np.dtype(np.float32)
        elif result == np.complex128:
            result = np.dtype(np.complex64)

    return result


def upcast(*args):
    a = np.array([0], dtype=args[0])
    for t in args[1:]:
        a = a + np.array([0], dtype=t)
    return a.dtype


# vim: foldmethod=marker
