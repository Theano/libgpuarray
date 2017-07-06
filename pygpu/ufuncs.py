﻿"""Ufuncs and reductions for GPU arrays."""

import mako
import numpy
from pkg_resources import parse_version
import sys
import warnings
from pygpu.dtypes import dtype_to_ctype, NAME_TO_DTYPE
from pygpu._elemwise import arg
from pygpu.elemwise import as_argument, GpuElemwise
from pygpu.reduction import reduce1
from pygpu.gpuarray import GpuArray, array, empty, get_default_context

PY3 = sys.version_info.major >= 3

# Save for later use since the original names are used for reductions
builtin_all = all
builtin_any = any


# --- Helper functions --- #


def restore_reduced_dims(shape, red_axes):
    """Return tuple from ``shape`` with size-1 axes at indices ``red_axes``."""
    newshape = list(shape)
    try:
        for ax in sorted(red_axes):
            newshape.insert(ax, 1)
    except TypeError:
        newshape.insert(red_axes, 1)
    return tuple(newshape)


def reduce_dims(shape, red_axes):
    """Return tuple from ``shape`` with ``red_axes`` removed."""
    newshape = list(shape)
    try:
        for ax in reversed(sorted(red_axes)):
            newshape.pop(ax)
    except TypeError:
        newshape.pop(red_axes)
    return tuple(newshape)


# --- Reductions with arithmetic operators --- #


def _prepare_array_for_reduction(a, out, context=None):
    """Return input array ready for usage in a reduction kernel."""
    # Lazy import to avoid circular dependency
    from pygpu._array import ndgpuarray

    # Get a context and an array class to work with. Use the "highest"
    # class present in the inputs.
    need_context = True
    cls = None
    for ary in (a, out):
        if isinstance(ary, GpuArray):
            if context is not None and ary.context != context:
                raise ValueError('cannot mix contexts')
            context = ary.context
            if cls is None or cls == GpuArray:
                cls = ary.__class__
            need_context = False

    if need_context and context is None:
        context = get_default_context()
        cls = ndgpuarray

    if not isinstance(a, GpuArray):
        a = numpy.asarray(a)
        if a.flags.f_contiguous and not a.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'
        a = array(a, dtype=a.dtype, copy=False, order=order, context=context,
                  cls=cls)

    return a


def reduce_with_op(a, op, neutral, axis=None, dtype=None, out=None,
                   keepdims=False, context=None):
    """Reduce ``a`` using the operation ``op``.

    This is a wrapper around `pygpu.reduction.reduce1` with signature
    closer to NumPy and parameter ``keepdims``.

    Parameters
    ----------
    a : `pygpu.gpuarray.GpuArray`
        Array that should be reduced.
    op : str
        Operation to be used for reduction. The reduction logic is::

            result = a op b

    neutral : scalar
        Neutral element of the operation fulfilling ``(n op a) == a``
        for all ``a``. It is used as initial state of the result.

    axis, dtype, out, keepdims :
        Arguments as in NumPy reductions. See e.g. `numpy.sum`.
    context : `pygpu.gpuarray.GpuContext`, optional
        Use this GPU context to evaluate the GPU kernel. For ``None``,
        and if neither ``out`` nor ``a`` are GPU arrays, a default
        GPU context must have been set.

    Returns
    -------
    reduced : `pygpu.gpuarray.GpuArray` or scalar
        If not all axes are reduced over or ``out`` was given, the result is
        an array, a reference to ``out`` if provided. Otherwise, i.e. for
        reductions along all axes without ``out`` parameter, the result
        is a scalar.
    """
    a = _prepare_array_for_reduction(a, out, context)

    if dtype is None:
        if numpy.issubsctype(a.dtype, numpy.unsignedinteger):
            # Avoid overflow for small integer types by default, as in Numpy
            out_type = max(a.dtype, numpy.dtype('uint'))
        elif numpy.issubsctype(a.dtype, numpy.integer):
            out_type = max(a.dtype, numpy.dtype('int'))
        else:
            out_type = a.dtype
    else:
        out_type = dtype

    axes = axis if axis is not None else tuple(range(a.ndim))
    if out is not None:
        out_arr = out.reshape(reduce_dims(a.shape, axes))
    else:
        out_arr = out

    r = reduce1(a, op=op, neutral=neutral, out_type=out_type, axis=axis,
                out=out_arr)
    if keepdims:
        newshape = restore_reduced_dims(r.shape, axes)
        r = r.reshape(newshape)
    if not r.shape and out is None:
        return numpy.asarray(r).reshape([1])[0]
    elif out is not None:
        return out
    else:
        return r


def sum(a, axis=None, dtype=None, out=None, keepdims=False, context=None):
    """Sum of array elements over a given axis.

    See Also
    --------
    numpy.sum
    """
    # Do what Numpy does with booleans, sensible or not
    if a.dtype == bool and dtype is None:
        dtype = int
    return reduce_with_op(a, '+', 0, axis, dtype, out, keepdims, context)


def prod(a, axis=None, dtype=None, out=None, keepdims=False, context=None):
    """Return the product of array elements over a given axis.

    See Also
    --------
    numpy.prod
    """
    # Do what Numpy does with booleans, sensible or not
    if a.dtype == bool and dtype is None:
        dtype = int
    return reduce_with_op(a, '*', 1, axis, dtype, out, keepdims, context)


def all(a, axis=None, out=None, keepdims=False, context=None):
    """Test whether all array elements along a given axis evaluate to True.

    See Also
    --------
    numpy.all
    """
    return reduce_with_op(a, '&&', 1, axis, numpy.bool, out, keepdims, context)


def any(a, axis=None, out=None, keepdims=False, context=None):
    """Test whether all array elements along a given axis evaluate to True.

    See Also
    --------
    numpy.all
    """
    return reduce_with_op(a, '||', 0, axis, numpy.bool, out, keepdims, context)


# --- Reductions with comparison operators --- #


def reduce_with_cmp(a, cmp, neutral, axis=None, out=None, keepdims=False,
                    context=None):
    """Reduce ``a`` by comparison using ``cmp``.

    This is a wrapper around `pygpu.reduction.reduce1` with signature
    closer to NumPy and parameter ``keepdims``.

    Parameters
    ----------
    a : `pygpu.gpuarray.GpuArray`
        Array that should be reduced.
    cmp : str
        Comparison operator to be used for reduction. The reduction
        logic is::

            result = (a cmp b) ? a : b

    neutral : scalar
        Neutral element of the comparison fulfilling ``(n cmp a) == True``
        for all ``a``. It is used as initial state of the result.

    axis, out, keepdims :
        Arguments as in NumPy reductions. See e.g. `numpy.amax`.
    context : `pygpu.gpuarray.GpuContext`, optional
        Use this GPU context to evaluate the GPU kernel. For ``None``,
        and if neither ``out`` nor ``a`` are GPU arrays, a default
        GPU context must have been set.

    Returns
    -------
    reduced : `pygpu.gpuarray.GpuArray` or scalar
        If not all axes are reduced over or ``out`` was given, the result is
        an array, a reference to ``out`` if provided. Otherwise, i.e. for
        reductions along all axes without ``out`` parameter, the result
        is a scalar.
    """
    a = _prepare_array_for_reduction(a, out, context)

    axes = axis if axis is not None else tuple(range(a.ndim))
    if out is not None:
        out_arr = out.reshape(reduce_dims(a.shape, axes))
    else:
        out_arr = out

    oper = '(a {} b) ? a : b'.format(cmp)
    r = reduce1(a, op=None, neutral=neutral, out_type=a.dtype, axis=axis,
                out=out_arr, oper=oper)
    if keepdims:
        axes = axis if axis is not None else tuple(range(a.ndim))
        newshape = restore_reduced_dims(r.shape, axes)
        r = r.reshape(newshape)
    if not r.shape and out is None:
        return numpy.asarray(r).reshape([1])[0]
    elif out is not None:
        return out
    else:
        return r


def amin(a, axis=None, out=None, keepdims=False, context=None):
    """Return the minimum of an array or minimum along an axis.

    See Also
    --------
    numpy.amin
    """
    a = _prepare_array_for_reduction(a, out)
    if a.dtype == numpy.bool:
        neutral = 1
    elif numpy.issubsctype(a.dtype, numpy.integer):
        neutral = numpy.iinfo(a.dtype).max
    elif numpy.issubsctype(a.dtype, numpy.floating):
        neutral = 'INFINITY'
    elif numpy.issubsctype(a.dtype, numpy.complexfloating):
        raise ValueError('array dtype {!r} not comparable'
                         ''.format(a.dtype.name))
    else:
        raise ValueError('array dtype {!r} not supported'
                         ''.format(a.dtype.name))
    return reduce_with_cmp(a, '<', neutral, axis, out, keepdims, context)


def amax(a, axis=None, out=None, keepdims=False, context=None):
    """Return the maximum of an array or minimum along an axis.

    See Also
    --------
    numpy.amax
    """
    a = _prepare_array_for_reduction(a, out)
    if a.dtype == numpy.bool:
        neutral = 0
    elif numpy.issubsctype(a.dtype, numpy.integer):
        neutral = numpy.iinfo(a.dtype).min
    elif numpy.issubsctype(a.dtype, numpy.floating):
        neutral = '-INFINITY'
    elif numpy.issubsctype(a.dtype, numpy.complexfloating):
        raise ValueError('array dtype {!r} not comparable'
                         ''.format(a.dtype.name))
    else:
        raise ValueError('array dtype {!r} not supported'
                         ''.format(a.dtype.name))
    return reduce_with_cmp(a, '>', neutral, axis, out, keepdims, context)


# --- Elementwise ufuncs --- #


# This dictionary is derived from Numpy's C99_FUNCS list, see
# https://github.com/numpy/numpy/search?q=C99_FUNCS
UNARY_C_FUNC_TO_UFUNC = {
    'abs': 'absolute',
    'acos': 'arccos',
    'acosh': 'arccosh',
    'asin': 'arcsin',
    'asinh': 'arcsinh',
    'atan': 'arctan',
    'atanh': 'arctanh',
    'cbrt': 'cbrt',
    'ceil': 'ceil',
    'cos': 'cos',
    'cosh': 'cosh',
    'exp': 'exp',
    'exp2': 'exp2',
    'expm1': 'expm1',
    'fabs': 'fabs',
    'floor': 'floor',
    'log': 'log',
    'log10': 'log10',
    'log1p': 'log1p',
    'log2': 'log2',
    'rint': 'rint',
    'sin': 'sin',
    'sinh': 'sinh',
    'sqrt': 'sqrt',
    'tan': 'tan',
    'tanh': 'tanh',
    'trunc': 'trunc',
    }
UNARY_UFUNC_TO_C_FUNC = {v: k for k, v in UNARY_C_FUNC_TO_UFUNC.items()}
UNARY_UFUNC_TO_C_OP = {
    'bitwise_not': '~',
    'logical_not': '!',
    'negative': '-',
    }
UNARY_UFUNCS = (list(UNARY_UFUNC_TO_C_FUNC.keys()) +
                list(UNARY_UFUNC_TO_C_OP.keys()))
UNARY_UFUNCS.extend(['deg2rad', 'rad2deg', 'reciprocal', 'sign', 'signbit',
                     'square', 'isinf', 'isnan', 'isfinite', 'spacing'])
UNARY_UFUNCS_TWO_OUT = ['frexp', 'modf']
BINARY_C_FUNC_TO_UFUNC = {
    'atan2': 'arctan2',
    'copysign': 'copysign',
    'hypot': 'hypot',
    'ldexp': 'ldexp',
    'pow': 'power',
    'nextafter': 'nextafter',
    'fmod': 'fmod',
}
BINARY_UFUNC_TO_C_FUNC = {v: k for k, v in BINARY_C_FUNC_TO_UFUNC.items()}
BINARY_UFUNC_TO_C_BINOP = {
    'add': '+',
    'bitwise_and': '&',
    'bitwise_or': '|',
    'bitwise_xor': '^',
    'left_shift': '<<',
    'logical_and': '&&',
    'logical_or': '||',
    'multiply': '*',
    'right_shift': '>>',
    'subtract': '-',
    }
BINARY_UFUNC_TO_C_CMP = {
    'equal': '==',
    'greater': '>',
    'greater_equal': '>=',
    'less': '<',
    'less_equal': '<=',
    'not_equal': '!=',
    }
BINARY_UFUNC_TO_C_OP = {}
BINARY_UFUNC_TO_C_OP.update(BINARY_UFUNC_TO_C_BINOP)
BINARY_UFUNC_TO_C_OP.update(BINARY_UFUNC_TO_C_CMP)
BINARY_UFUNCS = (list(BINARY_UFUNC_TO_C_FUNC.keys()) +
                 list(BINARY_UFUNC_TO_C_OP.keys()))
BINARY_UFUNCS.extend(['floor_divide', 'true_divide', 'logical_xor',
                      'maximum', 'minimum', 'remainder', 'logaddexp',
                      'logaddexp2'])


def ufunc_dtypes(ufunc_name, dtypes_in):
    """Return result dtypes for a ufunc and input dtypes.

    Parameters
    ----------
    ufunc_name : str
        Name of the Numpy ufunc.
    dtypes_in : sequence of `numpy.dtype`
        Data types of the arrays.

    Returns
    -------
    prom_in_dtypes : tuple of `numpy.dtype`
        Promoted input dtypes, different from ``dtypes_in`` if type
        promotion is necessary for the ufunc.
    result_dtypes : tuple of `numpy.dtype`
        Resulting data types of the ufunc. If a function is not suited
        for integers, the dtype is promoted to the smallest possible
        supported floating point data type.

    Examples
    --------
    Real floating point types:

    >>> ufunc_dtypes('absolute', [numpy.dtype('float64')])
    ((dtype('float64'),), (dtype('float64'),))
    >>> ufunc_dtypes('absolute', [numpy.dtype('float32')])
    ((dtype('float32'),), (dtype('float32'),))
    >>> ufunc_dtypes('power', [numpy.dtype('float32'), numpy.dtype('float64')])
    ((dtype('float64'), dtype('float64')), (dtype('float64'),))
    >>> ufunc_dtypes('power', [numpy.dtype('float32'), numpy.dtype('float32')])
    ((dtype('float32'), dtype('float32')), (dtype('float32'),))

    Integer types -- some functions produce integer results, others
    need to convert to floating point:

    >>> ufunc_dtypes('absolute', [numpy.dtype('int8')])
    ((dtype('int8'),), (dtype('int8'),))
    >>> ufunc_dtypes('absolute', [numpy.dtype('int16')])
    ((dtype('int16'),), (dtype('int16'),))
    >>> ufunc_dtypes('exp', [numpy.dtype('int8')])
    ((dtype('float32'),), (dtype('float32'),))
    >>> ufunc_dtypes('exp', [numpy.dtype('int16')])
    ((dtype('float32'),), (dtype('float32'),))
    >>> ufunc_dtypes('exp', [numpy.dtype('int32')])
    ((dtype('float64'),), (dtype('float64'),))
    >>> ufunc_dtypes('power', [numpy.dtype('int8'), numpy.dtype('int8')])
    ((dtype('int8'), dtype('int8')), (dtype('int8'),))
    >>> ufunc_dtypes('power', [numpy.dtype('int8'), numpy.dtype('float32')])
    ((dtype('float32'), dtype('float32')), (dtype('float32'),))
    """
    npy_ufunc = getattr(numpy, ufunc_name)
    supported_dtypes = set(NAME_TO_DTYPE.values())

    # Filter for dtypes larger than our input dtypes, using only supported ones
    def larger_eq_than_dtypes(sig):
        from_part = sig.split('->')[0]
        if len(from_part) != len(dtypes_in):
            return False
        else:
            dts = tuple(numpy.dtype(c) for c in from_part)
            # Currently unsupported, filtering out
            if builtin_any(dt not in supported_dtypes for dt in dts):
                return False
            else:
                return builtin_all(dt >= dt_in
                                   for dt, dt_in in zip(dts, dtypes_in))

    # List of ufunc signatures that are "larger" than our input dtypes
    larger_sig_list = list(filter(larger_eq_than_dtypes, npy_ufunc.types))
    if not larger_sig_list:
        # Numpy raises TypeError for bad data types, which is not quite right,
        # but we mirror that behavior
        raise TypeError('data types {} not supported for ufunc {}'
                        ''.format(tuple(dt.name for dt in dtypes_in),
                                  ufunc_name))

    # Key function for signature comparison. It results in comparison of
    # *all* typecodes in the signature since they are assembled in a tuple.
    def from_part_key(sig):
        from_part = sig.split('->')[0]
        return tuple(numpy.dtype(c) for c in from_part)

    # Get the smallest signature larger than our input dtypes
    smallest_sig = min(larger_sig_list, key=from_part_key)
    smallest_str_in, smallest_str_out = smallest_sig.split('->')
    prom_dtypes = tuple(numpy.dtype(c) for c in smallest_str_in)
    result_dtypes = tuple(numpy.dtype(c) for c in smallest_str_out)

    # Quad precision unsupported also on output side
    if builtin_any(dt in result_dtypes for dt in (numpy.dtype('float16'),
                                                  numpy.dtype('float128'))):
        # TODO: Numpy raises TypeError for bad data types, which is wrong,
        # but we mirror that behavior
        raise TypeError('data types {} not supported for ufunc {}'
                        ''.format(tuple(dt.name for dt in dtypes_in),
                                  ufunc_name))

    return prom_dtypes, result_dtypes


def ufunc_c_fname(ufunc_name, dtypes_in):
    """Return C function name for a ufunc and input dtypes.

    The names are only specialized for complex math and to distinguish
    between functions for absolute value. There is no name-mangling for
    specific precisions.

    Parameters
    ----------
    ufunc_name : str
        Name of the Numpy ufunc.
    dtype_in : sequence of `numpy.dtype`
        Data types of input arrays.

    Returns
    -------
    cname : str
        C name of the ufunc.

    Examples
    --------
    >>> ufunc_c_fname('exp', [numpy.dtype('float32')])
    'exp'
    >>> ufunc_c_fname('exp', [numpy.dtype('int16')])
    'exp'
    >>> ufunc_c_fname('absolute', [numpy.dtype('float64')])
    'fabs'
    >>> ufunc_c_fname('absolute', [numpy.dtype('float32')])
    'fabs'
    >>> ufunc_c_fname('absolute', [numpy.dtype('int8')])
    'abs'
    >>> ufunc_c_fname('power',
    ...               [numpy.dtype('float32'), numpy.dtype('float64')])
    'pow'

    See Also
    --------
    ufunc_result_dtype : Get dtype of a ufunc result.
    """
    _, result_dtypes = ufunc_dtypes(ufunc_name, dtypes_in)
    result_dtype = result_dtypes[0]
    c_base_name = UNARY_UFUNC_TO_C_FUNC.get(ufunc_name, None)
    if c_base_name is None:
        c_base_name = BINARY_UFUNC_TO_C_FUNC.get(ufunc_name)

    if c_base_name == 'abs':
        if numpy.issubsctype(dtypes_in[0], numpy.floating):
            c_base_name = 'fabs'

    if numpy.issubsctype(result_dtype, numpy.complexfloating):
        prefix = 'c'
        # Currently broken
        raise NotImplementedError('complex math kernels currently not '
                                  'available')
    else:
        prefix = ''

    return prefix + c_base_name


def unary_ufunc(a, ufunc_name, out=None, context=None):
    """Call a unary ufunc on an array ``a``.

    Parameters
    ----------
    a : `array-like`
        Input array.
    ufunc_name : str
        Name of the NumPy ufunc to be called on ``a``.
    out : `pygpu.gpuarray.GpuArray`, optional
        Array in which to store the result. Its shape must be equal to
        ``a.shape`` and its dtype must be the result dtype of the called
        function.
    context : `pygpu.gpuarray.GpuContext`, optional
        Use this GPU context to evaluate the GPU kernel. For ``None``,
        and if neither ``out`` nor ``a`` are GPU arrays, a default
        GPU context must have been set.

    Returns
    -------
    out : `pygpu.gpuarray.GpuArray`
        Result of the computation. If ``out`` was given, the returned
        object is a reference to it. If ``a`` was not a GPU arrays, the
        type of ``out`` is `pygpu._array.ndgpuarray`.

    See Also
    --------
    ufunc_result_dtype : Get dtype of a ufunc result.
    ufunc_c_funcname : Get C name for math ufuncs.
    """
    # Lazy import to avoid circular dependency
    from pygpu._array import ndgpuarray

    # Get a context and an array class to work with. Use the "highest"
    # class present in the inputs.
    need_context = True
    cls = None
    for ary in (a, out):
        if isinstance(ary, GpuArray):
            if context is not None and ary.context != context:
                raise ValueError('cannot mix contexts')
            context = ary.context
            if cls is None or cls == GpuArray:
                cls = ary.__class__
            need_context = False

    if need_context and context is None:
        context = get_default_context()
        cls = ndgpuarray

    if not isinstance(a, GpuArray):
        a = numpy.asarray(a)
        if a.flags.f_contiguous and not a.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'
        a = array(a, dtype=a.dtype, copy=False, order=order, context=context,
                  cls=cls)

    if a.dtype == numpy.dtype('float16'):
        # Gives wrong results currently, see
        # https://github.com/Theano/libgpuarray/issues/316
        raise NotImplementedError('float16 currently broken')

    prom_dtypes_in, result_dtypes = ufunc_dtypes(ufunc_name, [a.dtype])
    prom_dtype_in = prom_dtypes_in[0]
    result_dtype = result_dtypes[0]

    # This is the "fallback signature" case, for us it signals failure.
    # TypeError is what Numpy raises, too, which is kind of wrong
    if prom_dtype_in == numpy.dtype(object):
        raise TypeError('input dtype {!r} invalid for ufunc {!r}'
                        ''.format(a.dtype.name, ufunc_name))

    # Convert input such that the kernel runs
    # TODO: can this be avoided?
    a = a.astype(prom_dtype_in, copy=False)

    if out is None:
        out = empty(a.shape, dtype=result_dtype, context=context, cls=cls)
    else:
        # TODO: allow larger dtype
        if out.dtype != result_dtype:
            raise ValueError('`out.dtype` != result dtype ({!r} != {!r})'
                             ''.format(out.dtype.name, result_dtype.name))

    # C result dtype for casting
    c_res_dtype = dtype_to_ctype(result_dtype)

    oper = ''

    # Case 1: math function
    if ufunc_name in UNARY_UFUNC_TO_C_FUNC:
        c_func = ufunc_c_fname(ufunc_name, (a.dtype,))
        # Shortcut for abs() with unsigned int. This also fixes a CUDA quirk
        # that makes abs() fail with unsigned ints.
        if (ufunc_name == 'absolute' and
                numpy.issubsctype(a.dtype, numpy.unsignedinteger)):
            out[:] = a.copy()
            return out
        else:
            oper = 'res = ({}) {}(a)'.format(c_res_dtype, c_func)

    # Case 2: unary operator
    unop = UNARY_UFUNC_TO_C_OP.get(ufunc_name, None)
    if unop is not None:
        if a.dtype == numpy.bool and unop == '-':
            if parse_version(numpy.__version__) >= parse_version('1.13'):
                # Numpy >= 1.13 raises a TypeError
                raise TypeError(
                    'negation of boolean arrays is not supported, use '
                    '`logical_not` instead')
            else:
                # Warn and remap to logical not
                warnings.warn('using negation (`-`) with boolean arrays is '
                              'deprecated, use `logical_not` (`~`) instead; '
                              'the current behavior will be changed along '
                              "with NumPy's", FutureWarning)
                unop = '!'
        oper = 'res = ({}) {}a'.format(c_res_dtype, unop)

    # Other cases: specific functions
    if ufunc_name == 'deg2rad':
        oper = 'res = ({rdt})({:.45f}) * ({rdt}) a'.format(numpy.deg2rad(1.0),
                                                           rdt=c_res_dtype)

    elif ufunc_name == 'rad2deg':
        oper = 'res = ({rdt})({:.45f}) * ({rdt}) a'.format(numpy.rad2deg(1.0),
                                                           rdt=c_res_dtype)

    elif ufunc_name == 'reciprocal':
        oper = 'res = ({dt}) (({dt}) 1.0) / a'.format(dt=c_res_dtype)

    elif ufunc_name == 'sign':
        oper = 'res = ({}) ((a > 0) ? 1 : (a < 0) ? -1 : 0)'.format(
            c_res_dtype)

    elif ufunc_name == 'signbit':
        oper = 'res = ({}) (a < 0)'.format(c_res_dtype)

    elif ufunc_name == 'square':
        oper = 'res = ({}) (a * a)'.format(c_res_dtype)

    elif ufunc_name == 'isfinite':
        oper = '''
        res = ({}) (a != INFINITY && a != -INFINITY && !isnan(a))
        '''.format(c_res_dtype)

    elif ufunc_name == 'isinf':
        oper = 'res = ({}) (a == INFINITY || a == -INFINITY)'.format(
            c_res_dtype)

    elif ufunc_name == 'isnan':
        oper = 'res = ({}) (abs(isnan(a)))'.format(c_res_dtype)

    elif ufunc_name == 'spacing':
        if numpy.issubsctype(a.dtype, numpy.integer):
            # TODO: float16 as soon as it is properly supported
            cast_dtype = numpy.result_type(a.dtype, numpy.float32)
            c_cast_dtype = dtype_to_ctype(cast_dtype)
        else:
            c_cast_dtype = dtype_to_ctype(a.dtype)
        oper = '''
        res = ({}) ((a < 0) ?
                    nextafter(({ct}) a, ({ct}) a - 1) - a :
                    nextafter(({ct}) a, ({ct}) a + 1) - a)
        '''.format(c_res_dtype, ct=c_cast_dtype)

    if not oper:
        raise ValueError('`ufunc_name` {!r} does not represent a unary ufunc'
                         ''.format(ufunc_name))

    a_arg = as_argument(a, 'a', read=True)
    args = [arg('res', out.dtype, write=True), a_arg]

    ker = GpuElemwise(context, oper, args, convert_f16=True)
    ker(out, a)
    return out


def binary_ufunc(a, b, ufunc_name, out=None, context=None):
    """Call binary ufunc on ``a`` and ``b``.

    Parameters
    ----------
    a, b : `array-like`
        Input arrays.
    ufunc_name : str
        Name of the NumPy ufunc to be called on ``a`` and ``b``.
    out : `pygpu.gpuarray.GpuArray`, optional
        Array in which to store the result. Its shape must be equal to
        ``a.shape`` and its dtype must be the result dtype of the called
        function.
    context : `pygpu.gpuarray.GpuContext`, optional
        Use this GPU context to evaluate the GPU kernel. For ``None``,
        and if neither ``out`` nor ``a`` are GPU arrays, a default
        GPU context must have been set.

    Returns
    -------
    out : `pygpu.gpuarray.GpuArray`
        Result of the computation. If ``out`` was given, the returned
        object is a reference to it. If both input arrays were not
        GPU arrays, the type of ``out`` is `pygpu._array.ndgpuarray`.

    See Also
    --------
    pygpu.gpuarray.set_default_context
    """
    # Lazy import to avoid circular dependency
    from pygpu._array import ndgpuarray

    # Get a context and an array class to work with
    cls = None
    need_context = True
    for ary in (a, b, out):
        if isinstance(a, GpuArray):
            context = ary.context
            cls = ary.__class__
            need_context = False
            break

    if need_context and context is None:
        context = get_default_context()
        cls = ndgpuarray

    if not isinstance(a, GpuArray):
        if numpy.isscalar(a):
            # TODO: this is quite hacky, perhaps mixed input signatures
            # should be handled in ufunc_dtypes?
            if ufunc_name == 'ldexp':
                # Want signed type
                # TODO: this upcasts to a larger dtype than Numpy in
                # some cases (mostly unsigned int)
                a = numpy.asarray(a, dtype=numpy.min_scalar_type(-abs(a)))
            else:
                a = numpy.asarray(a, dtype=numpy.result_type(a, b))

        if a.flags.f_contiguous and not a.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'
        a = array(a, dtype=a.dtype, copy=False, order=order, context=context,
                  cls=cls)

    if not isinstance(b, GpuArray):
        if numpy.isscalar(b):
            # TODO: this is quite hacky, perhaps mixed input signatures
            # should be handled in ufunc_dtypes?
            if ufunc_name == 'ldexp':
                # Want signed type
                b = numpy.asarray(b, dtype=numpy.min_scalar_type(-abs(b)))
            else:
                b = numpy.asarray(b, dtype=numpy.result_type(a, b))

        if isinstance(b, numpy.ndarray):
            if b.flags.f_contiguous and not b.flags.c_contiguous:
                order = 'F'
            else:
                order = 'C'
        else:
            b = numpy.asarray(b)
            order = 'C'
        b = array(b, dtype=b.dtype, copy=False, order=order, context=context,
                  cls=cls)

    if builtin_any(ary.dtype == numpy.dtype('float16') for ary in (a, b)):
        # Gives wrong results currently, see
        # https://github.com/Theano/libgpuarray/issues/316
        raise NotImplementedError('float16 currently broken')

    prom_dtypes_in, result_dtypes = ufunc_dtypes(ufunc_name,
                                                 [a.dtype, b.dtype])
    prom_dtype_a, prom_dtype_b = prom_dtypes_in
    result_dtype = result_dtypes[0]

    # This is the "fallback signature" case, for us it signals failure
    if builtin_any(dt == numpy.dtype(object) for dt in prom_dtypes_in):
        raise TypeError('input dtypes {} invalid for ufunc {!r}'
                        ''.format((a.dtype.name, b.dtype.name), ufunc_name))

    # Convert input such that the kernel runs
    # TODO: can this be avoided?
    a = a.astype(prom_dtype_a, copy=False)
    b = b.astype(prom_dtype_b, copy=False)

    if a.ndim != b.ndim:
        nd = max(a.ndim, b.ndim)
        if a.ndim < nd:
            a = a.reshape(((1,) * (nd - a.ndim)) + a.shape)
        if b.ndim < nd:
            b = b.reshape(((1,) * (nd - b.ndim)) + b.shape)
    result_shape = tuple(max(sa, sb) for sa, sb in zip(a.shape, b.shape))

    if out is None:
        out = empty(result_shape, dtype=result_dtype, context=context, cls=cls)
    else:
        if out.shape != result_shape:
            raise ValueError('`out.shape` != result shape ({} != {})'
                             ''.format(out.shape, result_shape))
        if out.dtype != result_dtype:
            raise ValueError('`out.dtype` != result dtype ({!r} != {!r})'
                             ''.format(out.dtype.name, result_dtype.name))

    a_arg = as_argument(a, 'a', read=True)
    b_arg = as_argument(b, 'b', read=True)
    args = [arg('res', result_dtype, write=True), a_arg, b_arg]

    # C result dtype for casting
    c_res_dtype = dtype_to_ctype(result_dtype)

    # Set string for mapping operation and preamble for additional functions
    oper = ''
    preamble = ''

    # Case 1: math function
    if ufunc_name in BINARY_UFUNC_TO_C_FUNC:
        if ufunc_name == 'power':
            # Arguments to `pow` cannot be integer, need to cast
            if numpy.issubsctype(result_dtype, numpy.integer):
                tpl = 'res = ({rt}) (long) round(pow((double) a, (double) b))'
                oper = tpl.format(rt=c_res_dtype)
            else:
                oper = 'res = ({rt}) pow(({rt}) a, ({rt}) b)'.format(
                    rt=c_res_dtype)

        elif ufunc_name == 'fmod':
            # Arguments to `fmod` cannot be integer, need to cast
            if numpy.issubsctype(result_dtype, numpy.integer):
                oper = 'res = ({rt}) fmod((double) a, (double) b)'.format(
                    rt=c_res_dtype)
            else:
                oper = 'res = ({rt}) fmod(({rt}) a, ({rt}) b)'.format(
                    rt=c_res_dtype)

        else:
            c_fname = ufunc_c_fname(ufunc_name, [a.dtype, b.dtype])
            oper = 'res = ({}) {}(a, b)'.format(c_res_dtype, c_fname)

    # Case 2: binary operator
    binop = BINARY_UFUNC_TO_C_OP.get(ufunc_name, None)
    if binop is not None:
        if b.dtype == numpy.bool and binop == '-':
            if parse_version(numpy.__version__) >= parse_version('1.13'):
                # Numpy >= 1.13 raises a TypeError
                raise TypeError(
                    'subtraction of boolean arrays is not supported, use '
                    '`logical_not` instead')
            else:
                # Warn and remap to logical not
                warnings.warn('using subtraction (`-`) with boolean arrays is '
                              'deprecated, use `bitwise_xor` (`^`) or '
                              '`logical_xor` instead; '
                              'the current behavior will be changed along '
                              "with NumPy's", FutureWarning)
                binop = '^'

        oper = 'res = ({}) (a {} b)'.format(c_res_dtype, binop)
    else:
        # Other cases: specific functions
        if ufunc_name == 'floor_divide':
            # implement as sign(a/b) * int(abs(a/b) + shift(a,b))
            # where shift(a,b) = 0 if sign(a) == sign(b) else 1 - epsilon
            preamble = '''
            WITHIN_KERNEL long
            floor_div_dbl(double a, double b) {
                double quot = a / b;
                if ((a < 0) != (b < 0)) {
                    return - (long) (quot + 0.999);
                } else {
                    return (long) quot;
                }
            }
            '''
            oper = 'res = ({}) floor_div_dbl((double) a, (double) b)'.format(
                c_res_dtype)

        elif ufunc_name == 'true_divide':
            if result_dtype == numpy.dtype('float64'):
                flt = 'double'
            else:
                flt = 'float'
            oper = 'res = ({}) (({flt}) a / ({flt}) b)'.format(c_res_dtype,
                                                               flt=flt)

        elif ufunc_name == 'logical_xor':
            oper = 'res = ({}) (a ? !b : b)'.format(c_res_dtype)

        elif ufunc_name == 'maximum':
            oper = 'res = ({}) ((a > b) ? a : b)'.format(c_res_dtype)

        elif ufunc_name == 'minimum':
            oper = 'res = ({}) ((a < b) ? a : b)'.format(c_res_dtype)

        elif ufunc_name == 'remainder':
            # The same as `fmod` except for b < 0, where we have
            # remainder(a, b) = fmod(a, b) + b
            if numpy.issubsctype(result_dtype, numpy.integer):
                cast_type = 'double'
            else:
                cast_type = c_res_dtype

            preamble = mako.template.Template('''
            WITHIN_KERNEL ${ct}
            rem(${ct} a, ${ct} b) {
                ${ct} modval = fmod(a, b);
                if (b < 0 && modval != 0) {
                    return b + modval;
                } else {
                    return modval;
                }
            }
            ''').render(ct=cast_type)
            oper = 'res = ({rt}) rem(({ct}) a, ({ct}) b)'.format(
                rt=c_res_dtype, ct=cast_type)

        elif ufunc_name == 'logaddexp':
            oper = 'res = ({}) log(exp(a) + exp(b))'.format(c_res_dtype)

        elif ufunc_name == 'logaddexp2':
            oper = '''
            res = ({}) log(exp(a * log(2.0)) + exp(b * log(2.0))) / log(2.0)
            '''.format(c_res_dtype)

    if not oper:
        raise ValueError('`ufunc_name` {!r} does not represent a binary '
                         'ufunc'.format(ufunc_name))

    kernel = GpuElemwise(a.context, oper, args, convert_f16=True,
                         preamble=preamble)
    kernel(out, a, b, broadcast=True)
    return out


def unary_ufunc_two_out(a, ufunc_name, out1=None, out2=None, context=None):
    """Call a unary ufunc with two outputs on an array ``a``.

    Parameters
    ----------
    a : `array-like`
        Input array.
    ufunc_name : str
        Name of the NumPy ufunc to be called on ``a``.
    out1, out2 : `pygpu.gpuarray.GpuArray`, optional
        Arrays in which to store the result. Their shape must be equal to
        ``a.shape`` and their dtype must be the result dtypes of the
        called function.
        The arrays ``out1`` and ``out2`` must either both be provided,
        or none of them.
    context : `pygpu.gpuarray.GpuContext`, optional
        Use this GPU context to evaluate the GPU kernel. For ``None``,
        and if neither ``out1`` nor ``out2`` nor ``a`` are GPU arrays,
        a default GPU context must have been set.

    Returns
    -------
    out1, out2 : `pygpu.gpuarray.GpuArray`
        Results of the computation. If ``out1``, ``out2`` were given,
        the returned object is a reference to it.
        If ``out1`` and ``out2`` are ``None`` and ``a`` is not a GPU array,
        the result arrays ``out1`` and ``out2`` are instances of
        `pygpu._array.ndgpuarray`.

    See Also
    --------
    ufunc_result_dtype : Get dtype of a ufunc result.
    ufunc_c_funcname : Get C name for math ufuncs.
    """
    # Lazy import to avoid circular dependency
    from pygpu._array import ndgpuarray

    # Get a context and an array class to work with. Use the "highest"
    # class present in the inputs.
    need_context = True
    cls = None
    for ary in (a, out1, out2):
        if isinstance(ary, GpuArray):
            if context is not None and ary.context != context:
                raise ValueError('cannot mix contexts')
            context = ary.context
            if cls is None or cls == GpuArray:
                cls = ary.__class__
            need_context = False

    if need_context and context is None:
        context = get_default_context()
        cls = ndgpuarray

    if not isinstance(a, GpuArray):
        a = numpy.asarray(a)
        if a.flags.f_contiguous and not a.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'
        a = array(a, dtype=a.dtype, copy=False, order=order, context=context,
                  cls=cls)

    if a.dtype == numpy.dtype('float16'):
        # Gives wrong results currently, see
        # https://github.com/Theano/libgpuarray/issues/316
        raise NotImplementedError('float16 currently broken')

    prom_dtypes_in, result_dtypes = ufunc_dtypes(ufunc_name, [a.dtype])
    prom_dtype_in = prom_dtypes_in[0]
    result_dtype1, result_dtype2 = result_dtypes

    # This is the "fallback signature" case, for us it signals failure.
    # TypeError is what Numpy raises, too, which is kind of wrong
    if prom_dtype_in == numpy.dtype(object):
        raise TypeError('input dtype {!r} invalid for ufunc {!r}'
                        ''.format(a.dtype.name, ufunc_name))

    # Convert input such that the kernel runs
    # TODO: can this be avoided?
    a = a.astype(prom_dtype_in, copy=False)

    if out1 is None:
        out1 = empty(a.shape, dtype=result_dtype1, context=context, cls=cls)
    else:
        # TODO: allow larger dtype
        if out1.dtype != result_dtype1:
            raise ValueError('`out1.dtype` != result dtype: {!r} != {!r}'
                             ''.format(out1.dtype.name, result_dtype1.name))
    if out2 is None:
        out2 = empty(a.shape, dtype=result_dtype2, context=context, cls=cls)
    else:
        if out2.dtype != result_dtype2:
            raise ValueError('`out2.dtype` != result dtype: {!r} != {!r}'
                             ''.format(out2.dtype.name, result_dtype2.name))

    # C result dtypes for casting
    c_res_dtype1 = dtype_to_ctype(result_dtype1)

    oper = ''

    if ufunc_name == 'frexp':
        oper = 'res = ({rdt1}) frexp(a, &out2)'.format(rdt1=c_res_dtype1)
    elif ufunc_name == 'modf':
        oper = 'res = ({rdt1}) modf(a, &out2)'.format(rdt1=c_res_dtype1)

    if not oper:
        raise ValueError('`ufunc_name` {!r} does not represent a unary ufunc'
                         ''.format(ufunc_name))

    a_arg = as_argument(a, 'a', read=True)
    args = [arg('res', out1.dtype, write=True),
            arg('out2', out2.dtype, write=True),
            a_arg]

    ker = GpuElemwise(context, oper, args, convert_f16=True)
    ker(out1, out2, a)
    return out1, out2


# TODO: add ufuncs conditionally depending on Numpy version?
MISSING_UFUNCS = [
    'conjugate',  # no complex dtype yet
    'float_power',  # new in Numpy 1.12
    'divmod',  # new in Numpy 1.13
    'heaviside',  # new in Numpy 1.13
    'isnat',  # new in Numpy 1.13
    'positive',  # new in Numpy 1.13
    ]


UFUNC_SYNONYMS = [
    ('absolute', 'abs'),
    # ('conjugate', 'conj'),
    ('deg2rad', 'degrees'),
    ('rad2deg', 'radians'),
    ('true_divide', 'divide'),
    ('maximum', 'fmax'),  # TODO: differ in NaN propagation in numpy, doable?
    ('minimum', 'fmin'),  # TODO: differ in NaN propagation in numpy, doable?
    ('bitwise_not', 'invert'),
    ('remainder', 'mod')
    ]


# --- Reductions from binary ufuncs --- #


def make_binary_ufunc_reduce(name):
    npy_ufunc = getattr(numpy, name)
    binop = BINARY_UFUNC_TO_C_BINOP.get(name, None)
    if binop is not None:

        def reduce_wrapper(a, axis=0, dtype=None, out=None, keepdims=False,
                           context=None):
            return reduce_with_op(a, binop, npy_ufunc.identity, axis, dtype,
                                  out, keepdims, context)

        return reduce_wrapper

    cmp = BINARY_UFUNC_TO_C_CMP.get(name, None)
    if cmp is not None:

        def reduce_wrapper(a, axis=0, dtype=None, out=None, keepdims=False,
                           context=None):
            return reduce_with_cmp(a, cmp, npy_ufunc.identity, axis, dtype,
                                   out, keepdims, context)

        return reduce_wrapper

    # TODO: add reduction with binary function, not possible currently since
    # no sensible "neutral" can be defined. We need a variant of `reduce1`
    # that initializes the accumulator with the first element along a given
    # reduction axis.
    return None


# --- Add (incomplete) ufunc class --- #


# TODO: find out how to use one Ufunc class with __call__ signature depending
# on nin and nout. What doesn't work:
# - Setting __call__ on the class in __new__ since later instantiations of
#   the class overwrite previous ones (only one class object)
#   object will not be registered as callable
# - Setting __call__ on an instance (in __new__ or __init__), since the
#   object will not be registered as callable
# - Metaclass using __call__, need to find a more complete reference
#
# We need some way of setting the __call__ signature of an instance,
# probably by modifying __call__.__code__.


class UfuncBase(object):

    def __init__(self, name, nin, nout, call, **kwargs):
        self.__name__ = name
        self.nin = nin
        self.nout = nout
        self.nargs = self.nin + self.nout
        self._call = call

        self.identity = kwargs.pop('identity', None)

        # Wrappers for unimplemented stuff

        def _at_not_impl(a, indices, b=None, context=None):
            return NotImplemented

        def _accumulate_not_impl(array, axis=0, dtype=None, out=None,
                                 keepdims=None, context=None):
            return NotImplemented

        def _outer_not_impl(A, B, context=None, **kwargs):
            return NotImplemented

        def _reduce_not_impl(a, axis=0, dtype=None, out=None,
                             keepdims=False, context=None):
            return NotImplemented

        def _reduceat_not_impl(a, indices, axis=0, dtype=None, out=None,
                               context=None):
            return NotImplemented

        self.accumulate = kwargs.pop('accumulate', _accumulate_not_impl)
        self.accumulate.__name__ = 'accumulate'
        if PY3:
            self.accumulate.__qualname__ = name + '.accumulate'

        self.at = kwargs.pop('at', _at_not_impl)
        self.at.__name__ = 'at'
        if PY3:
            self.at.__qualname__ = name + '.at'

        self.outer = kwargs.pop('outer', _outer_not_impl)
        self.outer.__name__ = 'outer'
        if PY3:
            self.outer.__qualname__ = name + '.outer'

        reduce = kwargs.pop('reduce', None)
        if reduce is None:
            self.reduce = _reduce_not_impl
        else:
            self.reduce = reduce
        self.reduce.__name__ = 'reduce'
        if PY3:
            self.reduce.__qualname__ = name + '.reduce'

        self.reduceat = kwargs.pop('reduceat', _reduceat_not_impl)
        self.reduceat.__name__ = 'reduceat'
        if PY3:
            self.reduceat.__qualname__ = name + '.reduceat'

    def __repr__(self):
        return '<ufunc {}>'.format(self.__name__)


class Ufunc01(UfuncBase):

    def __call__(self, out=None, context=None):
        return self._call(out=out, context=context)


class Ufunc11(UfuncBase):

    def __call__(self, x, out=None, context=None):
        return self._call(x, out=out, context=context)


class Ufunc12(UfuncBase):

    def __call__(self, x, out1=None, out2=None, context=None):
        return self._call(x, out1=out1, out2=out2, context=context)


class Ufunc21(UfuncBase):

    def __call__(self, x1, x2, out=None, context=None):
        return self._call(x1, x2, out=out, context=context)


# --- Add ufuncs to global namespace --- #


def make_ufunc(name):
    npy_ufunc = getattr(numpy, name)
    nin = npy_ufunc.nin
    nout = npy_ufunc.nout

    if nin == 1 and nout == 1:
        cls = Ufunc11

        def call(a, out=None, context=None):
            return unary_ufunc(a, name, out, context)

    elif nin == 1 and nout == 2:
        cls = Ufunc12

        def call(a, out1=None, out2=None, context=None):
            return unary_ufunc_two_out(a, name, out1, out2, context)

    elif nin == 2 and nout == 1:
        cls = Ufunc21

        def call(a, b, out=None, context=None):
            return binary_ufunc(a, b, name, out, context)

    else:
        raise NotImplementedError('nin = {}, nout = {} not supported'
                                  ''.format(nin, nout))

    call.__name__ = name
    if PY3:
        call.__qualname__ = name
    # TODO: add docstring

    if nin == 1:
        ufunc = cls(name, nin, nout, call, identity=npy_ufunc.identity)
    else:
        ufunc = cls(name, nin, nout, call, identity=npy_ufunc.identity,
                    reduce=make_binary_ufunc_reduce(name))

    return ufunc


# Add the ufuncs to the module dictionary
for name in UNARY_UFUNCS + UNARY_UFUNCS_TWO_OUT + BINARY_UFUNCS:
    globals()[name] = make_ufunc(name)


for name, alt_name in UFUNC_SYNONYMS:
    globals()[alt_name] = globals()[name]


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    optionflags = NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE, extraglobs={'np': numpy})