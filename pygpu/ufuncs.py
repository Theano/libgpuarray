"""Ufuncs and reductions for GPU arrays."""

import mako
import numpy
import re
import warnings
from pygpu._array import ndgpuarray
from pygpu.dtypes import dtype_to_ctype, NAME_TO_DTYPE
from pygpu._elemwise import arg
from pygpu.elemwise import as_argument, GpuElemwise
from pygpu.reduction import reduce1
from pygpu.gpuarray import GpuArray, array, empty, get_default_context


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


def _prepare_array_for_reduction(a, out):
    """Return input array ready for usage in a reduction kernel."""
    # Get a context and an array class to work with. Use the "highest"
    # class present in the inputs.
    need_context = True
    ctx = None
    cls = None
    for ary in (a, out):
        if isinstance(ary, GpuArray):
            if ctx is not None and ary.context != ctx:
                raise ValueError('cannot mix contexts')
            ctx = ary.context
            if cls is None or cls == GpuArray:
                cls = ary.__class__
            need_context = False

    if need_context:
        ctx = get_default_context()
        cls = ndgpuarray  # TODO: sensible choice as default?

    # TODO: can CPU memory handed directly to kernels?
    if not isinstance(a, GpuArray):
        a = numpy.asarray(a)
        if a.flags.f_contiguous and not a.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'
        a = array(a, dtype=a.dtype, copy=False, order=order, context=ctx,
                  cls=cls)

    return a


def reduce_with_op(a, op, neutral, axis=None, dtype=None, out=None,
                   keepdims=False):
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

    Returns
    -------
    reduced : `pygpu.gpuarray.GpuArray` or scalar
        If not all axes are reduced over or ``out`` was given, the result is
        an array, a reference to ``out`` if provided. Otherwise, i.e. for
        reductions along all axes without ``out`` parameter, the result
        is a scalar.
    """
    a = _prepare_array_for_reduction(a, out)

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


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    """Sum of array elements over a given axis.

    See Also
    --------
    numpy.sum
    """
    # Do what Numpy does with booleans, sensible or not
    if a.dtype == bool and dtype is None:
        dtype = int
    return reduce_with_op(a, '+', 0, axis, dtype, out, keepdims)


def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    """Return the product of array elements over a given axis.

    See Also
    --------
    numpy.prod
    """
    # Do what Numpy does with booleans, sensible or not
    if a.dtype == bool and dtype is None:
        dtype = int
    return reduce_with_op(a, '*', 1, axis, dtype, out, keepdims)


def all(a, axis=None, out=None, keepdims=False):
    """Test whether all array elements along a given axis evaluate to True.

    See Also
    --------
    numpy.all
    """
    return reduce_with_op(a, '&&', 1, axis, numpy.bool, out, keepdims)


def any(a, axis=None, out=None, keepdims=False):
    """Test whether all array elements along a given axis evaluate to True.

    See Also
    --------
    numpy.all
    """
    return reduce_with_op(a, '||', 0, axis, numpy.bool, out, keepdims)


# --- Reductions with comparison operators --- #


def reduce_with_cmp(a, cmp, neutral, axis=None, out=None, keepdims=False):
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

    Returns
    -------
    reduced : `pygpu.gpuarray.GpuArray` or scalar
        If not all axes are reduced over or ``out`` was given, the result is
        an array, a reference to ``out`` if provided. Otherwise, i.e. for
        reductions along all axes without ``out`` parameter, the result
        is a scalar.
    """
    a = _prepare_array_for_reduction(a, out)

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


def amin(a, axis=None, out=None, keepdims=False):
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
    return reduce_with_cmp(a, '<', neutral, axis, out, keepdims)


def amax(a, axis=None, out=None, keepdims=False):
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
    return reduce_with_cmp(a, '>', neutral, axis, out, keepdims)


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
                     'square'])
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
BINARY_UFUNC_TO_C_OP = {
    'add': '+',
    'bitwise_and': '&',
    'bitwise_or': '|',
    'bitwise_xor': '^',
    'equal': '==',
    'greater': '>',
    'greater_equal': '>=',
    'left_shift': '<<',
    'less': '<',
    'less_equal': '<=',
    'logical_and': '&&',
    'logical_or': '||',
    'multiply': '*',
    'not_equal': '!=',
    'right_shift': '>>',
    'subtract': '-',
    }
BINARY_UFUNCS = (list(BINARY_UFUNC_TO_C_FUNC.keys()) +
                 list(BINARY_UFUNC_TO_C_OP.keys()))
BINARY_UFUNCS.extend(['floor_divide', 'true_divide', 'logical_xor',
                      'maximum', 'minimum', 'remainder'])


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
    from builtins import all, any
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
            if any(dt not in supported_dtypes for dt in dts):
                return False
            else:
                return all(dt >= dt_in for dt, dt_in in zip(dts, dtypes_in))

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
    if any(dt in result_dtypes for dt in (numpy.dtype('float16'),
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


def unary_ufunc(a, ufunc_name, out=None):
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
        If ``out=None`` and ``a`` is not a GPU array, a default
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
    # Get a context and an array class to work with. Use the "highest"
    # class present in the inputs.
    need_context = True
    ctx = None
    cls = None
    for ary in (a, out):
        if isinstance(ary, GpuArray):
            if ctx is not None and ary.context != ctx:
                raise ValueError('cannot mix contexts')
            ctx = ary.context
            if cls is None or cls == GpuArray:
                cls = ary.__class__
            need_context = False

    if need_context:
        ctx = get_default_context()
        cls = ndgpuarray  # TODO: sensible choice as default?

    # TODO: can CPU memory handed directly to kernels?
    if not isinstance(a, GpuArray):
        a = numpy.asarray(a)
        if a.flags.f_contiguous and not a.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'
        a = array(a, dtype=a.dtype, copy=False, order=order, context=ctx,
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
        out = empty(a.shape, dtype=result_dtype, context=ctx, cls=cls)
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
            warnings.warn('using negation (`-`) with boolean arrays is '
                          'deprecated, use logical not (`~`) instead; '
                          'the current behavior will be changed along with '
                          "NumPy's", FutureWarning)
            unop = '!'
        oper = 'res = ({}) {}a'.format(c_res_dtype, unop)

    # Other cases: specific functions
    if ufunc_name == 'deg2rad':
        oper = 'res = ({rdt})({}) * ({rdt}) a'.format(numpy.deg2rad(1),
                                                      rdt=c_res_dtype)

    if ufunc_name == 'rad2deg':
        oper = 'res = ({rdt})({}) * ({rdt}) a'.format(numpy.rad2deg(1),
                                                      rdt=c_res_dtype)

    if ufunc_name == 'reciprocal':
        oper = 'res = ({dt}) (({dt}) 1.0) / a'.format(dt=c_res_dtype)

    if ufunc_name == 'sign':
        oper = 'res = ({}) ((a > 0) ? 1 : (a < 0) ? -1 : 0)'.format(
            c_res_dtype)

    if ufunc_name == 'signbit':
        oper = 'res = ({}) (a < 0)'.format(c_res_dtype)

    if ufunc_name == 'square':
        oper = 'res = ({}) (a * a)'.format(c_res_dtype)

    if not oper:
        raise ValueError('`ufunc_name` {!r} does not represent a unary ufunc'
                         ''.format(ufunc_name))

    a_arg = as_argument(a, 'a', read=True)
    args = [arg('res', out.dtype, write=True), a_arg]

    ker = GpuElemwise(ctx, oper, args, convert_f16=True)
    ker(out, a)
    return out


def binary_ufunc(a, b, ufunc_name, out=None):
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
        If ``out=None`` and ``a, b`` are both not GPU arrays, a default
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
    from builtins import any
    # Get a context and an array class to work with
    need_context = True
    for ary in (a, b, out):
        if isinstance(a, GpuArray):
            ctx = ary.context
            cls = ary.__class__
            need_context = False
            break
    if need_context:
        ctx = get_default_context()
        # TODO: sensible choice? Makes sense to choose the more "feature-rich"
        # variant here perhaps.
        cls = ndgpuarray

    # TODO: can CPU memory handed directly to kernels?
    if not isinstance(a, GpuArray):
        if numpy.isscalar(a):
            # TODO: this is quite hacky, perhaps mixed input signatures
            # should be handled in ufunc_dtypes?
            if ufunc_name == 'ldexp':
                # Want signed type
                a = numpy.asarray(a, dtype=numpy.min_scalar_type(-abs(a)))
            else:
                a = numpy.asarray(a, dtype=numpy.result_type(a, b))

        if a.flags.f_contiguous and not a.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'
        a = array(a, dtype=a.dtype, copy=False, order=order, context=ctx,
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

        if b.flags.f_contiguous and not b.flags.c_contiguous:
            order = 'F'
        else:
            order = 'C'
        b = array(b, dtype=b.dtype, copy=False, order=order, context=ctx,
                  cls=cls)

    if any(ary.dtype == numpy.dtype('float16') for ary in (a, b)):
        # Gives wrong results currently, see
        # https://github.com/Theano/libgpuarray/issues/316
        raise NotImplementedError('float16 currently broken')

    prom_dtypes_in, result_dtypes = ufunc_dtypes(ufunc_name,
                                                 [a.dtype, b.dtype])
    prom_dtype_a, prom_dtype_b = prom_dtypes_in
    result_dtype = result_dtypes[0]

    # This is the "fallback signature" case, for us it signals failure
    if any(dt == numpy.dtype(object) for dt in prom_dtypes_in):
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
        out = empty(result_shape, dtype=result_dtype, context=ctx, cls=cls)
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

        if ufunc_name == 'true_divide':
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
            print(oper)

    if not oper:
        raise ValueError('`ufunc_name` {!r} does not represent a binary '
                         'ufunc'.format(ufunc_name))

    kernel = GpuElemwise(a.context, oper, args, convert_f16=True,
                         preamble=preamble)
    kernel(out, a, b, broadcast=True)
    return out


MISSING_UFUNCS = [
    'conjugate',  # no complex dtype yet
    'frexp',  # multiple output values, how to handle that?
    'isfinite',  # how to test in C?
    'isinf',  # how to test in C?
    'isnan',  # how to test in C?
    'logaddexp',  # not a one-liner (at least not in numpy)
    'logaddexp2',  # not a one-liner (at least not in numpy)
    'modf',  # multiple output values, how to handle that?
    'spacing',  # implementation?
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


# --- Add ufuncs to global namespace --- #


def make_unary_ufunc(name, doc):
    def wrapper(a, out=None):
        return unary_ufunc(a, name, out)
    wrapper.__qualname__ = wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


# Add the ufuncs to the module dictionary
for ufunc_name in UNARY_UFUNCS:
    npy_ufunc = getattr(numpy, ufunc_name)
    descr = npy_ufunc.__doc__.splitlines()[2]
    # Numpy occasionally uses single ticks for doc, we only use them for links
    descr = re.sub('`+', '``', descr)
    doc = descr + """

See Also
--------
numpy.{}
""".format(ufunc_name)

    globals()[ufunc_name] = make_unary_ufunc(ufunc_name, doc)


def make_binary_ufunc(name, doc):
    def wrapper(a, b, out=None):
        return binary_ufunc(a, b, name, out)

    wrapper.__qualname__ = wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


for ufunc_name in BINARY_UFUNCS:
    npy_ufunc = getattr(numpy, ufunc_name)
    descr = npy_ufunc.__doc__.splitlines()[2]
    # Numpy occasionally uses single ticks for doc, we only use them for links
    descr = re.sub('`+', '``', descr)
    doc = descr + """

See Also
--------
numpy.{}
""".format(ufunc_name)

    globals()[ufunc_name] = make_binary_ufunc(ufunc_name, doc)


for name, alt_name in UFUNC_SYNONYMS:
    globals()[alt_name] = globals()[name]


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    import numpy
    optionflags = NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE, extraglobs={'np': numpy})
