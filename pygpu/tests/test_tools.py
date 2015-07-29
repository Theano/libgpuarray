from pygpu.tools import (as_argument, Argument, ArrayArg, ScalarArg,
                         check_contig, check_args, Counter, lfu_cache)


from .support import (guard_devsup, rand, check_flags, check_meta, check_all,
                      context, gen_gpuarray, dtypes_no_complex)


def test_check_contig_1():
    ac, ag = gen_gpuarray((50, 1, 20), 'float32', ctx=context)
    bc, bg = gen_gpuarray((50, 1, 20), 'float32', ctx=context)
    n, offsets, contig = check_contig((ag, bg))
    assert n == 1000
    assert offsets == (0, 0)
    assert contig


def test_check_contig_2():
    ac, ag = gen_gpuarray((50, 1, 20), 'float32', ctx=context)
    bc, bg = gen_gpuarray((50, 1, 20), 'float32', ctx=context, sliced=2)
    n, offsets, contig = check_contig((ag, bg))
    assert n == None
    assert offsets == None
    assert not contig


def test_check_args_simple():
    ac, ag = gen_gpuarray((50,), 'float32', ctx=context)
    bc, bg = gen_gpuarray((50,), 'float32', ctx=context)
    n, nd, dims, strs, offsets = check_args((ag, bg))
    assert n == 50
    assert nd == 1
    assert dims == (50,)
    assert strs == ((4,), (4,))
    assert offsets == (0, 0)

    ac, ag = gen_gpuarray((50, 1, 20), 'float32', ctx=context)
    bc, bg = gen_gpuarray((50, 1, 20), 'float32', ctx=context)
    n, nd, dims, strs, offsets = check_args((ag, bg))
    assert n == 1000
    assert nd == 3
    assert dims == (50, 1, 20)
    assert strs == ((80, 80, 4), (80, 80, 4))
    assert offsets == (0, 0)


def test_check_args_collapse_1():
    ac, ag = gen_gpuarray((50, 1, 20), 'float32', ctx=context)
    bc, bg = gen_gpuarray((50, 1, 20), 'float32', ctx=context)
    n, nd, dims, strs, offsets = check_args((ag, bg), collapse=False)
    assert n == 1000
    assert nd == 3
    assert dims == (50, 1, 20)
    assert strs == ((80, 80, 4), (80, 80, 4))
    assert offsets == (0, 0)

    n, nd, dims, strs, offsets = check_args((ag, bg), collapse=True)
    assert n == 1000
    assert nd == 1
    assert dims == (1000,)
    assert strs == ((4,), (4,))
    assert offsets == (0, 0)


def test_check_args_collapse_2():
    ac, ag = gen_gpuarray((50, 1, 20), 'float32', ctx=context, sliced=2,
                          offseted_inner=True)
    bc, bg = gen_gpuarray((50, 1, 20), 'float32', ctx=context)
    n, nd, dims, strs, offsets = check_args((ag, bg), collapse=True)
    assert n == 1000
    assert nd == 2
    assert dims == (50, 20)
    assert strs == ((168, 4), (80, 4))
    assert offsets == (4, 0)


def test_check_args_collapse_3():
    ac, ag = gen_gpuarray((50, 2, 10), 'float32', ctx=context, sliced=2,
                          offseted_outer=True)
    bc, bg = gen_gpuarray((50, 2, 10), 'float32', ctx=context)
    n, nd, dims, strs, offsets = check_args((ag, bg), collapse=True)
    assert n == 1000
    assert nd == 2
    assert dims == (50, 20)
    assert strs == ((160, 4), (80, 4))
    assert offsets == (80, 0)


def test_check_args_collapse_4():
    ac, ag = gen_gpuarray((1,), 'float32', ctx=context)
    n, nd, dims, strs, offsets = check_args((ag,), collapse=False)
    assert n == 1
    assert nd == 1
    assert dims == (1,)
    assert strs == ((4,),)
    assert offsets == (0,)

    ac, ag = gen_gpuarray((1, 1), 'float32', ctx=context)
    n, nd, dims, strs, offsets = check_args((ag,), collapse=True)
    assert n == 1
    assert nd == 1
    assert dims == (1,)
    assert strs == ((4,),)
    assert offsets == (0,)


def test_check_args_broadcast_1():
    ac, ag = gen_gpuarray((1,), 'float32', ctx=context)
    bc, bg = gen_gpuarray((50,), 'float32', ctx=context)
    n, nd, dims, strs, offsets = check_args((ag, bg), broadcast=True)
    assert n == 50
    assert nd == 1
    assert dims == (50,)
    assert strs == ((0,), (4,))
    assert offsets == (0, 0)


def test_check_args_broadcast_2():
    ac, ag = gen_gpuarray((50, 1, 20), 'float32', ctx=context, sliced=2,
                          offseted_inner=True)
    bc, bg = gen_gpuarray((50, 1, 20), 'float32', ctx=context)
    n, nd, dims, strs, offsets = check_args((ag, bg), collapse=True,
                                                    broadcast=True)
    assert n == 1000
    assert nd == 2
    assert dims == (50, 20)
    assert strs == ((168, 4), (80, 4))
    assert offsets == (4, 0)


def test_check_args_broadcast_3():
    ac, ag = gen_gpuarray((10, 20, 30), 'float32', ctx=context)
    bc, bg = gen_gpuarray((1, 1, 1), 'float32', ctx=context)
    n, nd, dims, strs, offsets = check_args((ag, bg), broadcast=True)
    assert n == 6000
    assert nd == 3
    assert dims == (10, 20, 30)
    assert strs == ((2400, 120, 4), (0, 0, 0))
    assert offsets == (0, 0)
