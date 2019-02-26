def get_include():
    import os.path
    p = os.path.dirname(__file__)
    assert os.path.exists(os.path.join(p, 'gpuarray_api.h'))
    return p

from . import gpuarray, elemwise, reduction, ufuncs
from .gpuarray import (init, set_default_context, get_default_context,
                       array, zeros, empty, asarray, ascontiguousarray,
                       asfortranarray, register_dtype)
from .operations import (split, array_split, hsplit, vsplit, dsplit,
                         concatenate, hstack, vstack, dstack)
from ._array import ndgpuarray

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

def test():
    from . import tests
    from .tests import main
    if hasattr(main, "NoseTester"):
        main.NoseTester(package=tests).test()

