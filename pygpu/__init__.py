def get_include():
    import os.path
    p = os.path.dirname(__file__)
    assert os.path.exists(os.path.join(p, 'gpuarray_api.h'))
    return p

from . import gpuarray, elemwise, reduction
from .gpuarray import (init, set_default_context, get_default_context,
                       array, zeros, empty, asarray, ascontiguousarray,
                       asfortranarray, register_dtype)
from .operations import (split, array_split, hsplit, vsplit, dsplit,
                         concatenate, hstack, vstack, dstack)
from ._array import PyGpuArray

from .tests import main
if hasattr(main, "NoseTester"):
    test = main.NoseTester().test
else:
    def test():
        raise ImportError("The nose module is not installed."
                          " It is needed for pygpu tests.")
