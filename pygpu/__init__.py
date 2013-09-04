def get_include():
    import os.path
    p = os.path.dirname(__file__)
    assert os.path.exists(os.path.join(p, 'gpuarray_api.h'))
    return p

from . import gpuarray
from .gpuarray import (init, set_default_context, array, zeros, empty,
                       asarray, ascontiguousarray, asfortranarray,
                       register_dtype)
import elemwise
import reduction
