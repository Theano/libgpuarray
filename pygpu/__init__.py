def get_include():
    import os.path
    p = os.path.dirname(os.path.dirname(__file__))
    assert os.path.exists(os.path.join(p, 'pygpu/gpuarray.h'))
    return p

from . import gpuarray
from .gpuarray import init, set_default_context, array, zeros, empty
