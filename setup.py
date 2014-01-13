import sys

have_cython = False

try:
    import Cython
    if Cython.__version__ < '0.19':
        raise Exception('cython is too old or not installed '
                        '(at least 0.19 required)')
    from Cython.Build import cythonize
    have_cython = True
except Exception:
    # for devel version
    raise
    def cythonize(arg):
        return arg

try:
    from setuptools import setup, Extension as _Extension

    # setuptools is stupid and rewrites "sources" to change '.pyx' to '.c'
    # if it can't find Pyrex (and in recent versions, Cython).
    #
    # This is a really stupid thing to do behind the users's back (since
    # it breaks development builds) especially with no way of disabling it
    # short of the hack below.
    class Extension(_Extension):
        def __init__(self, *args, **kwargs):
            save_sources = kwargs.get('sources', None)
            _Extension.__init__(self, *args, **kwargs)
            self.sources = save_sources
except ImportError:
    from distutils.core import setup, Extension

import numpy as np

to_del = []

for i, a in enumerate(sys.argv):
    if a == '--disable-cython':
        to_del.append(i)
        have_cython = False

for i in reversed(to_del):
    del sys.argv[i]

del to_del

if have_cython:
    srcs = ['pygpu/gpuarray.pyx']
    blas_src = ['pygpu/blas.pyx']
else:
    srcs = ['pygpu/gpuarray.c']
    blas_src = ['pygpu/blas.c']

exts = [Extension('pygpu.gpuarray',
                  sources = srcs,
                  include_dirs = [np.get_include()],
                  libraries = ['compyte'],
                  define_macros = [('COMPYTE_SHARED', None)],
                  ),
        Extension('pygpu.blas',
                  sources = blas_src,
                  include_dirs = [np.get_include()],
                  libraries = ['compyte'],
                  define_macros = [('COMPYTE_SHARED', None)],
                  )]

setup(name='pygpu',
      version='0.2.1',
      description='numpy-like wrapper on libcompyte for GPU computations',
      packages = ['pygpu'],
      data_files = [('pygpu', ['pygpu/gpuarray.h', 'pygpu/gpuarray_api.h',
                               'pygpu/blas_api.h', 'pygpu/numpy_compat.h'])],
      ext_modules=cythonize(exts),
      install_requires=['mako>=0.7'],
      )
