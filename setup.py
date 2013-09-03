import sys

have_cython = False

try:
    import Cython
    if Cython.__version__ < '0.19':
        raise Exception('cython is too old')
    from Cython.Build import cythonize
    have_cython = True
except Exception:
    def cythonize(arg):
        return arg

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils import setup, Extension

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
else:
    srcs = ['pygpu/gpuarray.c']

exts = [Extension('pygpu.gpuarray',
                sources = srcs,
                include_dirs = [np.get_include()],
                libraries = ['compyte'],
                )]

setup(name='pygpu',
      version='0.2.1',
      description='numpy-like wrapper on libcompyte for GPU computations',
      packages = ['pygpu'],
      data_files = [('pygpu', ['pygpu/gpuarray.h', 'pygpu/gpuarray_api.h'])],
      ext_modules=cythonize(exts),
      )
