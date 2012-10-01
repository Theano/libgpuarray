import sys

have_cython = False

try:
    from Cython.Distutils import build_ext
    have_cython = True
except ImportError:
    try:
        from setuptools.command.build_ext import build_ext
    except ImportError:
        from distutils.command.build_ext import build_ext

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

if sys.platform == 'win32':
    import shutil
    shutil.copyfile('Release/compyte.lib', 'compyte.lib')
    shutil.copyfile('Release/compyte.dll', 'compyte.dll')

setup(name='compyte',
      cmdclass = {'build_ext': build_ext},
      packages = ['pygpu'],
      ext_modules=[Extension('pygpu.gpuarray',
                             sources = srcs,
                             include_dirs = [np.get_include(), 'src'],
                             libraries = ['compyte'],
                             library_dirs = ['lib']
                             )
                   ])
