import os, sys
import subprocess

have_cython = False
have_cuda = None
have_opencl = None

try:
    from Cython.Distutils import build_ext
    have_cython = True
except:
    from distutils.command.build_ext import build_ext

from distutils.core import setup, Extension

import numpy as np

# These are always there
srcs = ['compyte_types.c', 'compyte_util.c', 'compyte_buffer.c']
macros = []
include_dirs = [np.get_include(), '.']
lib_dirs = []
libraries = []
ext_link_args = []

if sys.platform.startswith('linux'):
    # stupid linux and its lack of strlcat/strlcpy
    srcs.append('stupid_linux.c')

fnull = open(os.devnull, 'r+')

# Detect CUDA install
def try_cuda(arg):
    global have_cuda
    print "Searching for CUDA..."
    if arg is None:
        cuda_root = os.getenv('CUDA_ROOT')
    else:
        cuda_root = arg
    if cuda_root is None:
        cuda_bin_prefix = ''
    else:
        cuda_bin_prefix = os.path.join(cuda_root, 'bin')+os.path.sep
    
    try:
        subprocess.check_call([os.path.join(cuda_bin_prefix, 'nvcc'),
                               '--version'], stdout=fnull, stderr=fnull)
    except Exception:
        have_cuda = False
        return

    print "Found nvcc at:", os.path.join(cuda_bin_prefix, 'nvcc')
    macros.append(('WITH_CUDA', '1'))
    macros.append(('CUDA_BIN_PATH', '"'+cuda_bin_prefix+'"'))
    macros.append(('call_compiler', 'call_compiler_python '))
    srcs.append('compyte_buffer_cuda.c')
    
    if sys.platform == 'darwin':
        print "Mac platform, using CUDA framework"
        ext_link_args.append('-framework')
        ext_link_args.append('CUDA')
    else:
        libraries.append('cuda')
        if cuda_root is None:
            print "WARNING: no CUDA_ROOT specified, assuming it is part of the default search path"
        else:
            include_dirs.append(os.path.join(cuda_root, 'include'))
            lib = os.path.join(cuda_root, 'lib')
            lib64 = os.path.join(cuda_root, 'lib64')
            if os.path.isdir(lib):
                lib_dirs.append(lib)
                ext_link_args.append('-Xlinker')
                ext_link_args.append('-rpath')
                ext_link_args.append('-Xlinker')
                ext_link_args.append(lib)
            if os.path.isdir(lib64):
                lib_dirs.append(lib64)
                ext_link_args.append('-Xlinker')
                ext_link_args.append('-rpath')
                ext_link_args.append('-Xlinker')
                ext_link_args.append(lib64)
    have_cuda = True

def enable_cuda(arg):
    try_cuda(arg)
    if not have_cuda:
        print "Could not find CUDA",
        if arg is not None:
            print "at the specified location (%s)"%(arg,)
        else:
            print "try specifying the cuda root (either as argument to --enable-cuda or by the envionnement variable CUDA_ROOT)"
        raise Exception("Could not find CUDA")

def try_opencl(arg):
    global have_opencl
    print "Searching for OpenCL..."
    if sys.platform == 'darwin':
        if int(os.uname()[2].split('.')[0]) <= 9:
            have_opencl = False
            return
        print "On Mac >= 10.6, using framework"
        ext_link_args.append('-framework')
        ext_link_args.append('OpenCL')
    else:
        print "Assuming OpenCL install in default search paths"
        libraries.append('OpenCL')
    srcs.append('compyte_buffer_opencl.c')
    macros.append(('WITH_OPENCL', '1'))
    have_opencl = True

def enable_opencl(arg):
    try_opencl(arg)
    if not have_opencl:
        print "Could not find OpenCL"
        if sys.platform == 'darwin':
            print "OpenCL support on Mac starts with 10.6, please upgrade"
        else:
            print "Install an OpenCL runtime."

to_del = []

for i, a in enumerate(sys.argv):
    if a.startswith('--enable-cuda'):
        to_del.append(i)
        s = a.split('=', 1)
        if len(s) == 1:
            enable_cuda(None)
        else:
            enable_cuda(s[1])
    elif a.startswith('--enable-opencl'):
        to_del.append(i)
        s = a.split('=', 1)
        if len(s) == 1:
            enable_opencl(None)
        else:
            enable_opencl(s[1])
    elif a == '--disable-cuda':
        to_del.append(i)
        have_cuda = False
    elif a == '--disable-opencl':
        to_del.append(i)
        have_opencl = False
    elif a == '--disable-cython':
        to_del.append(i)
        have_cython = False

for i in reversed(to_del):
    del sys.argv[i]

del to_del

if have_cuda is None:
    try_cuda(None)

if have_opencl is None:
    try_opencl(None)

if have_cython:
    srcs.append('pygpu_ndarray.pyx')
else:
    srcs.append('pygpu_ndarray.c')

setup(name='compyte',
      cmdclass = {'build_ext': build_ext},
      ext_modules=[Extension('pygpu_ndarray',
                             define_macros = macros,
                             sources = srcs,
                             include_dirs = include_dirs,
                             libraries = libraries,
                             library_dirs = lib_dirs,
                             extra_link_args = ext_link_args,
                             )
                   ])
