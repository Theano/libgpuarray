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
from distutils import ccompiler

cc = ccompiler.new_compiler()

import numpy as np

# These are always there
srcs = ['compyte_types.c', 'compyte_util.c', 'compyte_buffer.c']
macros = []
cython_env = {'WITH_CUDA': False, 'WITH_OPENCL': False}
include_dirs = [np.get_include(), '.']
lib_dirs = []
libraries = []
ext_link_args = []

def has_function(cc, func_call, includes=None, include_dirs=None,
                 frameworks=None, libraries=None, library_dirs=None,
                 macros=None):
    from distutils.errors import CompileError, LinkError
    import tempfile
    if includes is None:
        includes = []
    if include_dirs is None:
        include_dirs = []
    if frameworks is None:
        frameworks = []
    if libraries is None:
        librairies = []
    if library_dirs is None:
        libraries = []
    if macros is None:
        macros = []
    fd, fname, = tempfile.mkstemp(".c", 'configtest', text=True)
    f = os.fdopen(fd, "w")
    try:
        for incl in includes:
            f.write("""#include "%s"\n"""%(incl,))
        f.write("""int main() { %s; }"""%(func_call,))
    finally:
        f.close()

    for m in macros:
        cc.macros.append(m)
    try:
        objs = cc.compile([fname], include_dirs=include_dirs)
    except CompileError:
        return False
    finally:
        for m in macros:
            cc.macros.pop(-1)

    ext_a = []
    for f in frameworks:
        ext_a.append('-framework')
        ext_a.append(f)

    try:
        cc.link_executable(objs, "a.out", libraries=libraries,
                           library_dirs=library_dirs, extra_postargs=ext_a)
    except (LinkError, TypeError):
        return False
    return True

if not has_function(cc, 'strlcat((char *)NULL, "aaa", 3)',
                    includes=['string.h']):
    srcs.append('compyte_strl.c')

if not has_function(cc, 'asprintf((char **)NULL, "aaa", "b", 1.0, 2)',
                       includes=['stdio.h']):
    if has_function(cc, 'asprintf((char **)NULL, "aaa", "b", 1.0, 2)',
                       includes=['stdio.h'], macros=[('_GNU_SOURCE', 1)]):
        macros.append(('_GNU_SOURCE', 1))
    else:
        srcs.append('compyte_asprintf.c')

fnull = open(os.devnull, 'r+')

# Detect CUDA install
def find_cuda_lib(cuda_root):
    if sys.platform == 'darwin':
        if has_function(cc, 'cuInit', includes=['CUDA/cuda.h'],
                         frameworks=['CUDA'], cc=cc):
            ext_link_args = []
            ext_link_args.append('-framework')
            ext_link_args.append('CUDA')
            return {'extra_link_args', ext_link_args}
    if cuda_root:
        inc = os.path.join(cuda_root, 'include')
        lib = os.path.join(cuda_root, 'lib')
        lib64 = os.path.join(cuda_root, 'lib64')
        if has_function(cc, 'cuInit', includes=['cuda.h'], include_dirs=[inc],
                           libraries=['cuda']):
            return {'library': ['cuda'], 'include_dirs': [inc]}
        elif has_function(cc, 'cuInit', includes=['cuda.h'],
                          include_dirs=[inc], libraries=['cuda'],
                          library_dirs=[lib]):
            ext_link_args = []
            if not sys.platform.startswith('win'):
                ext_link_args.append('-Xlinker')
                ext_link_args.append('-rpath')
                ext_link_args.append('-Xlinker')
                ext_link_args.append(lib)
            return {'library': ['cuda'], 'include_dirs': [inc],
                    'library_dirs': [lib], 'extra_link_args': ext_link_args}
        elif has_function(cc, 'cuInit', includes=['cuda.h'],
                          include_dirs=[inc], libraries=['cuda'],
                          library_dirs=[lib64]):
            ext_link_args = []
            if not sys.platform.startswith('win'):
                ext_link_args.append('-Xlinker')
                ext_link_args.append('-rpath')
                ext_link_args.append('-Xlinker')
                ext_link_args.append(lib64)
            return {'library': ['cuda'], 'include_dirs': [inc],
                    'library_dirs': [lib64], 'extra_link_args': ext_link_args}
    else:
        if has_function(cc, 'cuInit', includes=['cuda.h'], libraries=['cuda']):
            return {'libraries': ['cuda']}
    return None

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

    res = find_cuda_lib(cuda_root)
    if res is None:
        have_cuda = False
        return

    macros.append(('WITH_CUDA', '1'))
    cython_env['WITH_CUDA'] = True
    macros.append(('CUDA_BIN_PATH', '"'+cuda_bin_prefix+'"'))
    macros.append(('call_compiler', 'call_compiler_python'))
    srcs.append('compyte_buffer_cuda.c')
    
    ext_link_args.extend(res.pop('extra_link_args', []))
    libraries.extend(res.pop('libraries', []))
    lib_dirs.extend(res.pop('library_dirs', []))
    include_dirs.extend(res.pop('include_dirs', []))
    
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
        if has_function(cc, 'clCreateContext', includes=['opencl.h'],
                        libraries=['OpenCL']):
            libraries.append('OpenCL')
        else:
            have_opencl = False
            return
    srcs.append('compyte_buffer_opencl.c')
    macros.append(('WITH_OPENCL', '1'))
    cython_env['WITH_OPENCL'] = True
    have_opencl = True

def enable_opencl(arg):
    try_opencl(arg)
    if not have_opencl:
        print "Could not find OpenCL"
        if sys.platform == 'darwin':
            print "OpenCL support on Mac starts with 10.6, upgrade or", \
                "install a runtime"
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

# update definitions
with open('defs.pxi', 'w') as f:
    f.write("""
DEF WITH_CUDA = %(WITH_CUDA)r
DEF WITH_OPENCL = %(WITH_OPENCL)r
""" % cython_env)

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
