import os, sys
import subprocess

have_cython = False
have_cuda = None
have_opencl = None

import distribute_setup
distribute_setup.use_setuptools()

try:
    from Cython.Distutils import build_ext
    have_cython = True
except:
    from setuptools.command.build_ext import build_ext

from setuptools import setup, Extension
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
        libraries = []
    if library_dirs is None:
        libraries = []
    if macros is None:
        macros = []
    fd, fname, = tempfile.mkstemp(".c", 'configtest', text=True)
    f = os.fdopen(fd, "w")
    try:
        try:
            for incl in includes:
                f.write("""#include "%s"\n"""%(incl,))
            f.write("""int main() { %s; }"""%(func_call,))
        finally:
            f.close()

        try:
            objs = cc.compile([fname], macros=macros, include_dirs=include_dirs)
        except CompileError:
            return False
    finally:
        try: os.unlink(fname)
        except Exception: pass

    try:
        ext_a = []
        for f in frameworks:
            ext_a.append('-framework')
            ext_a.append(f)

        try:
            cc.link_executable(objs, "a.out", libraries=libraries,
                               library_dirs=library_dirs, extra_postargs=ext_a)
        except (LinkError, TypeError):
            return False
    finally:
        for o in objs:
            try: os.unlink(o)
            except Exception: pass
        try: os.unlink('a.out')
        except Exception: pass
    return True

if not has_function(cc, 'strlcat((char *)NULL, "aaa", 3)',
                    includes=['string.h']):
    srcs.append('compyte_strl.c')
    macros.append(('NO_STRL', ''))

if not has_function(cc, 'asprintf((char **)NULL, "aaa", "b", 1.0, 2)',
                       includes=['stdio.h']):
    if has_function(cc, 'asprintf((char **)NULL, "aaa", "b", 1.0, 2)',
                       includes=['stdio.h'], macros=[('_GNU_SOURCE', '')]):
        macros.append(('_GNU_SOURCE', ''))
    else:
        srcs.append('compyte_asprintf.c')
	macros.append(('NO_ASPRINTF', ''))

if not has_function(cc, 'mkstemp((char *)NULL)', includes=['stdlib.h']):
    srcs.append('compyte_mkstemp.c')
    macros.append(('NO_MKSTEMP', ''))

fnull = open(os.devnull, 'r+')

def find_cuda_root():
    root = os.getenv('CUDA_ROOT')
    if root is not None:
        return root
    for loc in ('/usr/local/cuda',
                'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v4.2'
                'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v4.1'
                'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v4.0'
                'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v3.2'):
        if os.path.isdir(loc):
            return loc
    return None

# Detect CUDA install
def find_cuda_lib(cuda_root):
    if sys.platform == 'darwin':
        if has_function(cc, 'cuInit(0)', includes=['CUDA/cuda.h'],
                         frameworks=['CUDA']):
            return {'extra_link_args': ['-framework', 'CUDA']}
    if cuda_root:
        inc = os.path.join(cuda_root, 'include')
        if sys.platform == 'win32':
            lib = os.path.join(cuda_root, 'lib', 'Win32')
            lib64 = os.path.join(cuda_root, 'lib', 'x64')
        else:
            lib = os.path.join(cuda_root, 'lib')
            lib64 = os.path.join(cuda_root, 'lib64')
        if has_function(cc, 'cuInit(0)', includes=['cuda.h'], include_dirs=[inc],
                           libraries=['cuda']):
            return {'libraries': ['cuda'], 'include_dirs': [inc]}
        elif has_function(cc, 'cuInit(0)', includes=['cuda.h'],
                          include_dirs=[inc], libraries=['cuda'],
                          library_dirs=[lib]):
            ext_link_args = []
            if not sys.platform.startswith('win'):
                ext_link_args.append('-Xlinker')
                ext_link_args.append('-rpath')
                ext_link_args.append('-Xlinker')
                ext_link_args.append(lib)
            return {'libraries': ['cuda'], 'include_dirs': [inc],
                    'library_dirs': [lib], 'extra_link_args': ext_link_args}
        elif has_function(cc, 'cuInit(0)', includes=['cuda.h'],
                          include_dirs=[inc], libraries=['cuda'],
                          library_dirs=[lib64]):
            ext_link_args = []
            if not sys.platform.startswith('win'):
                ext_link_args.append('-Xlinker')
                ext_link_args.append('-rpath')
                ext_link_args.append('-Xlinker')
                ext_link_args.append(lib64)
            return {'libraries': ['cuda'], 'include_dirs': [inc],
                    'library_dirs': [lib64], 'extra_link_args': ext_link_args}
    else:
        if has_function(cc, 'cuInit(0)', includes=['cuda.h'], libraries=['cuda']):
            return {'libraries': ['cuda']}
    return None

def try_cuda(arg):
    global have_cuda
    print "Searching for CUDA..."
    if arg is None:
        cuda_root = find_cuda_root()
    else:
        cuda_root = arg

    nvcc_bin = 'nvcc'
    if sys.platform == 'win32':
        nvcc_bin = nvcc_bin + '.exe'

    if cuda_root:
        nvcc_bin = os.path.join(cuda_root, 'bin', nvcc_bin)

    try:
        subprocess.check_call([nvcc_bin, '--version'],
                              stdout=fnull, stderr=fnull)
    except Exception:
        have_cuda = False
        return

    print "Found nvcc at:", nvcc_bin

    res = find_cuda_lib(cuda_root)
    if res is None:
        have_cuda = False
        return

    macros.append(('WITH_CUDA', '1'))
    cython_env['WITH_CUDA'] = True
    if sys.platform == 'win32':
        macros.append(('NVCC_BIN', '\\"'+nvcc_bin+'\\"'))
    else:
        macros.append(('NVCC_BIN', '"'+nvcc_bin+'"'))
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
            print "try specifying the cuda root (either as argument to --with-cuda or by the envionnement variable CUDA_ROOT)"
        raise Exception("Could not find CUDA")

def find_ocl_root():
    # Detect the AMD runtime
    root = os.getenv('AMDAPPSDKROOT')
    if root is not None:
        return root
    return None

def find_ocl_lib(ocl_root):
    if ocl_root is None:
        if sys.platform == 'darwin':
            if has_function(cc, 'clGetPlatformIDs(0, NULL, NULL)', includes=['OpenCL/opencl.h'],
                            frameworks=['OpenCL']):
                return {'extra_link_args': ['-framework', 'OpenCL']}
        if has_function(cc, 'clGetPlatformIDs(0, NULL, NULL)', includes=['CL/opencl.h'],
                        libraries=['OpenCL']):
            return {'libraries': ['OpenCL']}
    else:
        inc = os.path.join(ocl_root, 'include')
        lib = os.path.join(ocl_root, 'lib', 'x86')
        lib64 = os.path.join(ocl_root, 'lib', 'x86_64')
        if has_function(cc, 'clGetPlatformIDs(0, NULL, NULL)', includes=['CL/opencl.h'],
                        include_dirs=[inc], libraries=['OpenCL'], library_dirs=[lib]):
            return {'libraries': ['OpenCL'], 'include_dirs': [inc], 'library_dirs': [lib]}
        if has_function(cc, 'clGetPlatformIDs(0, NULL, NULL)', includes=['CL/opencl.h'],
                        include_dirs=[inc], libraries=['OpenCL'], library_dirs=[lib64]):
            return {'libraries': ['OpenCL'], 'include_dirs': [inc], 'library_dirs': [lib64]}
    return None

def try_opencl(arg):
    global have_opencl
    if arg is None:
        ocl_root = find_ocl_root()
    else:
        ocl_root = arg
	
    res = find_ocl_lib(ocl_root)
    if res is None:
        have_opencl = False
        return

    ext_link_args.extend(res.pop('extra_link_args', []))
    libraries.extend(res.pop('libraries', []))
    lib_dirs.extend(res.pop('library_dirs', []))
    include_dirs.extend(res.pop('include_dirs', []))

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
    if a.startswith('--with-cuda'):
        to_del.append(i)
        s = a.split('=', 1)
        if len(s) == 1:
            enable_cuda(None)
        else:
            enable_cuda(s[1])
    elif a.startswith('--with-opencl'):
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
