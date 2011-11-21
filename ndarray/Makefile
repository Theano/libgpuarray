all: pygpu_ndarray.so

PYTHONVERSION ?= $(shell python -c "import sys; print '%d.%d'%(sys.version_info[0], sys.version_info[1]"))
CUDA_ROOT ?= /opt/lisa/os/cuda
THEANO_ROOT ?= /u/bastienf/repos/Theano


CFLAGS=-g -DDEBUG -DOFFSET
# By default enable the OFFSET usage. Otherwise some test fail.
CFLAGS=-g -DOFFSET
#BINDIR=--compiler-bindir ${HOME}/.theano.nvcc-bindir

#NPY_PATH!=python -c "import numpy;print numpy.__path__"
#NPY_INCLUDE=-I${NPY_PATH}/core/include
CUDA_INCLUDE=-I${CUDA_ROOT}/include
PYTHON_INCLUDE=-I$(shell python -c "import distutils.sysconfig;print distutils.sysconfig.get_python_inc()")
INCLUDES=${CUDA_INCLUDE} ${PYTHON_INCLUDE}
CUDA_FLAGS=-Xlinker -rpath,${CUDA_ROOT}/lib64 -Xlinker -rpath,${CUDA_ROOT}/lib

pygpu_language_cuda.o: pygpu_language_cuda.cu pygpu_language.h
	nvcc -c ${CFLAGS} -m64 -Xcompiler -fPIC,-m64 ${CUDA_FLAGS} ${INCLUDES} ${BINDIR} -o $@ $<

pygpu_ndarray.so: pygpu_ndarray.cpp pygpu_ndarray.h pygpu_language_cuda.o pygpu_ndarray_object.h
	nvcc -shared ${CFLAGS} -m64 -Xcompiler -fPIC,-m64 ${CUDA_FLAGS} ${INCLUDES} ${BINDIR} -o $@ pygpu_language_cuda.o $< -lpython${PYTHONVERSION} -lcublas -lcudart

clean:
	rm -f pygpu_ndarray.so core.* *.o *~
	rm -rf build

cleantmp:
	rm -f core.* *.o *~