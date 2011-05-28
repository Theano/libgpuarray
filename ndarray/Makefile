all: pygpu_ndarray.so

PYTHON_VERSION=2.5
NPY_INCLUDE=-I/opt/lisa/byhost/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include
THEANO_CUDA_INCLUDE=-I/u/bastienf/repos/theano/sandbox/cuda
CUDA_INCLUDE=-I/opt/lisa/os/cuda/include
PYTHON_INCLUDE=-I/home/nouiz/local_python${PYTHON_VERSION}.1/include/python${PYTHON_VERSION}
#PYTHON_INCLUDE=-I/usr/include/python${PYTHON_VERSION}
#BINDIR=--compiler-bindir /home/nouiz/.theano.nvcc-bindir

pygpu_language_cuda.o: pygpu_language_cuda.cu pygpu_language.h
	nvcc -c -g -m64 -Xcompiler -fPIC,-m64 -Xlinker -rpath,/opt/lisa/os/cuda/lib -Xlinker -rpath,/opt/lisa/os/cuda/lib64 ${PYTHON_INCLUDE} ${CUDA_INCLUDE} ${NPY_INCLUDE} ${BINDIR} -I/usr/include/python${PYTHON_VERSION} -o $@ $<

pygpu_ndarray.so: pygpu_ndarray.cu pygpu_ndarray.cuh pygpu_language_cuda.o
	nvcc -shared -g -m64 -Xcompiler -fPIC,-m64 -Xlinker -rpath,/opt/lisa/os/cuda/lib -Xlinker -rpath,/opt/lisa/os/cuda/lib64 ${PYTHON_INCLUDE} ${CUDA_INCLUDE} ${NPY_INCLUDE} ${BINDIR} -I/usr/include/python${PYTHON_VERSION} -o $@ $< pygpu_language_cuda.o -L/opt/lisa/os/cuda/lib -L/opt/lisa/os/cuda/lib64 -L/usr/lib64 -lpython${PYTHON_VERSION} -lcublas -lcudart

clean:
	rm -f pygpu_ndarray.so core.* *.o *~