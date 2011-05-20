all: pygpu_ndarray.so

NPY_INCLUDE=-I/opt/lisa/byhost/lib/python2.5/site-packages/numpy/core/include
THEANO_CUDA_INCLUDE=-I/u/bastienf/repos/theano/sandbox/cuda
CUDA_INCLUDE=-I/opt/lisa/os/cuda/include

pygpu_language_cuda.o: pygpu_language_cuda.cu pygpu_language.h
	nvcc -c -g -m64 -Xcompiler -fPIC,-m64 -Xlinker -rpath,/opt/lisa/os/cuda/lib -Xlinker -rpath,/opt/lisa/os/cuda/lib64 ${CUDA_INCLUDE} ${NPY_INCLUDE} -I/usr/include/python2.5 -o $@ $<

pygpu_ndarray.so: pygpu_ndarray.cu pygpu_ndarray.cuh pygpu_language_cuda.o
	nvcc -shared -g -m64 -Xcompiler -fPIC,-m64 -Xlinker -rpath,/opt/lisa/os/cuda/lib -Xlinker -rpath,/opt/lisa/os/cuda/lib64 ${CUDA_INCLUDE} ${NPY_INCLUDE} -I/usr/include/python2.5 -o $@ $< pygpu_language_cuda.o -L/opt/lisa/os/cuda/lib -L/opt/lisa/os/cuda/lib64 -L/usr/lib64 -lpython2.5 -lcublas -lcudart

clean:
	rm -f pygpu_ndarray.so core.* *.o *~