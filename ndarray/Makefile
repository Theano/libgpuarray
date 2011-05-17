all: pygpu_ndarray.so

NPY_INCLUDE="-I/opt/lisa/byhost/lib/python2.5/site-packages/numpy/core/include"
CUDA_INCLUDE="-I/u/bastienf/repos/theano/sandbox/cuda"

pygpu_ndarray.so: pygpu_ndarray.cu pygpu_ndarray.cuh
	nvcc -shared -g -m64 -Xcompiler -fPIC,-m64 -Xlinker -rpath,/opt/lisa/os/cuda/lib -Xlinker -rpath,/opt/lisa/os/cuda/lib64 ${CUDA_INCLUDE} ${NPY_INCLUDE} -I/usr/include/python2.5 -o $@ $< -L/opt/lisa/os/cuda/lib -L/opt/lisa/os/cuda/lib64 -L/usr/lib64 -lpython2.5 -lcublas -lcudart

clean:
	rm pygpu_ndarray.so core.*