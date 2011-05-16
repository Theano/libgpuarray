all: gpu_ndarray.so

gpu_ndarray.so: gpu_ndarray.cu gpu_ndarray.cuh
	nvcc -shared -g -m64 -Xcompiler -fPIC,-m64 -Xlinker -rpath,/opt/lisa/os/cuda/lib -Xlinker -rpath,/opt/lisa/os/cuda/lib64 -I/u/bastienf/repos/theano/sandbox/cuda -I/opt/lisa/byhost/lib/python2.5/site-packages/numpy/core/include -I/usr/include/python2.5 -o gpu_ndarray.so gpu_ndarray.cu -L/opt/lisa/os/cuda/lib -L/opt/lisa/os/cuda/lib64 -L/usr/lib64 -lpython2.5 -lcublas -lcudart

clean:
	rm gpu_ndarray.so