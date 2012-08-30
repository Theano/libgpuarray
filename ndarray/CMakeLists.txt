PROJECT(compyte C)

cmake_minimum_required(VERSION 2.8)
#set(CMAKE_VERBOSE_MAKEFILE true)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} .)

find_package(CUDA)
find_package(OpenCL)

SET(COMPYTE_SRC
compyte_types.c
compyte_error.c
compyte_util.c
compyte_buffer.c
compyte_array.c
compyte_kernel.c
compyte_extension.c
)

if(CUDA_FOUND)
  set(COMPYTE_SRC ${COMPYTE_SRC} compyte_buffer_cuda.c)
  add_definitions(-DNVCC_BIN="${CUDA_NVCC_EXECUTABLE}" -DWITH_CUDA)
endif()

if(OPENCL_FOUND)
  set(COMPYTE_SRC ${COMPYTE_SRC} compyte_buffer_opencl.c)
  add_definitions(-DWITH_OPENCL)
endif()

ADD_LIBRARY(compyte SHARED ${COMPYTE_SRC})

if(CUDA_FOUND)
  target_link_libraries(compyte ${CUDA_CUDA_LIBRARY})
endif()

if(OPENCL_FOUND)
  target_link_libraries(compyte ${OPENCL_LIBRARIES})
endif()
