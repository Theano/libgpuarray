PROJECT(compyte C)

include(CheckFunctionExists)

cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} .)

#################################################
## Set this to False to build a static library ##
#################################################
set(COMPYTE_SHARED True)

set(CMAKE_OSX_ARCHITECTURES i386 x86_64)
set(CMAKE_OSX_SYSROOT /)

find_package(CUDA)
find_package(OpenCL)

configure_file(
  ${PROJECT_SOURCE_DIR}/compyte_compat.h.in
  ${PROJECT_BINARY_DIR}/compyte_compat.h
  )

include_directories("${PROJECT_BINARY_DIR}")

SET(COMPYTE_SRC
compyte_types.c
compyte_error.c
compyte_util.c
compyte_buffer.c
compyte_array.c
compyte_kernel.c
compyte_extension.c
)

check_function_exists(strlcat HAVE_STRL)
check_function_exists(asprintf HAVE_ASPRINTF)
check_function_exists(mkstemp HAVE_MKSTEMP)

if(NOT HAVE_STRL)
  set(COMPYTE_SRC ${COMPYTE_SRC} compyte_strl.c)
endif()

if(NOT HAVE_ASPRINTF)
  set(COMPYTE_SRC ${COMPYTE_SRC} compyte_asprintf.c)
endif()

if(NOT HAVE_MKSTEMP)
  set(COMPYTE_SRC ${COMPYTE_SRC} compyte_mkstemp.c)
endif()

if(CUDA_FOUND)
  set(COMPYTE_SRC ${COMPYTE_SRC} compyte_buffer_cuda.c)
  add_definitions(-DNVCC_BIN="${CUDA_NVCC_EXECUTABLE}" -DWITH_CUDA)
  include_directories("${CUDA_TOOLKIT_INCLUDE}")
endif()

if(OPENCL_FOUND)
  set(COMPYTE_SRC ${COMPYTE_SRC} compyte_buffer_opencl.c)
  add_definitions(-DWITH_OPENCL)
  include_directories("${OPENCL_INCLUDE_DIRS}")
endif()

if(COMPYTE_SHARED)
  add_definitions(-DCOMPYTE_BUILDING_DLL)
  ADD_LIBRARY(compyte SHARED ${COMPYTE_SRC})
else()
  add_library(compyte STATIC ${COMPYTE_SRC})
endif()

if(CUDA_FOUND)
  target_link_libraries(compyte ${CUDA_CUDA_LIBRARY})
endif()

if(OPENCL_FOUND)
  target_link_libraries(compyte ${OPENCL_LIBRARIES})
endif()
