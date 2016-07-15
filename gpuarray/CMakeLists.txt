include(CheckFunctionExists)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG")

find_package(CUDA)
find_package(OpenCL)
if(OPENCL_FOUND)
find_package(clBLAS)
endif()
if(CUDA_FOUND)
find_package(NCCL)
endif()

include_directories(include)
include_directories(src)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/gpuarray/types.c
         ${CMAKE_CURRENT_SOURCE_DIR}/include/gpuarray/types.h
  COMMAND python gen_types.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_types.py)

set(GPUARRAY_SRC
src/cache/lru.c
src/cache/twoq.c
src/util/strb.c
src/util/xxhash.c
src/gpuarray/types.c
src/gpuarray/error.c
src/gpuarray/util.c
src/gpuarray/buffer.c
src/gpuarray/buffer_blas.c
src/gpuarray/buffer_collectives.c
src/gpuarray/array.c
src/gpuarray/blas.c
src/gpuarray/collectives.c
src/gpuarray/kernel.c
src/gpuarray/extension.c
src/gpuarray/elemwise.c
)

check_function_exists(strlcat HAVE_STRL)
check_function_exists(mkstemp HAVE_MKSTEMP)

if(UNIX)
  add_definitions(-D_GNU_SOURCE)
endif()

if(NOT HAVE_STRL)
  list(APPEND GPUARRAY_SRC src/strl.c)
endif()

if(NOT HAVE_MKSTEMP)
  list(APPEND GPUARRAY_SRC src/mkstemp.c)
endif()

if (CUDA_FOUND)
  if(NCCL_FOUND)
    if (CUDA_VERSION_MAJOR LESS 7)
      message( WARNING "This package requires CUDA 7.0 or more (building with NCCL).  Found version ${CUDA_VERSION_STRING}")
      set(CUDA_FOUND 0)
    endif()
  else(NCCL_FOUND)
    if (CUDA_VERSION_MAJOR LESS 6 OR
        (CUDA_VERSION_MAJOR EQUAL 6 AND CUDA_VERSION_MINOR EQUAL 0))
      message( WARNING "This package requires CUDA 6.5 or more.  Found version ${CUDA_VERSION_STRING}")
      set(CUDA_FOUND 0)
    endif()
  endif(NCCL_FOUND)
endif()

if (CUDA_FOUND)
  if (APPLE)
    FIND_LIBRARY(CUDADRV_LIBRARY CUDA)
    FIND_PATH(CUDADRV_INCLUDE CUDA/cuda.h)
    # this is somewhat a hack, but otherwise cublas_v2.h isn't found
    set(CUDADRV_INCLUDE ${CUDADRV_INCLUDE} ${CUDA_TOOLKIT_INCLUDE})
  endif()
  if(NOT CUDADRV_LIBRARY)
    set(CUDADRV_LIBRARY ${CUDA_CUDA_LIBRARY})
    set(CUDADRV_INCLUDE ${CUDA_TOOLKIT_INCLUDE})
  endif()

  find_cuda_helper_libs(nvrtc)

  if(CUDA_nvrtc_LIBRARY)
    message(STATUS "Building with NVRTC")
    add_definitions(-DWITH_NVRTC)
    set(CUDADRV_LIBRARY ${CUDADRV_LIBRARY} ${CUDA_nvrtc_LIBRARY})
  else()
    add_definitions(-DNVCC_BIN=${CUDA_NVCC_EXECUTABLE})
  endif()

  list(APPEND GPUARRAY_SRC src/gpuarray/cuda/buffer.c)
  add_definitions(-DWITH_CUDA)
  include_directories(${CUDADRV_INCLUDE})

  list(APPEND GPUARRAY_SRC src/gpuarray/cuda/blas_cublas.c)
  add_definitions(-DWITH_CUDA_CUBLAS)

  set(CMAKE_REQUIRED_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})

  check_function_exists(cublasSgemmEx CUBLAS_SGEMMEX)
  if (CUBLAS_SGEMMEX)
    add_definitions(-DHAVE_CUBLAS_SGEMMEX)
  endif()

  if(NCCL_FOUND)
    message(STATUS "Building with NCCL")
    add_definitions(-DWITH_CUDA_NCCL)
    list(APPEND GPUARRAY_SRC src/gpuarray/cuda/collectives_nccl.c)
    include_directories(${NCCL_INCLUDE_DIR})
  endif()
endif()

if(OPENCL_FOUND)
  list(APPEND GPUARRAY_SRC src/gpuarray/opencl/buffer.c)
  add_definitions(-DWITH_OPENCL)
  include_directories(${OPENCL_INCLUDE_DIRS})

  if(CLBLAS_FOUND)
    message(STATUS "Building with CLBLAS")
    list(APPEND GPUARRAY_SRC src/gpuarray/opencl/blas_clblas.c)
    add_definitions(-DWITH_OPENCL_CLBLAS)
    include_directories(${CLBLAS_INCLUDE_DIRS})
  endif()
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/private_config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/src/private_config.h
  )

add_library(gpuarray SHARED ${GPUARRAY_SRC})
set_target_properties(gpuarray PROPERTIES
  COMPILE_FLAGS "-DGPUARRAY_BUILDING_DLL -DGPUARRAY_SHARED"
  INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}/lib
  MACOSX_RPATH OFF
  )

add_library(gpuarray-static STATIC ${GPUARRAY_SRC})

if(CUDA_FOUND)
  target_link_libraries(gpuarray ${CUDADRV_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
  target_link_libraries(gpuarray-static ${CUDADRV_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
  if (NCCL_FOUND)
    target_link_libraries(gpuarray ${NCCL_LIBRARY})
    target_link_libraries(gpuarray-static ${NCCL_LIBRARY})
  endif()
endif()

if(OPENCL_FOUND)
  target_link_libraries(gpuarray ${OPENCL_LIBRARIES})
  target_link_libraries(gpuarray-static ${OPENCL_LIBRARIES})
  if (CLBLAS_FOUND)
    target_link_libraries(gpuarray ${CLBLAS_LIBRARIES})
    target_link_libraries(gpuarray-static ${CLBLAS_LIBRARIES})
  endif()
endif()

add_subdirectory(test)

set(headers
  include/gpuarray/array.h
  include/gpuarray/blas.h
  include/gpuarray/collectives.h
  include/gpuarray/buffer.h
  include/gpuarray/buffer_blas.h
  include/gpuarray/buffer_collectives.h
  include/gpuarray/config.h
  include/gpuarray/elemwise.h
  include/gpuarray/error.h
  include/gpuarray/extension.h
  include/gpuarray/ext_cuda.h
  include/gpuarray/kernel.h
  include/gpuarray/types.h
  include/gpuarray/util.h
)

install(FILES ${headers} DESTINATION include/gpuarray)

if(NOT UNIX)
  install(FILES include/gpuarray/wincompat/stdint.h DESTINATION include/gpuarray/wincompat)
endif()

install(TARGETS gpuarray gpuarray-static
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
