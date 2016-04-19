include(CheckFunctionExists)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG")

find_package(CUDA)
find_package(OpenCL)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/gpuarray_types.c
         ${CMAKE_CURRENT_SOURCE_DIR}/gpuarray/types.h
  COMMAND python gen_types.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_types.py)

macro (set_rel var)
  file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
  # clear previous list (if any)
  set(SET_REL_TMP)
  foreach (_src ${ARGN})
    if (_relPath)
      list(APPEND SET_REL_TMP "${_relPath}/${_src}")
    else()
      list(APPEND SET_REL_TMP "${_src}")
    endif()
  endforeach()
  if (_relPath)
    set(${var} ${SET_REL_TMP} PARENT_SCOPE)
  else()
    set(${var} ${SET_REL_TMP})
  endif()
endmacro()

set_rel(GPUARRAY_SRC
cache/lru.c
cache/twoq.c
gpuarray_types.c
gpuarray_error.c
gpuarray_util.c
gpuarray_buffer.c
gpuarray_array.c
gpuarray_array_blas.c
gpuarray_kernel.c
gpuarray_extension.c
gpuarray_elemwise.c
)

check_function_exists(strlcat HAVE_STRL)
check_function_exists(mkstemp HAVE_MKSTEMP)

if(UNIX)
  add_definitions(-D_GNU_SOURCE)
endif()

if(NOT HAVE_STRL)
  set(GPUARRAY_SRC ${GPUARRAY_SRC} gpuarray_strl.c)
endif()

if(NOT HAVE_MKSTEMP)
  set(GPUARRAY_SRC ${GPUARRAY_SRC} gpuarray_mkstemp.c)
endif()

if (CUDA_FOUND)
  if (CUDA_VERSION_MAJOR LESS 6 OR
      (CUDA_VERSION_MAJOR EQUAL 6 AND CUDA_VERSION_MINOR EQUAL 0))
    message( WARNING "This package requires CUDA 6.5 or more.  Found version ${CUDA_VERSION_STRING}")
    set(CUDA_FOUND 0)
  endif()
endif()

if (CUDA_FOUND)
  if (APPLE)
    FIND_LIBRARY(CUDADRV_LIBRARY CUDA)
    FIND_PATH(CUDADRV_INCLUDE CUDA/cuda.h)
    # this is somewhat a hack, but otherwise cublas_v2.h isn't found
    set(CUDADRV_INCLUDE "${CUDADRV_INCLUDE}" "${CUDA_TOOLKIT_INCLUDE}")
  endif()
  if(NOT CUDADRV_LIBRARY)
    SET(CUDADRV_LIBRARY ${CUDA_CUDA_LIBRARY})
    SET(CUDADRV_INCLUDE ${CUDA_TOOLKIT_INCLUDE})
  endif()

  find_cuda_helper_libs(nvrtc)

  if(CUDA_nvrtc_LIBRARY)
    message("Building with NVRTC")
    add_definitions(-DWITH_NVRTC)
    set(CUDADRV_LIBRARY ${CUDADRV_LIBRARY} ${CUDA_nvrtc_LIBRARY})
  else()
    add_definitions(-DNVCC_BIN="${CUDA_NVCC_EXECUTABLE}")
  endif()

  set(GPUARRAY_SRC ${GPUARRAY_SRC} gpuarray_buffer_cuda.c)
  add_definitions(-DWITH_CUDA)
  include_directories("${CUDADRV_INCLUDE}")
  set(GPUARRAY_SRC ${GPUARRAY_SRC} gpuarray_blas_cuda_cublas.c)
  add_definitions(-DWITH_CUDA_CUBLAS)
  set(CMAKE_REQUIRED_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
  check_function_exists(cublasSgemmEx CUBLAS_SGEMMEX)
  if (CUBLAS_SGEMMEX)
    add_definitions(-DHAVE_CUBLAS_SGEMMEX)
  endif()
endif()

if(OPENCL_FOUND)
  set(GPUARRAY_SRC ${GPUARRAY_SRC} gpuarray_buffer_opencl.c)
  add_definitions(-DWITH_OPENCL)
  include_directories("${OPENCL_INCLUDE_DIRS}")
  find_package(clBLAS)
  if (CLBLAS_FOUND)
    set(GPUARRAY_SRC ${GPUARRAY_SRC} gpuarray_blas_opencl_clblas.c)
    add_definitions(-DWITH_OPENCL_CLBLAS)
    include_directories("${CLBLAS_INCLUDE_DIRS}")
  endif()
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/private_config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/private_config.h
  )

add_subdirectory(util)

set(GPUARRAY_SRC ${GPUARRAY_SRC} ${UTIL_SRC})

add_library(gpuarray SHARED ${GPUARRAY_SRC})
set_target_properties(gpuarray PROPERTIES
  COMPILE_FLAGS "-DGPUARRAY_BUILDING_DLL -DGPUARRAY_SHARED"
  INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}/lib
  MACOSX_RPATH OFF
  )

add_library(gpuarray-static STATIC ${GPUARRAY_SRC})

if(CUDA_FOUND)
  target_link_libraries(gpuarray ${CUDADRV_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
  target_link_libraries(gpuarray-static ${CUDADRV_LIBRARY} ${CUDA_CUBLAS_LIBRARY})
endif()

if(OPENCL_FOUND)
  target_link_libraries(gpuarray ${OPENCL_LIBRARIES})
  target_link_libraries(gpuarray-static ${OPENCL_LIBRARIES})
  if (CLBLAS_FOUND)
    target_link_libraries(gpuarray ${CLBLAS_LIBRARIES})
    target_link_libraries(gpuarray-static ${CLBLAS_LIBRARIES})
  endif()
endif()

SET(headers
  gpuarray/array.h
  gpuarray/blas.h
  gpuarray/buffer.h
  gpuarray/buffer_blas.h
  gpuarray/config.h
  gpuarray/elemwise.h
  gpuarray/error.h
  gpuarray/extension.h
  gpuarray/ext_cuda.h
  gpuarray/kernel.h
  gpuarray/types.h
  gpuarray/util.h
)

INSTALL(FILES ${headers} DESTINATION include/gpuarray)

if(NOT UNIX)
  INSTALL(FILES gpuarray/wincompat/stdint.h DESTINATION include/gpuarray/wincompat)
endif()

INSTALL(TARGETS gpuarray gpuarray-static
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
