include(CheckFunctionExists)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG")
if(CMAKE_COMPILER_IS_GNUCC)
  add_definitions(-Wdeclaration-after-statement)
endif()

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

set(_GPUARRAY_SRC
cache/lru.c
cache/twoq.c
cache/disk.c
gpuarray_types.c
gpuarray_error.c
gpuarray_util.c
gpuarray_buffer.c
gpuarray_buffer_blas.c
gpuarray_buffer_collectives.c
gpuarray_array.c
gpuarray_array_blas.c
gpuarray_array_collectives.c
gpuarray_kernel.c
gpuarray_extension.c
gpuarray_elemwise.c
gpuarray_reduction.c
gpuarray_buffer_cuda.c
gpuarray_blas_cuda_cublas.c
gpuarray_collectives_cuda_nccl.c
gpuarray_buffer_opencl.c
gpuarray_blas_opencl_clblas.c
gpuarray_blas_opencl_clblast.c
)

check_function_exists(strlcat HAVE_STRL)
check_function_exists(mkstemp HAVE_MKSTEMP)

if(UNIX)
  add_definitions(-D_GNU_SOURCE)
endif()

if(NOT HAVE_STRL)
  list(APPEND _GPUARRAY_SRC gpuarray_strl.c)
endif()

if(NOT HAVE_MKSTEMP)
  list(APPEND _GPUARRAY_SRC gpuarray_mkstemp.c)
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/private_config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/private_config.h
  )

add_subdirectory(util)
add_subdirectory(loaders)

set_rel(GPUARRAY_SRC ${_GPUARRAY_SRC})
list(APPEND GPUARRAY_SRC ${UTIL_SRC} ${LOADERS_SRC})

add_library(gpuarray SHARED ${GPUARRAY_SRC})
set_target_properties(gpuarray PROPERTIES
  COMPILE_FLAGS "-DGPUARRAY_BUILDING_DLL -DGPUARRAY_SHARED"
  INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}/lib
  MACOSX_RPATH OFF
  # This is the shared library version
  VERSION 2.1
  )

add_library(gpuarray-static STATIC ${GPUARRAY_SRC})

target_link_libraries(gpuarray ${CMAKE_DL_LIBS})
target_link_libraries(gpuarray-static ${CMAKE_DL_LIBS})

# Generate gpuarray/abi_version.h that contains the ABI version number.
get_target_property(GPUARRAY_ABI_VERSION gpuarray VERSION)
string(REPLACE "." ";" GPUARRAY_ABI_VERSION_NUMBERS ${GPUARRAY_ABI_VERSION})
list(GET GPUARRAY_ABI_VERSION_NUMBERS 0 GPUARRAY_ABI_VERSION_MAJOR)
list(GET GPUARRAY_ABI_VERSION_NUMBERS 1 GPUARRAY_ABI_VERSION_MINOR)
math(EXPR GPUARRAY_ABI_NUMBER "1000*${GPUARRAY_ABI_VERSION_MAJOR} + ${GPUARRAY_ABI_VERSION_MINOR}")
FILE(WRITE gpuarray/abi_version.h
"\#ifndef GPUARRAY_ABI_VERSION\n\#define GPUARRAY_ABI_VERSION ${GPUARRAY_ABI_NUMBER}\n\#endif\n"
)

# set SOVERSION and ensure it is the first part of VERSION.
set_property(TARGET gpuarray PROPERTY SOVERSION ${GPUARRAY_ABI_VERSION_MAJOR})

set(headers
  gpuarray/array.h
  gpuarray/blas.h
  gpuarray/collectives.h
  gpuarray/buffer.h
  gpuarray/buffer_blas.h
  gpuarray/buffer_collectives.h
  gpuarray/abi_version.h
  gpuarray/config.h
  gpuarray/elemwise.h
  gpuarray/error.h
  gpuarray/extension.h
  gpuarray/ext_cuda.h
  gpuarray/kernel.h
  gpuarray/types.h
  gpuarray/util.h
)

install(FILES ${headers} DESTINATION include/gpuarray)

if(NOT UNIX)
  install(FILES gpuarray/wincompat/stdint.h DESTINATION include/gpuarray/wincompat)
endif()

install(TARGETS gpuarray gpuarray-static
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
