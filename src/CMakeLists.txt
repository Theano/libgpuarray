PROJECT(compyte C)

include(CheckFunctionExists)

cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_OSX_ARCHITECTURES i386 x86_64)
set(CMAKE_OSX_SYSROOT /)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../lib)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG")

find_package(CUDA)
find_package(OpenCL)

include_directories("${PROJECT_SOURCE_DIR}")

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

if(UNIX)
  add_definitions(-D_GNU_SOURCE)
endif()

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
  if (APPLE)
    FIND_LIBRARY(CUDADRV_LIBRARY CUDA)
    FIND_PATH(CUDADRV_INCLUDE CUDA/cuda.h)
  endif()
  if(NOT CUDADRV_LIBRARY)
    SET(CUDADRV_LIBRARY ${CUDA_CUDA_LIBRARY})
    SET(CUDADRV_INCLUDE ${CUDA_TOOLKIT_INCLUDE})
  endif()

  set(COMPYTE_SRC ${COMPYTE_SRC} compyte_buffer_cuda.c)
  add_definitions(-DNVCC_BIN="${CUDA_NVCC_EXECUTABLE}" -DWITH_CUDA)
  include_directories("${CUDADRV_INCLUDE}")
endif()

if(OPENCL_FOUND)
  set(COMPYTE_SRC ${COMPYTE_SRC} compyte_buffer_opencl.c)
  add_definitions(-DWITH_OPENCL)
  include_directories("${OPENCL_INCLUDE_DIRS}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

configure_file(
  ${PROJECT_SOURCE_DIR}/compyte/compat.h.in
  ${PROJECT_SOURCE_DIR}/compyte/compat.h
  )

add_library(compyte SHARED ${COMPYTE_SRC})
set_target_properties(compyte PROPERTIES COMPILE_FLAGS "-DCOMPYTE_BUILDING_DLL -DCOMPYTE_SHARED")

add_library(compyte-static STATIC ${COMPYTE_SRC})

if(CUDA_FOUND)
  target_link_libraries(compyte ${CUDADRV_LIBRARY})
  target_link_libraries(compyte-static ${CUDADRV_LIBRARY})
endif()

if(OPENCL_FOUND)
  target_link_libraries(compyte ${OPENCL_LIBRARIES})
  target_link_libraries(compyte-static ${OPENCL_LIBRARIES})
endif()

FILE(GLOB headers compyte/*.h)

INSTALL(FILES ${headers} DESTINATION include/compyte)

if(NOT UNIX)
  INSTALL(FILES compyte/wincompat/stdint.h DESTINATION include/compyte/wincompat)
endif()

INSTALL(TARGETS compyte compyte-static
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
