enable_testing()

find_package(PkgConfig)
find_package(MPI)

pkg_search_module(CHECK check)

if (NOT CHECK_FOUND)

find_path(CHECK_INCLUDE_DIRS check.h)
find_library(CHECK_LIBRARIES NAMES check)

if (CHECK_INCLUDE_DIRS AND CHECK_LIBRARIES)
  set(CHECK_CFLAGS)
  set(CHECK_LIBRARY_DIRS)
  set(CHECK_FOUND 1)
endif()

endif()

if (CHECK_FOUND)

include_directories(${CHECK_INCLUDE_DIRS})
link_directories(${CHECK_LIBRARY_DIRS})

set(LIBS ${LIBS} ${CHECK_LIBRARIES})

foreach(flag ${CHECK_C_FLAGS})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}")
endforeach()

foreach(flag ${CHECK_LDFLAGS_OTHER})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag}")
endforeach()

include_directories(. ../src)

add_executable(check_types main.c check_types.c)
target_link_libraries(check_types ${LIBS} gpuarray)
add_test(test_types ${CMAKE_CURRENT_BINARY_DIR}/check_types)

add_executable(check_util main.c check_util.c)
target_link_libraries(check_util ${LIBS} gpuarray)
add_test(test_util ${CMAKE_CURRENT_BINARY_DIR}/check_util)

add_executable(check_array main.c device.c check_array.c)
target_link_libraries(check_array ${LIBS} gpuarray)
add_test(test_array ${CMAKE_CURRENT_BINARY_DIR}/check_array)

add_executable(check_elemwise main.c device.c check_elemwise.c)
target_link_libraries(check_elemwise ${LIBS} gpuarray)
add_test(test_elemwise ${CMAKE_CURRENT_BINARY_DIR}/check_elemwise)

add_executable(check_error main.c check_error.c)
target_link_libraries(check_error ${LIBS} gpuarray)
add_test(test_error ${CMAKE_CURRENT_BINARY_DIR}/check_error)

add_executable(check_buffer main.c device.c check_buffer.c)
target_link_libraries(check_buffer ${LIBS} gpuarray)
add_test(test_buffer ${CMAKE_CURRENT_BINARY_DIR}/check_buffer)

# TODO Add tests 1. Only build execs (to find compilation errs)
# 2. Run in servers with multi-process/gpu
ELSE(CHECK_FOUND)

MESSAGE("Tests disabled because Check was not found")

ENDIF(CHECK_FOUND)
