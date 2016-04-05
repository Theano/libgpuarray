enable_testing()

find_package(PkgConfig)

pkg_search_module(CHECK check)

if (CHECK_FOUND)

include_directories(${CHECK_INCLUDE_DIRS})
link_directories(${CHECK_LIBRARY_DIRS})

set(LIBS ${LIBS} ${CHECK_LIBRARIES})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CHECK_CFLAGS}")

include_directories(. ../src)

add_executable(check_types main.c check_types.c)
target_link_libraries(check_types ${LIBS} gpuarray)
add_test(test_types ${CMAKE_CURRENT_BINARY_DIR}/check_types)

add_executable(check_util main.c check_util.c)
target_link_libraries(check_util ${LIBS} gpuarray)
add_test(test_util ${CMAKE_CURRENT_BINARY_DIR}/check_util)

add_executable(check_array main.c check_array.c)
target_link_libraries(check_array ${LIBS} gpuarray)
add_test(test_array ${CMAKE_CURRENT_BINARY_DIR}/check_array)

add_executable(check_error main.c check_error.c)
target_link_libraries(check_error ${LIBS} gpuarray)
add_test(test_error ${CMAKE_CURRENT_BINARY_DIR}/check_error)

add_executable(check_buffer main.c check_buffer.c)
target_link_libraries(check_buffer ${LIBS} gpuarray)
add_test(test_buffer ${CMAKE_CURRENT_BINARY_DIR}/check_buffer)

ELSE(CHECK_FOUND)

MESSAGE("Tests disabled because Check was not found")

ENDIF(CHECK_FOUND)
