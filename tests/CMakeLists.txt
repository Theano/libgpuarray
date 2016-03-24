enable_testing()

find_package(Check)

if (CHECK_FOUND)

include_directories(${CHECK_INCLUDE_DIRS})

set(LIBS ${LIBS} ${CHECK_LIBRARIES})

# This is a workaround because cmake is stupid when dealing with
# pkg-config output
STRING(REPLACE ";" " " CHK_FLAGS "${CHECK_CFLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CHK_FLAGS}")

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

ELSE(CHECK_FOUND)

MESSAGE("Tests disabled because Check was not found")

ENDIF(CHECK_FOUND)
