enable_testing()

find_package(Check)

if (CHECK_FOUND)

include_directories(${CHECK_INCLUDE_DIRS})

set(LIBS ${LIBS} ${CHECK_LIBRARIES})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CHECK_CFLAGS}")

include_directories(. ../src)

add_executable(check_types main.c check_types.c)
target_link_libraries(check_types ${LIBS} gpuarray)
add_test(test_types ${CMAKE_CURRENT_BINARY_DIR}/check_types)

add_executable(check_error main.c check_error.c)
target_link_libraries(check_error ${LIBS} gpuarray)
add_test(test_error ${CMAKE_CURRENT_BINARY_DIR}/check_error)

add_executable(check_buffer main.c check_buffer.c)
target_link_libraries(check_buffer ${LIBS} gpuarray)
add_test(test_buffer ${CMAKE_CURRENT_BINARY_DIR}/check_buffer)

ELSE(CHECK_FOUND)

MESSAGE("Tests disabled because Check was not found")

ENDIF(CHECK_FOUND)
