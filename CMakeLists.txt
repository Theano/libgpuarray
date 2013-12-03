cmake_minimum_required(VERSION 2.8)

PROJECT(compyte C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

enable_testing()

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_OSX_ARCHITECTURES i386 x86_64)

add_subdirectory(src)
add_subdirectory(tests)
