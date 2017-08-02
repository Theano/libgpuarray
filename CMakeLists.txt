cmake_minimum_required(VERSION 3.0)

PROJECT(libgpuarray C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

# -Wall is unbelieveably noisy with Visual Studio:
# https://stackoverflow.com/q/4001736/3257826
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W3")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-parameter -Werror=format-security -Wdeclaration-after-statement -std=gnu89")
endif()

enable_testing()

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_OSX_ARCHITECTURES x86_64)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# Make recent cmake not spam about stuff
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 OLD)
endif()

add_subdirectory(src)
add_subdirectory(tests)

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
