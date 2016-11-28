include(CheckSymbolExists)
find_package(PkgConfig)

pkg_search_module(CHECK check)

if(NOT CHECK_FOUND)

  find_path(CHECK_INCLUDE_DIRS check.h)
  find_library(CHECK_LIBRARIES NAMES check)

  if(CHECK_INCLUDE_DIRS AND CHECK_LIBRARIES)
    set(CHECK_CFLAGS)
    set(CHECK_LIBRARY_DIRS)
    set(CHECK_FOUND 1)
  endif()

endif()

if(CHECK_FOUND)
  set(CMAKE_REQUIRED_INCLUDE ${CHECK_INCLUDE_DIRS})
  CHECK_SYMBOL_EXISTS(ck_assert_ptr_ne "check.h" CHECK_FUNCS)
  if (NOT CHECK_FUNCS)
    set(CHECK_FOUND 0)
  endif()
endif()

if(CHECK_FOUND)
enable_testing()

include_directories(${CHECK_INCLUDE_DIRS})
link_directories(${CHECK_LIBRARY_DIRS})

foreach(flag ${CHECK_C_FLAGS})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}")
endforeach()

foreach(flag ${CHECK_LDFLAGS_OTHER})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag}")
endforeach()

include_directories("${CMAKE_SOURCE_DIR}/src")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(check_types main.c check_types.c)
target_link_libraries(check_types ${CHECK_LIBRARIES} gpuarray)
add_test(test_types "${CMAKE_CURRENT_BINARY_DIR}/check_types")

add_executable(check_util main.c check_util.c)
target_link_libraries(check_util ${CHECK_LIBRARIES} gpuarray)
add_test(test_util "${CMAKE_CURRENT_BINARY_DIR}/check_util")

add_executable(check_util_integerfactoring main.c check_util_integerfactoring.c)
target_link_libraries(check_util_integerfactoring ${CHECK_LIBRARIES} gpuarray-static)
add_test(test_util_integerfactoring "${CMAKE_CURRENT_BINARY_DIR}/check_util_integerfactoring")

add_executable(check_reduction main.c device.c check_reduction.c)
target_link_libraries(check_reduction ${CHECK_LIBRARIES} gpuarray)
add_test(test_reduction "${CMAKE_CURRENT_BINARY_DIR}/check_reduction")

add_executable(check_array main.c device.c check_array.c)
target_link_libraries(check_array ${CHECK_LIBRARIES} gpuarray)
add_test(test_array "${CMAKE_CURRENT_BINARY_DIR}/check_array")

add_executable(check_blas main.c device.c check_blas.c)
target_link_libraries(check_blas ${CHECK_LIBRARIES} gpuarray)
add_test(test_blas "${CMAKE_CURRENT_BINARY_DIR}/check_blas")

add_executable(check_elemwise main.c device.c check_elemwise.c)
target_link_libraries(check_elemwise ${CHECK_LIBRARIES} gpuarray)
add_test(test_elemwise "${CMAKE_CURRENT_BINARY_DIR}/check_elemwise")

add_executable(check_error main.c check_error.c)
target_link_libraries(check_error ${CHECK_LIBRARIES} gpuarray)
add_test(test_error "${CMAKE_CURRENT_BINARY_DIR}/check_error")

add_executable(check_buffer main.c device.c check_buffer.c)
target_link_libraries(check_buffer ${CHECK_LIBRARIES} gpuarray)
add_test(test_buffer "${CMAKE_CURRENT_BINARY_DIR}/check_buffer")

find_package(MPI)

if (MPI_C_FOUND)

  add_executable(check_buffer_collectives
    main.c device.c communicator.c check_buffer_collectives.c
    )
  target_link_libraries(check_buffer_collectives
    ${CHECK_LIBRARIES} ${MPI_C_LIBRARIES} gpuarray
    )
  target_include_directories(check_buffer_collectives
    PRIVATE ${MPI_C_INCLUDE_PATH}
    )

  add_executable(check_collectives
    main.c device.c communicator.c check_collectives.c
    )
  target_link_libraries(check_collectives
    ${CHECK_LIBRARIES} ${MPI_C_LIBRARIES} gpuarray
    )
  target_include_directories(check_collectives
    PRIVATE ${MPI_C_INCLUDE_PATH}
    )

  set_target_properties(check_buffer_collectives check_collectives PROPERTIES
    COMPILE_DEFINITIONS TEST_COLLECTIVES
    COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}"
    LINK_FLAGS "${MPI_C_LINK_FLAGS}"
    )

  set(_NUM_DEVS $ENV{NUM_DEVS})
  if(NOT _NUM_DEVS)
    set(_NUM_DEVS 1)
  endif()

  set(_DEV_NAMES $ENV{DEV_NAMES})
  if(NOT _DEV_NAMES)
    set(_DEV_NAMES "cuda")
  endif()
  separate_arguments(_DEV_NAMES)

  add_test(NAME test_buffer_collectives
    COMMAND "${MPIEXEC}" ${MPIEXEC_NUMPROC_FLAG} ${_NUM_DEVS} ${MPIEXEC_PREFLAGS}
    "${CMAKE_CURRENT_BINARY_DIR}/check_buffer_collectives" ${MPIEXEC_POSTFLAGS} ${_DEV_NAMES})
  add_test(NAME test_collectives
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_NUM_DEVS} ${MPIEXEC_PREFLAGS}
    "${CMAKE_CURRENT_BINARY_DIR}/check_collectives" ${MPIEXEC_POSTFLAGS} ${_DEV_NAMES})

else()

  message(WARNING "Cannot find MPI")
  message(WARNING "Checks on collectives and buffer_collectives will not be built or performed.")

endif()

ELSE(CHECK_FOUND)

MESSAGE("Tests disabled because Check was not found")

ENDIF(CHECK_FOUND)
