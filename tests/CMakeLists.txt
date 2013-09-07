enable_testing()

find_package(Check)

if (CHECK_FOUND)

include_directories(${CHECK_INCLUDE_DIRS})

set(LIBS ${LIBS} ${CHECK_LIBRARIES})

include_directories(. ../src)

add_executable(check_buf main.c check_buf.c)

target_link_libraries(check_buf ${LIBS})

add_test(test_buf ${CMAKE_CURRENT_BINARY_DIR}/check_buf)

ELSE(CHECK_FOUND)

MESSAGE("Tests disabled because Check was not found")

ENDIF(CHECK_FOUND)
