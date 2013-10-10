set(GEN_BLAS_FILES
  ${CMAKE_SOURCE_DIR}/src/generic_blas.inc.c
  ${CMAKE_SOURCE_DIR}/src/compyte/buffer_blas.h
  ${CMAKE_SOURCE_DIR}/src/compyte/blas.h
  ${CMAKE_SOURCE_DIR}/src/compyte_array_blas.c
  ${CMAKE_SOURCE_DIR}/pygpu/blas.pyx
)

add_custom_command(
  OUTPUT ${GEN_BLAS_FILES}
  COMMAND python ${CMAKE_SOURCE_DIR}/gen_blas.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS ${CMAKE_SOURCE_DIR}/gen_blas.py
)
