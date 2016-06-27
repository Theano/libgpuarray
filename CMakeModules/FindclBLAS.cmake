# - Try to find clBLAS
#  Once done this will define
#
#  CLBLAS_FOUND - system has clBLAS
#  CLBLAS_INCLUDE_DIRS - location of clBLAS.h
#  CLBLAS_LIBRARIES - location of libclBLAS

IF(CLBLAS_INCLUDE_DIRS)
  # Already in cache, be silent
  set (CLBLAS_FIND_QUIETLY TRUE)
ENDIF (CLBLAS_INCLUDE_DIRS)

FIND_PATH(CLBLAS_ROOT_DIR
    NAMES include/clBLAS.h
    HINTS /usr/local/ $ENV{CLBLAS_ROOT}
    DOC "clBLAS root directory.")

FIND_PATH(_CLBLAS_INCLUDE_DIRS
    NAMES clBLAS.h
    HINTS ${CLBLAS_ROOT_DIR}/include
    DOC "clBLAS Include directory")

FIND_LIBRARY(_CLBLAS_LIBRARY
    NAMES CLBLAS clBLAS
    HINTS ${CLBLAS_ROOT_DIR}/lib ${CLBLAS_ROOT_DIR}/lib64 ${CLBLAS_ROOT_DIR}/lib32
    PATH_SUFFIXES import
    DOC "clBLAS lib directory")

SET(CLBLAS_INCLUDE_DIRS ${_CLBLAS_INCLUDE_DIRS})
SET(CLBLAS_LIBRARIES ${_CLBLAS_LIBRARY})

# handle the QUIETLY and REQUIRED arguments and set CLBLAS_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CLBLAS DEFAULT_MSG CLBLAS_LIBRARIES CLBLAS_INCLUDE_DIRS)
MARK_AS_ADVANCED(CLBLAS_LIBRARIES CLBLAS_INCLUDE_DIRS)
