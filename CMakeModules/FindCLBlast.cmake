# - Try to find CLBlast
#  Once done this will define
#
#  CLBLAST_FOUND - system has CLBlast
#  CLBLAST_INCLUDE_DIRS - location of CLBlast.h
#  CLBLAST_LIBRARIES - location of libCLBlast

IF(CLBLAST_INCLUDE_DIRS)
  # Already in cache, be silent
  set (CLBLAST_FIND_QUIETLY TRUE)
ENDIF (CLBLAST_INCLUDE_DIRS)

FIND_PATH(CLBLAST_ROOT_DIR
    NAMES include/clblast_c.h
    HINTS /usr/local/ $ENV{CLBLAST_ROOT}
    DOC "CLBlast root directory.")

FIND_PATH(_CLBLAST_INCLUDE_DIRS
    NAMES clblast_c.h
    HINTS ${CLBLAST_ROOT_DIR}/include
    DOC "CLBlast Include directory")

FIND_LIBRARY(_CLBLAST_LIBRARY
	NAMES libclblast.so
    HINTS ${CLBLAST_ROOT_DIR}/lib ${CLBLAST_ROOT_DIR}/lib64 ${CLBLAST_ROOT_DIR}/lib32
    DOC "CLBlast lib directory")

SET(CLBLAST_INCLUDE_DIRS ${_CLBLAST_INCLUDE_DIRS})
SET(CLBLAST_LIBRARIES ${_CLBLAST_LIBRARY})

# handle the QUIETLY and REQUIRED arguments and set CLBLAST_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CLBLAST DEFAULT_MSG CLBLAST_LIBRARIES CLBLAST_INCLUDE_DIRS)
MARK_AS_ADVANCED(CLBLAST_LIBRARIES CLBLAST_INCLUDE_DIRS)
