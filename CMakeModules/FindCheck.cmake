# This will find check and create the following variables
#
#  CHECK_FOUND
#  CHECK_INCLUDE_DIR
#  CHECK_LIBRARIES
#  CHECK_DEFINITIONS
#
#  This is public domain. Reuse as you wish.

find_package(PkgConfig)

pkg_check_modules(PC_CHECK QUIET check)

set(CHECK_DEFINITIONS ${PC_CHECK_CFLAGS_OTHER})

find_path(CHECK_INCLUDE_DIR check.h
          HINTS ${PC_CHECK_INCLUDEDIR} ${PC_CHECK_INCLUDE_DIRS})

find_library(CHECK_LIBRARY NAMES check
             HINTS ${PC_CHECK_LIBRARY_DIRS} ${PC_CHECK_LIBDIR})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Check DEFAULT_MSG
                                  CHECK_LIBRARY CHECK_INCLUDE_DIR)

mark_as_advanced(CHECK_INCLUDE_DIR CHECK_LIBRARY)

set(CHECK_LIBRARIES ${CHECK_LIBRARY})
set(CHECK_INCLUDE_DIRS ${CHECK_INCLUDE_DIR})
