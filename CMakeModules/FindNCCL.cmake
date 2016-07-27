# Find the NCCL libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR:    Base directory where all NCCL components are found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIR
#  NCCL_LIBRARY

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS
      ENV CUDA_PATH
      ENV NCCL_ROOT_DIR
    PATH_SUFFIXES
      include)

find_library(NCCL_LIBRARY
    NAMES nccl
    PATHS
      ENV CUDA_PATH
      ENV NCCL_ROOT_DIR
    PATH_SUFFIXES
      lib64
      lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NCCL
  FOUND_VAR NCCL_FOUND
  REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR)

mark_as_advanced(
  NCCL_INCLUDE_DIR
  NCCL_LIBRARY)
