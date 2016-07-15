#ifndef LIBGPUARRAY_GPUARRAY_CONFIG_H
#define LIBGPUARRAY_GPUARRAY_CONFIG_H

#ifdef GPUARRAY_SHARED
  #ifdef _WIN32
    #ifdef GPUARRAY_BUILDING_DLL
      #define GPUARRAY_PUBLIC __declspec(dllexport)
    #else
      #define GPUARRAY_PUBLIC __declspec(dllimport)
    #endif
    #define GPUARRAY_LOCAL
  #else  // _WIN32
    #if __GNUC__ >= 4
      #define GPUARRAY_PUBLIC __attribute__((visibility ("default")))
      #define GPUARRAY_LOCAL  __attribute__((visibility ("hidden")))
    #else
      #define GPUARRAY_PUBLIC
      #define GPUARRAY_LOCAL
    #endif
  #endif  // _WIN32
#else  // GPUARRAY_SHARED
  #define GPUARRAY_PUBLIC
  #define GPUARRAY_LOCAL
#endif  // GPUARRAY_SHARED

#ifdef _MSC_VER
  #include <stddef.h>
  #if _MSC_VER < 1600
    #include <gpuarray/wincompat/stdint.h>
  #endif
  #define ssize_t intptr_t
  #define SSIZE_MAX INTPTR_MAX
#else  // _MSC_VER
  #include <sys/types.h>
  #include <stdint.h>
#endif  // _MSC_VER

#endif  // LIBGPUARRAY_GPUARRAY_CONFIG_H
