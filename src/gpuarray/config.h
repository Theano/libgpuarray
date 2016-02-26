#ifndef GPUARRAY_CONFIG
#define GPUARRAY_CONFIG

#ifdef GPUARRAY_SHARED
 #ifdef _WIN32
  #ifdef GPUARRAY_BUILDING_DLL
   #define GPUARRAY_PUBLIC __declspec(dllexport)
  #else
   #define GPUARRAY_PUBLIC __declspec(dllimport)
  #endif
  #define GPUARRAY_LOCAL
 #else
  #if __GNUC__ >= 4
   #define GPUARRAY_PUBLIC __attribute__((visibility ("default")))
   #define GPUARRAY_LOCAL  __attribute__((visibility ("hidden")))
  #else
   #define GPUARRAY_PUBLIC
   #define GPUARRAY_LOCAL
  #endif
 #endif
#else
 #define GPUARRAY_PUBLIC
 #define GPUARRAY_LOCAL
#endif

#ifdef _MSC_VER
#include <stddef.h>
#if _MSC_VER < 1600
#include <gpuarray/wincompat/stdint.h>
#endif
#define ssize_t intptr_t
#define SSIZE_MAX INTPTR_MAX
#else
#include <sys/types.h>
#include <stdint.h>
#endif

#endif
