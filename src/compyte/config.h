#ifndef COMPYTE_CONFIG
#define COMPYTE_CONFIG

#ifdef COMPYTE_SHARED
 #ifdef _WIN32
  #ifdef COMPYTE_BUILDING_DLL
   #define COMPYTE_PUBLIC __declspec(dllexport)
  #else
   #define COMPYTE_PUBLIC __declspec(dllimport)
  #endif
  #define COMPYTE_LOCAL
 #else
  #if __GNUC__ >= 4
   #define COMPYTE_PUBLIC __attribute__((visibility ("default")))
   #define COMPYTE_LOCAL  __attribute__((visibility ("hidden")))
  #else
   #define COMPYTE_PUBLIC
   #define COMPYTE_LOCAL
  #endif
 #endif
#else
 #define COMPYTE_PUBLIC
 #define COMPYTE_LOCAL
#endif

#ifdef _MSC_VER
#include <stddef.h>
#include <compyte/wincompat/stdint.h>
#define ssize_t intptr_t
#define SSIZE_MAX INTPTR_MAX
#else
#include <sys/types.h>
#include <stdint.h>
#endif

#endif
