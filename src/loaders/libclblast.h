#ifndef LOADER_LIBCLBLAST_H
#define LOADER_LIBCLBLAST_H

#include "libopencl.h"

typedef enum Layout_ {
  kRowMajor = 101,
  kColMajor = 102
} Layout;

typedef enum Transpose_ {
  kNo = 111,
  kYes = 112,
  kConjugate = 113
} Transpose;

typedef enum StatusCode_ {
  kSuccess = 0,
  /* Rest is not exposed from here */
} StatusCode;

int load_libclblast(void);

#define DEF_PROC(ret, name, args) typedef ret t##name args

#include "libclblast.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args) extern t##name *name

#include "libclblast.fn"

#undef DEF_PROC

#endif
