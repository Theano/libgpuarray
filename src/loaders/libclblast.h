#ifndef LOADER_LIBCLBLAST_H
#define LOADER_LIBCLBLAST_H

#include "util/error.h"
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

typedef enum CLBLastStatusCode_ {
  kSuccess = 0,
  /* Rest is not exposed from here */
  CLBlastNotImplemented            = -1024,
  CLBlastInvalidMatrixA            = -1022,
  CLBlastInvalidMatrixB            = -1021,
  CLBlastInvalidMatrixC            = -1020,
  CLBlastInvalidVectorX            = -1019,
  CLBlastInvalidVectorY            = -1018,
  CLBlastInvalidDimension          = -1017,
  CLBlastInvalidLeadDimA           = -1016,
  CLBlastInvalidLeadDimB           = -1015,
  CLBlastInvalidLeadDimC           = -1014,
  CLBlastInvalidIncrementX         = -1013,
  CLBlastInvalidIncrementY         = -1012,
  CLBlastInsufficientMemoryA       = -1011,
  CLBlastInsufficientMemoryB       = -1010,
  CLBlastInsufficientMemoryC       = -1009,
  CLBlastInsufficientMemoryX       = -1008,
  CLBlastInsufficientMemoryY       = -1007,

  CLBlastInvalidLocalMemUsage      = -2046,
  CLBlastNoHalfPrecision           = -2045,
  CLBlastNoDoublePrecision         = -2044,
  CLBlastInvalidVectorScalar       = -2043,
  CLBlastInsufficientMemoryScalar  = -2042,
  CLBlastDatabaseError             = -2041,
  CLBlastUnknownError              = -2040,
  CLBlastUnexpectedError           = -2039,
} CLBlastStatusCode;

int load_libclblast(error *);

#define DEF_PROC(ret, name, args) typedef ret t##name args

#include "libclblast.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args) extern t##name *name

#include "libclblast.fn"

#undef DEF_PROC

#endif
