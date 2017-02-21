#ifndef LOADER_LIBNVRTC_H
#define LOADER_LIBNVRTC_H

#include "util/error.h"

/** @cond NEVER */

typedef enum {
  NVRTC_SUCCESS = 0,
} nvrtcResult;

typedef struct _nvrtcProgram *nvrtcProgram;

/** @endcond */

int load_libnvrtc(int major, int minor, error *e);

/** @cond NEVER */

#define DEF_PROC(rt, name, args) typedef rt t##name args

#include "libnvrtc.fn"

#undef DEF_PROC

#define DEF_PROC(rt, name, args) extern t##name *name

#include "libnvrtc.fn"

#undef DEF_PROC

/** @endcond */

#endif
