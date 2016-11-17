#ifndef LOADER_LIBNVRTC_H
#define LOADER_LIBNVRTC_H

typedef enum {
  NVRTC_SUCCESS = 0,
} nvrtcResult;

typedef struct _nvrtcProgram *nvrtcProgram;

int load_libnvrtc(int major, int minor);

#define DEF_PROC(name, args) typedef nvrtcResult t##name args

#include "libnvrtc.fn"

#undef DEF_PROC

#define DEF_PROC(name, args) extern t##name *name

#include "libnvrtc.fn"

#undef DEF_PROC

#endif
