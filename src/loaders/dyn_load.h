#ifndef UTIL_DYN_LOAD_H
#define UTIL_DYN_LOAD_H

#include "util/error.h"

void *ga_load_library(const char *name, error *e);
void *ga_func_ptr(void *h, const char *name, error *e);

#endif
