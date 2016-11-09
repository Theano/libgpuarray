#ifndef UTIL_DYN_LOAD_H
#define UTIL_DYN_LOAD_H

void *ga_load_library(const char *name);
void *ga_func_ptr(void *h, const char *name);

#endif
