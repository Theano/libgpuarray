/*
 * This file is a part of Hierarchical Allocator library.
 * Copyright (c) 2004-2011 Alex Pankratov.
 * Copyright (c) 2014 Arnaud Bergeron <abergeron@gmail.com>
 * All rights reserved.
 *
 * http://swapped.cc/halloc
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the following
 *   disclaimer in the documentation and/or other materials provided
 *   with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef GPUARRAY_UTIL_HALLOC_H
#define GPUARRAY_UTIL_HALLOC_H

#include "private_config.h"

/*
 *	Core API
 */
GPUARRAY_LOCAL void * halloc(void *block, size_t len);
GPUARRAY_LOCAL void   hattach(void *block, void *parent);

/*
 *	standard malloc/free api
 */
GPUARRAY_LOCAL void * h_malloc(size_t len);
GPUARRAY_LOCAL void * h_calloc(size_t n, size_t len);
GPUARRAY_LOCAL void * h_realloc(void * p, size_t len);
GPUARRAY_LOCAL void * h_reallocarray(void *p, size_t n, size_t len);
GPUARRAY_LOCAL void   h_free(void * p);
GPUARRAY_LOCAL char * h_strdup(const char * str);
GPUARRAY_LOCAL void * h_memdup(const void *p, size_t s);

#endif

