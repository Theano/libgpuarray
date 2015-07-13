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

/* Small hack to make NDEBUG respect DEBUG */
#ifdef DEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif
#endif

#include <assert.h>
#include <stdlib.h>  /* realloc */
#include <string.h>  /* memset & co */
#include <stddef.h>

#include "util/halloc.h"

#define structof(p,t,f) ((t*)(- offsetof(t,f) + (void*)(p)))

#ifndef _GCC_MAX_ALIGN_T
#define _GCC_MAX_ALIGN_T
union max_align
{
  char   c;
  short  s;
  long   l;
  int    i;
  float  f;
  double d;
  void * v;
  void (*q)(void);
};

typedef union max_align max_align_t;
#endif

/*
 *      weak double-linked list w/ tail sentinel
 */
typedef struct hlist_head  hlist_head_t;
typedef struct hlist_item  hlist_item_t;

/*
 *
 */
struct hlist_head
{
  hlist_item_t * next;
};

struct hlist_item
{
  hlist_item_t * next;
  hlist_item_t ** prev;
};

/*
 *      shared tail sentinel
 */
struct hlist_item hlist_null;

#define __hlist_init(h)      { &hlist_null }
#define __hlist_init_item(i) { &hlist_null, NULL }

#define hlist_for_each(i, h) \
  for (i = (h)->next; i != &hlist_null; i = i->next)

#define hlist_for_each_safe(i, tmp, h) \
  for (i = (h)->next, tmp = i->next; \
       i!= &hlist_null; \
       i = tmp, tmp = i->next)

static void hlist_init(hlist_head_t * h)
{
  assert(h);
  h->next = &hlist_null;
}

static void hlist_init_item(hlist_item_t * i)
{
  assert(i);
  i->prev = NULL;
  i->next = &hlist_null;
}

static void hlist_add(hlist_head_t * h, hlist_item_t * i)
{
  hlist_item_t * next;
  assert(h && i);

  next = i->next = h->next;
  next->prev = &i->next;
  h->next = i;
  i->prev = &h->next;
}

static void hlist_del(hlist_item_t * i)
{
  hlist_item_t * next;
  assert(i);

  next = i->next;
  next->prev = i->prev;
  if (i->prev)
    *i->prev = next;

  hlist_init_item(i);
}

static void hlist_relink(hlist_item_t * i)
{
  assert(i);
  if (i->prev)
    *i->prev = i;
  i->next->prev = &i->next;
}

static void hlist_relink_head(hlist_head_t * h)
{
  assert(h);
  h->next->prev = &h->next;
}

/*
 *	block control header
 */
typedef struct hblock
{
#ifdef DEBUG
#define HH_MAGIC    0x20040518L
	long          magic;
#endif
	hlist_item_t  siblings; /* 2 pointers */
	hlist_head_t  children; /* 1 pointer  */
	max_align_t   data[1];  /* not allocated, see below */

} hblock_t;

#define sizeof_hblock offsetof(hblock_t, data)

/*
 *
 */
static void * _realloc(void * ptr, size_t n);
typedef void *(*realloc_t)(void *, size_t n);
static realloc_t halloc_allocator = _realloc;

#define allocator halloc_allocator

/*
 *	static methods
 */
#ifndef NDEBUG
static int  _relate(hblock_t * b, hblock_t * p);
#endif
static void _free_children(hblock_t * p);

/*
 *	Core API
 */
void * halloc(void * ptr, size_t len) {
  hblock_t * p;

  assert(allocator);

  /* calloc */
  if (! ptr) {
    if (! len)
      return NULL;

    p = allocator(0, len + sizeof_hblock);
    if (! p)
      return NULL;
#ifdef DEBUG
    p->magic = HH_MAGIC;
#endif
    hlist_init(&p->children);
    hlist_init_item(&p->siblings);

    return p->data;
  }

  p = structof(ptr, hblock_t, data);
#ifdef DEBUG
  assert(p->magic == HH_MAGIC);
#endif

  /* realloc */
  if (len) {
    p = allocator(p, len + sizeof_hblock);
    if (p == NULL)
      return NULL;

    hlist_relink(&p->siblings);
    hlist_relink_head(&p->children);

    return p->data;
  }

  /* free */
  _free_children(p);
  hlist_del(&p->siblings);
  allocator(p, 0);

  return NULL;
}

void hattach(void * block, void * parent)
{
	hblock_t * b, * p;

	if (! block)
		return;

	/* detach */
	b = structof(block, hblock_t, data);
#ifdef DEBUG
	assert(b->magic == HH_MAGIC);
#endif

	hlist_del(&b->siblings);

	if (! parent)
		return;

	/* attach */
	p = structof(parent, hblock_t, data);
#ifdef DEBUG
	assert(p->magic == HH_MAGIC);
#endif

	/* sanity checks */
	assert(b != p);          /* trivial */
	assert(! _relate(p, b)); /* heavy ! */

	hlist_add(&p->children, &b->siblings);
}

/*
 *	malloc/free api
 */
void * h_malloc(size_t len)
{
	return halloc(0, len);
}

/*
 * This is sqrt(SIZE_MAX+1), as s1*s2 <= SIZE_MAX
 * if both s1 < MUL_NO_OVERFLOW and s2 < MUL_NO_OVERFLOW
 */
#define MUL_NO_OVERFLOW (1UL << (sizeof(size_t) * 4))

void * h_calloc(size_t n, size_t len)
{
  if ((n >= MUL_NO_OVERFLOW || len >= MUL_NO_OVERFLOW) &&
      n > 0 && SIZE_MAX / n < len) {
    return NULL;
  }
  len *= n;
  void * ptr = halloc(0, len);
  return ptr ? memset(ptr, 0, len) : NULL;
}

void * h_realloc(void * ptr, size_t len)
{
  return halloc(ptr, len);
}

void *h_reallocarray(void *ptr, size_t n, size_t len)
{
  if ((n >= MUL_NO_OVERFLOW || len >= MUL_NO_OVERFLOW) &&
      n > 0 && SIZE_MAX / n < len) {
    return NULL;
  }
  return halloc(ptr, n*len);
}

void   h_free(void * ptr)
{
	halloc(ptr, 0);
}

char * h_strdup(const char * str)
{
	size_t len = strlen(str);
	char * ptr = halloc(0, len + 1);
	return ptr ? (ptr[len] = 0, memcpy(ptr, str, len)) : NULL;
}

void * h_memdup(const void * p, size_t s)
{
       void * ptr = halloc(NULL, s);
       return ptr ? memcpy(ptr, p, s) : NULL;
}

static void * _realloc(void * ptr, size_t n)
{
  /* free'ing realloc() */
  if (n != 0) {
    return realloc(ptr, n);
  } else {
    free(ptr);
    return NULL;
  }
}

/* Only used in asserts */
#ifndef NDEBUG
static int _relate(hblock_t * b, hblock_t * p)
{
	hlist_item_t * i;

	if (!b || !p)
		return 0;

	/*
	 *  since there is no 'parent' pointer, which would've allowed
	 *  O(log(n)) upward traversal, the check must use O(n) downward
	 *  iteration of the entire hierarchy; and this can be VERY SLOW
	 */
	hlist_for_each(i, &p->children)
	{
		hblock_t * q = structof(i, hblock_t, siblings);
		if (q == b || _relate(b, q))
			return 1;
	}
	return 0;
}
#endif

static void _free_children(hblock_t * p)
{
	hlist_item_t * i, * tmp;

#ifdef DEBUG
	/*
	 *	this catches loops in hierarchy with almost zero
	 *	overhead (compared to _relate() running time)
	 */
	assert(p && p->magic == HH_MAGIC);
	p->magic = 0;
#endif
	hlist_for_each_safe(i, tmp, &p->children)
	{
		hblock_t * q = structof(i, hblock_t, siblings);
		_free_children(q);
		allocator(q, 0);
	}
}
