/*
 * This whole cache business is ugly, but fast.
 */
typedef struct _extcopy_args {
  unsigned int ind;
  unsigned int ond;
  size_t ioff;
  size_t ooff;
  int itype;
  int otype;
  const size_t *idims;
  const size_t *odims;
  const ssize_t *istr;
  const ssize_t *ostr;
  size_t hash;
} cache_key_t;

typedef gpukernel *cache_val_t;

#define key_hash(k) (k)->hash

static inline int key_eq(const cache_key_t *k1, const cache_key_t *k2) {
  return (k1->ind == k2->ind && k1->ond == k2->ond &&
	  k1->ioff == k2->ioff && k1->ooff == k2->ooff &&
	  k1->itype == k2->itype && k1->otype == k2->otype &&
	  memcmp(k1->idims, k2->idims, k1->ind * sizeof(size_t)) == 0 &&
	  memcmp(k1->odims, k2->odims, k1->ond * sizeof(size_t)) == 0 &&
	  memcmp(k1->istr, k2->istr, k1->ind * sizeof(ssize_t)) == 0 &&
	  memcmp(k1->ostr, k2->ostr, k1->ond * sizeof(ssize_t)) == 0);
}

static inline void key_free(const cache_key_t *k) {
  free((void *)k->idims);
  free((void *)k->odims);
  free((void *)k->istr);
  free((void *)k->ostr);
}

#include <assert.h>
#include <stdlib.h>

#include "cache_impl.h"

#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define mix(a,b,c)				\
  {						\
    a -= c;  a ^= rot(c, 4);  c += b;		\
    b -= a;  b ^= rot(a, 6);  a += c;		\
    c -= b;  c ^= rot(b, 8);  b += a;		\
    a -= c;  a ^= rot(c,16);  c += b;		\
    b -= a;  b ^= rot(a,19);  a += c;		\
    c -= b;  c ^= rot(b, 4);  b += a;		\
  }

#define final(a,b,c)				\
  {						\
    c ^= b; c -= rot(b,14);			\
    a ^= c; a -= rot(c,11);			\
    b ^= a; b -= rot(a,25);			\
    c ^= b; c -= rot(b,16);			\
    a ^= c; a -= rot(c,4);			\
    b ^= a; b -= rot(a,14);			\
    c ^= b; c -= rot(b,24);			\
  }

static inline void hashword2(const uint32_t *k, size_t l,
			     uint32_t *pb, uint32_t *pc)  {
  uint32_t a, b, c;
  a = b = c = 0xdeadbeef + ((uint32_t)(l<<2)) + *pc;
  c += *pb;

  while (l > 3) {
    a += k[0];
    b += k[1];
    c += k[2];
    mix(a,b,c);
    l -= 3;
    k += 3;
  }

  switch (l) {
  case 3: c += k[2];
  case 2: c += k[1];
  case 1: c += k[0];
    final(a,b,c);
  case 0:
    break;
  }
  
  *pb = b; *pc = c;
}

static inline void do_key_hash(cache_key_t *k) {
  uint32_t b = k->ind, c = k->ond;
  hashword2((uint32_t *)(((char *)k) + offsetof(cache_key_t, ioff)),
	    (offsetof(cache_key_t, idims) - offsetof(cache_key_t, ioff))/4, &b, &c);
  hashword2((uint32_t *)k->idims, k->ind*(sizeof(size_t)/4), &b, &c);
  hashword2((uint32_t *)k->odims, k->ond*(sizeof(size_t)/4), &b, &c);
  hashword2((uint32_t *)k->istr, k->ind*(sizeof(size_t)/4), &b, &c);
  hashword2((uint32_t *)k->ostr, k->ond*(sizeof(size_t)/4), &b, &c);
  if (sizeof(size_t) == 4)
    k->hash = c;
  else
    k->hash = ((size_t)b) << 32 | c;
}

