#ifndef CACHE_H
#define CACHE_H

#include <stdlib.h>
#include <gpuarray/config.h>
#include "private_config.h"
#include "util/strb.h"
#include "util/error.h"

typedef void *cache_key_t;
typedef void *cache_value_t;

typedef int (*cache_eq_fn)(cache_key_t, cache_key_t);
typedef uint32_t (*cache_hash_fn)(cache_key_t);
typedef void (*cache_freek_fn)(cache_key_t);
typedef void (*cache_freev_fn)(cache_value_t);

typedef int (*kwrite_fn)(strb *res, cache_key_t key);
typedef int (*vwrite_fn)(strb *res, cache_value_t val);
typedef cache_key_t (*kread_fn)(const strb *b);
typedef cache_value_t (*vread_fn)(const strb *b);

typedef struct _cache cache;

struct _cache {
  /**
   * Add the specified value to the cache under the key k, replacing
   * any previous value.
   *
   * The value and key belong to the cache and will be freed with the
   * supplied free functions whether the add is successful or not.
   *
   * The key and value data must stay valid until they are explicitely
   * released by the cache when it calls the supplied free functions.
   *
   * NULL is not a valid value or key.
   *
   * Returns 0 if value was added sucessfully and some other value otherwise.
   */
  int (*add)(cache *c, cache_key_t k, cache_value_t v);

  /**
   * Remove the data associated with k from the cache.  The value and
   * the key will be free with the supplied free functions.
   *
   * The passed in key is not claimed by the cache and need only be
   * valid until the call returns. It will not be freed through the
   * key free function.
   *
   * Returns 1 if the key was in the cache and 0 if not.
   */
  int (*del)(cache *c, const cache_key_t k);

  /**
   * Get the data entry associated with k.
   *
   * The passed in key is not claimed by the cache and need only be
   * valid until the call returns. It will not be freed through the
   * key free function.
   *
   * Returns NULL if the key is not found, a value otherwise.
   */
  cache_value_t (*get)(cache *c, const cache_key_t k);

  /**
   * Releases all entries in the cache as well as all of the support
   * structures.
   *
   * This must NOT free the passed in pointer.
   */
  void (*destroy)(cache *c);
  cache_eq_fn keq;
  cache_hash_fn khash;
  cache_freek_fn kfree;
  cache_freev_fn vfree;
  /* Extra data goes here depending on cache type */
};

cache *cache_lru(size_t max_size, size_t elasticity,
                 cache_eq_fn keq, cache_hash_fn khash,
                 cache_freek_fn kfree, cache_freev_fn vfree,
                 error *e);

cache *cache_twoq(size_t hot_size, size_t warm_size,
                  size_t cold_size, size_t elasticity,
                  cache_eq_fn keq, cache_hash_fn khash,
                  cache_freek_fn kfree, cache_freev_fn vfree,
                  error *e);

cache *cache_disk(const char *dirpath, cache *mem,
                  kwrite_fn kwrite, vwrite_fn vwrite,
                  kread_fn kread, vread_fn vread,
                  error *e);

/* API functions */
static inline int cache_add(cache *c, cache_key_t k, cache_value_t v) {
  return c->add(c, k, v);
}

static inline int cache_del(cache *c, cache_key_t k) {
  return c->del(c, k);
}

static inline cache_value_t cache_get(cache *c, cache_key_t k) {
  return c->get(c, k);
}

static inline void cache_destroy(cache *c) {
  c->destroy(c);
  free(c);
}

#endif
