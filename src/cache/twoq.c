#include <assert.h>
#include <stdlib.h>

#include "cache.h"
#include "private_config.h"

typedef struct _node node;
typedef struct _list list;
typedef struct _hash hash;
typedef struct _twoq_cache twoq_cache;

#define HOT 0
#define WARM 1
#define COLD 2

struct _node {
  node *prev;
  node *next;
  node *h_next;
  cache_key_t key;
  cache_value_t val;
  int temp;
};

static inline void node_init(node *n, const cache_key_t k,
                             const cache_value_t v) {
  n->prev = NULL;
  n->next = NULL;
  n->h_next = NULL;
  n->key = k;
  n->val = v;
  n->temp = HOT;
}

static inline node *node_alloc(const cache_key_t key,
                               const cache_value_t val) {
  node *res = malloc(sizeof(node));
  if (res != NULL)
    node_init(res, key, val);
  return res;
}

static inline void node_free(node *n, cache_freek_fn kfree,
                             cache_freev_fn vfree) {
  kfree(n->key);
  vfree(n->val);
  if (n->h_next != NULL)
    node_free(n->h_next, kfree, vfree);
  free(n);
}

static inline void node_unlink(node *n) {
  if (n->next != NULL)
    n->next->prev = n->prev;
  if (n->prev != NULL)
    n->prev->next = n->next;
  n->next = NULL;
  n->prev = NULL;
}

struct _list {
  node *head;
  node *tail;
  size_t size;
};

static inline void list_init(list *l) {
  l->head = NULL;
  l->tail = NULL;
  l->size = 0;
}

static inline void list_clear(list *l) {
  l->head = NULL;
  l->tail = NULL;
  l->size = 0;
}

static inline node *list_pop(list *l) {
  if (l->head == NULL)
    return NULL;
  else {
    node *oldHead = l->head;
    l->head = l->head->next;
    node_unlink(oldHead);
    l->size--;
    if (l->size == 0) {
      l->tail = NULL;
    }
    return oldHead;
  }
}

static inline node *list_remove(list *l, node *n) {
  if (n == l->head)
    l->head = n->next;
  if (n == l->tail)
    l->tail = n->prev;
  node_unlink(n);
  l->size--;
  return n;
}

static inline void list_push(list *l, node *n) {
  node_unlink(n);
  if (l->head == NULL) {
    l->head = n;
  } else if (l->head == l->tail) {
    l->head->next = n;
    n->prev = l->head;
  } else {
    l->tail->next = n;
    n->prev = l->tail;
  }
  l->tail = n;
  l->size++;
}

struct _hash {
  node **keyval;
  size_t nbuckets;
  size_t size;
};

static inline size_t roundup2(size_t s) {
  s--;
  s |= s >> 1;
  s |= s >> 2;
  s |= s >> 4;
  s |= s >> 8;
  s |= s >> 16;
  if (sizeof(size_t) >= 8)
    s |= s >> 32;
  s++;
  return s;
}

static inline int hash_init(hash *h, size_t size) {
  h->nbuckets = roundup2(size + (size/6));
  h->keyval = calloc(h->nbuckets, sizeof(*h->keyval));
  if (h->keyval == NULL) {
    return -1;
  }
  h->size = 0;
  return 0;
}

static inline void hash_clear(hash *h, cache_freek_fn kfree,
                              cache_freev_fn vfree) {
  size_t i;
  for (i = 0; i < h->nbuckets; i++) {
    if (h->keyval[i] != NULL)
      node_free(h->keyval[i], kfree, vfree);
  }
  free(h->keyval);
  h->nbuckets = 0;
  h->size = 0;
  h->keyval = NULL;
}

static inline node *hash_find(hash *h, const cache_key_t key,
                              cache_eq_fn keq, cache_hash_fn khash) {
  size_t p = khash(key) & (h->nbuckets - 1);
  node *n;
  if (h->keyval[p] != NULL) {
    n = h->keyval[p];
    do {
      if (keq(n->key, key))
        return n;
      n = n->h_next;
    } while (n != NULL);
  }
  return NULL;
}

static inline node *hash_add(hash *h, const cache_key_t key,
                             const cache_value_t val,
                             cache_hash_fn khash) {
  size_t p = khash(key) & (h->nbuckets - 1);
  node *n = node_alloc(key, val);
  if (n == NULL) return NULL;
  if (h->keyval[p] == NULL) {
    h->keyval[p] = n;
  } else {
    n->h_next = h->keyval[p];
    h->keyval[p] = n;
  }
  h->size++;
  return n;
}

static inline void hash_del(hash *h, node *n,
                            cache_freek_fn kfree,
                            cache_freev_fn vfree,
                            cache_hash_fn khash) {
  size_t p = khash(n->key) & (h->nbuckets - 1);
  node *np;
  if (n == h->keyval[p]) {
    h->keyval[p] = n->h_next;
    n->h_next = NULL;
    node_free(n, kfree, vfree);
    h->size--;
  } else {
    np = h->keyval[p];
    while (np->h_next != NULL) {
      if (np->h_next == n) {
        np->h_next = n->h_next;
        n->h_next = NULL;
        node_free(n, kfree, vfree);
        h->size--;
        break;
      }
      np = np->h_next;
    }
  }
}

struct _twoq_cache {
  cache c;
  hash data;
  list hot;
  list warm;
  list cold;
  size_t hot_size;
  size_t warm_size;
  size_t cold_size;
  size_t elasticity;
};

static inline void twoq_prune(twoq_cache *c) {
  while (c->hot.size > c->hot_size) {
    node *n = list_pop(&c->hot);
    n->temp = COLD;
    list_push(&c->cold, n);
  }
  if (c->cold.size > c->cold_size + c->elasticity) {
    while (c->cold.size > c->cold_size) {
      node *n = list_pop(&c->cold);
      hash_del(&c->data, n, c->c.kfree, c->c.vfree, c->c.khash);
    }
  }
}

static int twoq_del(cache *_c, const cache_key_t k) {
  twoq_cache *c = (twoq_cache *)_c;
  node *n = hash_find(&c->data, k, c->c.keq, c->c.khash);
  if (n != NULL) {
    switch (n->temp) {
    case HOT:
      list_remove(&c->hot, n);
      break;
    case WARM:
      list_remove(&c->warm, n);
      break;
    case COLD:
      list_remove(&c->cold, n);
      break;
    default:
      assert(0 && "node temperature is not within expected values");
    }
    hash_del(&c->data, n, c->c.kfree, c->c.vfree, c->c.khash);
    return 1;
  }
  return 0;
}

static int twoq_add(cache *_c, cache_key_t key, cache_value_t val) {
  twoq_cache *c = (twoq_cache *)_c;
  node *n;
  /* XXX: possible optimization here to combine remove and add.
          currently needs to be done this way since hash_add does not
          overwrite previous values */
  twoq_del(_c, key);
  n = hash_add(&c->data, key, val, c->c.khash);
  if (n == NULL) {
    return -1;
  }
  list_push(&c->hot, n);
  twoq_prune(c);
  return 0;
}

static cache_value_t twoq_get(cache *_c, const cache_key_t key) {
  twoq_cache *c = (twoq_cache *)_c;
  node *nn;
  node *n = hash_find(&c->data, key, c->c.keq, c->c.khash);
  if (n == NULL) {
    return NULL;
  } else {
    switch (n->temp) {
    case HOT:
      list_remove(&c->hot, n);
      list_push(&c->hot, n);
      break;
    case WARM:
      list_remove(&c->warm, n);
      list_push(&c->warm, n);
      break;
    case COLD:
      list_remove(&c->cold, n);
      n->temp = WARM;
      list_push(&c->warm, n);
      if (c->warm.size > c->warm_size) {
        nn = list_pop(&c->warm);
        nn->temp = COLD;
        list_push(&c->cold, nn);
      }
      break;
    default:
      assert(0 && "node temperature is not within expected values");
    }
    return n->val;
  }
}

static void twoq_destroy(cache *_c) {
  twoq_cache *c = (twoq_cache *)_c;
  hash_clear(&c->data, c->c.kfree, c->c.vfree);
  list_clear(&c->hot);
  list_clear(&c->warm);
  list_clear(&c->cold);
}

cache *cache_twoq(size_t hot_size, size_t warm_size, size_t cold_size,
                 size_t elasticity, cache_eq_fn keq, cache_hash_fn khash,
                 cache_freek_fn kfree, cache_freev_fn vfree) {
  twoq_cache *res;
  if (hot_size == 0 || warm_size == 0 || cold_size == 0)
    return NULL;

  res = malloc(sizeof(*res));
  if (res == NULL) return NULL;

  if (hash_init(&res->data, hot_size+warm_size+cold_size+elasticity)) {
    free(res);
    return NULL;
  }
  list_init(&res->hot);
  list_init(&res->warm);
  list_init(&res->cold);
  res->hot_size = hot_size;
  res->warm_size = warm_size;
  res->cold_size = cold_size;
  res->elasticity = elasticity;

  res->c.add = twoq_add;
  res->c.del = twoq_del;
  res->c.get = twoq_get;
  res->c.destroy = twoq_destroy;
  res->c.keq = keq;
  res->c.khash = khash;
  res->c.kfree = kfree;
  res->c.vfree = vfree;
  return (cache *)res;
}
