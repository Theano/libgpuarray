#include <stdlib.h>
#include "cache.h"
#include "private_config.h"

typedef struct _node node;
typedef struct _list list;
typedef struct _hash hash;
typedef struct _lru_cache lru_cache;

struct _node {
  node *prev;
  node *next;
  node *h_next;
  cache_key_t key;
  cache_value_t val;
};

static inline void node_init(node *n, const cache_key_t k,
                             const cache_value_t v) {
  n->prev = NULL;
  n->next = NULL;
  n->h_next = NULL;
  n->key = k;
  n->val = v;
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

static inline int hash_init(hash *h, size_t size, error *e) {
  h->nbuckets = roundup2(size + (size/6));
  h->keyval = calloc(h->nbuckets, sizeof(*h->keyval));
  if (h->keyval == NULL) {
    error_sys(e, "calloc");
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

static inline size_t hash_size(hash *h) {
  return h->size;
}

struct _lru_cache {
  cache c;
  hash data;
  list order;
  size_t maxSize;
  size_t elasticity;
};

static inline void lru_prune(lru_cache *c) {
  if (c->maxSize > 0 &&
      hash_size(&c->data) > (c->maxSize + c->elasticity)) {
    while (hash_size(&c->data) > c->maxSize) {
      node *n = list_pop(&c->order);
      hash_del(&c->data, n, c->c.kfree, c->c.vfree, c->c.khash);
    }
  }
}

static int lru_del(cache *_c, const cache_key_t k) {
  lru_cache *c = (lru_cache *)_c;
  node *n = hash_find(&c->data, k, c->c.keq, c->c.khash);
  if (n != NULL) {
    list_remove(&c->order, n);
    hash_del(&c->data, n, c->c.kfree, c->c.vfree, c->c.khash);
    return 1;
  }
  return 0;
}

static int lru_add(cache *_c, cache_key_t key, cache_value_t val) {
  lru_cache *c = (lru_cache *)_c;
  node *n;
  /* XXX: possible optimization here to combine remove and add.
          currently needs to be done this way since hash_add does not
          overwrite previous values */
  lru_del(_c, key);
  n = hash_add(&c->data, key, val, c->c.khash);
  if (n == NULL) {
    return -1;
  }
  list_push(&c->order, n);
  lru_prune(c);
  return 0;
}

static cache_value_t lru_get(cache *_c, const cache_key_t key) {
  lru_cache *c = (lru_cache *)_c;
  node *n = hash_find(&c->data, key, c->c.keq, c->c.khash);
  if (n == NULL) {
    return NULL;
  } else {
    list_remove(&c->order, n);
    list_push(&c->order, n);
    return n->val;
  }
}

static void lru_destroy(cache *_c) {
  lru_cache *c = (lru_cache *)_c;
  hash_clear(&c->data, c->c.kfree, c->c.vfree);
  list_clear(&c->order);
}

cache *cache_lru(size_t max_size, size_t elasticity,
                 cache_eq_fn keq, cache_hash_fn khash,
                 cache_freek_fn kfree, cache_freev_fn vfree,
                 error *e) {
  lru_cache *res = malloc(sizeof(*res));
  if (res == NULL) {
    error_sys(e, "malloc");
    return NULL;
  }

  if (hash_init(&res->data, max_size+elasticity, e)) {
    free(res);
    return NULL;
  }
  list_init(&res->order);
  res->maxSize = max_size;
  res->elasticity = elasticity;

  res->c.add = lru_add;
  res->c.del = lru_del;
  res->c.get = lru_get;
  res->c.destroy = lru_destroy;
  res->c.keq = keq;
  res->c.khash = khash;
  res->c.kfree = kfree;
  res->c.vfree = vfree;
  return (cache *)res;
}
