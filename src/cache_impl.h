/*
 * Need to declare cache_key_t, cache_val_t,
 * size_t key_hash(cache_key_t *), int key_eq(cache_key_t *, cache_key_t *),
 * void key_free(cache_key_t), void val_free(cache_val_t)
 * DECL(proto, body)
 * include assert.h stdlib.h
 *
 */

typedef struct node node;

struct node {
  node *prev;
  node *next;
  node *h_next;
  cache_key_t key;
  cache_val_t val;
};

static inline void node_init(node *n, const cache_key_t *k,
			     const cache_val_t *v) {
  n->prev = NULL;
  n->next = NULL;
  n->h_next = NULL;
  n->key = *k;
  n->val = *v;
}

static inline node *node_alloc(const cache_key_t *key,
			       const cache_val_t *val) {
  node *res = malloc(sizeof(node));
  if (res != NULL)
    node_init(res, key, val);
  return res;
}

static inline void node_free(node *n) {
  key_free(&n->key);
  val_free(&n->val);
  if (n->h_next != NULL)
    node_free(n->h_next);
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

typedef struct _list {
  node *head;
  node *tail;
  size_t size;
} list;

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

typedef struct hash {
  node **keyval;
  size_t nbuckets;
  size_t size;
} hash;

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

static inline void hash_init(hash *h, size_t size) {
  h->nbuckets = roundup2(size + (size/6));
  h->keyval = calloc(h->nbuckets, sizeof(*h->keyval));
  h->size = 0;
}

static inline void hash_clear(hash *h) {
  size_t i;
  for (i = 0; i < h->nbuckets; i++) {
    if (h->keyval[i] != NULL)
      node_free(h->keyval[i]);
  }
  free(h->keyval);
  h->nbuckets = 0;
  h->size = 0;
  h->keyval = NULL;
}

static inline node *hash_find(hash *h, const cache_key_t *key) {
  size_t p = key_hash(key) & (h->nbuckets - 1);
  node *n;
  if (h->keyval[p] != NULL) {
    n = h->keyval[p];
    do {
      if (key_eq(&n->key, key))
	return n;
      n = n->h_next;
    } while (n != NULL);
  }
  return NULL;
}

static inline node *hash_add(hash *h, const cache_key_t *key,
			     const cache_val_t *val) {
  size_t p = key_hash(key) & (h->nbuckets - 1);
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

static inline void hash_del(hash *h, node *n) {
  size_t p = key_hash(&n->key) & (h->nbuckets - 1);
  node *np;
  if (n == h->keyval[p]) {
    h->keyval[p] = n->h_next;
    n->h_next = NULL;
    node_free(n);
    h->size--;
  } else {
    np = h->keyval[p];
    while (np->h_next != NULL) {
      if (np->h_next == n) {
	np->h_next = n->h_next;
	n->h_next = NULL;
	node_free(n);
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

typedef struct _cache {
  hash cache;
  list keys;
  size_t maxSize;
  size_t elasticity;
} cache;

static inline void cache_init(cache *c, size_t maxSize, size_t elasticity) {
  hash_init(&c->cache, maxSize+elasticity);
  list_init(&c->keys);
  c->maxSize = maxSize;
  c->elasticity = elasticity;
}

static inline cache *cache_alloc(size_t maxSize, size_t elasticity) {
  cache *res = malloc(sizeof(cache));
  if (res != NULL)
    cache_init(res, maxSize, elasticity);
  return res;
}

static inline void cache_clear(cache *c) {
  hash_clear(&c->cache);
  list_clear(&c->keys);
}

static inline void cache_free(cache *c) {
  cache_clear(c);
  free(c);
}

static inline void cache_prune(cache *c);

static inline int cache_insert(cache *c, const cache_key_t *key,
			       const cache_val_t *val) {
  node *n = hash_add(&c->cache, key, val);
  if (n == NULL) {
    return -1;
  }
  list_push(&c->keys, n);
  cache_prune(c);
  return 0;
}

static inline cache_val_t *cache_get(cache *c, const cache_key_t *key) {
  node *n = hash_find(&c->cache, key);
  if (n == NULL) {
    return NULL;
  } else {
    list_remove(&c->keys, n);
    list_push(&c->keys, n);
    return &n->val;
  }
}

static inline void cache_remove(cache *c, const cache_key_t *key) {
  node *n = hash_find(&c->cache, key);
  if (n != NULL) {
    list_remove(&c->keys, n);
    hash_del(&c->cache, n);
  }
}

static inline int cache_contains(cache *c, const cache_key_t *key) {
  return hash_find(&c->cache, key) != NULL;
}

static inline void cache_prune(cache *c) {
  if (c->maxSize > 0 &&
      hash_size(&c->cache) > (c->maxSize + c->elasticity)) {
    while (hash_size(&c->cache) > c->maxSize) {
      node *n = list_pop(&c->keys);
      hash_del(&c->cache, n);
    }
  }
}
