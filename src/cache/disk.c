#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/stat.h>

#include "cache.h"
#include "private_config.h"
#include "util/strb.h"
#include "util/skein.h"

#define HEXP_LEN (128 + 2)

typedef int (*kwrite_fn)(strb *res, cache_key_t key);
typedef int (*vwrite_fn)(strb *res, cache_value_t val);
typedef cache_key_t (*kread_fn)(const strb *b);
typedef cache_value_t (*vread_fn)(const strb *b);

typedef struct _disk_cache {
  cache c;
  cache * mem;
  kwrite_fn kwrite;
  vwrite_fn vwrite;
  kread_fn kread;
  vread_fn vread;
  int dirfd;
} disk_cache;


static unsigned long long ntohull(const char *in) {
  return ((unsigned long long)in[0] << 56 | (unsigned long long)in[1] << 48 |
          (unsigned long long)in[2] << 40 | (unsigned long long)in[3] << 32 |
          (unsigned long long)in[4] << 24 | (unsigned long long)in[5] << 16 |
          (unsigned long long)in[6] << 8 | (unsigned long long)in[7]);
}

static void htonull(unsigned long long in, char *out) {
  out[0] = in >> 56;
  out[1] = in >> 48;
  out[2] = in >> 40;
  out[3] = in >> 32;
  out[4] = in >> 24;
  out[5] = in >> 16;
  out[6] = in >> 8;
  out[7] = in;
}

static int mkstempat(int dfd, char *template) {
  static const char letters[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  size_t length;
  char *XXXXXX;
  struct timeval tv;
  unsigned long long  randnum, working;
  int i, tries, fd;

  length = strlen(template);
  if (length < 6) {
    errno = EINVAL;
    return -1;
  }
  XXXXXX = template + length - 6;
  if (strcmp(XXXXXX, "XXXXXX") != 0) {
    errno = EINVAL;
    return -1;
  }

  /* This is kind of crappy, but the point is to not step on each
     other's feet */
  gettimeofday(&tv, NULL);
  randnum = ((unsigned long long) tv.tv_usec << 16) ^ tv.tv_sec ^ getpid();

  for (tries = 0; tries < TMP_MAX; tries++) {
    for (working = randnum, i = 0; i < 6; i++) {
      XXXXXX[i] = letters[working % 62];
      working /= 62;
    }
    fd = openat(dfd, template, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0 || (errno != EEXIST && errno != EISDIR))
      return fd;

    randnum += (tv.tv_usec >> 10) & 0xfff;
  }
  errno = EEXIST;
  return -1;
}

static int key_path(disk_cache *c, const cache_key_t key, char *out) {
  strb kb = STRB_STATIC_INIT;
  unsigned char hash[64];
  int i;

  if (c->kwrite(&kb, key)) return -1;
  if (Skein_512((unsigned char *)kb.s, kb.l, hash)) return -1;
  if (snprintf(out, 6, "%02x%02x/%02x%02x",
               hash[0], hash[1], hash[2], hash[3]) != 5)
    return -1;
  for (i = 4; i < 64; i += 4) {
    if (snprintf(out+(i * 2 + 1), 9, "%02x%02x%02x%02x",
                 hash[i], hash[i+1], hash[i+2], hash[i+3]) != 8)
      return -1;
  }
  return 0;
}

static int write_entry(disk_cache *c, const cache_key_t k,
                       const cache_value_t v) {
  char hexp[HEXP_LEN];
  char tmp_path[] = "tmp.XXXXXXXX";
  strb b = STRB_STATIC_INIT;
  size_t kl, vl;
  int fd, err;

  if (key_path(c, k, hexp)) return -1;

  if (!strb_ensure(&b, 16)) return -1;
  b.l = 16;
  c->kwrite(&b, k);
  kl = b.l - 16;
  c->vwrite(&b, v);
  vl = b.l - kl - 16;
  htonull(kl, b.s);
  htonull(vl, b.s + 8);
  if (strb_error(&b)) {
    strb_clear(&b);
    return -1;
  }

  fd = mkstempat(c->dirfd, tmp_path);
  if (fd == -1) {
    strb_clear(&b);
    return -1;
  }

  err = strb_write(fd, &b);
  strb_clear(&b);
  close(fd);
  if (err) {
    unlinkat(c->dirfd, tmp_path, 0);
    return -1;
  }
  
  if (renameat(c->dirfd, tmp_path, c->dirfd, hexp)) {
    unlinkat(c->dirfd, tmp_path, 0);
    return -1;
  }

  return 0;
}

static int find_entry(disk_cache *c, const cache_key_t key,
                      cache_key_t *_k, cache_value_t *_v) {
  struct stat st;
  strb b = STRB_STATIC_INIT;
  char *ts;
  size_t kl, vl;
  cache_key_t k;
  char hexp[HEXP_LEN];
  int fd;

  if (key_path(c, key, hexp)) return 0;

  fd = openat(c->dirfd, hexp, O_RDONLY);

  if (fd == -1) return 0;

  if (fstat(fd, &st)) {
    close(fd);
    return 0;
  }

  if (!(st.st_mode & S_IFREG)) {
    close(fd);
    return 0;
  }

  strb_read(&b, fd, st.st_size);
  close(fd);

  if (strb_error(&b) || b.l < 16) {
    strb_clear(&b);
    return 0;
  }

  kl = ntohull(b.s);
  vl = ntohull(b.s + 8);

  if (b.l < 16 + kl + vl) {
    strb_clear(&b);
    return 0;
  }

  ts = b.s;

  b.s += 16;
  b.l = kl;

  k = c->kread(&b);
  if (k && c->c.keq(key, k)) {
    if (_v) {
      b.s += kl;
      b.l = vl;
      *_v = c->vread(&b);
      if (*_v == NULL)
        goto error;
    }
    if (_k)
      *_k = k;
    else
      c->c.kfree(k);
    b.s = ts;
    strb_clear(&b);
    return 1;
  }
 error:
  c->c.kfree(k);
  b.s = ts;
  strb_clear(&b);
  return 0;
}

static int disk_add(cache *_c, cache_key_t k, cache_value_t v) {
  disk_cache *c = (disk_cache *)_c;

  /* Ignore write errors */
  write_entry(c, k, v);

  return cache_add(c->mem, k, v);
}

static int disk_del(cache *_c, const cache_key_t key) {
  disk_cache *c = (disk_cache *)_c;
  char hexp[HEXP_LEN] = {0};
  
  cache_del(c->mem, key);

  key_path(c, key, hexp);

  return (unlinkat(c->dirfd, hexp, 0) == 0);
}

static cache_value_t disk_get(cache *_c, const cache_key_t key) {
  disk_cache *c = (disk_cache *)_c;
  cache_key_t k;
  cache_value_t v;

  v = cache_get(c->mem, key);
  if (v != NULL)
    return v;

  if (find_entry(c, key, &k, &v)) {
    if (cache_add(c->mem, k, v)) return NULL;
    return v;
  }
  return NULL;
}

static void disk_destroy(cache *_c) {
  disk_cache *c = (disk_cache *)_c;
  cache_destroy(c->mem);
  close(c->dirfd);
}

cache *cache_disk(const char *dirpath, cache *mem,
                  kwrite_fn kwrite, vwrite_fn vwrite,
                  kread_fn kread, vread_fn vread) {
  struct stat st;
  disk_cache *res;

  mkdir(dirpath, 0777); /* This may fail, but we don't care */
  if (lstat(dirpath, &st) != 0)
    return NULL;
  if (!(st.st_mode & S_IFDIR))
    return NULL;

  res = calloc(sizeof(*res), 1);
  if (res == NULL) return NULL;

  res->dirfd = open(dirpath, O_RDWR|O_CLOEXEC);
  if (res->dirfd == -1) {
    free(res);
    return NULL;
  }

  res->mem = mem;
  res->kwrite = kwrite;
  res->vwrite = vwrite;
  res->kread = kread;
  res->vread = vread;
  res->c.add = disk_add;
  res->c.del = disk_del;
  res->c.get = disk_get;
  res->c.destroy = disk_destroy;
  res->c.keq = mem->keq;
  res->c.khash = mem->khash;
  res->c.kfree = mem->kfree;
  res->c.vfree = mem->vfree;
  return (cache *)res;
}
