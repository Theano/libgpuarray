#define _CRT_SECURE_NO_WARNINGS
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>

#include "private_config.h"

#ifdef _WIN32
#define PATH_MAX 255

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <process.h>
#include <direct.h>
#include <io.h>
#include <sys/types.h>
#include <sys/stat.h>

struct timezone;

struct timeval {
  long tv_sec;
  long tv_usec;
} timeval;

static int gettimeofday(struct timeval *tp, struct timezone *tzp) {
  /*
   * Note: some broken versions only have 8 trailing zero's, the
   * correct epoch has 9 trailing zero's This magic number is the
   * number of 100 nanosecond intervals since January 1, 1601 (UTC)
   * until 00:00:00 January 1, 1970
   */
  static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

  SYSTEMTIME system_time;
  FILETIME file_time;
  uint64_t time;

  GetSystemTime(&system_time);
  SystemTimeToFileTime(&system_time, &file_time);
  time = ((uint64_t)file_time.dwLowDateTime);
  time += ((uint64_t)file_time.dwHighDateTime) << 32;

  tp->tv_sec = (long)((time - EPOCH) / 10000000L);
  tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
  return 0;
}

#define open _open
#define unlink _unlink
#define mkdir(p, f) _mkdir(p)
#define close _close
#define strdup _strdup
#define lstat _stat64
#define fstat _fstat64
#define stat __stat64

#else
#define PATH_MAX 1024
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>

#define O_BINARY 0
#define _setmode(a, b)

#endif


#include "cache.h"
#include "util/skein.h"

#define HEXP_LEN (128 + 2)

typedef struct _disk_cache {
  cache c;
  cache * mem;
  kwrite_fn kwrite;
  vwrite_fn vwrite;
  kread_fn kread;
  vread_fn vread;
  const char *dirp;
} disk_cache;


/* Convert unsigned long long from network to host order */
static unsigned long long ntohull(const char *_in) {
  const unsigned char *in = (const unsigned char *)_in;
  return ((unsigned long long)in[0] << 56 | (unsigned long long)in[1] << 48 |
          (unsigned long long)in[2] << 40 | (unsigned long long)in[3] << 32 |
          (unsigned long long)in[4] << 24 | (unsigned long long)in[5] << 16 |
          (unsigned long long)in[6] << 8 | (unsigned long long)in[7]);
}

/* Convert unsigned long long from host to network order */
static void htonull(unsigned long long in, char *out) {
  out[0] = (unsigned char)(in >> 56);
  out[1] = (unsigned char)(in >> 48);
  out[2] = (unsigned char)(in >> 40);
  out[3] = (unsigned char)(in >> 32);
  out[4] = (unsigned char)(in >> 24);
  out[5] = (unsigned char)(in >> 16);
  out[6] = (unsigned char)(in >> 8);
  out[7] = (unsigned char)(in);
}

/* Concatenate prefix and suffix into a single path string while
   checking for overflow */
static int catp(char *path, const char *dirp, const char *rpath) {
  if (strlcpy(path, dirp, PATH_MAX) >= PATH_MAX) {
    errno = ENAMETOOLONG;
    return -1;
  }
  if (strlcat(path, rpath, PATH_MAX) >= PATH_MAX) {
    errno = ENAMETOOLONG;
    return -1;
  }
  return 0;
}

/* open() for a path specifed by the concatenation of dirp and rpath */
static int openp(const char *dirp, const char *rpath, int flags, int mode) {
  char path[PATH_MAX];

  if (catp(path, dirp, rpath))
    return -1;

  return open(path, flags, mode);
}

static int mkstempp(const char *dirp, char *template) {
  char path[PATH_MAX];
  int res;

  if (catp(path, dirp, template))
    return -1;

  res = mkstemp(path);

  /* We need to copy the result path back and set binary mode (for windows) */
  if (res != -1) {
    _setmode(res, O_BINARY);
    memcpy(template, &path[strlen(dirp)], strlen(template));
  }

  return res;
}

static int unlinkp(const char *dirp, const char *rpath) {
  char path[PATH_MAX];

  if (catp(path, dirp, rpath))
    return -1;

  return unlink(path);
}

static int renamep(const char *dirp, const char *ropath, const char *rnpath) {
  char opath[PATH_MAX];
  char npath[PATH_MAX];

  if (catp(opath, dirp, ropath))
    return -1;
  if (catp(npath, dirp, rnpath))
    return -1;

  return rename(opath, npath);
}

/* Ensure that a path exists by creating all intermediate directories */
int ensurep(const char *dirp, const char *rpath) {
  char path[PATH_MAX];
  char *pp;
  char sep;

  if (dirp == NULL) {
    if (strlcpy(path, rpath, PATH_MAX) >= PATH_MAX) {
      errno = ENAMETOOLONG;
      return -1;
    }
#ifdef _WIN32
    /* Skip root dir (windows) */
    pp = strchr(path, '\\');
    if (pp)
      while (*pp == '\\') pp++;
    else
      pp = path;
#else
    pp = path;
    /* Skip root dir (unix) */
    while (*pp == '/') pp++;
#endif
  } else {
    if (catp(path, dirp, rpath))
      return -1;

    pp = path + strlen(dirp);
  }
  while ((pp = strpbrk(pp + 1, "\\/")) != NULL) {
    sep = *pp;
    *pp = '\0';
    if (mkdir(path, 0777)) {
      if (errno != EEXIST) return -1;
      /* For now we suppose that EEXIST means that the directory is
       * already there. */
    }
    *pp = sep;
  }

  return 0;
}

static int key_path(disk_cache *c, const cache_key_t key, char *out) {
  strb kb = STRB_STATIC_INIT;
  unsigned char hash[64];
  int i;

  if (c->kwrite(&kb, key)) return -1;
  if (Skein_512((unsigned char *)kb.s, kb.l, hash)) return -1;
  if (snprintf(out, 10, "%02x%02x/%02x%02x",
               hash[0], hash[1], hash[2], hash[3]) != 9)
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

  if (ensurep(c->dirp, hexp)) return -1;

  if (strb_ensure(&b, 16)) return -1;
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

  fd = mkstempp(c->dirp, tmp_path);
  if (fd == -1) {
    strb_clear(&b);
    return -1;
  }

  err = strb_write(fd, &b);
  strb_clear(&b);
  close(fd);
  if (err) {
    unlinkp(c->dirp, tmp_path);
    return -1;
  }

  if (renamep(c->dirp, tmp_path, hexp)) {
    unlinkp(c->dirp, tmp_path);
#ifdef _WIN32
    /* On windows we can't rename over an existing file */
    return (errno != EACCES) ? -1 : 0;
#else
    return -1;
#endif
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

  fd = openp(c->dirp, hexp, O_RDONLY|O_BINARY, 0);

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
        goto error_find_entry;
    }
    if (_k)
      *_k = k;
    else
      c->c.kfree(k);
    b.s = ts;
    strb_clear(&b);
    return 1;
  }
 error_find_entry:
  if (k)
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

  return (unlinkp(c->dirp, hexp) == 0);
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
  free((void *)c->dirp);
}

cache *cache_disk(const char *dirpath, cache *mem,
                  kwrite_fn kwrite, vwrite_fn vwrite,
                  kread_fn kread, vread_fn vread, error *e) {
  struct stat st;
  disk_cache *res;
  char *dirp;
  size_t dirl = strlen(dirpath);
  char sep = '/';

  /* This trickery is to make sure the path ends with a separator */
#ifdef _WIN32
  if (dirpath[dirl - 1] == '\\')
    sep = '\\';
#endif

  if (dirpath[dirl - 1] != sep) dirl++;

  dirp = malloc(dirl + 1);  /* With the NUL */

  if (dirp == NULL) {
    error_sys(e, "malloc");
    return NULL;
  }

  strlcpy(dirp, dirpath, dirl + 1);

  if (dirp[dirl - 1] != sep) {
    dirp[dirl - 1] = sep;
    dirp[dirl] = '\0';
  }

  if (ensurep(NULL, dirp) != 0) {
    free(dirp);
    error_sys(e, "ensurep");
    return NULL;
  }

  /* For Windows mkdir and lstat which can't handle trailing separator */
  dirp[dirl -  1] = '\0';

  mkdir(dirp, 0777); /* This may fail, but it's ok */

  if (lstat(dirp, &st) != 0) {
    error_sys(e, "lstat");
    return NULL;
  }

  /* Restore the good path at the end */
  dirp[dirl - 1] = sep;

  if (!(st.st_mode & S_IFDIR)) {
    error_set(e, GA_SYS_ERROR, "Cache path exists but is not a directory");
    return NULL;
  }

  res = calloc(sizeof(*res), 1);
  if (res == NULL) {
    error_sys(e, "calloc");
    return NULL;
  }

  res->dirp = dirp;
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
