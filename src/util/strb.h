#ifndef STRB_H
#define STRB_H

#include "private_config.h"
#include "util/halloc.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

/*
 * Main strb structure.
 * `s`: pointer to character data, not guaranteed to be nul-terminated.
 * `l`: current length of valid data in `s`.
 * `a`: current length of allocated data in `s`.  Always >= l.
 */
typedef struct _strb {
  char *s;
  size_t l;
  size_t a;
} strb;

/*
 * Static initializer for a stack or globalc declaration of an strb.
 * Usage:
 *   strb sb = STRB_STATIC_INIT;
 *
 * It is an error to leave an strb uninitialized.
 */
#define STRB_STATIC_INIT {NULL, 0, 0}

/*
 * Return a pointer to a dynamically allocated strb with `s` bytes
 * preallocated in its data member.
 *
 * The returned pointer needs to be freed with strb_free().
 *
 * Returns NULL on error.
 */
GPUARRAY_LOCAL strb *strb_alloc(size_t s);
/*
 * Frees an strb that was dynamically allocated.
 *
 * Don't call this for stack of global declarations, see strb_clear() instead.
 */
GPUARRAY_LOCAL void strb_free(strb *);

/*
 * Return a pointer to a dynamically allocated strb with a default
 * initial size.  See strb_alloc() for defails.
 */
#define strb_new() strb_alloc(1024)

/*
 * Resets the length to 0.  Also clears error mode.
 */
static inline void strb_reset(strb *sb) {
  sb->l = 0;
}

/*
 * Place the strb in error mode where further attempts to append
 * data will silently fail.
 */
static inline int strb_seterror(strb *sb) {
  sb->l = (size_t)-1;
  return -1;
}

/*
 * Returns true if the strb is in error mode.
 */
static inline int strb_error(strb *sb) {
  return sb->l == (size_t)-1;
}


/*
 * Clear any allocation the strb may have done and reset all of its
 * members to the initial state.  The strb can be used as new after
 * this call.
 */
static inline void strb_clear(strb *sb) {
  h_free(sb->s);
  sb->s = NULL;
  sb->a = 0;
  sb->l = 0;
}

/*
 * Grow the allocation of the strb by at least `s`.
 *
 * This should almost never be called directly.  Use strb_ensure()
 * instead.
 */
GPUARRAY_LOCAL int strb_grow(strb *, size_t s);

/*
 * Make sure there is space to store at least `s` bytes of data after
 * the current data.
 *
 * Since the auto-allocation algorithm is tuned to small-ish strings
 * (below 4kb), it may be better from a performance point of view to
 * preallocate space yourself, using strb_ensure() with a large
 * number.
 */
static inline int strb_ensure(strb *sb, size_t s) {
  if (strb_error(sb)) return -1;
  if (sb->a - sb->l < s) return strb_grow(sb, s);
  return 0;
}

/*
 * Append a character to the data.
 */
static inline void strb_appendc(strb *sb, char c) {
  if (strb_ensure(sb, 1)) return;
  sb->s[sb->l++] = c;
}

/*
 * Append a NUL ('\0') to the data.
 */
#define strb_append0(s) strb_appendc(s, '\0')

/*
 * Appends `n` bytes from buffer `s`.
 */
static inline void strb_appendn(strb *sb, const char *s, size_t n) {
  if (strb_ensure(sb, n)) return;
  memcpy(sb->s+sb->l, s, n);
  sb->l += n;
}

/*
 * Appends the content of the nul-terminated string `s`, excluding the
 * final nul.
 */
static inline void strb_appends(strb *sb, const char *s) {
  strb_appendn(sb, s, strlen(s));
}

/*
 * Appends the content of another strb.
 */
static inline void strb_appendb(strb *sb, strb *sb2) {
  strb_appendn(sb, sb2->s, sb2->l);
}

/*
 * Appends the result of a sprintf using the format string `f` and
 * following arguments, excluding terminating nul.
 *
 * Unlike sprintf, this function makes sure not to run off the end of
 * memory and behaves like asprintf in that respect.
 *
 * A format error will place the strb in error mode.
 */
GPUARRAY_LOCAL void strb_appendf(strb *, const char *f, ...);

/*
 * Returns a C string from the content of the strb.
 *
 * Returns the `s` member of the strb after ensuring that a
 * terminating nul is appended.  This value must be freed with
 * h_free().
 *
 * If the strb is in error mode, this function will clear it and
 * return NULL.
 *
 * The strb should not be reused after this function is called (nor
 * should it be cleared).
 *
 * This behavior makes it easy for functions that build a string and
 * return the result as a C string.
 */
static inline char *strb_cstr(strb *sb) {
  strb_append0(sb);
  if (strb_error(sb)) {
    strb_clear(sb);
    return NULL;
  }
  sb->l--;
  return sb->s;
}

#ifdef __cplusplus
}
#endif

#endif
