#ifndef GPUARRAY_UTIL
#define GPUARRAY_UTIL
/** \file util.h
 *  \brief Utility functions.
 */

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

#include <gpuarray/config.h>
#include <gpuarray/elemwise.h>
#include <gpuarray/types.h>

/**
 * Registers a type with the kernel machinery.
 *
 * \param t is a preallocated and filled gpuarray_type structure. The
 *   memory can be allocated from static memory as it will never be
 *   freed.
 * \param ret is a pointer where the error code (if any) will be
 *   stored.  It can be NULL in which case no error code will be
 *   returned.  If there is no error then the memory pointed to by
 *   `ret` will be untouched.
 *
 * \returns The type code that corresponds to the registered type.
 * This code is only valid for the duration of the application and
 * cannot be reused between invocation.
 *
 * On error this function will return -1.
 */
GPUARRAY_PUBLIC int gpuarray_register_type(gpuarray_type *t, int *ret);

/**
 * Get the type structure for a type.
 *
 * The resulting structure MUST NOT be modified.
 *
 * \param typecode the typecode to get structure for
 *
 * \returns A type structure pointer or NULL
 */
GPUARRAY_PUBLIC const gpuarray_type *gpuarray_get_type(int typecode);

/**
 * Get the size of one element of a type.
 *
 * If the type does not exists this function returns (size_t)-1.
 *
 * \param typecode the type to get the element size for
 *
 * \returns the size
 */
GPUARRAY_PUBLIC size_t gpuarray_get_elsize(int typecode);

/**
 * Return the type use flags for the specified typecodes.
 *
 * The flags for each type passed in are OR-ed together.
 *
 * To check for a single typecode, you have to pass the final -1 also.
 *
 * Passing a -1 as the sole argument is allowed and returns 0, however
 * useful that is.
 *
 * \param init a typecode
 * \param ... list of typecodes terminated by -1
 *
 * \returns flags for all passed-in types.
 */
GPUARRAY_PUBLIC int gpuarray_type_flags(int init, ...);

GPUARRAY_PUBLIC int gpuarray_type_flagsa(unsigned int n, gpuelemwise_arg *arg);

/**
 * Perform dimension collapsing on the specified arguments.
 *
 * This function will check for dimension that are next to each other
 * and contiguous for all inputs and merge them together. This allows
 * to reduce the complexity of the indexing code in kernels and
 * therefore enables faster runtime for kernels.
 *
 * On return the nd, dims and strs will be overwritten with the
 * collapsed versions.
 *
 * For scalar arguments, strs[k] can be NULL.
 *
 * \param n The number of arguments
 * \param nd The number of dimensions of all arguments
 * \param dim The compute shape
 * \param strs The strides for all arguments
 *
 */
GPUARRAY_PUBLIC void gpuarray_elemwise_collapse(unsigned int n,
                                                unsigned int *nd,
                                                size_t *dim, ssize_t **strs);


typedef struct _ga_half_t { uint16_t h; } ga_half_t;

/* code strongly inspired from
   https://github.com/numpy/numpy/blob/master/numpy/core/src/npymath/halffloat.c#L246 */

static inline ga_half_t ga_float2half(float f) {
  union {
    float f;
    uint32_t bits;
  } bf;
  union {
    ga_half_t h;
    uint16_t bits;
  } bh;

  uint32_t f_exp, f_sig;
  uint16_t h_sgn, h_exp, h_sig;

  bf.f = f;

  h_sgn = (bf.bits&0x80000000u) >> 16;
  f_exp = (bf.bits&0x7f800000u);

  /* Exponent overflow/NaN converts to signed inf/NaN */
  if (f_exp >= 0x47800000u) {
    if (f_exp == 0x7f800000u) {
      /* Inf or NaN */
      f_sig = (bf.bits&0x007fffffu);
      if (f_sig != 0) {
	/* NaN - propagate the flag in the significand... */
	bh.bits = (uint16_t) (0x7c00u + (f_sig >> 13));
	/* ...but make sure it stays a NaN */
	if (bh.bits == 0x7c00u) {
	  bh.bits++;
	}
	bh.bits += h_sgn;
	return bh.h;
      } else {
	/* signed inf */
	bh.bits = h_sgn + 0x7c00u;
	return bh.h;
      }
    } else {
      bh.bits = h_sgn + 0x7c00u;
      return bh.h;
    }
  }

  if (f_exp <= 0x38000000u) {
    /*
     * Signed zeros, subnormal floats, and floats with small
     * exponents all convert to signed zero halfs.
     */
    if (f_exp < 0x33000000u) {
      bh.bits = h_sgn;
      return bh.h;
    }
    /* Make the subnormal significand */
    f_exp >>= 23;
    f_sig = (0x00800000u + (bf.bits&0x007fffffu));
    f_sig >>= (113 - f_exp);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig += 0x00001000u;
    h_sig = (uint16_t) (f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp from zero to one and h_sig will be zero.
     * This is the correct result.
     */
    bh.bits = h_sgn + h_sig;
    return bh.h;
  }

  /* Regular case with no overflow or underflow */
  h_exp = (uint16_t) ((f_exp - 0x38000000u) >> 13);
  /* Handle rounding by adding 1 to the bit beyond half precision */
  f_sig = (bf.bits&0x007fffffu);
  f_sig += 0x00001000u;
  h_sig = (uint16_t) (f_sig >> 13);
  bh.bits = h_sgn + h_exp + h_sig;
  return bh.h;
}

#ifdef __cplusplus
}
#endif

#endif /* GPUARRAY_UTIL */
