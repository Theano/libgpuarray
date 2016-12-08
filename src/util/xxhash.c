/*
xxHash - Fast Hash algorithm
Copyright (C) 2012-2015, Yann Collet

BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You can contact the author at :
- xxHash source repository : https://github.com/Cyan4973/xxHash
*/

/* XXH_FORCE_NATIVE_FORMAT :
 * By default, xxHash library provides endian-independant Hash values, based on little-endian convention.
 * Results are therefore identical for little-endian and big-endian CPU.
 * This comes at a performance cost for big-endian CPU, since some swapping is required to emulate little-endian format.
 * Should endian-independance be of no importance for your application, you may set the #define below to 1,
 * to improve speed for Big-endian CPU.
 * This option has no impact on Little_Endian CPU.
 */
#define XXH_FORCE_NATIVE_FORMAT 1

/* XXH_USELESS_ALIGN_BRANCH :
 * This is a minor performance trick, only useful with lots of very small keys.
 * It means : don't make a test between aligned/unaligned, because performance will be the same.
 * It saves one initial branch per hash.
 */
#if defined(__i386) || defined(_M_IX86) || defined(__x86_64__) || defined(_M_X64)
#  define XXH_USELESS_ALIGN_BRANCH 1
#endif


/**************************************
*  Compiler Specific Options
***************************************/
#ifdef _MSC_VER    /* Visual Studio */
#  pragma warning(disable : 4127)      /* disable: C4127: conditional expression is constant */
#  define FORCE_INLINE static __forceinline
#else
#  if defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L   /* C99 */
#    ifdef __GNUC__
#      define FORCE_INLINE static inline __attribute__((always_inline))
#    else
#      define FORCE_INLINE static inline
#    endif
#  else
#    define FORCE_INLINE static
#  endif /* __STDC_VERSION__ */
#endif


/**************************************
*  Includes & Memory related functions
***************************************/
#include "xxhash.h"
/* Modify the local functions below should you wish to use some other memory routines */
/* for memcpy() */
#include <string.h>
static void* XXH_memcpy(void* dest, const void* src, size_t size) { return memcpy(dest,src,size); }


/**************************************
*  Basic Types
***************************************/
#if defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L   /* C99 */
# include <stdint.h>
  typedef uint8_t  BYTE;
  typedef uint16_t U16;
  typedef uint32_t U32;
  typedef  int32_t S32;
  typedef uint64_t U64;
#else
  typedef unsigned char      BYTE;
  typedef unsigned short     U16;
  typedef unsigned int       U32;
  typedef   signed int       S32;
  typedef unsigned long long U64;
#endif


static U32 XXH_read32(const void* memPtr)
{
    U32 val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}


/******************************************
*  Compiler-specific Functions and Macros
******************************************/
#define GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)

/* Note : although _rotl exists for minGW (GCC under windows), performance seems poor */
#if defined(_MSC_VER)
#  define XXH_rotl32(x,r) _rotl(x,r)
#else
#  define XXH_rotl32(x,r) ((x << r) | (x >> (32 - r)))
#endif

#if defined(_MSC_VER)     /* Visual Studio */
#  define XXH_swap32 _byteswap_ulong
#elif GCC_VERSION >= 403
#  define XXH_swap32 __builtin_bswap32
#else
static U32 XXH_swap32 (U32 x)
{
    return  ((x << 24) & 0xff000000 ) |
            ((x <<  8) & 0x00ff0000 ) |
            ((x >>  8) & 0x0000ff00 ) |
            ((x >> 24) & 0x000000ff );
}
#endif


/***************************************
*  Architecture Macros
***************************************/
typedef enum { XXH_bigEndian=0, XXH_littleEndian=1 } XXH_endianess;

/* XXH_CPU_LITTLE_ENDIAN can be defined externally, for example one the compiler command line */
#ifndef XXH_CPU_LITTLE_ENDIAN
    static const int one = 1;
#   define XXH_CPU_LITTLE_ENDIAN   (*(const char*)(&one))
#endif


/*****************************
*  Memory reads
*****************************/
typedef enum { XXH_aligned, XXH_unaligned } XXH_alignment;

FORCE_INLINE U32 XXH_readLE32_align(const void* ptr, XXH_endianess endian, XXH_alignment align)
{
    if (align==XXH_unaligned)
        return endian==XXH_littleEndian ? XXH_read32(ptr) : XXH_swap32(XXH_read32(ptr));
    else
        return endian==XXH_littleEndian ? *(const U32*)ptr : XXH_swap32(*(const U32*)ptr);
}

FORCE_INLINE U32 XXH_readLE32(const void* ptr, XXH_endianess endian)
{
    return XXH_readLE32_align(ptr, endian, XXH_unaligned);
}

/***************************************
*  Macros
***************************************/
#define XXH_STATIC_ASSERT(c)   { enum { XXH_static_assert = 1/(!!(c)) }; }    /* use only *after* variable declarations */


/***************************************
*  Constants
***************************************/
#define PRIME32_1   2654435761U
#define PRIME32_2   2246822519U
#define PRIME32_3   3266489917U
#define PRIME32_4    668265263U
#define PRIME32_5    374761393U

/*****************************
*  Simple Hash Functions
*****************************/
FORCE_INLINE U32 XXH32_endian_align(const void* input, size_t len, U32 seed, XXH_endianess endian, XXH_alignment align)
{
    const BYTE* p = (const BYTE*)input;
    const BYTE* bEnd = p + len;
    U32 h32;
#define XXH_get32bits(p) XXH_readLE32_align(p, endian, align)

    if (len>=16)
    {
        const BYTE* const limit = bEnd - 16;
        U32 v1 = seed + PRIME32_1 + PRIME32_2;
        U32 v2 = seed + PRIME32_2;
        U32 v3 = seed + 0;
        U32 v4 = seed - PRIME32_1;

        do
        {
            v1 += XXH_get32bits(p) * PRIME32_2;
            v1 = XXH_rotl32(v1, 13);
            v1 *= PRIME32_1;
            p+=4;
            v2 += XXH_get32bits(p) * PRIME32_2;
            v2 = XXH_rotl32(v2, 13);
            v2 *= PRIME32_1;
            p+=4;
            v3 += XXH_get32bits(p) * PRIME32_2;
            v3 = XXH_rotl32(v3, 13);
            v3 *= PRIME32_1;
            p+=4;
            v4 += XXH_get32bits(p) * PRIME32_2;
            v4 = XXH_rotl32(v4, 13);
            v4 *= PRIME32_1;
            p+=4;
        }
        while (p<=limit);

        h32 = XXH_rotl32(v1, 1) + XXH_rotl32(v2, 7) + XXH_rotl32(v3, 12) + XXH_rotl32(v4, 18);
    }
    else
    {
        h32  = seed + PRIME32_5;
    }

    h32 += (U32) len;

    while (p+4<=bEnd)
    {
        h32 += XXH_get32bits(p) * PRIME32_3;
        h32  = XXH_rotl32(h32, 17) * PRIME32_4 ;
        p+=4;
    }

    while (p<bEnd)
    {
        h32 += (*p) * PRIME32_5;
        h32 = XXH_rotl32(h32, 11) * PRIME32_1 ;
        p++;
    }

    h32 ^= h32 >> 15;
    h32 *= PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= PRIME32_3;
    h32 ^= h32 >> 16;

    return h32;
}


unsigned int XXH32 (const void* input, size_t len, unsigned int seed)
{
#if 0
    /* Simple version, good for code maintenance, but unfortunately slow for small inputs */
    XXH32_state_t state;
    XXH32_reset(&state, seed);
    XXH32_update(&state, input, len);
    return XXH32_digest(&state);
#else
    XXH_endianess endian_detected = (XXH_endianess)XXH_CPU_LITTLE_ENDIAN;

#  if !defined(XXH_USELESS_ALIGN_BRANCH)
    if ((((size_t)input) & 3) == 0)   /* Input is 4-bytes aligned, leverage the speed benefit */
    {
        if ((endian_detected==XXH_littleEndian) || XXH_FORCE_NATIVE_FORMAT)
            return XXH32_endian_align(input, len, seed, XXH_littleEndian, XXH_aligned);
        else
            return XXH32_endian_align(input, len, seed, XXH_bigEndian, XXH_aligned);
    }
#  endif

    if ((endian_detected==XXH_littleEndian) || XXH_FORCE_NATIVE_FORMAT)
        return XXH32_endian_align(input, len, seed, XXH_littleEndian, XXH_unaligned);
    else
        return XXH32_endian_align(input, len, seed, XXH_bigEndian, XXH_unaligned);
#endif
}

/****************************************************
*  Advanced Hash Functions
****************************************************/

/*** Allocation ***/
typedef struct
{
    U64 total_len;
    U32 seed;
    U32 v1;
    U32 v2;
    U32 v3;
    U32 v4;
    U32 mem32[4];   /* defined as U32 for alignment */
    U32 memsize;
} XXH_istate32_t;

/*** Hash feed ***/

XXH_errorcode XXH32_reset(XXH32_state_t* state_in, unsigned int seed)
{
    XXH_istate32_t* state = (XXH_istate32_t*) state_in;
    state->seed = seed;
    state->v1 = seed + PRIME32_1 + PRIME32_2;
    state->v2 = seed + PRIME32_2;
    state->v3 = seed + 0;
    state->v4 = seed - PRIME32_1;
    state->total_len = 0;
    state->memsize = 0;
    return XXH_OK;
}

FORCE_INLINE XXH_errorcode XXH32_update_endian (XXH32_state_t* state_in, const void* input, size_t len, XXH_endianess endian)
{
    XXH_istate32_t* state = (XXH_istate32_t *) state_in;
    const BYTE* p = (const BYTE*)input;
    const BYTE* const bEnd = p + len;

#ifdef XXH_ACCEPT_NULL_INPUT_POINTER
    if (input==NULL) return XXH_ERROR;
#endif

    state->total_len += len;

    if (state->memsize + len < 16)   /* fill in tmp buffer */
    {
        XXH_memcpy((BYTE*)(state->mem32) + state->memsize, input, len);
        state->memsize += (U32)len;
        return XXH_OK;
    }

    if (state->memsize)   /* some data left from previous update */
    {
        XXH_memcpy((BYTE*)(state->mem32) + state->memsize, input, 16-state->memsize);
        {
            const U32* p32 = state->mem32;
            state->v1 += XXH_readLE32(p32, endian) * PRIME32_2;
            state->v1 = XXH_rotl32(state->v1, 13);
            state->v1 *= PRIME32_1;
            p32++;
            state->v2 += XXH_readLE32(p32, endian) * PRIME32_2;
            state->v2 = XXH_rotl32(state->v2, 13);
            state->v2 *= PRIME32_1;
            p32++;
            state->v3 += XXH_readLE32(p32, endian) * PRIME32_2;
            state->v3 = XXH_rotl32(state->v3, 13);
            state->v3 *= PRIME32_1;
            p32++;
            state->v4 += XXH_readLE32(p32, endian) * PRIME32_2;
            state->v4 = XXH_rotl32(state->v4, 13);
            state->v4 *= PRIME32_1;
            p32++;
        }
        p += 16-state->memsize;
        state->memsize = 0;
    }

    if (p <= bEnd-16)
    {
        const BYTE* const limit = bEnd - 16;
        U32 v1 = state->v1;
        U32 v2 = state->v2;
        U32 v3 = state->v3;
        U32 v4 = state->v4;

        do
        {
            v1 += XXH_readLE32(p, endian) * PRIME32_2;
            v1 = XXH_rotl32(v1, 13);
            v1 *= PRIME32_1;
            p+=4;
            v2 += XXH_readLE32(p, endian) * PRIME32_2;
            v2 = XXH_rotl32(v2, 13);
            v2 *= PRIME32_1;
            p+=4;
            v3 += XXH_readLE32(p, endian) * PRIME32_2;
            v3 = XXH_rotl32(v3, 13);
            v3 *= PRIME32_1;
            p+=4;
            v4 += XXH_readLE32(p, endian) * PRIME32_2;
            v4 = XXH_rotl32(v4, 13);
            v4 *= PRIME32_1;
            p+=4;
        }
        while (p<=limit);

        state->v1 = v1;
        state->v2 = v2;
        state->v3 = v3;
        state->v4 = v4;
    }

    if (p < bEnd)
    {
        XXH_memcpy(state->mem32, p, bEnd-p);
        state->memsize = (int)(bEnd-p);
    }

    return XXH_OK;
}

XXH_errorcode XXH32_update (XXH32_state_t* state_in, const void* input, size_t len)
{
    XXH_endianess endian_detected = (XXH_endianess)XXH_CPU_LITTLE_ENDIAN;

    if ((endian_detected==XXH_littleEndian) || XXH_FORCE_NATIVE_FORMAT)
        return XXH32_update_endian(state_in, input, len, XXH_littleEndian);
    else
        return XXH32_update_endian(state_in, input, len, XXH_bigEndian);
}



FORCE_INLINE U32 XXH32_digest_endian (const XXH32_state_t* state_in, XXH_endianess endian)
{
    const XXH_istate32_t* state = (const XXH_istate32_t*) state_in;
    const BYTE * p = (const BYTE*)state->mem32;
    const BYTE* bEnd = (const BYTE*)(state->mem32) + state->memsize;
    U32 h32;

    if (state->total_len >= 16)
    {
        h32 = XXH_rotl32(state->v1, 1) + XXH_rotl32(state->v2, 7) + XXH_rotl32(state->v3, 12) + XXH_rotl32(state->v4, 18);
    }
    else
    {
        h32  = state->seed + PRIME32_5;
    }

    h32 += (U32) state->total_len;

    while (p+4<=bEnd)
    {
        h32 += XXH_readLE32(p, endian) * PRIME32_3;
        h32  = XXH_rotl32(h32, 17) * PRIME32_4;
        p+=4;
    }

    while (p<bEnd)
    {
        h32 += (*p) * PRIME32_5;
        h32 = XXH_rotl32(h32, 11) * PRIME32_1;
        p++;
    }

    h32 ^= h32 >> 15;
    h32 *= PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= PRIME32_3;
    h32 ^= h32 >> 16;

    return h32;
}


unsigned int XXH32_digest (const XXH32_state_t* state_in)
{
    XXH_endianess endian_detected = (XXH_endianess)XXH_CPU_LITTLE_ENDIAN;

    if ((endian_detected==XXH_littleEndian) || XXH_FORCE_NATIVE_FORMAT)
        return XXH32_digest_endian(state_in, XXH_littleEndian);
    else
        return XXH32_digest_endian(state_in, XXH_bigEndian);
}
