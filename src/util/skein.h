#ifndef _SKEIN_H_
#define _SKEIN_H_     1
/**************************************************************************
**
** Interface declarations and internal definitions for Skein hashing.
**
** Source code author: Doug Whiting, 2008.
**
** This algorithm and source code is released to the public domain.
**
***************************************************************************
**
** The following compile-time switches may be defined to control some
** tradeoffs between speed, code size, error checking, and security.
**
** The "default" note explains what happens when the switch is not defined.
**
**  SKEIN_ERR_CHECK        -- how error checking is handled inside Skein
**                            code. If not defined, most error checking
**                            is disabled (for performance). Otherwise,
**                            the switch value is interpreted as:
**                                0: use assert()      to flag errors
**                                1: return SKEIN_FAIL to flag errors
**
***************************************************************************/
#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>                          /* get size_t definition */
#include <gpuarray/config.h>
typedef unsigned int uint_t;
typedef uint8_t  u08b_t;
typedef uint64_t u64b_t;

enum {
  SKEIN_SUCCESS         =      0,          /* return codes from Skein calls */
  SKEIN_FAIL            =      1
};

#define  SKEIN_MODIFIER_WORDS  ( 2)     /* number of modifier (tweak) words */

#define  SKEIN_512_STATE_WORDS ( 8)

#define  SKEIN_512_STATE_BYTES ( 8*SKEIN_512_STATE_WORDS)
#define  SKEIN_512_STATE_BITS  (64*SKEIN_512_STATE_WORDS)
#define  SKEIN_512_BLOCK_BYTES ( 8*SKEIN_512_STATE_WORDS)

typedef struct {
  size_t  hashBitLen;                        /* size of hash result, in bits */
  size_t  bCnt;                          /* current byte count in buffer b[] */
  u64b_t  T[SKEIN_MODIFIER_WORDS]; /* tweak words: T[0]=byte cnt, T[1]=flags */
} Skein_Ctxt_Hdr_t;

typedef struct {                     /* 512-bit Skein hash context structure */
  Skein_Ctxt_Hdr_t h;                     /* common header context variables */
  u64b_t  X[SKEIN_512_STATE_WORDS];                    /* chaining variables */
  union Skein_512_Ctxt_b_u {
    u08b_t b[SKEIN_512_BLOCK_BYTES]; /* partial block buffer (8-byte aligned) */
    u64b_t l[SKEIN_512_BLOCK_BYTES/8];
  } bb;
} Skein_512_Ctxt_t;

/*   Skein APIs for (incremental) "straight hashing" */
int  Skein_512_Init  (Skein_512_Ctxt_t *ctx);
int  Skein_512_Update(Skein_512_Ctxt_t *ctx, const u08b_t *msg, size_t msgByteCnt);
int  Skein_512_Final (Skein_512_Ctxt_t *ctx, u08b_t * hashVal);
int  Skein_512(const u08b_t *msg, size_t msgByteCnt, u08b_t *hashVal);

/*****************************************************************
** "Internal" Skein definitions
**    -- not needed for sequential hashing API, but will be
**           helpful for other uses of Skein (e.g., tree hash mode).
**    -- included here so that they can be shared between
**           reference and optimized code.
******************************************************************/

/* tweak word T[1]: bit field starting positions */
#define SKEIN_T1_BIT(BIT)       ((BIT) - 64)            /* offset 64 because it's the second word  */

#define SKEIN_T1_POS_BLK_TYPE   SKEIN_T1_BIT(120)       /* bits 120..125: type field               */
#define SKEIN_T1_POS_FIRST      SKEIN_T1_BIT(126)       /* bits 126     : first block flag         */
#define SKEIN_T1_POS_FINAL      SKEIN_T1_BIT(127)       /* bit  127     : final block flag         */

/* tweak word T[1]: flag bit definition(s) */
#define SKEIN_T1_FLAG_FIRST     (((u64b_t)  1 ) << SKEIN_T1_POS_FIRST)
#define SKEIN_T1_FLAG_FINAL     (((u64b_t)  1 ) << SKEIN_T1_POS_FINAL)

/* tweak word T[1]: block type field */
#define SKEIN_BLK_TYPE_MSG      (48)              /* message processing */
#define SKEIN_BLK_TYPE_OUT      (63)                    /* output stage */

#define SKEIN_T1_BLK_TYPE(T)   (((u64b_t) (SKEIN_BLK_TYPE_##T)) << SKEIN_T1_POS_BLK_TYPE)
#define SKEIN_T1_BLK_TYPE_MSG   SKEIN_T1_BLK_TYPE(MSG) /* message processing */
#define SKEIN_T1_BLK_TYPE_OUT   SKEIN_T1_BLK_TYPE(OUT)       /* output stage */

#define SKEIN_T1_BLK_TYPE_OUT_FINAL       (SKEIN_T1_BLK_TYPE_OUT | SKEIN_T1_FLAG_FINAL)

#define SKEIN_MK_64(hi32,lo32)  ((lo32) + (((u64b_t) (hi32)) << 32))
#define SKEIN_KS_PARITY         SKEIN_MK_64(0x1BD11BDA,0xA9FC1A22)

/*
**   Skein macros for setting tweak words, etc.
**/
#define Skein_Set_Tweak(ctxPtr,TWK_NUM,tVal)    {(ctxPtr)->h.T[TWK_NUM] = (tVal);}

#define Skein_Set_T0(ctxPtr,T0) Skein_Set_Tweak(ctxPtr,0,T0)
#define Skein_Set_T1(ctxPtr,T1) Skein_Set_Tweak(ctxPtr,1,T1)

/* set both tweak words at once */
#define Skein_Set_T0_T1(ctxPtr,T0,T1)         \
    {                                           \
    Skein_Set_T0(ctxPtr,(T0));                  \
    Skein_Set_T1(ctxPtr,(T1));                  \
    }

/* set up for starting with a new type: h.T[0]=0; h.T[1] = NEW_TYPE; h.bCnt=0; */
#define Skein_Start_New_Type(ctxPtr,BLK_TYPE)                         \
  { Skein_Set_T0_T1(ctxPtr,0,SKEIN_T1_FLAG_FIRST | SKEIN_T1_BLK_TYPE_##BLK_TYPE); (ctxPtr)->h.bCnt=0; }

/**************************************************
** "Internal" Skein definitions for error checking
***************************************************/

#include <assert.h>
#define Skein_Assert(x,retCode) { if (!(x)) return retCode; } /*  caller  error */
#define Skein_assert(x)         assert(x)                     /* internal error */

/*****************************************************************
** Skein block function constants (shared across Ref and Opt code)
******************************************************************/
enum {
  /* Skein_512 round rotation constants */
  R_512_0_0=46, R_512_0_1=36, R_512_0_2=19, R_512_0_3=37,
  R_512_1_0=33, R_512_1_1=27, R_512_1_2=14, R_512_1_3=42,
  R_512_2_0=17, R_512_2_1=49, R_512_2_2=36, R_512_2_3=39,
  R_512_3_0=44, R_512_3_1= 9, R_512_3_2=54, R_512_3_3=56,
  R_512_4_0=39, R_512_4_1=30, R_512_4_2=34, R_512_4_3=24,
  R_512_5_0=13, R_512_5_1=50, R_512_5_2=10, R_512_5_3=17,
  R_512_6_0=25, R_512_6_1=29, R_512_6_2=39, R_512_6_3=43,
  R_512_7_0= 8, R_512_7_1=35, R_512_7_2=56, R_512_7_3=22,
};

#ifdef __cplusplus
}
#endif

#endif  /* ifndef _SKEIN_H_ */
