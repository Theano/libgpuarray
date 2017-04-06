/***********************************************************************
**
** Implementation of the Skein hash function.
**
** Source code author: Doug Whiting, 2008.
**
** This algorithm and source code is released to the public domain.
**
************************************************************************/

#include <string.h>      /* get the memcpy/memset functions */
#include "skein.h"       /* get the Skein API definitions   */

#define MK_64 SKEIN_MK_64

/* blkSize =  512 bits. hashSize =  512 bits */
static const u64b_t SKEIN_512_IV_512[] =
  {
    MK_64(0x4903ADFF,0x749C51CE),
    MK_64(0x0D95DE39,0x9746DF03),
    MK_64(0x8FD19341,0x27C79BCE),
    MK_64(0x9A255629,0xFF352CB1),
    MK_64(0x5DB62599,0xDF6CA7B0),
    MK_64(0xEABE394C,0xA9D5C3F4),
    MK_64(0x991112C7,0x1A75B523),
    MK_64(0xAE18A40B,0x660FCC33)
  };

static void Skein_Put64_LSB_First(u08b_t *dst,const u64b_t *src,size_t bCnt) {
  size_t n;

  for (n = 0; n < bCnt; n++)
    dst[n] = (u08b_t)(src[n>>3] >> (8*(n&7)));
}

static void Skein_Get64_LSB_First(u64b_t *dst, const u08b_t *src,
                                  size_t wCnt) {
  size_t n;

  for (n=0; n<8*wCnt; n+=8)
    dst[n/8] = (((u64b_t) src[n  ])) +
      (((u64b_t) src[n+1]) <<  8) +
      (((u64b_t) src[n+2]) << 16) +
      (((u64b_t) src[n+3]) << 24) +
      (((u64b_t) src[n+4]) << 32) +
      (((u64b_t) src[n+5]) << 40) +
      (((u64b_t) src[n+6]) << 48) +
      (((u64b_t) src[n+7]) << 56) ;
}

static u64b_t Skein_Swap64(u64b_t in) {
  u64b_t o;
  u08b_t *out = (u08b_t *)&o;
  out[7] = in >> 56;
  out[6] = in >> 48;
  out[5] = in >> 40;
  out[4] = in >> 32;
  out[3] = in >> 24;
  out[2] = in >> 16;
  out[1] = in >> 8;
  out[0] = in;
  return o;
}

/*****************************************************************/
/* Function to process blkCnt (nonzero) full block(s) of data. */
#define BLK_BITS        (WCNT*64)               /* some useful definitions for \
                                                   code here */
#define KW_TWK_BASE     (0)
#define KW_KEY_BASE     (3)
#define ks              (kw + KW_KEY_BASE)
#define ts              (kw + KW_TWK_BASE)

#define RotL_64(x,N)    (((x) << (N)) | ((x) >> (64-(N))))

static void Skein_512_Process_Block(Skein_512_Ctxt_t *ctx, const u08b_t *blkPtr,
                             size_t blkCnt, size_t byteCntAdd) {
  enum {
      WCNT = SKEIN_512_STATE_WORDS
  };
#define RCNT  (SKEIN_512_ROUNDS_TOTAL/8)

  u64b_t  kw[WCNT+4];                         /* key schedule words : chaining vars + tweak */
  u64b_t  X0,X1,X2,X3,X4,X5,X6,X7;            /* local copy of vars, for speed */
  u64b_t  w [WCNT];                           /* local copy of input block */

  Skein_assert(blkCnt != 0);                  /* never call with blkCnt == 0! */
  ts[0] = ctx->h.T[0];
  ts[1] = ctx->h.T[1];
  do  {
        /* this implementation only supports 2**64 input bytes (no carry out here) */
    ts[0] += byteCntAdd;                    /* update processed length */

    /* precompute the key schedule for this block */
    ks[0] = ctx->X[0];
    ks[1] = ctx->X[1];
    ks[2] = ctx->X[2];
    ks[3] = ctx->X[3];
    ks[4] = ctx->X[4];
    ks[5] = ctx->X[5];
    ks[6] = ctx->X[6];
    ks[7] = ctx->X[7];
    ks[8] = ks[0] ^ ks[1] ^ ks[2] ^ ks[3] ^
      ks[4] ^ ks[5] ^ ks[6] ^ ks[7] ^ SKEIN_KS_PARITY;

    ts[2] = ts[0] ^ ts[1];

    Skein_Get64_LSB_First(w,blkPtr,WCNT); /* get input block in little-endian format */

    X0   = w[0] + ks[0];                    /* do the first full key injection */
    X1   = w[1] + ks[1];
    X2   = w[2] + ks[2];
    X3   = w[3] + ks[3];
    X4   = w[4] + ks[4];
    X5   = w[5] + ks[5] + ts[0];
    X6   = w[6] + ks[6] + ts[1];
    X7   = w[7] + ks[7];

    blkPtr += SKEIN_512_BLOCK_BYTES;

    /* run the rounds */
#define Round512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)                  \
    X##p0 += X##p1; X##p1 = RotL_64(X##p1,ROT##_0); X##p1 ^= X##p0; \
    X##p2 += X##p3; X##p3 = RotL_64(X##p3,ROT##_1); X##p3 ^= X##p2; \
    X##p4 += X##p5; X##p5 = RotL_64(X##p5,ROT##_2); X##p5 ^= X##p4; \
    X##p6 += X##p7; X##p7 = RotL_64(X##p7,ROT##_3); X##p7 ^= X##p6; \

#define R512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)      /* unrolled */  \
    Round512(p0,p1,p2,p3,p4,p5,p6,p7,ROT,rNum)

#define I512(R)                                                     \
    X0   += ks[((R)+1) % 9];   /* inject the key schedule value */  \
    X1   += ks[((R)+2) % 9];                                        \
    X2   += ks[((R)+3) % 9];                                        \
    X3   += ks[((R)+4) % 9];                                        \
    X4   += ks[((R)+5) % 9];                                        \
    X5   += ks[((R)+6) % 9] + ts[((R)+1) % 3];                      \
    X6   += ks[((R)+7) % 9] + ts[((R)+2) % 3];                      \
    X7   += ks[((R)+8) % 9] +     (R)+1;

    {

#define R512_8_rounds(R)  /* do 8 full rounds */  \
        R512(0,1,2,3,4,5,6,7,R_512_0,8*(R)+ 1);   \
        R512(2,1,4,7,6,5,0,3,R_512_1,8*(R)+ 2);   \
        R512(4,1,6,3,0,5,2,7,R_512_2,8*(R)+ 3);   \
        R512(6,1,0,7,2,5,4,3,R_512_3,8*(R)+ 4);   \
        I512(2*(R));                              \
        R512(0,1,2,3,4,5,6,7,R_512_4,8*(R)+ 5);   \
        R512(2,1,4,7,6,5,0,3,R_512_5,8*(R)+ 6);   \
        R512(4,1,6,3,0,5,2,7,R_512_6,8*(R)+ 7);   \
        R512(6,1,0,7,2,5,4,3,R_512_7,8*(R)+ 8);   \
        I512(2*(R)+1);        /* and key injection */

      R512_8_rounds( 0);

#define R512_Unroll_R(NN) (SKEIN_512_ROUNDS_TOTAL/8 > (NN))

  #if   R512_Unroll_R( 1)
      R512_8_rounds( 1);
  #endif
  #if   R512_Unroll_R( 2)
      R512_8_rounds( 2);
  #endif
  #if   R512_Unroll_R( 3)
      R512_8_rounds( 3);
  #endif
  #if   R512_Unroll_R( 4)
      R512_8_rounds( 4);
  #endif
  #if   R512_Unroll_R( 5)
      R512_8_rounds( 5);
  #endif
  #if   R512_Unroll_R( 6)
      R512_8_rounds( 6);
  #endif
  #if   R512_Unroll_R( 7)
      R512_8_rounds( 7);
  #endif
  #if   R512_Unroll_R( 8)
      R512_8_rounds( 8);
  #endif
  #if   R512_Unroll_R( 9)
      R512_8_rounds( 9);
  #endif
  #if   R512_Unroll_R(10)
      R512_8_rounds(10);
  #endif
  #if   R512_Unroll_R(11)
      R512_8_rounds(11);
  #endif
  #if   R512_Unroll_R(12)
      R512_8_rounds(12);
  #endif
  #if   R512_Unroll_R(13)
      R512_8_rounds(13);
  #endif
  #if   R512_Unroll_R(14)
      R512_8_rounds(14);
  #endif
    }

    /* do the final "feedforward" xor, update context chaining vars */
    ctx->X[0] = X0 ^ w[0];
    ctx->X[1] = X1 ^ w[1];
    ctx->X[2] = X2 ^ w[2];
    ctx->X[3] = X3 ^ w[3];
    ctx->X[4] = X4 ^ w[4];
    ctx->X[5] = X5 ^ w[5];
    ctx->X[6] = X6 ^ w[6];
    ctx->X[7] = X7 ^ w[7];

    ts[1] &= ~SKEIN_T1_FLAG_FIRST;
  }
  while (--blkCnt);
  ctx->h.T[0] = ts[0];
  ctx->h.T[1] = ts[1];
}

/*****************************************************************/
/*     512-bit Skein                                             */
/*****************************************************************/

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* init the context for a straight hashing operation  */
int Skein_512_Init(Skein_512_Ctxt_t *ctx) {
  ctx->h.hashBitLen = 512;         /* output hash bit count */
  memcpy(ctx->X,SKEIN_512_IV_512,sizeof(ctx->X));

  /* Set up to process the data message portion of the hash (default) */
  Skein_Start_New_Type(ctx,MSG);              /* T0=0, T1= MSG type */

  return SKEIN_SUCCESS;
}

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* process the input bytes */
int Skein_512_Update(Skein_512_Ctxt_t *ctx, const u08b_t *msg,
                     size_t msgByteCnt) {
  size_t n;

  Skein_Assert(ctx->h.bCnt <= SKEIN_512_BLOCK_BYTES,SKEIN_FAIL);    /* catch uninitialized context */

  /* process full blocks, if any */
  if (msgByteCnt + ctx->h.bCnt > SKEIN_512_BLOCK_BYTES) {
    if (ctx->h.bCnt) {                              /* finish up any buffered message data */
      n = SKEIN_512_BLOCK_BYTES - ctx->h.bCnt;  /* # bytes free in buffer b[] */
      if (n) {
        Skein_assert(n < msgByteCnt);         /* check on our logic here */
        memcpy(&ctx->bb.b[ctx->h.bCnt],msg,n);
        msgByteCnt  -= n;
        msg         += n;
        ctx->h.bCnt += n;
      }
      Skein_assert(ctx->h.bCnt == SKEIN_512_BLOCK_BYTES);
      Skein_512_Process_Block(ctx,ctx->bb.b,1,SKEIN_512_BLOCK_BYTES);
      ctx->h.bCnt = 0;
    }
    /* now process any remaining full blocks, directly from input message data */
    if (msgByteCnt > SKEIN_512_BLOCK_BYTES) {
      n = (msgByteCnt-1) / SKEIN_512_BLOCK_BYTES;   /* number of full blocks to process */
      Skein_512_Process_Block(ctx,msg,n,SKEIN_512_BLOCK_BYTES);
      msgByteCnt -= n * SKEIN_512_BLOCK_BYTES;
      msg        += n * SKEIN_512_BLOCK_BYTES;
    }
    Skein_assert(ctx->h.bCnt == 0);
  }

  /* copy any remaining source message data bytes into b[] */
  if (msgByteCnt) {
    Skein_assert(msgByteCnt + ctx->h.bCnt <= SKEIN_512_BLOCK_BYTES);
    memcpy(&ctx->bb.b[ctx->h.bCnt],msg,msgByteCnt);
    ctx->h.bCnt += msgByteCnt;
  }

  return SKEIN_SUCCESS;
}

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/* finalize the hash computation and output the result */
int Skein_512_Final(Skein_512_Ctxt_t *ctx, u08b_t *hashVal) {
  size_t i,n,byteCnt;
  u64b_t X[SKEIN_512_STATE_WORDS];
  Skein_Assert(ctx->h.bCnt <= SKEIN_512_BLOCK_BYTES,SKEIN_FAIL);    /* catch uninitialized context */

  ctx->h.T[1] |= SKEIN_T1_FLAG_FINAL;                 /* tag as the final block */
  if (ctx->h.bCnt < SKEIN_512_BLOCK_BYTES)            /* zero pad b[] if necessary */
    memset(&ctx->bb.b[ctx->h.bCnt],0,SKEIN_512_BLOCK_BYTES - ctx->h.bCnt);

  Skein_512_Process_Block(ctx,ctx->bb.b,1,ctx->h.bCnt);  /* process the final block */

  /* now output the result */
  byteCnt = (ctx->h.hashBitLen + 7) >> 3;             /* total number of output bytes */

  /* run Threefish in "counter mode" to generate output */
  memset(ctx->bb.b,0,sizeof(ctx->bb.b));  /* zero out b[], so it can hold the counter */
  memcpy(X,ctx->X,sizeof(X));       /* keep a local copy of counter mode "key" */
  for (i=0;i*SKEIN_512_BLOCK_BYTES < byteCnt;i++) {
    ctx->bb.l[0] = Skein_Swap64((u64b_t) i); /* build the counter block */
    Skein_Start_New_Type(ctx,OUT_FINAL);
    Skein_512_Process_Block(ctx,ctx->bb.b,1,sizeof(u64b_t)); /* run "counter mode" */
    n = byteCnt - i*SKEIN_512_BLOCK_BYTES;   /* number of output bytes left to go */
    if (n >= SKEIN_512_BLOCK_BYTES)
      n  = SKEIN_512_BLOCK_BYTES;
    Skein_Put64_LSB_First(hashVal+i*SKEIN_512_BLOCK_BYTES,ctx->X,n);   /* "output" the ctr mode bytes */
    memcpy(ctx->X,X,sizeof(X));   /* restore the counter mode key for next time */
  }
  return SKEIN_SUCCESS;
}

int Skein_512(const u08b_t *msg, size_t msgByteCnt, u08b_t *hashVal) {
  Skein_512_Ctxt_t ctx;
  if (Skein_512_Init(&ctx)) return SKEIN_FAIL;
  if (Skein_512_Update(&ctx, msg, msgByteCnt)) return SKEIN_FAIL;
  if (Skein_512_Final(&ctx, hashVal)) return SKEIN_FAIL;
  return SKEIN_SUCCESS;
}
