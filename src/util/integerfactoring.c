/* Includes */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "integerfactoring.h"


/* Detect when to avoid VLAs. */
#if defined(_MSC_VER) || defined(__STDC_NO_VLA__)
#define GA_USING_MALLOC_FOR_VLA 1
#endif


/* Defines */
#define GA_IS_COMPOSITE      0
#define GA_IS_PRIME          1
#define GA_IS_PROBABLY_PRIME 2


/**
 * Static Function Prototypes
 */

/**
 * @brief Count trailing zeros of a 64-bit integer.
 *
 * @param [in] n  The integer whose trailing zero count is to be computed.
 * @return     If n != 0, returns trailing zero count; Else returns 64.
 */

static int      gaICtz(uint64_t n);

/**
 * @brief Count leading zeros of a 64-bit integer.
 *
 * @param [in] n  The integer whose leading zero count is to be computed.
 * @return     If n != 0, returns leading zero count; Else returns 64.
 */

static int      gaIClz(uint64_t n);

/**
 * @brief Integer Modular Addition.
 * 
 * Computes
 * 
 *     $$a+b \pmod m$$
 * 
 * efficiently for 64-bit unsigned integers a, b, m.
 */

static uint64_t gaIAddMod    (uint64_t a, uint64_t b, uint64_t m);

/**
 * @brief Integer Modular Subtraction.
 * 
 * Computes
 * 
 *     $$a-b \pmod m$$
 * 
 * efficiently for 64-bit unsigned integers a, b, m.
 */

static uint64_t gaISubMod    (uint64_t a, uint64_t b, uint64_t m);

/**
 * @brief Integer Modular Average.
 * 
 * Computes
 * 
 *     $$\frac{a+b}{2} \pmod m$$
 * 
 * efficiently for 64-bit unsigned integers a, b, m.
 */

static uint64_t gaIAvgMod    (uint64_t a, uint64_t b, uint64_t m);

/**
 * @brief Integer Modular Multiplication.
 *
 * Computes
 *
 *     $$a*b \pmod m$$
 *
 * efficiently for 64-bit unsigned integers a, b, m.
 */

static uint64_t gaIMulMod    (uint64_t a, uint64_t b, uint64_t m);

/**
 * @brief Integer Modular Exponentiation.
 *
 * Computes
 *
 *     $$x^a \pmod m$$
 *
 * efficiently for 64-bit unsigned integers x, a, m.
 */

static uint64_t gaIPowMod    (uint64_t x, uint64_t a, uint64_t m);

/**
 * @brief Jacobi Symbol
 * 
 * Computes the Jacobi symbol, notated
 * 
 *     $$(a/n)$$
 * 
 * efficiently for 64-bit unsigned integers a, n.
 */

static int      gaIJacobiSymbol(uint64_t a, uint64_t n);

/**
 * @brief Strong Fermat base-a probable prime test.
 * 
 * @param [in] n  An odd integer >= 3.
 * @param [in] a  A witness integer > 0.
 * @return Non-zero if n is a strong probable prime to base a and zero if n is
 *         composite.
 */

static int      gaIIsPrimeStrongFermat(uint64_t n, uint64_t a);

/**
 * @brief Strong Lucas probable prime test.
 * 
 * The function uses Selfridge's Method A for selecting D,P,Q.
 * 
 * @param [in] n  An odd integer >= 3.
 * @return Non-zero if n is a strong probable prime and zero if n is composite.
 */

static int      gaIIsPrimeStrongLucas(uint64_t n);

/**
 * @brief Round up positive n to next 2-, 3- or 5-smooth number and report its
 *        factorization.
 */

static int      gaIFactorize2Smooth(uint64_t n, ga_factor_list* fl);
static int      gaIFactorize3Smooth(uint64_t n, ga_factor_list* fl);
static int      gaIFactorize5Smooth(uint64_t n, ga_factor_list* fl);

/**
 * @brief Satisfy individual product limits on "from" by moving factors to
 *        corresponding "to" list.
 */

static void     gaIFLScheduleSatisfyInd(const int       n,
                                        ga_factor_list* from,
                                        ga_factor_list* to,
                                        const uint64_t* maxInd);

/**
 * @brief Satisfy global product limit on "from" by moving factors to
 *        corresponding "to" list.
 */

static void     gaIFLScheduleSatisfyTot(const int       n,
                                        ga_factor_list* from,
                                        ga_factor_list* to,
                                        const uint64_t  maxTot);

/**
 * @brief Optimize "to" by moving factors from "from", under both individual
 *        and global limits.
 */

static void     gaIFLScheduleOpt(const int       n,
                                 ga_factor_list* from,
                                 ga_factor_list* to,
                                 const uint64_t  maxTot,
                                 const uint64_t* maxInd);

/**
 * @brief Schedule block/grid/chunk size, integer version, n checked >= 0.
 */

static void     gaIScheduleChecked(const int       n,
                                   const uint64_t  maxBtot,
                                   const uint64_t* maxBind,
                                   const uint64_t  maxGtot,
                                   const uint64_t* maxGind,
                                   uint64_t*       bs,
                                   uint64_t*       gs,
                                   uint64_t*       cs);



/**
 * Function Definitions
 */

static int      gaICtz       (uint64_t n){
#if __GNUC__ >= 4
	return n ? __builtin_ctzll(n) : 64;
#else
	int z;

	for(z=0;z<64;z++){
		if((n>>z) & 1){break;}
	}

	return z;
#endif
}

static int      gaIClz       (uint64_t n){
#if __GNUC__ >= 4
	return n ? __builtin_clzll(n) : 64;
#else
	int z;

	for(z=63;z>=0;z--){
		if((n>>z) & 1){break;}
	}

	return 63-z;
#endif
}

static uint64_t gaIAddMod    (uint64_t a, uint64_t b, uint64_t m){
	a %= m;
	b %= m;
	
	if(m-a > b){
		return a+b;
	}else{
		return a+b-m;
	}
}

static uint64_t gaISubMod    (uint64_t a, uint64_t b, uint64_t m){
	a %= m;
	b %= m;
	
	if(a >= b){
		return a-b;
	}else{
		return a-b+m;
	}
}

static uint64_t gaIAvgMod    (uint64_t a, uint64_t b, uint64_t m){
	uint64_t s = gaIAddMod(a,b,m);
	
	if(s&1){
		return (s>>1)+(m>>1)+(s&m&1);
	}else{
		return s>>1;
	}
}

static uint64_t gaIMulMod    (uint64_t a, uint64_t b, uint64_t m){
#if (__GNUC__ >= 4) && defined(__x86_64__) && !defined(__STRICT_ANSI__)
	uint64_t r;

	asm(
	    "mul %2\n\t"
	    "div %3\n\t"
	    : "=&d"(r), "+a"(a)   /* Outputs */
	    : "r"(b),  "r"(m)     /* Inputs */
	    : "cc"
	);

	return r;
#elif (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
	/* Hardcore GCC 4.6+ optimization jazz */
	return ((unsigned __int128)a * (unsigned __int128)b) % m;
#else
	const uint64_t TWOPOW32 = (uint64_t)1<<32;
	int i;

	a %= m;
	b %= m;

	if(m <= TWOPOW32){
		/**
		 * Fast path: When performing modulo arithmetic on values <= 2^32,
		 * (a*b) % m gives the correct answer.
		 */

		return (a*b) % m;
	}else{
		/**
		 * Slow path: Have to simulate 128-bit arithmetic long division.
		 */

		uint64_t ah   = a>>32;
		uint64_t al   = (uint32_t)a;
		uint64_t bh   = b>>32;
		uint64_t bl   = (uint32_t)b;

		uint64_t ahbh = ah*bh;
		uint64_t ahbl = ah*bl;
		uint64_t albh = al*bh;
		uint64_t albl = al*bl;

		uint64_t md   = ahbl+albh;

		uint64_t lo   = albl + (md<<32);
		uint64_t hi   = ahbh + (md>>32);

		/* Propagate carry-outs from `md` and `lo` into `hi` */
		if(lo < albl){hi++;}
		if(md < ahbl){hi+=TWOPOW32;}

		/**
		 * Begin 128-bit-by-64-bit remainder.
		 *
		 * 1) Cut down `hi` mod `m`. This implements the first few iterations
		 *    of a shift-and-subtract loop, leaving only 64 iterations to go.
		 * 2) Iterate 64 times:
		 *     2.1) Shift left [hi:lo] by 1 bit, into [newHi:newLo].
		 *     2.2) If:
		 *         2.2.1) newHi < hi, then there was an overflow into bit 128.
		 *                The value [1:newHi:newLo] is definitely larger than
		 *                m, so we subtract. This situation can only occur if
		 *                m > 2^63.
		 *         2.2.2) newHi > m, then we must subtract m out of newHi in
		 *                order to bring back newHi within the range [0, m).
		 * 3) The modulo is in hi.
		 */

		hi %= m;
		for(i=0;i<64;i++){
			uint64_t newLo = (lo<<1);
			uint64_t newHi = (hi<<1) + (newLo<lo);

			if(newHi < hi || newHi > m){newHi -= m;}

			hi = newHi;
			lo = newLo;
		}

		return hi;
	}
#endif
}

static uint64_t gaIPowMod    (uint64_t x, uint64_t a, uint64_t m){
	uint64_t r;

	/**
	 * Special cases (order matters!):
	 * - A modulo of 0 makes no sense and a modulo of 1 implies a return value
	 *   of 0, since the result must be integer.
	 * - An exponent of 0 requires a return value of 1.
	 * - A base of 0 or 1 requires a return value of 0 or 1.
	 * - An exponent of 1 requires a return value of x.
	 * - An exponent of 2 can be handled by the modulo multiplication directly.
	 */

	if(m<=1){
		return 0;
	}

	x %= m;

	if(a==0){
		return 1;
	}else if(x<=1){
		return x;
	}else if(a==1){
		return x;
	}else if(a==2){
		return gaIMulMod(x,x,m);
	}

	/**
	 * Otherwise, perform modular exponentiation by squaring.
	 */

	r = 1;
	while(a){
		if(a&1){
			r = gaIMulMod(r, x, m);
		}

		x = gaIMulMod(x, x, m);
		a >>= 1;
	}

	return r;
}

static int      gaIJacobiSymbol(uint64_t a, uint64_t n){
	int      s=0;
	uint64_t e, a1, n1;
	
	a %= n;
	
	if(a == 1 || n == 1){
		return 1;
	}
	
	if(a == 0){
		return 0;
	}
	
	e  = gaICtz(a);
	a1 = a >> e;
	
	if(e%2 == 0){
		s =  1;
	}else if(n%8 == 1 || n%8 == 7){
		s =  1;
	}else if(n%8 == 3 || n%8 == 5){
		s = -1;
	}
	
	if(n%4 == 3 && a1%4 == 3){
		s = -s;
	}
	
	n1 = n%a1;
	return s*gaIJacobiSymbol(n1,a1);
}

static int      gaIIsPrimeStrongFermat(uint64_t n, uint64_t a){
	/**
	 * The Fermat strong probable prime test the Miller-Rabin test relies upon
	 * uses integer "witnesses" in an attempt at proving the number composite.
	 * Should it fail to prove an integer composite, it reports the number as
	 * "probably prime". However, if the witnesses are chosen carefully, the
	 * Miller-Rabin test can be made deterministic below a chosen threshold.
	 * 
	 * One can use the primes 2 to 37 in order to ensure the correctness of the
	 * identifications for integers under 2^64.
	 * 
	 * Jim Sinclair has found that the seven witnesses
	 *     2, 325, 9375, 28178, 450775, 9780504, 1795265022
	 * also deterministically classify all integers <2^64.
	 * 
	 * 
	 * The Fermat strong probable prime test states that, for integers
	 *             n = d*2^s+1,  d odd, s integer >= 0
	 *             a             integer (chosen witness)
	 * n is a Fermat strong probable prime if
	 *     a^(d    ) =  1 mod n       or
	 *     a^(d*2^r) = -1 mod n       for any integer r, 0 <= r < s.
	 * 
	 * 
	 * The justification for this comes from Fermat's Little Theorem: If n is
	 * prime and a is any integer, then the following always holds:
	 *           a^n =  a mod n
	 * If n is prime and a is coprime to n, then the following always holds:
	 *       a^(n-1) =  1 mod n
	 * 
	 * 
	 * In effect, the logic goes
	 * 
	 *   A:   The number  n  is prime.                               (Statement)
	 *   B:   The number  n  does not divide a.                      (Statement)
	 *   C:   a^(  n-1)       =  1 mod n                             (Statement)
	 *   D:   The commutative ring Z/nZ is a finite field.           (Statement)
	 *   E:   Finite fields are unique factorization domains.        (Statement)
	 *   F:   x^2 = 1 mod n factorizes as (x+1)(x-1) = 0 mod n.      (Statement)
	 *   G:   x^2 mod n only has the trivial square roots 1 and -1   (Statement)
	 *   H:   The number  n  is odd and >= 3.                        (Statement)
	 *   I:   The number n-1 equals d*2^s, with d,s int > 0, d odd.  (Statement)
	 *   J:   a^(    d)       =   1 mod n                            (Statement)
	 *   K:   a^(d*2^r)       =  -1 mod n   for some 0 <= r < s.     (Statement)
	 *   L:   a^(d*2^(r+1))   =   1 mod n   for some 0 <= r < s.     (Statement)
	 *   M:   a^(d*2^r)      != +-1 mod n   AND                      (Statement)
	 *        a^(d*2^(r+1))   =   1 mod n   for some 0 <= r < s.
	 *   
	 *   A&B           -->  C                 (Proposition:     Fermat's Little Theorem)
	 *   !C            -->  !(A&B) = !A|!B    (Contrapositive:  Fermat's Little Theorem)
	 *   A             <->  D                 (Proposition)
	 *   E                                    (Proposition:     By definition)
	 *   F                                    (Proposition:     x^2-x+x-1 = x^2-1 mod n)
	 *   D&E&F         -->  G                 (Proposition:     (x+1)(x-1) is the only
	 *                                                           factorization)
	 *   !G            -->  !D|!E|!F          (Contrapositive:  See above)
	 *   H&I&J         -->  C                 (Proposition:     Squaring  1 gives 1)
	 *   H&I&K         -->  L                 (Proposition:     Squaring -1 gives 1)
	 *   H&I&L         -->  C                 (Proposition:     1, squared or not, gives 1)
	 *   H&I&K         -->  C                 (Hypothetical Syllogism)
	 *   H&I&(J|K)     -->  C                 (Union)
	 *   H&I&!(J|K)    -->  M|!C              (Proposition:     Either squaring
	 *                                                            a^(d*2^(s-1)) != +-1 mod n
	 *                                                          gives a 1, in which case
	 *                                                          M holds, or it does not
	 *                                                          give 1 and therefore
	 *                                                            a^(n-1) != 1 mod n)
	 *                                                          and thus !C holds.
	 *   H&I&!(J|K)    -->  H&I&M | !A | !B   (Absorbtion, Hypothetical Syllogism)
	 *   H&I&M         -->  !G                (Proposition:     x^2 = 1 mod n but x!=+1,
	 *                                                          so x^2 - 1 has roots
	 *                                                          other than +-1)
	 *   H&I&M         -->  !D|!E|!F          (Modus Tollens)
	 *   H&I&M         -->  !D                (Disjunctive Syllogism)
	 *   H&I&M         -->  !A                (Biconditional)
	 *   H&I&!(J|K)    -->  !A | !A | !B      (Hypothethical Syllogism)
	 *   H&I&!(J|K)&B  -->  !A | !A           (Absorbtion)
	 *   H&I&!(J|K)&B  -->  !A | !A           (Disjunctive Syllogism)
	 *   H&I&!(J|K)&B  -->  !A                (Disjunctive Simplification)
	 *                           ***** Conclusions: *****
	 *                            H&I&M         -->  !A
	 *                            H&I&!(J|K)&B  -->  !A
	 * 
	 * Broadly speaking, what the above tells us is:
	 *   - We can't prove n prime (A), but we can prove it composite (!A).
	 *   - Either H&I&M or H&I&!(J|K)&B prove compositeness.
	 *   - If H&I&(J|K) for any r, then we've proven C true. If we prove C true,
	 *     we can't use the contrapositive of Fermat's Little Theorem, so no
	 *     conclusions about the truth-value of A can be made. The test is
	 *     inconclusive. Thus this function returns "probably prime".
	 */
	
	uint64_t d, x;
	int64_t  s, r;
	
	a %= n;
	if(a==0){
		return GA_IS_PROBABLY_PRIME;
	}
	
	s  = gaICtz(n-1);
	d  = (n-1) >> s;
	x  = gaIPowMod(a,d,n);
	
	if(x==1 || x==n-1){
		return GA_IS_PROBABLY_PRIME;
	}
	
	for(r=0;r<s-1;r++){
		x = gaIMulMod(x,x,n);
		if(x==1){
			return GA_IS_COMPOSITE;
		}else if(x == n-1){
			return GA_IS_PROBABLY_PRIME;
		}
	}
	
	return GA_IS_COMPOSITE;
}

static int      gaIIsPrimeStrongLucas(uint64_t n){
	uint64_t Dp, Dm, D, K, U, Ut, V, Vt;
	int      J, r, i;
	
	/**
	 * FIPS 186-4 C.3.3 (General) Lucas Probabilistic Primality Test
	 * 
	 * 1. Test if n is perfect square. If so, return "composite".
	 * 
	 *     NOTE: The only strong base-2 Fermat pseudoprime squares are
	 *           1194649 and 12327121;
	 */
	
	if(n==1194649 || n==12327121){
		return GA_IS_COMPOSITE;
	}
	
	/**
	 * 2. Find first D in sequence 5,-7,9,-11,... s.t. Jacobi symbol (D/n) < 1.
	 *     Iff Jacobi symbol is 0, return "composite".
	 */
	
	Dp = gaIAddMod(0, 5, n);
	Dm = gaISubMod(0, 7, n);
	while(1){
		J = gaIJacobiSymbol(Dp, n);
		if     (J ==  0){return GA_IS_COMPOSITE;}
		else if(J == -1){D = Dp;break;}
		
		J = gaIJacobiSymbol(Dm, n);
		if     (J ==  0){return GA_IS_COMPOSITE;}
		else if(J == -1){D = Dm;break;}
		
		Dp = gaIAddMod(Dp, 4, n);
		Dm = gaISubMod(Dm, 4, n);
	}
	
	/**
	 * 3. K = n+1
	 * 
	 *     NOTE: Cannot overflow, since 2^64-1 is eliminated by strong Fermat
	 *           base-2 test.
	 */
	
	K = n+1;
	
	/**
	 * 4. Let Kr, Kr–1, ..., K0 be the binary expansion of K, with Kr = 1.
	 */
	
	r = 63-gaIClz(K);
	
	/**
	 * 5. Set Ur = 1 and Vr = 1.
	 */
	
	U = V = 1;
	
	/**
	 * 6. For i=r–1 to 0, do
	 */
	
	for(i=r-1;i>=0;i--){
		Ut = gaIMulMod(U,V,n);
		Vt = gaIAvgMod(gaIMulMod(V,V,n), gaIMulMod(D,gaIMulMod(U,U,n),n), n);
		if((K>>i)&1){
			U = gaIAvgMod(Ut,Vt,n);
			V = gaIAvgMod(Vt,gaIMulMod(D,Ut,n),n);
		}else{
			U = Ut;
			V = Vt;
		}
	}
	
	/**
	 * 7. If U0==0, then return "probably prime". Otherwise, return "composite".
	 */
	
	return U==0 ? GA_IS_PROBABLY_PRIME : GA_IS_COMPOSITE;
}

int      gaIIsPrime   (uint64_t n){
	int            hasNoSmallFactors, hasSmallFactors;

	/**
	 * Check if it is 2, the oddest prime.
	 */

	if(n==2){return GA_IS_PRIME;}

	/**
	 * Check if it is an even integer.
	 */

	if((n&1) == 0){return GA_IS_COMPOSITE;}

	/**
	 * For small integers, read directly the answer in a table.
	 */

	if(n<256){
		return "nnyynynynnnynynnnynynnnynnnnnyny"
		       "nnnnnynnnynynnnynnnnnynnnnnynynn"
		       "nnnynnnynynnnnnynnnynnnnnynnnnnn"
		       "nynnnynynnnynynnnynnnnnnnnnnnnny"
		       "nnnynnnnnynynnnnnnnnnynynnnnnynn"
		       "nnnynnnynnnnnynnnnnynynnnnnnnnny"
		       "nynnnynynnnnnnnnnnnynnnnnnnnnnny"
		       "nnnynynnnynnnnnynynnnnnnnnnynnnn"[n] == 'y';
	}

	/**
	 * Test small prime factors.
	 */

	hasNoSmallFactors = n% 3 && n% 5 && n% 7 && n%11 && n%13 && n%17 && n%19 &&
	                    n%23 && n%29 && n%31 && n%37 && n%41 && n%43 && n%47 &&
	                    n%53 && n%59 && n%61 && n%67 && n%71 && n%73 && n%79;
	hasSmallFactors   = !hasNoSmallFactors;
	if(hasSmallFactors){
		return GA_IS_COMPOSITE;
	}

	/**
	 * We implement the Baillie-Pomerance-Selfridge-Wagstaff primality checker.
	 *   1) A Fermat base-2 strong probable prime that is also
	 *   2) A Lucas strong probable prime is
	 *   3) Prime.
	 * The BPSW test has no known failure cases and is proven to have no failures
	 * for all numbers under 2^64. It is expected to have failures (composites
	 * classified as "probably prime") but they are expected to be enormous.
	 *
	 * We begin with the Fermat base-2 strong primality test
	 * (Miller-Rabin test with one witness only, a=2).
	 */

	return gaIIsPrimeStrongFermat(n,          2) &&

	/**
	 * Assuming this is one of the base-2 Fermat strong probable primes, we run
	 * the Lucas primality test with Selfridge's Method A for selecting D.
	 */

	       gaIIsPrimeStrongLucas (n            );
}

int      gaIFactorize (uint64_t n, uint64_t maxN, uint64_t k, ga_factor_list* fl){
	int      infiniteSlack,  finiteSlack,   greaterThanMaxN,
	         exactFactoring, noKSmoothness, kSmoothness;
	uint64_t i, x, newX, p, f, c;


	/**
	 * Insane argument handling.
	 */

	if(!fl || (k == 1) || (maxN > 0 && maxN < n)){
		return 0;
	}


	/**
	 * Handle special cases of n = 0,1,2.
	 */

	if(n<=2){
		gaIFLInit(fl);
		gaIFLAddFactors(fl, n, 1);
		return 1;
	}


	/**
	 * Magic-value arguments interpreted and canonicalized.
	 */

	exactFactoring  = (maxN == (uint64_t) 0);
	infiniteSlack   = (maxN == (uint64_t)-1);
	noKSmoothness   = (k    == 0) || (k >= n);
	finiteSlack     = !infiniteSlack;
	kSmoothness     = !noKSmoothness;
	maxN            = exactFactoring ? n : maxN;
	k               = noKSmoothness  ? n :    k;


	/**
	 * Try optimal k-smooth optimizers.
	 */

	if     (k <= 2){gaIFactorize2Smooth(n, fl);}
	else if(k <= 4){gaIFactorize3Smooth(n, fl);}
	else           {gaIFactorize5Smooth(n, fl);}
	greaterThanMaxN = finiteSlack && (gaIFLIsOverflowed(fl)       ||
	                                  gaIFLGetProduct  (fl) > maxN);
	if(greaterThanMaxN){
		if(kSmoothness && k<=6){
			/**
			 * We've *proven* there exists no k-smooth n <= maxN, k <= 6.
			 * No use wasting more time here.
			 */

			return 0;
		}

		/* Otherwise fall-through to factorizer. */
	}else{
		/**
		 * Either the slack was infinite, or the product did not overflow and
		 * was <= maxN. The k-smoothness criterion is guaranteed by the
		 * factorizer we chose earlier.
		 *
		 * Therefore we have a satisfactory, optimal 2-, 3- or 5-smooth
		 * factorization (although not necessarily an exact one unless it is
		 * the case that maxN == n). We return it.
		 */

		return 1;
	}


	/**
	 * Master loop.
	 * 
	 * We arrive here with finite slack and all optimal 2-, 3- and 5-smooth
	 * factorizers unable to produce a factorization whose product is less
	 * than or equal to maxN.
	 */

	for(i=n; i <= maxN; i++){
		/**
		 * Do not manipulate the loop index!
		 * Initial subfactor to cut down is x=i.
		 */

		x = i;
		gaIFLInit(fl);

		/**
		 * Subfactorization always begins with an attempt at an initial
		 * cut-down by factors of 2. Should this result in a 1 (which isn't
		 * technically prime, but indicates a complete factorization), we
		 * report success.
		 */

		subfactorize:
		gaIFLAddFactors(fl, 2, gaICtz(x));
		x >>= gaICtz(x);
		f = 3;

		/**
		 * Primality test.
		 *
		 * If the remaining factor x is a prime number, it's decision time. One
		 * of two things is true:
		 *
		 *  1) We have a smoothness constraint k and x is <= than it, or we
		 *     don't have a smoothness constraint at all (k==n). Both cases are
		 *     covered by checking x<=k.
		 *
		 *     In this case we add x as the last factor to the factor list and
		 *     return affirmatively.
		 *
		 *  2) We have a smoothness constraint and x>k.
		 *
		 *     In this case we have to inc/decrement x and begin anew the
		 *     sub-factorization. This may cause us to fail out of factorizing
		 *     the current i, by exceeding our slack limit. If this happens we
		 *     abort the factorization rooted at i and move to the next i.
		 */

		primetest:
		if(x==1 || gaIIsPrime(x)){
			if(x <= k){
				gaIFLAddFactors(fl, x, 1);
				return 1;
			}else{
				p     = gaIFLGetProduct(fl);
				newX  = n/p;
				newX += newX*p < n; 
				if(newX < x){
					x = newX;
					goto subfactorize;
				}else if((maxN - p*x) < p){/* Overflow-free check maxN >= p*(x+1) */
					goto nextI;
				}else{
					x++;
					goto subfactorize;
				}
			}
		}

		/**
		 * Composite number handler.
		 *
		 * We continue by trying to cut down x by factors of 3+. Should a trial
		 * division by a factor f succeed, all powers of f are factored out of
		 * x and once f no longer divides x evenly, a new primality test is
		 * run. The primality test will be invoked at most 15 times from this loop.
		 */

		for(;f<=k && f*f<=x && f<=0xFFFFFFFFU;f+=2){/* Overflow-safe f*f */
			if(x%f == 0){
				c = 0;
				do{
					x /= f;
					c++;
				}while(x%f == 0);

				gaIFLAddFactors(fl, f, c);

				goto primetest;
			}
		}

		/* Check before next iteration for 64-bit integer overflow. */
		nextI: if(i == 0xFFFFFFFFFFFFFFFF){break;}
	}

	/* Failed to factorize. */
	return 0;
}

static int      gaIFactorize2Smooth(uint64_t n, ga_factor_list* fl){
	n--;
	n |= n >>  1;
	n |= n >>  2;
	n |= n >>  4;
	n |= n >>  8;
	n |= n >> 16;
	n |= n >> 32;
	n++;

	gaIFLInit(fl);
	gaIFLAddFactors(fl, 2, gaICtz(n));

	return 1;
}

static int      gaIFactorize3Smooth(uint64_t n, ga_factor_list* fl){
	uint64_t nBest=-1, i3Best=0, i3, p3, nCurr;
	int nlz = gaIClz(n), isBest2to64 = 1;

	/**
	 * Iterate over all powers of 3, scaling them by the least power-of-2 such
	 * that the result is greater than or equal to n. Report the smallest nBest
	 * so obtained.
	 */

	for(i3=0, p3=1;i3<=40;i3++, p3*=3){
		nCurr = p3;

		/**
		 * If the current power of 3 is >= n, then this must be the last
		 * iteration, but perhaps a pure power of 3 is the best choice, so
		 * check for this.
		 */

		if(nCurr >= n){
			if(isBest2to64 || nBest >= nCurr){
				isBest2to64 = 0;
				nBest       = nCurr;
				i3Best      = i3;
			}
			break;
		}

		/**
		 * Otherwise we have a pure power of 3, p3, less than n, and must
		 * derive the least power of 2 such that p3 multiplied by that power of
		 * 2 is greater than or equal to n. We then compute the product of
		 * both.
		 */

		nCurr <<= gaIClz(nCurr) - nlz;
		if(nCurr<n){
			/**
			 * The line above only guarantees we get a value within a factor of
			 * 2 from n. We may have to boost nCurr by another factor of 2, if
			 * this is still possible without overflow.
			 */

			nCurr<<=1;
			if(nCurr<n){
				/**
				 * If we enter this branch, overflow occured. Moreover, we know
				 * that (before overflow) it was the case that 2^63 <= nCurr < n,
				 * and thus 2**64 is a superior factorization to this one. Skip.
				 */

				continue;
			}
		}

		/**
		 * By here we know that nCurr is >= n. But is it the best factorization
		 * so far?
		 */

		if(isBest2to64 || nBest >= nCurr){
			isBest2to64 = 0;
			nBest       = nCurr;
			i3Best      = i3;

			if(nCurr == n){
				break;
			}
		}
	}


	/**
	 * Return the smallest n found above.
	 *
	 * nBest and i3Best must be set.
	 */

	gaIFLInit(fl);
	if(isBest2to64){
		gaIFLAddFactors(fl, 2, 64);
	}else{
		gaIFLAddFactors(fl, 2, gaICtz(nBest));
		gaIFLAddFactors(fl, 3, i3Best);
	}
	return 1;
}

static int      gaIFactorize5Smooth(uint64_t n, ga_factor_list* fl){
	uint64_t nBest=-1, i3Best=0, i3, p3, i5Best=0, i5, p5, nCurr;
	int nlz = gaIClz(n), isBest2to64 = 1;

	/**
	 * Iterate over all products of powers of 5 and 3, scaling them by the
	 * least power-of-2 such that the result is greater than or equal to n.
	 * Report the smallest nBest so obtained.
	 */

	for(i5=0, p5=1;i5<=27;i5++, p5*=5){
		nCurr = p5;

		/**
		 * If the current power of 5 is >= n, then this must be the last
		 * iteration, but perhaps a pure power of 5 is the best choice, so
		 * check for this.
		 */

		if(nCurr >= n){
			if(isBest2to64 || nBest >= nCurr){
				isBest2to64 = 0;
				nBest       = nCurr;
				i3Best      = 0;
				i5Best      = i5;
			}
			break;
		}

		for(i3=0, p3=1;i3<=40;i3++, p3*=3){
			/**
			 * Detect when the product p3*p5 would overflow 2^64.
			 */

			if(i3){
				nCurr = (p3/3)*p5;
				if(nCurr+nCurr < nCurr || nCurr+nCurr+nCurr < nCurr+nCurr){
					break;
				}
			}
			nCurr = p3*p5;

			/**
			 * If the current product of powers of 3 and 5 is >= n, then this
			 * must be the last iteration, but perhaps a pure product of powers
			 * of 3 and 5 is the best choice, so check for this.
			 */

			if(nCurr >= n){
				if(isBest2to64 || nBest >= nCurr){
					isBest2to64 = 0;
					nBest       = nCurr;
					i3Best      = i3;
					i5Best      = i5;
				}
				break;
			}

			/**
			 * Otherwise we have a number nCurr, composed purely of factors 3
			 * and 5, that is less than n. We must derive the least power of 2
			 * such that nCurr multiplied by that power of 2 is greater than or
			 * equal to n. We then compute the product of both.
			 */

			nCurr <<= gaIClz(nCurr) - nlz;
			if(nCurr<n){
				/**
				 * The line above only guarantees we get a value within a factor of
				 * 2 from n. We may have to boost nCurr by another factor of 2, if
				 * this is still possible without overflow.
				 */

				nCurr<<=1;
				if(nCurr<n){
					/**
					 * If we enter this branch, overflow occured. Moreover, we know
					 * that (before overflow) it was the case that 2^63 <= nCurr < n,
					 * and thus 2**64 is a superior factorization to this one. Skip.
					 */

					continue;
				}
			}

			/**
			 * By here we know that nCurr is >= n. But is it the best factorization
			 * so far?
			 */

			if(isBest2to64 || nBest >= nCurr){
				isBest2to64 = 0;
				nBest       = nCurr;
				i3Best      = i3;
				i5Best      = i5;

				if(nCurr == n){
					goto exit;
				}
			}
		}
	}


	/**
	 * Return the smallest n found above.
	 *
	 * nBest and i3Best must be set.
	 */

    exit:
	gaIFLInit(fl);
	if(isBest2to64){
		gaIFLAddFactors(fl, 2, 64);
	}else{
		gaIFLAddFactors(fl, 2, gaICtz(nBest));
		gaIFLAddFactors(fl, 3, i3Best);
		gaIFLAddFactors(fl, 5, i5Best);
	}
	return 1;
}

void     gaIFLInit(ga_factor_list* fl){
	memset(fl, 0, sizeof(*fl));
}

int      gaIFLFull(const ga_factor_list* fl){
	return fl->d >= 15;/* Strictly speaking, fl->d never exceeds 15. */
}

int      gaIFLAddFactors(ga_factor_list* fl, uint64_t f, int p){
	int i;

	/**
	 * Fast case: We're adding 0 powers of f, or any powers of 1. The
	 * value of the factor list (and the integer it represents) is thus
	 * unchanged.
	 */

	if(p == 0 || f == 1){
		return 1;
	}

	/**
	 * Otherwise, the factor list has to change. We scan linearly the factor
	 * list for either a pre-existing spot or an insertion spot. Scanning
	 * linearly over a 15-element array is faster and less complex than binary
	 * search.
	 */

	for(i=0;i<fl->d;i++){
		if(fl->f[i] == f){
			/**
			 * Factor is already in list.
			 */

			fl->p[i] += p;
			if(fl->p[i] == 0){
				/**
				 * We removed all factors f. Bump leftwards the remainder to
				 * maintain sorted order.
				 */

				memmove(&fl->f[i], &fl->f[i+1], sizeof(fl->f[i])*(fl->d-i));
				memmove(&fl->p[i], &fl->p[i+1], sizeof(fl->p[i])*(fl->d-i));
				fl->d--;
			}
			return 1;
		}else if(fl->f[i] > f){
			/* Inject the factor at this place in order to keep list sorted,
			   if we have the capacity. */

			if(gaIFLFull(fl)){
				/* We can't bump the list rightwards, it's full already! */
				return 0;
			}

			memmove(&fl->f[i+1], &fl->f[i], sizeof(fl->f[i])*(fl->d-i));
			memmove(&fl->p[i+1], &fl->p[i], sizeof(fl->p[i])*(fl->d-i));
			fl->f[i] = f;
			fl->p[i] = p;
			fl->d++;
			return 1;
		}
	}

	/**
	 * We looked at every factor in the list and f is strictly greater than
	 * all of them.
	 *
	 * If the list is full, we cannot insert f, but if it isn't, we can simply
	 * tack it at the end.
	 */

	if(gaIFLFull(fl)){
		return 0;
	}else{
		fl->f[fl->d] = f;
		fl->p[fl->d] = p;
		fl->d++;
		return 1;
	}
}

int      gaIFLGetFactorPower(const ga_factor_list* fl, uint64_t f){
	int i;

	for(i=0;i<fl->d;i++){
		if(fl->f[i] == f){
			return fl->p[i];
		}
	}

	return 0;
}

uint64_t gaIFLGetProduct(const ga_factor_list* fl){
	uint64_t p = 1;
	int i, j;

	for(i=0;i<fl->d;i++){
		for(j=0;j<fl->p[i];j++){
			p *= fl->f[i];
		}
	}

	return p;
}

int      gaIFLIsOverflowed(const ga_factor_list* fl){
	uint64_t p = 1, MAX=-1;
	int i, j;

	if(gaIFLGetFactorPower(fl, 0) >=  1){
		return 0;
	}
	if(gaIFLGetFactorPower(fl, 2) >= 64){
		return 1;
	}

	for(i=0;i<fl->d;i++){
		for(j=0;j<fl->p[i];j++){
			if(MAX/p < fl->f[i]){
				return 1;
			}
			p *= fl->f[i];
		}
	}

	return 0;
}

uint64_t gaIFLGetGreatestFactor(const ga_factor_list* fl){
	return fl->d ? fl->f[fl->d-1] : 1;
}

uint64_t gaIFLGetSmallestFactor(const ga_factor_list* fl){
	return fl->d ? fl->f[0]         : 1;
}

static uint64_t gaIFLGetProductv(int n, const ga_factor_list* fl){
	uint64_t p = 1;
	int i;

	for(i=0;i<n;i++){
		p *= gaIFLGetProduct(fl+i);
	}

	return p;
}

static uint64_t gaIFLGetGreatestFactorv(int n, const ga_factor_list* fl, int* idx){
	uint64_t f = 0, currF;
	int i, hasFactors=0;

	if(idx){*idx = 0;}

	for(i=0;i<n;i++){
		if(fl[i].d > 0){
			hasFactors = 1;
			currF = gaIFLGetGreatestFactor(fl+i);
			if(f <= currF){
				f = currF;
				if(idx){*idx = i;}
			}
		}
	}

	return hasFactors ? f : 1;
}

static uint64_t gaIFLGetSmallestFactorv(int n, const ga_factor_list* fl, int* idx){
	uint64_t f = -1, currF;
	int i, hasFactors=0;

	if(idx){*idx = 0;}

	for(i=0;i<n;i++){
		if(fl[i].d > 0){
			hasFactors = 1;
			currF = gaIFLGetSmallestFactor(fl+i);
			if(f >= currF){
				f = currF;
				if(idx){*idx = i;}
			}
		}
	}

	return hasFactors ? f : 1;
}

int      gaIFLsprintf(char* str, const ga_factor_list* fl){
	int    i, j;
	int    total = 0;
	char*  ptr   = str;

	/* Loop over all factors and spit them out. */
	for(i=0;i<fl->d;i++){
		for(j=0;j<fl->p[i];j++){
			total += sprintf(ptr, "%llu*", (unsigned long long)fl->f[i]);
			if(ptr){
				ptr   += strlen(ptr);
			}
		}
	}

	/* If no factors were printed, print 1. */
	if(total == 0){
		total += sprintf(ptr, "1*");
		if(ptr){
			ptr   += strlen(ptr);
		}
	}

	/* Terminate buffer ('*' -> '\0') and deduct one character. */
	total--;
	if(str){
		str[total]  = '\0';
	}

	return total;
}

void gaIFLappend(strb *sb, const ga_factor_list* fl){
	int  i, j;
	int  noFactorsPrinted = 1;

	/* Loop over all factors and spit them out. */
	for(i=0;i<fl->d;i++){
		for(j=0;j<fl->p[i];j++){
			noFactorsPrinted = 0;
			strb_appendf(sb, "%llu*", (unsigned long long)fl->f[i]);
		}
	}

	/**
	 * If no factors were printed, print 1.
	 * Otherwise, delete final '*'.
	 */

	if(noFactorsPrinted){
		strb_appendf(sb, "1");
	}else{
		sb->s[--sb->l] = '\0';
	}
}

static void     gaIScheduleChecked(const int       n,
                                   const uint64_t  maxBtot,
                                   const uint64_t* maxBind,
                                   const uint64_t  maxGtot,
                                   const uint64_t* maxGind,
                                   uint64_t*       bs,
                                   uint64_t*       gs,
                                   uint64_t*       cs){
	int      i;
	uint64_t kBS, kGS, k;

	/**
	 * Allocate a VLA or similar.
	 *
	 * C89 neither allows VLAs nor a check beforehand that n>0 to avoid UB. The
	 * check for n>0 was thus done in our caller.
	 */

#if GA_USING_MALLOC_FOR_VLA
	ga_factor_list* factBS = malloc(n * sizeof(*factBS));
	ga_factor_list* factGS = malloc(n * sizeof(*factGS));
	ga_factor_list* factCS = malloc(n * sizeof(*factCS));
#else
	ga_factor_list factBS[n];
	ga_factor_list factGS[n];
	ga_factor_list factCS[n];
#endif




	/**
	 * Factorize the provided integers under their k-smoothness constraint.
	 * Use the strictest of either the block or grid constraints on each
	 * dimension.
	 */

	for(i=0;i<n;i++){
		kBS = maxBtot < maxBind[i] ? maxBtot : maxBind[i];
		kGS = maxGtot < maxGind[i] ? maxGtot : maxGind[i];
		k   =   kBS   <     kGS    ?   kBS   :     kGS;

		gaIFactorize(bs[i], -1, k, factBS+i);
		gaIFactorize(gs[i], -1, k, factGS+i);
		gaIFactorize(cs[i], -1, k, factCS+i);
	}

	/**
	 * Invoke scheduler core with factor-list version of our arguments.
	 */

	gaIFLSchedule(n,
	              maxBtot,
	              maxBind,
	              maxGtot,
	              maxGind,
	              factBS,
	              factGS,
	              factCS);


	/**
	 * Convert factor lists to products and place them in output arguments.
	 */

	for(i=0;i<n;i++){
		bs[i] = gaIFLGetProduct(factBS+i);
		gs[i] = gaIFLGetProduct(factGS+i);
		cs[i] = gaIFLGetProduct(factCS+i);
	}


	/**
	 * Eliminate VLA-like storage if it was allocated with malloc().
	 */

#if GA_USING_MALLOC_FOR_VLA
	free(factBS);
	free(factGS);
	free(factCS);
#endif
}

void     gaISchedule(const int       n,
                     const uint64_t  maxBtot,
                     const uint64_t* maxBind,
                     const uint64_t  maxGtot,
                     const uint64_t* maxGind,
                     uint64_t*       bs,
                     uint64_t*       gs,
                     uint64_t*       cs){
	if(n<=0){return;}

	gaIScheduleChecked(n,
	                   maxBtot,
	                   maxBind,
	                   maxGtot,
	                   maxGind,
	                   bs,
	                   gs,
	                   cs);
}

void     gaIFLSchedule(const int       n,
                       const uint64_t  maxBtot,
                       const uint64_t* maxBind,
                       const uint64_t  maxGtot,
                       const uint64_t* maxGind,
                       ga_factor_list* factBS,
                       ga_factor_list* factGS,
                       ga_factor_list* factCS){
	/**
	 * If we have zero dimensions, the scheduling job is easy.
	 */

	if(n<=0){return;}

	/**
	 * First, we move factors from factBS[i] and factGS[i] to factCS[i], in
	 * order of largest to smallest, until their product is at or below
	 * maxBind[i] and maxGind[i] respectively.
	 */

	gaIFLScheduleSatisfyInd(n, factBS, factCS, maxBind);
	gaIFLScheduleSatisfyInd(n, factGS, factCS, maxGind);

	/**
	 * Then we move out more factors from factBS[i] and factGS[i], in order of
	 * smallest to largest, until their common product is at or below maxBtot
	 * and maxGtot respectively.
	 */

	gaIFLScheduleSatisfyTot(n, factBS, factCS, maxBtot);
	gaIFLScheduleSatisfyTot(n, factGS, factCS, maxGtot);

	/**
	 * At this point, the scheduling is guaranteed to be valid, but may be
	 * nowhere close to optimal.
	 *
	 * So we start moving in factors from factCS[i] to factBS[i], in order of
	 * largest to smallest, while remaining below maxBtot and maxBind[i].
	 *
	 * Lastly, we move in factors from factCS[i] to factBG[i], in order of
	 * largest to smallest, while remaining below maxGtot and maxGind[i].
	 */

	gaIFLScheduleOpt(n, factCS, factBS, maxBtot, maxBind);
	gaIFLScheduleOpt(n, factCS, factGS, maxGtot, maxGind);
}

static void     gaIFLScheduleSatisfyInd(const int       n,
                                        ga_factor_list* from,
                                        ga_factor_list* to,
                                        const uint64_t* maxInd){
	int      i;
	uint64_t f, p;

	for(i=0;i<n;i++){
		p = gaIFLGetProduct       (from+i);
		f = gaIFLGetGreatestFactor(from+i);
		while(p > maxInd[i]){
			if(p%f){
				f  = gaIFLGetGreatestFactor(from+i);
			}
			p /= f;
			gaIFLAddFactors(from+i, f, -1);
			gaIFLAddFactors(to  +i, f, +1);
		}
	}
}

static void     gaIFLScheduleSatisfyTot(const int       n,
                                        ga_factor_list* from,
                                        ga_factor_list* to,
                                        const uint64_t  maxTot){
	int      a, i, c;
	uint64_t f, p;

	p = gaIFLGetProductv(n, from);
	a = 0;

	while(p > maxTot){
		f = gaIFLGetSmallestFactorv(n, from, &a);
		c = gaIFLGetFactorPower    (from+a, f);

		for(i=c-1;i>=0 && p>maxTot;i--){
			p /= f;
			gaIFLAddFactors(from+a, f, -1);
			gaIFLAddFactors(to  +a, f, +1);
		}
	}
}

static void     gaIFLScheduleOpt(const int       n,
                                 ga_factor_list* from,
                                 ga_factor_list* to,
                                 const uint64_t  maxTot,
                                 const uint64_t* maxInd){
	int i, j, k;
	uint64_t maxFTot, maxFInd, currF, f, pTot = 1;
#if GA_USING_MALLOC_FOR_VLA
	uint64_t* pInd = malloc(n * sizeof(*pInd));
#else
	uint64_t  pInd[n];
#endif

	/* Muzzle compiler about a random function being unused. */
	(void)gaIFLGetGreatestFactorv;

	/**
	 * Check whether optimization is possible.
	 */

	for(i=0;i<n;i++){
		pTot *= pInd[i] = gaIFLGetProduct(to+i);
	}
	maxFTot = maxTot/pTot;
	if(maxFTot <= 1){
		return;
	}

	/* Optimize. */
	do{
		/**
		 * At the beginning of each iteration, maxFTot is preset to maxTot/p,
		 * the largest factor that can legitimately be added into `to` without
		 * exceeding the *global* limit.
		 *
		 * We select, amongst all dimensions, the largest f such that
		 *     f <= maxFTot     and
		 *     f <= maxFInd[k]
		 * and record both f and k.
		 */

		f =  1;
		k = -1;
		for(i=0;i<n;i++){
			maxFInd = maxInd[i]/pInd[i];

			for(j=from[i].d-1;j>=0;j--){
				currF = from[i].f[j];

				if(currF <= maxFTot && currF <= maxFInd && currF >= f){
					f = currF;
					k = i;
					break;
				}
			}
		}

		if(k == -1){
			break;
		}

		gaIFLAddFactors(from+k, f, -1);
		gaIFLAddFactors(to  +k, f, +1);
		pInd[k] *= f;
		pTot    *= f;
		maxFTot  = maxTot/pTot;
	}while(maxFTot>1 && f>1);

#if GA_USING_MALLOC_FOR_VLA
	free(pInd);
#endif
}
