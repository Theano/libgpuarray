/* Includes */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "integerfactoring.h"



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
 * @brief Round up positive n to next power-of-2 and report its factorization.
 */

static int      gaIFactorizeNextPow2(uint64_t n, ga_factor_list* fl);

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

static uint64_t gaIMulMod    (uint64_t a, uint64_t b, uint64_t m){
#if (__GNUC__ >= 4) && defined(__x86_64__)
	uint64_t r;
	
	asm(
	    "mul %2\n\t"
	    "div %3\n\t"
	    : "=&d"(r)                 /* Outputs */
	    : "a"(a), "r"(b), "r"(m)   /* Inputs */
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

int      gaIIsPrime   (uint64_t n){
	size_t         i, j;
	int            hasNoSmallFactors, hasSmallFactors;
	uint64_t       r, d;
	const uint64_t WITNESSES[]  = {2,3,5,7,11,13,17,19,23,29,31,37};
	const int      NUMWITNESSES = sizeof(WITNESSES)/sizeof(WITNESSES[0]);
	
	
	/**
	 * Check if it is 2, the oddest prime.
	 */
	
	if(n==2){return 1;}
	
	/**
	 * Check if it is an even integer.
	 */
	
	if((n&1) == 0){return 0;}
	
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
	
	hasNoSmallFactors = n%3 && n%5 && n%7 && n%11 && n%13;
	hasSmallFactors   = !hasNoSmallFactors;
	if(hasSmallFactors){
		return 0;
	}
	
	/**
	 * Otherwise proceed to the Miller-Rabin test.
	 * 
	 * The Miller-Rabin test uses integer "witnesses" in an attempt at
	 * proving the number composite. Should it fail to prove an integer
	 * composite, it reports the number as "probably prime". However, if
	 * the witnesses are chosen carefully, the Miller-Rabin test can be made
	 * deterministic below a chosen threshold. In our case, we use the primes
	 * 2 to 37 in order to ensure the correctness of the identifications for
	 * integers under 2^64.
	 */
	
	r = gaICtz(n-1);
	d = (n-1)>>r;
	
	/* For each witness... */
	for(i=0;i<NUMWITNESSES;i++){
		uint64_t a = WITNESSES[i];
		
		/* Modular exponentiation of witness by d, modulo the prime. */
		uint64_t x = gaIPowMod(a,d,n);
		
		/**
		 * If result is 1 or n-1, test inconclusive (can't prove
		 * compositeness).
		 */
		
		if(x==1 || x==n-1){goto continueWitnessLoop;}
		
		/**
		 * Otherwise, modulo-square x r-1 times. If result is ever 1, it's
		 * composite. If result is ever n-1, it's inconclusive. If after r-1
		 * iterations neither 1 nor n-1 came up, it's composite.
		 */
		
		for(j=0;j<r-1;j++){
			x = gaIPowMod(x,2,n);
			
			if(x==1){
				/* Composite! */
				return 0;
			}else if(x == n-1){
				/* Inconclusive (can't prove compositeness) */
				goto continueWitnessLoop;
			}
		}
		
		/* Composite! */
		return 0;
		
		continueWitnessLoop:;
	}
	
	/**
	 * Having failed to prove this is a composite, and given our choice of
	 * witnesses, we know we've identified a prime.
	 */
	
	return 1;
}

int      gaIFactorize (uint64_t n, uint64_t maxN, uint64_t k, ga_factor_list* fl){
	uint64_t i, x, p, f, c;
	
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
	
	if(maxN == (uint64_t)-1 || gaIClz(maxN) < gaIClz(n)){
		/**
		 * Either we are allowed unlimited growth of n, or the slack space
		 * [n, maxN] is big enough to contain a power of 2. We identify, round
		 * up to and factorize the next higher power of 2 greater than or equal
		 * to n trivially. Since powers of 2 are by definition 2-smooth, we
		 * automatically satisfy the most stringent possible smoothness
		 * constraint.
		 */
		
		return gaIFactorizeNextPow2(n, fl);
	}else if(maxN == 0){
		/**
		 * We are asked for a strict factoring.
		 */
		
		maxN = n;
	}
	
	if(k == 0 || k >= n){
		/**
		 * We want no k-smoothness constraint.
		 */
		
		k = n;
	}
	
	
	/**
	 * Master loop.
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
		 *     In this case we have to increment x and begin anew the
		 *     sub-factorization. This may cause us to fail out of factorizing
		 *     the current i, by exceeding our slack limit. If this happens we
		 *     abort the factorization rooted at i and move to the next i.
		 */
		
		primetest:
		if(x==1 || gaIIsPrime(x)){
			if(x<=k){
				gaIFLAddFactors(fl, x, 1);
				return 1;
			}else{
				p = gaIFLGetProduct(fl);
				if((maxN - p*x) < p){/* Overflow-free check maxN >= p*(x+1) */
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

static int      gaIFactorizeNextPow2(uint64_t n, ga_factor_list* fl){
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

void     gaIFLInit(ga_factor_list* fl){
	memset(fl, 0, sizeof(*fl));
}

int      gaIFLFull(ga_factor_list* fl){
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

int      gaIFLGetFactorPower(ga_factor_list* fl, uint64_t f){
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

int      gaIFLsnprintf(char* str, size_t size, const ga_factor_list* fl){
	int    i, j;
	
	int    total = 0;
	size_t left = size;
	char*  ptr  = size ? str : NULL;
	
	/* Loop over all factors and spit them out. */
	for(i=0;i<fl->d;i++){
		for(j=0;j<fl->p[i];j++){
			total += snprintf(ptr, left, "%llu*", (unsigned long long)fl->f[i]);
			if(ptr){
				left  -= strlen(ptr);
				ptr   += strlen(ptr);
			}
		}
	}
	
	/* If no factors were printed, print 1. */
	if(total == 0){
		total += snprintf(ptr, left, "1*");
		if(ptr){
			left  -= strlen(ptr);
			ptr   += strlen(ptr);
		}
	}
	
	/* Terminate buffer ('*' -> '\0') and deduct one character. */
	total--;
	if(str && size > 0){
		if(total >= size){
			str[size-1] = '\0';
		}else{
			str[total]  = '\0';
		}
	}
	
	return total;
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
	uint64_t maxFTot, maxFInd, currF, f;
	uint64_t pInd[n], pTot = 1;
	
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
}
