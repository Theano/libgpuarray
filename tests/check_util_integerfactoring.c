/* Includes */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <check.h>
#include <gpuarray/util.h>
#include "util/integerfactoring.h"


/**
 * Primality Checker
 */

START_TEST(test_primalitychecker){
	/* Tiny numbers */
	ck_assert(!gaIIsPrime(                   0ULL));
	ck_assert(!gaIIsPrime(                   1ULL));
	ck_assert( gaIIsPrime(                   2ULL));
	ck_assert( gaIIsPrime(                   3ULL));
	ck_assert(!gaIIsPrime(                   4ULL));
	ck_assert( gaIIsPrime(                   5ULL));
	ck_assert(!gaIIsPrime(                   6ULL));
	ck_assert( gaIIsPrime(                   7ULL));
	ck_assert(!gaIIsPrime(                   8ULL));
	ck_assert(!gaIIsPrime(                   9ULL));
	ck_assert(!gaIIsPrime(                  10ULL));
	ck_assert( gaIIsPrime(                  11ULL));
	ck_assert(!gaIIsPrime(                  12ULL));
	ck_assert( gaIIsPrime(                  13ULL));
	ck_assert(!gaIIsPrime(                  14ULL));
	ck_assert(!gaIIsPrime(                  15ULL));
	ck_assert(!gaIIsPrime(                  16ULL));
	ck_assert( gaIIsPrime(                  17ULL));
	ck_assert(!gaIIsPrime(                  18ULL));
	ck_assert( gaIIsPrime(                  19ULL));
	ck_assert(!gaIIsPrime(                  20ULL));
	/* Small primes */
	ck_assert( gaIIsPrime(                4987ULL));
	ck_assert( gaIIsPrime(                4993ULL));
	ck_assert( gaIIsPrime(                4999ULL));
	/* Squares of primes */
	ck_assert(!gaIIsPrime(            24870169ULL));
	ck_assert(!gaIIsPrime(            24930049ULL));
	ck_assert(!gaIIsPrime(            24990001ULL));
	/* Catalan pseudoprimes */
	ck_assert(!gaIIsPrime(                5907ULL));
	ck_assert(!gaIIsPrime(             1194649ULL));
	ck_assert(!gaIIsPrime(            12327121ULL));
	/* Fermat base-2 pseudoprimes */
	ck_assert(!gaIIsPrime(                 341ULL));
	ck_assert(!gaIIsPrime(                 561ULL));
	ck_assert(!gaIIsPrime(                 645ULL));
	ck_assert(!gaIIsPrime(                1105ULL));
	ck_assert(!gaIIsPrime(                1387ULL));
	ck_assert(!gaIIsPrime(                1729ULL));
	ck_assert(!gaIIsPrime(                1905ULL));
	ck_assert(!gaIIsPrime(                2047ULL));
	ck_assert(!gaIIsPrime(                2465ULL));
	ck_assert(!gaIIsPrime(              486737ULL));
	/* Strong Lucas pseudoprimes */
	ck_assert(!gaIIsPrime(                5459ULL));
	ck_assert(!gaIIsPrime(                5459ULL));
	ck_assert(!gaIIsPrime(                5459ULL));
	ck_assert(!gaIIsPrime(                5777ULL));
	ck_assert(!gaIIsPrime(               10877ULL));
	ck_assert(!gaIIsPrime(               16109ULL));
	ck_assert(!gaIIsPrime(               18971ULL));
	ck_assert(!gaIIsPrime(               22499ULL));
	ck_assert(!gaIIsPrime(               24569ULL));
	ck_assert(!gaIIsPrime(               25199ULL));
	ck_assert(!gaIIsPrime(               40309ULL));
	ck_assert(!gaIIsPrime(               58519ULL));
	ck_assert(!gaIIsPrime(               75077ULL));
	ck_assert(!gaIIsPrime(               97439ULL));
	ck_assert(!gaIIsPrime(              100127ULL));
	ck_assert(!gaIIsPrime(              113573ULL));
	ck_assert(!gaIIsPrime(              115639ULL));
	ck_assert(!gaIIsPrime(              130139ULL));
	/* Medium, prime. */
	ck_assert( gaIIsPrime(          2100000011ULL));
	ck_assert( gaIIsPrime(          2100000017ULL));
	/* Large, non-smooth, composite */
	ck_assert(!gaIIsPrime( 2196095973992233039ULL));
	/* Largest prime < 2**64: */
	ck_assert( gaIIsPrime(18446744073709551557ULL));
	/* Largest integers */
	ck_assert(!gaIIsPrime(18446744073709551613ULL));
	ck_assert(!gaIIsPrime(18446744073709551614ULL));
	ck_assert(!gaIIsPrime(18446744073709551615ULL));
}END_TEST

/**
 * Integer Factorization test
 */

START_TEST(test_integerfactorization){
	ga_factor_list fl;
	uint64_t       n;

	/**
	 * Attempt exact factorization for 2^64-1, no k-smoothness constraint.
	 * Expected PASS with 3*5*17*257*641*65537*6700417
	 */

	n = 18446744073709551615ULL;
	ck_assert_int_ne (gaIFactorize(n,         0,     0, &fl), 0);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    3ULL),  1);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    5ULL),  1);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                   17ULL),  1);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                  257ULL),  1);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                  641ULL),  1);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                65537ULL),  1);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,              6700417ULL),  1);
	ck_assert_uint_eq(gaIFLGetProduct(&fl), n);

	/**
	 * Attempt exact factorization for 2^64-1, 4096-smooth constraint.
	 * Expected FAIL, because 2^64-1 possesses prime factors in excess of 4096.
	 */

	n = 18446744073709551615ULL;
	ck_assert_int_eq (gaIFactorize(n,         0,  4096, &fl), 0);

	/**
	 * Attempt approximate factorization for 2^64-1, no k-smoothness constraint.
	 * Unlimited growth permitted.
	 * Expected PASS, since 2^64-1 rounds up to 2^64 and 2^64 trivially factorizes.
	 */

	n = 18446744073709551615ULL;
	ck_assert_int_ne (gaIFactorize(n,        -1,     0, &fl), 0);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    2ULL), 64);
	ck_assert_uint_eq(gaIFLGetGreatestFactor(&fl), 2);
	ck_assert_int_ne (gaIFLIsOverflowed(&fl), 0);

	/**
	 * Attempt exact factorization for 2196095973992233039, no k-smoothness constraint.
	 * 2196095973992233039 is a large, highly non-smooth number, with three enormous
	 * factors.
	 * Expected PASS *very quickly*, since it factorizes as 1299817*1299821*1299827
	 */

	n =  2196095973992233039ULL;
	ck_assert_int_ne (gaIFactorize(n,         0,     0, &fl), 0);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,              1299817ULL),  1);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,              1299821ULL),  1);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,              1299827ULL),  1);
	ck_assert_uint_eq(gaIFLGetGreatestFactor(&fl), 1299827);
	ck_assert_uint_eq(gaIFLGetProduct(&fl), n);

	/**
	 * Attempt approximate factorization for 2196095973992233039, 16-smooth constraint.
	 * 2196095973992233039 is a large, highly non-smooth number, with three enormous
	 * factors. It is not 64-smooth, so code paths that attempt approximate
	 * factorization within the growth limits (.005%) are exercised.
	 *
	 * Expected PASS *relatively quickly*.
	 */

	n =  2196095973992233039ULL;
	ck_assert_int_ne (gaIFactorize(n, n*1.00005,    16, &fl), 0);
	ck_assert_uint_ge(gaIFLGetProduct(&fl), n);
	ck_assert_uint_le(gaIFLGetProduct(&fl), n*1.00005);

	/**
	 * Attempt exact factorization of 7438473388800000000, 5-smooth constraint.
	 * It is a large, 5-smooth number. This should exercise the 5-smooth
	 * factorization path.
	 */

	n =  7438473388800000000ULL;
	ck_assert_int_ne (gaIFactorize(n,         0,     5, &fl), 0);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    2ULL), 14);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    3ULL), 19);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    5ULL),  8);
	ck_assert_uint_eq(gaIFLGetGreatestFactor(&fl), 5);
	ck_assert_uint_eq(gaIFLGetProduct(&fl), n);

	/**
	 * Attempt approximate factorization of 7438473388799999997, 2-smooth constraint.
	 * It is a large, non-smooth number. This should exercise the optimal 2-smooth
	 * factorizer in spite of the available, unlimited slack.
	 */

	n =  7438473388799999997ULL;
	ck_assert_int_ne (gaIFactorize(n,        -1,      2, &fl), 0);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    2ULL), 63);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    3ULL),  0);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    5ULL),  0);
	ck_assert_uint_eq(gaIFLGetGreatestFactor(&fl), 2);
	ck_assert_uint_eq(gaIFLGetProduct(&fl),  9223372036854775808ULL);

	/**
	 * Attempt approximate factorization of 7438473388799999997, 3-smooth constraint.
	 * It is a large, non-smooth number. This should exercise the optimal 3-smooth
	 * factorizer in spite of the available, unlimited slack.
	 */

	n =  7438473388799999997ULL;
	ck_assert_int_ne (gaIFactorize(n,        -1,      3, &fl), 0);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    2ULL), 31);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    3ULL), 20);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    5ULL),  0);
	ck_assert_uint_eq(gaIFLGetGreatestFactor(&fl), 3);
	ck_assert_uint_eq(gaIFLGetProduct(&fl),  7487812485248974848ULL);

	/**
	 * Attempt approximate factorization of 7438473388799999997, 5-smooth constraint.
	 * It is a large, non-smooth number, but 3 integers above it is a 5-smooth
	 * integer, 7438473388800000000. This should exercise the optimal 5-smooth
	 * factorizer in spite of the available, unlimited slack.
	 */

	n =  7438473388799999997ULL;
	ck_assert_int_ne (gaIFactorize(n,        -1,     5, &fl), 0);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    2ULL), 14);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    3ULL), 19);
	ck_assert_int_eq (gaIFLGetFactorPower(&fl,                    5ULL),  8);
	ck_assert_uint_eq(gaIFLGetGreatestFactor(&fl), 5);
	ck_assert_uint_eq(gaIFLGetProduct(&fl), 7438473388800000000ULL);

	/**
	 * Toughest challenge: Attempt very tight approximate factorization of
	 * 9876543210987654321 with .01% slack and 43-smooth constraint.
	 *
	 * This forces a bypass of the optimal 5-smooth factorizers and heavily
	 * exercises the nextI:, subfactorize:, primetest: and newX jumps and
	 * calculations.
	 *
	 * Expected PASS, "reasonably fast".
	 */

	n =  9876543210987654321ULL;
	ck_assert_int_ne (gaIFactorize(n, n*1.0001,    43, &fl), 0);
	ck_assert_uint_ge(gaIFLGetProduct(&fl), n);
	ck_assert_uint_le(gaIFLGetProduct(&fl), n*1.0001);
	ck_assert_uint_le(gaIFLGetGreatestFactor(&fl), 43);
}END_TEST

START_TEST(test_scheduler){
	/* We use here the CUDA limits of a CC 3.0 GPU as an example. */
	uint64_t maxBTot  =       1024, maxBInd[] = {      1024,      1024,        64},
	         maxGTot  = 0xFFFFFFFF, maxGInd[] = {2147483647,     65535,     65535},
	         warpSize =         32;

	int                warpAxis;
	uint64_t           dims[3];
	ga_factor_list     factBS[3], factGS[3], factCS[3];
	unsigned long long intbBS[3], intbGS[3], intbCS[3];
	unsigned long long intaBS[3], intaGS[3], intaCS[3];

	/**
	 * NOTE: If you want to view befores-and-afters of scheduling, #define PRINT
	 *       to something non-0.
	 */
#define PRINT 0

	/**
	 *
	 * Testcase: (895,1147,923) job, warpSize on axis 0.
	 *
	 */

	{
		warpAxis       =          0;
		dims[0]        =        895;
		dims[1]        =       1141;
		dims[2]        =        923;
		dims[warpAxis] = (dims[warpAxis]+warpSize-1) / warpSize;

		/**
		 * Factorization job must be successful.
		 */

		ck_assert(gaIFactorize(warpAxis==0?warpSize:1,           0, maxBInd[0], factBS+0));
		ck_assert(gaIFactorize(warpAxis==1?warpSize:1,           0, maxBInd[1], factBS+1));
		ck_assert(gaIFactorize(warpAxis==2?warpSize:1,           0, maxBInd[2], factBS+2));
		ck_assert(gaIFactorize(                     1,           0, maxBInd[0], factGS+0));
		ck_assert(gaIFactorize(                     1,           0, maxBInd[1], factGS+1));
		ck_assert(gaIFactorize(                     1,           0, maxBInd[2], factGS+2));
		ck_assert(gaIFactorize(               dims[0], dims[0]*1.1, maxBInd[0], factCS+0));
		ck_assert(gaIFactorize(               dims[1], dims[1]*1.1, maxBInd[1], factCS+1));
		ck_assert(gaIFactorize(               dims[2], dims[2]*1.1, maxBInd[2], factCS+2));

		intbBS[0] = gaIFLGetProduct(factBS+0);
		intbBS[1] = gaIFLGetProduct(factBS+1);
		intbBS[2] = gaIFLGetProduct(factBS+2);
		intbGS[0] = gaIFLGetProduct(factGS+0);
		intbGS[1] = gaIFLGetProduct(factGS+1);
		intbGS[2] = gaIFLGetProduct(factGS+2);
		intbCS[0] = gaIFLGetProduct(factCS+0);
		intbCS[1] = gaIFLGetProduct(factCS+1);
		intbCS[2] = gaIFLGetProduct(factCS+2);

		/**
		 * Ensure that factorization only *increases* the size of the problem.
		 */

		ck_assert_uint_ge(intbCS[0], dims[0]);
		ck_assert_uint_ge(intbCS[1], dims[1]);
		ck_assert_uint_ge(intbCS[2], dims[2]);


		/**
		 * Run scheduler.
		 */

#if PRINT
		printf("Before:\n");
		printf("BS: (%6llu, %6llu, %6llu)\n", intbBS[0], intbBS[1], intbBS[2]);
		printf("GS: (%6llu, %6llu, %6llu)\n", intbGS[0], intbGS[1], intbGS[2]);
		printf("CS: (%6llu, %6llu, %6llu)\n", intbCS[0], intbCS[1], intbCS[2]);
#endif
		gaIFLSchedule(3, maxBTot, maxBInd, maxGTot, maxGInd, factBS, factGS, factCS);
		intaBS[0] = gaIFLGetProduct(factBS+0);
		intaBS[1] = gaIFLGetProduct(factBS+1);
		intaBS[2] = gaIFLGetProduct(factBS+2);
		intaGS[0] = gaIFLGetProduct(factGS+0);
		intaGS[1] = gaIFLGetProduct(factGS+1);
		intaGS[2] = gaIFLGetProduct(factGS+2);
		intaCS[0] = gaIFLGetProduct(factCS+0);
		intaCS[1] = gaIFLGetProduct(factCS+1);
		intaCS[2] = gaIFLGetProduct(factCS+2);
#if PRINT
		printf("After:\n");
		printf("BS: (%6llu, %6llu, %6llu)\n", intaBS[0], intaBS[1], intaBS[2]);
		printf("GS: (%6llu, %6llu, %6llu)\n", intaGS[0], intaGS[1], intaGS[2]);
		printf("CS: (%6llu, %6llu, %6llu)\n", intaCS[0], intaCS[1], intaCS[2]);
#endif

		/**
		 * Scheduling is only about moving factors between block/grid/chunk factor
		 * lists. Therefore, the three dimensions must not have changed size.
		 */

		ck_assert_uint_eq(intbBS[0]*intbGS[0]*intbCS[0], intaBS[0]*intaGS[0]*intaCS[0]);
		ck_assert_uint_eq(intbBS[1]*intbGS[1]*intbCS[1], intaBS[1]*intaGS[1]*intaCS[1]);
		ck_assert_uint_eq(intbBS[2]*intbGS[2]*intbCS[2], intaBS[2]*intaGS[2]*intaCS[2]);

		/**
		 * Verify that the individual limits and global limits on threads in a
		 * block and blocks in a grid are met.
		 */

		ck_assert_uint_le(intaBS[0],                     maxBInd[0]);
		ck_assert_uint_le(intaBS[1],                     maxBInd[1]);
		ck_assert_uint_le(intaBS[2],                     maxBInd[2]);
		ck_assert_uint_le(intaGS[0],                     maxGInd[0]);
		ck_assert_uint_le(intaGS[1],                     maxGInd[1]);
		ck_assert_uint_le(intaGS[2],                     maxGInd[2]);
		ck_assert_uint_le(intaBS[0]*intaBS[1]*intaBS[2], maxBTot);
		ck_assert_uint_le(intaGS[0]*intaGS[1]*intaGS[2], maxGTot);
	}


	/**
	 *
	 * Testcase: (1,1,121632959) job, warpSize on axis 2.
	 *
	 */

	{
		warpAxis       =         2;
		dims[0]        =         1;
		dims[1]        =         1;
		dims[2]        = 121632959;
		dims[warpAxis] = (dims[warpAxis]+warpSize-1) / warpSize;

		/**
		 * Factorization job must be successful.
		 */

		ck_assert(gaIFactorize(warpAxis==0?warpSize:1,           0, maxBInd[0], factBS+0));
		ck_assert(gaIFactorize(warpAxis==1?warpSize:1,           0, maxBInd[1], factBS+1));
		ck_assert(gaIFactorize(warpAxis==2?warpSize:1,           0, maxBInd[2], factBS+2));
		ck_assert(gaIFactorize(                     1,           0, maxBInd[0], factGS+0));
		ck_assert(gaIFactorize(                     1,           0, maxBInd[1], factGS+1));
		ck_assert(gaIFactorize(                     1,           0, maxBInd[2], factGS+2));
		ck_assert(gaIFactorize(               dims[0], dims[0]*1.1, maxBInd[0], factCS+0));
		ck_assert(gaIFactorize(               dims[1], dims[1]*1.1, maxBInd[1], factCS+1));
		ck_assert(gaIFactorize(               dims[2], dims[2]*1.1, maxBInd[2], factCS+2));

		intbBS[0] = gaIFLGetProduct(factBS+0);
		intbBS[1] = gaIFLGetProduct(factBS+1);
		intbBS[2] = gaIFLGetProduct(factBS+2);
		intbGS[0] = gaIFLGetProduct(factGS+0);
		intbGS[1] = gaIFLGetProduct(factGS+1);
		intbGS[2] = gaIFLGetProduct(factGS+2);
		intbCS[0] = gaIFLGetProduct(factCS+0);
		intbCS[1] = gaIFLGetProduct(factCS+1);
		intbCS[2] = gaIFLGetProduct(factCS+2);

		/**
		 * Ensure that factorization only *increases* the size of the problem.
		 */

		ck_assert_uint_ge(intbCS[0], dims[0]);
		ck_assert_uint_ge(intbCS[1], dims[1]);
		ck_assert_uint_ge(intbCS[2], dims[2]);


		/**
		 * Run scheduler.
		 */

#if PRINT
		printf("Before:\n");
		printf("BS: (%6llu, %6llu, %6llu)\n", intbBS[0], intbBS[1], intbBS[2]);
		printf("GS: (%6llu, %6llu, %6llu)\n", intbGS[0], intbGS[1], intbGS[2]);
		printf("CS: (%6llu, %6llu, %6llu)\n", intbCS[0], intbCS[1], intbCS[2]);
#endif
		gaIFLSchedule(3, maxBTot, maxBInd, maxGTot, maxGInd, factBS, factGS, factCS);
		intaBS[0] = gaIFLGetProduct(factBS+0);
		intaBS[1] = gaIFLGetProduct(factBS+1);
		intaBS[2] = gaIFLGetProduct(factBS+2);
		intaGS[0] = gaIFLGetProduct(factGS+0);
		intaGS[1] = gaIFLGetProduct(factGS+1);
		intaGS[2] = gaIFLGetProduct(factGS+2);
		intaCS[0] = gaIFLGetProduct(factCS+0);
		intaCS[1] = gaIFLGetProduct(factCS+1);
		intaCS[2] = gaIFLGetProduct(factCS+2);
#if PRINT
		printf("After:\n");
		printf("BS: (%6llu, %6llu, %6llu)\n", intaBS[0], intaBS[1], intaBS[2]);
		printf("GS: (%6llu, %6llu, %6llu)\n", intaGS[0], intaGS[1], intaGS[2]);
		printf("CS: (%6llu, %6llu, %6llu)\n", intaCS[0], intaCS[1], intaCS[2]);
#endif

		/**
		 * Scheduling is only about moving factors between block/grid/chunk factor
		 * lists. Therefore, the three dimensions must not have changed size.
		 */

		ck_assert_uint_eq(intbBS[0]*intbGS[0]*intbCS[0], intaBS[0]*intaGS[0]*intaCS[0]);
		ck_assert_uint_eq(intbBS[1]*intbGS[1]*intbCS[1], intaBS[1]*intaGS[1]*intaCS[1]);
		ck_assert_uint_eq(intbBS[2]*intbGS[2]*intbCS[2], intaBS[2]*intaGS[2]*intaCS[2]);

		/**
		 * Verify that the individual limits and global limits on threads in a
		 * block and blocks in a grid are met.
		 */

		ck_assert_uint_le(intaBS[0],                     maxBInd[0]);
		ck_assert_uint_le(intaBS[1],                     maxBInd[1]);
		ck_assert_uint_le(intaBS[2],                     maxBInd[2]);
		ck_assert_uint_le(intaGS[0],                     maxGInd[0]);
		ck_assert_uint_le(intaGS[1],                     maxGInd[1]);
		ck_assert_uint_le(intaGS[2],                     maxGInd[2]);
		ck_assert_uint_le(intaBS[0]*intaBS[1]*intaBS[2], maxBTot);
		ck_assert_uint_le(intaGS[0]*intaGS[1]*intaGS[2], maxGTot);
	}
}END_TEST



Suite *get_suite(void){
	Suite *s  = suite_create("util_integerfactoring");
	TCase *tc = tcase_create("All");

	tcase_set_timeout(tc, 10.0);

	tcase_add_test(tc, test_primalitychecker);
	tcase_add_test(tc, test_integerfactorization);
	tcase_add_test(tc, test_scheduler);

	suite_add_tcase(s, tc);

	return s;
}

