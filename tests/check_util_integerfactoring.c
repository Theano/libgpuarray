/* Includes */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <check.h>
#include <gpuarray/util.h>
#include "util/integerfactoring.h"


/**
 * Integer Factorization test
 */

START_TEST(test_integerfactorization){
	ga_factor_list fl;
	
	/**
	 * Attempt exact factorization for 2^64-1, no k-smoothness constraint.
	 * Expected PASS with 3*5*17*257*641*65537*6700417
	 */
	
	ck_assert_int_ne(gaIFactorize(18446744073709551615ULL,                            0,     0, &fl), 0);
	
	/**
	 * Attempt exact factorization for 2^64-1, 4096-smooth constraint.
	 * Expected FAIL, because 2^64-1 possesses prime factors in excess of 4096.
	 */
	
	ck_assert_int_eq(gaIFactorize(18446744073709551615ULL,                            0,  4096, &fl), 0);
	
	/**
	 * Attempt approximate factorization for 2^64-1, no k-smoothness constraint.
	 * Unlimited growth permitted.
	 * Expected PASS, since 2^64-1 rounds up to 2^64 and 2^64 trivially factorizes.
	 */
	
	ck_assert_int_ne(gaIFactorize(18446744073709551615ULL,                           -1,     0, &fl), 0);
	
	/**
	 * Attempt exact factorization for 2196095973992233039, no k-smoothness constraint.
	 * 2196095973992233039 is a large, highly non-smooth number, with three enormous
	 * factors.
	 * Expected PASS *very quickly*, since it factorizes as 1299817*1299821*1299827
	 */
	
	ck_assert_int_ne(gaIFactorize( 2196095973992233039ULL,                            0,     0, &fl), 0);
	
	/**
	 * Attempt approximate factorization for 2196095973992233039, 64-smooth constraint.
	 * 2196095973992233039 is a large, highly non-smooth number, with three enormous
	 * factors. It is not 64-smooth, so code paths that attempt approximate
	 * factorization within the growth limits (1%) are exercised.
	 * 
	 * Expected PASS *relatively quickly*.
	 */
	
	ck_assert_int_ne(gaIFactorize( 2196095973992233039ULL,  2196095973992233039ULL*1.01,    64, &fl), 0);
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
	
	tcase_add_test(tc, test_integerfactorization);
	tcase_add_test(tc, test_scheduler);
	
	suite_add_tcase(s, tc);
	
	return s;
}

