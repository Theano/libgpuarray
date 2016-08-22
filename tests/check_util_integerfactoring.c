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

START_TEST(test_integerfactorization)
{
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
}
END_TEST



Suite *get_suite(void){
	Suite *s  = suite_create("util_integerfactoring");
	TCase *tc = tcase_create("All");
	
	tcase_add_test(tc, test_integerfactorization);
	
	suite_add_tcase(s, tc);
	
	return s;
}

