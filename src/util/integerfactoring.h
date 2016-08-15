/* Include Guards */
#ifndef INTEGERFACTORING_H
#define INTEGERFACTORING_H


/* Includes */
#include <stdio.h>
#include <stdint.h>
#include <stdint-gcc.h>


/* Defines */



/* C++ Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structure Prototypes & Typedefs */
struct GA_FACTOR_LIST;
typedef struct GA_FACTOR_LIST GA_FACTOR_LIST;



/* Data Structures */

/**
 * @brief The GA_FACTOR_LIST struct.
 * 
 * Contains the list of distinct prime factors of a 64-bit unsigned integer, as
 * well as the powers of those factors.
 * 
 * There can be at most 15 such distinct factors, since the product of the
 * first 16 primes (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53) exceeds
 * the maximum unsigned number of 2^64-1. Moreover, there can be at most 63
 * factors all together, since 2^64 exceeds 2^64-1, so only an 8-bit number is
 * required to store the powers.
 * 
 * The 15th (last) element of the factor list is always 0 and has power 0,
 * and serves as a sort of sentinel.
 */

struct GA_FACTOR_LIST{
	uint64_t f[16];/* Factors */
	uint8_t  p[16];/* Powers of factors */
};



/* Functions */

/**
 * @brief Count trailing zeros of a 64-bit integer.
 * 
 * @param [in] n  The integer whose trailing zero count is to be computed.
 * @return     If n != 0, returns trailing zero count; Else returns 64.
 */

int      gaICtz(uint64_t n);

/**
 * @brief Count leading zeros of a 64-bit integer.
 * 
 * @param [in] n  The integer whose leading zero count is to be computed.
 * @return     If n != 0, returns leading zero count; Else returns 64.
 */

int      gaIClz(uint64_t n);

/**
 * @brief Integer Modular Multiplication.
 * 
 * Computes
 * 
 *     $$a*b \pmod m$$
 * 
 * efficiently for 64-bit unsigned integers a, b, m.
 */

uint64_t gaIMulMod    (uint64_t a, uint64_t b, uint64_t m);

/**
 * @brief Integer Modular Exponentiation.
 * 
 * Computes
 * 
 *     $$x^a \pmod m$$
 * 
 * efficiently for 64-bit unsigned integers x, a, m.
 */

uint64_t gaIPowMod    (uint64_t x, uint64_t a, uint64_t m);

/**
 * @brief Checks whether an integer is prime.
 * 
 * @param [in] n   The integer whose primality is to be checked.
 * @return 1 if prime; 0 if not prime.
 * 
 * NB: This is *not* a probabilistic primality checker. For all integers it can
 *     be given as input, it will correctly report "prime" or "composite".
 * NB: Internally, this function uses the Miller-Rabin test, which *is*
 *     probabilistic, and may falsely report a number as prime when in fact it
 *     is composite. However, this function uses a deterministic set of
 *     Miller-Rabin "witnesses", which ensures that there are no strong
 *     probable primes equal to or below 2^64-1 (the size of the input
 *     argument). This set of witnesses is
 *     
 *         $$a = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, and 37$$
 *     
 *     See https://oeis.org/A014233
 */

int      gaIIsPrime(uint64_t n);

/**
 * @brief Factorize a positive integer into a list of factors satisfying
 * certain properties.
 * 
 * The function factorizes a 64-bit, positive integer into a list of factors.
 * This factorization can be made "approximate"; That is, the product of the
 * factors returned can be slightly greater than the input number. The
 * maximum increase is controlled by a "slack" parameter maxN, as follows:
 * 
 *     $$\texttt{n} \le \prod(\mathrm{fact}(\texttt{n}) \le \texttt{maxN}$$
 * 
 * The advantage of offering some slack to the factorizer is that in return,
 * the factorizer may succeed in outputting a factorization with smaller
 * factors. The maxN slack parameter must be 0 or be greater than or equal to
 * n, but it is useless to set it beyond twice the value of n.
 * 
 * When maxN is equal to -1 (2^64 - 1), or is greater than or equal to 2n,
 * there is a guarantee that there exists a power of two that lies between n
 * and 2n. Since this factorization involves only powers of the smallest prime
 * (2), it is a valid factorization under any valid k-smoothness constraint,
 * and so will be returned.
 * 
 * When maxN is equal to 0 or n (no increase in value allowed), this implies
 * that an exact factoring is requested.
 * 
 * The factorization can also be constrained by a (k)-smoothness constraint.
 * A k-smooth number n has no prime factors greater than k. If the factorizer
 * is asked to factor with k-smoothness a number with prime factors greater
 * than k, it will search, within the slack space, for a slightly larger
 * number that is k-smooth and return that number's factoring. With maxN == n
 * and a k-smoothness constraint, this function reports whether or not the
 * number is k-smooth.
 * 
 * When k is equal to 0, equal to -1 (2^64 - 1), or is greater than or equal
 * to n, no k-smoothness constraints are imposed. An exact factoring is
 * required.
 * 
 * @param [in]  n       The integer to be factorized. Must be >0.
 * @param [in]  maxN    The "slack" parameter. The factor list returned will not
 *                      have a product that exceeds this number.
 * @param [in]  k       The k-smoothness constraint. k is the largest
 *                      acceptable factor in the output factor list. The
 *                      factorizer will, effectively, treat any number all of
 *                      whose prime factors exceed k as a prime.
 * @param [out] fl      The output factor list.
 * @return Non-zero if a factorization is found that satisfies both slack and
 *         smoothness constraints; Zero if no such factorization is found.
 *         If this function returns zero, the last factor in the factor
 *         list is not guaranteed to be prime.
 */

int      gaIFactorize(uint64_t n, uint64_t maxN, uint64_t k, GA_FACTOR_LIST* fl);

/**
 * @brief Initialize a factors list to all-factors- and all-powers-zero.
 * 
 * Such a factors list represents 1, since 0^0 = 1.
 */

void     gaIFLInit(GA_FACTOR_LIST* fl);

/**
 * @brief Add a factor f with power p to the factor list.
 * 
 * If factor f was already present in the factor list, increments
 * the corresponding power by p. Otherwise, adds the new factor f to
 * the list, if there is still space, and sets the power to p.
 * 
 * Maintains factor list in sorted order.
 * 
 * @return Non-zero if factor successfully added; Zero otherwise.
 */

int      gaIFLAddFactors(GA_FACTOR_LIST* fl, uint64_t f, uint8_t p);

/**
 * @brief Get the power of a given factor within a factor list.
 * 
 * @return The number of times a factor occurs within the current
 *         factorization. If it does not occur, return 0.
 */

int      gaIFLGetFactorPower(GA_FACTOR_LIST* fl, uint64_t f);

/**
 * @brief Compute the product of the factors stored in the factors list.
 */

uint64_t gaIFLGetProduct(const GA_FACTOR_LIST* fl);

/**
 * @brief Get the greatest factor in the factors list.
 */

uint64_t gaIFLGetGreatestFactor(const GA_FACTOR_LIST* fl);

/**
 * @brief Print out the factor list in a human-readable form, snprintf()-style.
 * 
 * @param [out] str   A string into which to print out the factor list. If the
 *                    factor list is a result of gaIFactorize(), then the
 *                    maximum length of buffer required is 128 bytes.
 *                    If str is NULL, nothing is printed.
 * @param [in]  size  The maximum number of bytes written, including the
 *                    terminating NUL (\0) character.
 * @param [in]  fl    The factor list to be printed.
 * @return            The number of characters that would have been printed
 *                    out, assuming an unbounded, non-NULL buffer.
 */

int   gaIFLsnprintf(char* str, size_t size, const GA_FACTOR_LIST* fl);


/* End C++ Extern "C" Guard */
#ifdef __cplusplus
}
#endif


/* End Include Guards */
#endif

