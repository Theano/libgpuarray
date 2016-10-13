/* Include Guards */
#ifndef INTEGERFACTORING_H
#define INTEGERFACTORING_H


/* Includes */
#include <stdio.h>
#include <stdint.h>

#include "util/strb.h"


/* Defines */



/* C++ Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structure Prototypes & Typedefs */
struct ga_factor_list_;
typedef struct ga_factor_list_ ga_factor_list;



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

struct ga_factor_list_{
	uint64_t f[16];/* Factors */
	uint8_t  p[16];/* Powers of factors */
	int      d;    /* Number of distinct factors. */
};



/* Functions */

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
 * n, but it is completely useless to set it beyond 2n.
 *
 * When maxN is equal to -1 (2^64 - 1), or is greater than or equal to 2n, no
 * upper limit is placed on the output factor list's product, but this
 * implementation guarantees its product will not exceed 2n. This is because
 * there always exists a power of two that lies between n and 2n, and since
 * this factorization involves only powers of the smallest prime (2), it is a
 * valid factorization under any valid k-smoothness constraint, and so may be
 * returned.
 *
 * When maxN is equal to 0 (no increase in value allowed), an exact factoring
 * is requested.
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
 * @param [in]  maxN    The "slack" parameter. The factor list returned will
 *                      not have a product that exceeds this number.
 * @param [in]  k       The k-smoothness constraint. k is the largest
 *                      acceptable factor in the output factor list. The
 *                      factorizer will, effectively, treat any number all of
 *                      whose prime factors exceed k as a prime.
 * @param [out] fl      The output factor list. Does *NOT* need to be
 *                      initialized.
 * @return Non-zero if a factorization is found that satisfies both slack and
 *         smoothness constraints; Zero if no such factorization is found.
 *         If this function returns zero, the last factor in the factor
 *         list is not guaranteed to be prime.
 */

int      gaIFactorize(uint64_t n, uint64_t maxN, uint64_t k, ga_factor_list* fl);

/**
 * @brief Initialize a factors list to all-factors- and all-powers-zero.
 *
 * Such a factors list represents 1, since 0^0 = 1.
 */

void     gaIFLInit(ga_factor_list* fl);

/**
 * @brief Reports whether another *distinct* factor can be added to the factor
 *        list safely.
 *
 * @return Returns zero if there are less than 15 distinct factors in the list
 *         and non-zero otherwise.
 */

int      gaIFLFull(const ga_factor_list* fl);

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

int      gaIFLAddFactors(ga_factor_list* fl, uint64_t f, int p);

/**
 * @brief Get the power of a given factor within a factor list.
 *
 * @return The number of times a factor occurs within the current
 *         factorization. If it does not occur, return 0.
 */

int      gaIFLGetFactorPower(const ga_factor_list* fl, uint64_t f);

/**
 * @brief Compute the product of the factors stored in the factors list.
 * 
 * NB: This function may return an overflowed result. To detect if it will,
 *     please call gaIFLIsOverflowed(fl).
 */

uint64_t gaIFLGetProduct(const ga_factor_list* fl);

/**
 * @brief Check whether the factor list produces a number >= 2^64.
 */

int      gaIFLIsOverflowed(const ga_factor_list* fl);

/**
 * @brief Get the greatest factor in the factors list.
 */

uint64_t gaIFLGetGreatestFactor(const ga_factor_list* fl);

/**
 * @brief Get the smallest factor in the factors list.
 */

uint64_t gaIFLGetSmallestFactor(const ga_factor_list* fl);

/**
 * @brief Print out the factor list in a human-readable form, sprintf()-style.
 * 
 * @param [out] str   A string into which to print out the factor list. If the
 *                    factor list is a result of gaIFactorize(), then the
 *                    maximum length of buffer required is 128 bytes.
 *                    If str is NULL, nothing is printed.
 * @param [in]  fl    The factor list to be printed.
 * @return            The number of characters that would have been printed
 *                    out, assuming an unbounded, non-NULL buffer.
 */

int gaIFLsprintf(char* str, const ga_factor_list* fl);

/**
 * @brief Print out the factor list in a human-readable form.
 *
 * @param [out] sb   A string into which to print out the factor list. If the
 *                   factor list is a result of gaIFactorize(), then the
 *                   maximum length of buffer required is 128 bytes.
 * @param [in]  fl   The factor list to be printed.
 */

void gaIFLappend(strb *sb, const ga_factor_list* fl);

/**
 * @brief Schedule block size, grid size and what's left over that fits in
 *        neither, which will be called "chunk" size, subject to constraints.
 *
 * @param [in]     n        Number of dimensions of the problem. The arrays
 *                          maxBind, maxGind, factBS, factGS, factCS must have
 *                          n elements.
 * @param [in]     maxBtot  The product of the block sizes in all n dimensions
 *                          will not exceed this value.
 * @param [in]     maxBind  The block size in dimensions i=0..n-1 will not
 *                          exceed maxBind[i].
 * @param [in]     maxGtot  The product of the grid sizes in all n dimensions
 *                          will not exceed this value.
 * @param [in]     maxGind  The grid size in dimensions i=0..n-1 will not
 *                          exceed maxGind[i].
 * @param [in,out] factBS   The block size for dimensions 0..n-1, as a factor list.
 * @param [in,out] factGS   The grid  size for dimensions 0..n-1, as a factor list.
 * @param [in,out] factCS   The chunk size for dimensions 0..n-1, as a factor list.
 * @param [in,out] bs       The block size for dimensions 0..n-1, as an integer.
 * @param [in,out] gs       The grid  size for dimensions 0..n-1, as an integer.
 * @param [in,out] cs       The chunk size for dimensions 0..n-1, as an integer.
 */

void     gaIFLSchedule(const int       n,
                       const uint64_t  maxBtot,
                       const uint64_t* maxBind,
                       const uint64_t  maxGtot,
                       const uint64_t* maxGind,
                       ga_factor_list* factBS,
                       ga_factor_list* factGS,
                       ga_factor_list* factCS);
void     gaISchedule  (const int       n,
                       const uint64_t  maxBtot,
                       const uint64_t* maxBind,
                       const uint64_t  maxGtot,
                       const uint64_t* maxGind,
                       uint64_t*       bs,
                       uint64_t*       gs,
                       uint64_t*       cs);


/* End C++ Extern "C" Guard */
#ifdef __cplusplus
}
#endif


/* End Include Guards */
#endif

