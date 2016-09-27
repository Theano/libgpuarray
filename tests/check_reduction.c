#include <check.h>

#include <gpuarray/buffer.h>
#include <gpuarray/array.h>
#include <gpuarray/error.h>
#include <gpuarray/types.h>

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>


extern void *ctx;

void setup(void);
void teardown(void);


/* Defines */
#define ga_assert_ok(e) ck_assert_int_eq(e, GA_NO_ERROR)




/**
 * PRNG based on PCG XSH RR 64/32 (LCG)
 * 
 * Used to generate random data for the kernel tests.
 */

/* Forward Declarations */
static       uint32_t pcgRor32 (uint32_t x, uint32_t n);
static       void     pcgSeed  (uint64_t seed);
static       uint32_t pcgRand  (void);
static       double   pcgRand01(void);
/* Definitions */
static       uint64_t pcgS =                   1;/* State */
static const uint64_t pcgM = 6364136223846793005;/* Multiplier */
static const uint64_t pcgA = 1442695040888963407;/* Addend */
static       uint32_t pcgRor32 (uint32_t x, uint32_t n){
	return (n &= 0x1F) ? x>>n | x<<(32-n) : x;
}
static       void     pcgSeed  (uint64_t seed){
	pcgS = seed;
}
static       uint32_t pcgRand  (void){
	pcgS = pcgS*pcgM + pcgA;
	
	/**
	 * PCG does something akin to an unbalanced Feistel round to blind the LCG
	 * state:
	 * 
	 * The rightmost 59 bits are involved in an xorshift by 18.
	 * The leftmost   5 bits select a rotation of the 32 bits 58:27.
	 */
	
	return pcgRor32((pcgS^(pcgS>>18))>>27, pcgS>>59);
}
static       double   pcgRand01(void){
	uint64_t u = pcgRand(), l = pcgRand();
	uint64_t x = u<<32 | l;
	return x /18446744073709551616.0;
}



START_TEST(test_reduction){
	pcgSeed(1);
	
	/**
	 * We test here a reduction of some random 3D tensor on the first and
	 * third dimensions.
	 */
	
	size_t i,j,k;
	size_t dims[3] = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {0,2};
	
	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMax    = calloc(1, sizeof(*pMax)    *         dims[1]        );
	size_t* pArgmax = calloc(1, sizeof(*pArgmax) *         dims[1]        );
	
	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);
	ck_assert_ptr_ne(pArgmax, NULL);
	
	
	/**
	 * Initialize source data.
	 */
	
	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}
	
	
	/**
	 * Run the kernel.
	 */
	
	GpuArray gaSrc;
	GpuArray gaMax;
	GpuArray gaArgmax;
	
	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMax,    ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmax, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));
	
	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMax,    -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaArgmax, -1));
	
	ga_assert_ok(GpuArray_maxandargmax(&gaMax, &gaArgmax, &gaSrc, 2, reduxList));
	
	ga_assert_ok(GpuArray_read(pMax,    sizeof(*pMax)   *dims[1], &gaMax));
	ga_assert_ok(GpuArray_read(pArgmax, sizeof(*pArgmax)*dims[1], &gaArgmax));
	
	
	/**
	 * Check that the destination tensors are correct.
	 */
	
	for(j=0;j<dims[1];j++){
		size_t gtArgmax = 0;
		float  gtMax    = pSrc[(0*dims[1] + j)*dims[2] + 0];
		
		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];
				
				if(v > gtMax){
					gtMax    = v;
					gtArgmax = i*dims[2] + k;
				}
			}
		}
		
		ck_assert_msg(gtMax    == pMax[j],    "Max value mismatch!");
		ck_assert_msg(gtArgmax == pArgmax[j], "Argmax value mismatch!");
	}
}END_TEST

START_TEST(test_idxtranspose){
	pcgSeed(1);
	
	/**
	 * We test here the same reduction as test_reduction, except with a
	 * reversed reduxList {2,0} instead of {0,2}. That should lead to a
	 * transposition of the argmax "coordinates" and thus a change in its
	 * "flattened" output version.
	 */
	
	size_t i,j,k;
	size_t dims[3] = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const unsigned reduxList[] = {2,0};
	
	float*  pSrc    = calloc(1, sizeof(*pSrc)    * dims[0]*dims[1]*dims[2]);
	float*  pMax    = calloc(1, sizeof(*pMax)    *         dims[1]        );
	size_t* pArgmax = calloc(1, sizeof(*pArgmax) *         dims[1]        );
	
	ck_assert_ptr_ne(pSrc,    NULL);
	ck_assert_ptr_ne(pMax,    NULL);
	ck_assert_ptr_ne(pArgmax, NULL);
	
	
	/**
	 * Initialize source data.
	 */
	
	for(i=0;i<prodDims;i++){
		pSrc[i] = pcgRand01();
	}
	
	
	/**
	 * Run the kernel.
	 */
	
	GpuArray gaSrc;
	GpuArray gaMax;
	GpuArray gaArgmax;
	
	ga_assert_ok(GpuArray_empty(&gaSrc,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaMax,    ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaArgmax, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));
	
	ga_assert_ok(GpuArray_write(&gaSrc,    pSrc, sizeof(*pSrc)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaMax,    -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaArgmax, -1));
	
	ga_assert_ok(GpuArray_maxandargmax(&gaMax, &gaArgmax, &gaSrc, 2, reduxList));
	
	ga_assert_ok(GpuArray_read(pMax,    sizeof(*pMax)   *dims[1], &gaMax));
	ga_assert_ok(GpuArray_read(pArgmax, sizeof(*pArgmax)*dims[1], &gaArgmax));
	
	
	/**
	 * Check that the destination tensors are correct.
	 */
	
	for(j=0;j<dims[1];j++){
		size_t gtArgmax = 0;
		float  gtMax    = pSrc[(0*dims[1] + j)*dims[2] + 0];
		
		for(k=0;k<dims[2];k++){
			for(i=0;i<dims[0];i++){
				float v = pSrc[(i*dims[1] + j)*dims[2] + k];
				
				if(v > gtMax){
					gtMax    = v;
					gtArgmax = k*dims[0] + i;
				}
			}
		}
		
		ck_assert_msg(gtMax    == pMax[j],    "Max value mismatch!");
		ck_assert_msg(gtArgmax == pArgmax[j], "Argmax value mismatch!");
	}
}END_TEST

Suite *get_suite(void) {
	Suite *s  = suite_create("reduction");
	TCase *tc = tcase_create("basic");
	tcase_add_checked_fixture(tc, setup, teardown);
	tcase_set_timeout(tc, 8.0);
	
	tcase_add_test(tc, test_reduction);
	tcase_add_test(tc, test_idxtranspose);
	
	suite_add_tcase(s, tc);
	return s;
}

