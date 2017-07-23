#include <check.h>

#include <gpuarray/error.h>
#include <gpuarray/reduction.h>
#include <gpuarray/types.h>

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>


extern void *ctx;

void setup(void);
void teardown(void);


/* Defines */
#define MAXERRPRINT  16
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


/**
 * Test cases.
 */

START_TEST(test_maxandargmax_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on the first and
	 * third dimensions.
	 */
	
	GpuArray gaS0;
	GpuArray gaD0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	float*  pS0 = calloc(1, sizeof(*pS0) * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0) *         dims[1]        );
	size_t* pD1 = calloc(1, sizeof(*pD1) *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */
	
	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0, ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MAXANDARGMAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, &gaD1, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0, sizeof(*pD0)*dims[1], &gaD0));
	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1)*dims[1], &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtD1 = 0;
		float  gtD0 = pS0[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v > gtD0){
					gtD0 = v;
					gtD1 = i*dims[2] + k;
				}
			}
		}
		
		if(gtD0 != pD0[j] || gtD1 != pD1[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %f[%zu] != %f[%zu] UUT @ %zu!\n",
				        __func__, __LINE__, gtD0, gtD1, pD0[j], pD1[j], j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_maxandargmax_idxtranspose){
	/**
	 * We test here the same reduction as test_reduction, except with a
	 * reversed reduxList {2,0} instead of {0,2}. That should lead to a
	 * transposition of the argmax "coordinates" and thus a change in its
	 * "flattened" output version.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]     = {32,50,79};
	size_t prodDims    = dims[0]*dims[1]*dims[2];
	size_t rdxDims[1]  = {50};
	size_t rdxProdDims = rdxDims[0];
	const int reduxList[] = {2,0};

	float*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	float*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);
	size_t* pD1 = calloc(1, sizeof(*pD1) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0, ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MAXANDARGMAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, &gaD1, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0, sizeof(*pD0)*dims[1], &gaD0));
	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1)*dims[1], &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtD1 = 0;
		float  gtD0 = pS0[(0*dims[1] + j)*dims[2] + 0];

		for(k=0;k<dims[2];k++){
			for(i=0;i<dims[0];i++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v > gtD0){
					gtD0 = v;
					gtD1 = k*dims[0] + i;
				}
			}
		}

		if(gtD0 != pD0[j] || gtD1 != pD1[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %f[%zu] != %f[%zu] UUT @ %zu!\n",
				__func__, __LINE__, gtD0, gtD1, pD0[j], pD1[j], j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_maxandargmax_bigdestination){
	/**
	 * We test here a reduction of some random 3D tensor on the first and
	 * third dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j;
	size_t dims[2]  = {2,131072};
	size_t prodDims = dims[0]*dims[1];
	const int reduxList[] = {0};

	float*  pS0 = calloc(1, sizeof(*pS0) * dims[0]*dims[1]);
	float*  pD0 = calloc(1, sizeof(*pD0) *         dims[1]);
	size_t* pD1 = calloc(1, sizeof(*pD1) *         dims[1]);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 2, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0, ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MAXANDARGMAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, &gaD1, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0,    sizeof(*pD0)   *dims[1], &gaD0));
	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1)*dims[1], &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtD1 = 0;
		float  gtD0 = pS0[0*dims[1] + j];

		for(i=0;i<dims[0];i++){
			float v = pS0[i*dims[1] + j];

			if(v > gtD0){
				gtD0 = v;
				gtD1 = i;
			}
		}

		if(gtD0 != pD0[j] || gtD1 != pD1[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %f[%zu] != %f[%zu] UUT @ %zu!\n",
				__func__, __LINE__, gtD0, gtD1, pD0[j], pD1[j], j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_maxandargmax_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	float*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	float*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);
	size_t* pD1 = calloc(1, sizeof(*pD1) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0, ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MAXANDARGMAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, &gaD1, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0, sizeof(*pD0)*rdxProdDims, &gaD0));
	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1)*rdxProdDims, &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					size_t gtD1 = 0;
					float  gtD0 = pS0[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v > gtD0){
										gtD0 = v;
										gtD1 = (((k)*dims[4] + m)*dims[7] + p)*dims[5] + n;
									}
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD0 != pD0[dstIdx] || gtD1 != pD1[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %f[%zu] != %f[%zu] UUT @ %zu!\n",
							__func__, __LINE__, gtD0, gtD1, pD0[dstIdx], pD1[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_maxandargmax_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	size_t gtD1;
	float  gtD0;

	float*  pS0 = calloc(1, sizeof(*pS0) * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0)                          );
	size_t* pD1 = calloc(1, sizeof(*pD1)                          );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0, ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MAXANDARGMAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, &gaD1, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0, sizeof(*pD0), &gaD0));
	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1), &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD1 = 0;
	gtD0 = pS0[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v > gtD0){
					gtD0 = v;
					gtD1 = (i*dims[1] + j)*dims[2] + k;
				}
			}
		}
	}
	if(gtD0 != pD0[0] || gtD1 != pD1[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %f[%zu] != %f[%zu] UUT @ %zu!\n",
			__func__, __LINE__, gtD0, gtD1, pD0[0], pD1[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_minandargmin_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on the first and
	 * third dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	float*  pS0 = calloc(1, sizeof(*pS0) * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0) *         dims[1]        );
	size_t* pD1 = calloc(1, sizeof(*pD1) *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0, ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MINANDARGMIN);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, &gaD1, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0, sizeof(*pD0)*dims[1], &gaD0));
	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1)*dims[1], &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtD1 = 0;
		float  gtD0 = pS0[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v < gtD0){
					gtD0 = v;
					gtD1 = i*dims[2] + k;
				}
			}
		}

		if(gtD0 != pD0[j] || gtD1 != pD1[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %f[%zu] != %f[%zu] UUT @ %zu!\n",
				__func__, __LINE__, gtD0, gtD1, pD0[j], pD1[j], j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_minandargmin_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	float*  pS0    = calloc(1, sizeof(*pS0)    * prodDims);
	float*  pD0    = calloc(1, sizeof(*pD0)    * rdxProdDims);
	size_t* pD1 = calloc(1, sizeof(*pD1) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0, ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MINANDARGMIN);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, &gaD1, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0, sizeof(*pD0)*rdxProdDims, &gaD0));
	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1)*rdxProdDims, &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					size_t gtD1 = 0;
					float  gtD0 = pS0[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v < gtD0){
										gtD0 = v;
										gtD1 = (((k)*dims[4] + m)*dims[7] + p)*dims[5] + n;
									}
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD0 != pD0[dstIdx] || gtD1 != pD1[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %f[%zu] != %f[%zu] UUT @ %zu!\n",
							__func__, __LINE__, gtD0, gtD1, pD0[dstIdx], pD1[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_minandargmin_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	size_t gtD1;
	float  gtD0;

	float*  pS0 = calloc(1, sizeof(*pS0)* dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0)                         );
	size_t* pD1 = calloc(1, sizeof(*pD1)                         );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0, ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MINANDARGMIN);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, &gaD1, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0, sizeof(*pD0), &gaD0));
	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1), &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD1 = 0;
	gtD0 = pS0[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v < gtD0){
					gtD0 = v;
					gtD1 = (i*dims[1] + j)*dims[2] + k;
				}
			}
		}
	}
	if(gtD0 != pD0[0] || gtD1 != pD1[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %f[%zu] != %f[%zu] UUT @ %zu!\n",
			__func__, __LINE__, gtD0, gtD1, pD0[0], pD1[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_argmax_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on the first and
	 * third dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	float*  pS0 = calloc(1, sizeof(*pS0) * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0) *         dims[1]        );
	size_t* pD1 = calloc(1, sizeof(*pD1) *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ARGMAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD1.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, NULL, &gaD1, &gaS0, gaS0.nd-gaD1.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1)*dims[1], &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtD1 = 0;
		float  gtD0 = pS0[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v > gtD0){
					gtD0 = v;
					gtD1 = i*dims[2] + k;
				}
			}
		}

		if(gtD1 != pD1[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT [%zu] != [%zu] UUT @ %zu!\n",
				__func__, __LINE__, gtD1, pD1[j], j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_argmax_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	float*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	float*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);
	size_t* pD1 = calloc(1, sizeof(*pD1) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ARGMAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD1.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, NULL, &gaD1, &gaS0, gaS0.nd-gaD1.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1)*rdxProdDims, &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					size_t gtD1 = 0;
					float  gtD0 = pS0[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v > gtD0){
										gtD0 = v;
										gtD1 = (((k)*dims[4] + m)*dims[7] + p)*dims[5] + n;
									}
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD1 != pD1[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT [%zu] != [%zu] UUT @ %zu!\n",
							__func__, __LINE__, gtD1, pD1[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_argmax_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	size_t gtD1;
	float  gtD0;

	float*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0)                             );
	size_t* pD1 = calloc(1, sizeof(*pD1)                          );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ARGMAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD1.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, NULL, &gaD1, &gaS0, gaS0.nd-gaD1.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1), &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD1 = 0;
	gtD0 = pS0[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v > gtD0){
					gtD0 = v;
					gtD1 = (i*dims[1] + j)*dims[2] + k;
				}
			}
		}
	}
	if(gtD1 != pD1[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT [%zu] != [%zu] UUT @ %zu!\n",
			__func__, __LINE__, gtD1, pD1[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_argmin_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on the first and
	 * third dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	float*  pS0    = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0    = calloc(1, sizeof(*pD0)    *         dims[1]        );
	size_t* pD1 = calloc(1, sizeof(*pD1) *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ARGMIN);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD1.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, NULL, &gaD1, &gaS0, gaS0.nd-gaD1.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1)*dims[1], &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		size_t gtD1 = 0;
		float  gtD0 = pS0[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v < gtD0){
					gtD0 = v;
					gtD1 = i*dims[2] + k;
				}
			}
		}

		if(gtD1 != pD1[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT [%zu] != [%zu] UUT @ %zu!\n",
				__func__, __LINE__, gtD1, pD1[j], j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_argmin_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	float*  pS0    = calloc(1, sizeof(*pS0)    * prodDims);
	float*  pD0    = calloc(1, sizeof(*pD0)    * rdxProdDims);
	size_t* pD1 = calloc(1, sizeof(*pD1) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0,    ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0,    pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ARGMIN);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD1.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, NULL, &gaD1, &gaS0, gaS0.nd-gaD1.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1)*rdxProdDims, &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					size_t gtD1 = 0;
					float  gtD0 = pS0[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v < gtD0){
										gtD0 = v;
										gtD1 = (((k)*dims[4] + m)*dims[7] + p)*dims[5] + n;
									}
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD1 != pD1[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT [%zu] != [%zu] UUT @ %zu!\n",
							__func__, __LINE__, gtD1, pD1[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_argmin_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD1;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	size_t gtD1;
	float  gtD0;

	float*  pS0 = calloc(1, sizeof(*pS0) * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0)                          );
	size_t* pD1 = calloc(1, sizeof(*pD1)                          );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);
	ck_assert_ptr_nonnull(pD1);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD1, ctx, GA_SIZE,  0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD1, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ARGMIN);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD1.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd1type(grAttr, gaD1.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, NULL, &gaD1, &gaS0, gaS0.nd-gaD1.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD1, sizeof(*pD1), &gaD1));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD1 = 0;
	gtD0 = pS0[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v < gtD0){
					gtD0 = v;
					gtD1 = (i*dims[1] + j)*dims[2] + k;
				}
			}
		}
	}
	if(gtD1 != pD1[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT [%zu] != [%zu] UUT @ %zu!\n",
			__func__, __LINE__, gtD1, pD1[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	free(pD1);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD1);
}END_TEST

START_TEST(test_max_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	float*  pS0    = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0    = calloc(1, sizeof(*pD0)    *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * axitialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0,    ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0,    pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0,    -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0, sizeof(*pD0)*dims[1], &gaD0));


	/**
	 * Check that the destaxation tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		float  gtD0 = pS0[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v > gtD0){
					gtD0 = v;
				}
			}
		}

		if(gtD0 != pD0[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu!\n",
				__func__, __LINE__, gtD0, pD0[j], j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_max_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	float*  pS0    = calloc(1, sizeof(*pS0)    * prodDims);
	float*  pD0    = calloc(1, sizeof(*pD0)    * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0,    ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0,    ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0,    pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0,    -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0,    sizeof(*pD0)   *rdxProdDims, &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					float  gtD0 = pS0[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v > gtD0){
										gtD0 = v;
									}
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD0 != pD0[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu!\n",
							__func__, __LINE__, gtD0, pD0[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_max_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	float  gtD0;

	float*  pS0    = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0    = calloc(1, sizeof(*pD0)                             );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * axitialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0,    ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0,    pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0,    -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MAX);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0,    sizeof(*pD0),    &gaD0));


	/**
	 * Check that the destaxation tensors are correct.
	 */

	gtD0 = pS0[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v > gtD0){
					gtD0 = v;
				}
			}
		}
	}
	if(gtD0 != pD0[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_min_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	float*  pS0    = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0    = calloc(1, sizeof(*pD0)    *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0,    ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0,    pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0,    -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MIN);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0,    sizeof(*pD0)   *dims[1], &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		float  gtD0 = pS0[(0*dims[1] + j)*dims[2] + 0];

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v < gtD0){
					gtD0 = v;
				}
			}
		}

		if(gtD0 != pD0[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu!\n",
				__func__, __LINE__, gtD0, pD0[j], j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_min_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	float*  pS0    = calloc(1, sizeof(*pS0)    * prodDims);
	float*  pD0    = calloc(1, sizeof(*pD0)    * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0,    ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0,    ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0,    pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0,    -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MIN);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0,    sizeof(*pD0)   *rdxProdDims, &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					float  gtD0 = pS0[(((((((i)*dims[1] + j)*dims[2] + 0)*dims[3] + l)*dims[4] + 0)*dims[5] + 0)*dims[6] + o)*dims[7] + 0];

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];

									if(v < gtD0){
										gtD0 = v;
									}
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD0 != pD0[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu!\n",
							__func__, __LINE__, gtD0, pD0[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_min_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	float  gtD0;

	float*  pS0    = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0    = calloc(1, sizeof(*pD0)                             );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01();
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty(&gaS0,    ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty(&gaD0,    ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write(&gaS0,    pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0,    -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_MIN);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read(pD0,    sizeof(*pD0),    &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD0 = pS0[0];

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];

				if(v < gtD0){
					gtD0 = v;
				}
			}
		}
	}
	if(gtD0 != pD0[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_sum_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};
	const float TOL = 1e-4;

	float*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0)    *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01()-0.5;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_SUM);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*dims[1], &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		float  gtD0 = 0;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 += v;
			}
		}

		if(fabs(gtD0-pD0[j]) >= TOL){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu (TOL=%f)!\n",
				__func__, __LINE__, gtD0, pD0[j], j, TOL);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_sum_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};
	const float TOL    = 1e-4;

	float*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	float*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01()-0.5;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_SUM);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*rdxProdDims, &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					float gtD0 = 0;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD0 += v;
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(fabs(gtD0-pD0[dstIdx]) >= TOL){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu (TOL=%f)!\n",
							__func__, __LINE__, gtD0, pD0[dstIdx], dstIdx, TOL);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_sum_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	const float TOL = 1e-4;
	float  gtD0;

	float*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0)                             );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01()-0.5;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_SUM);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0), &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD0 = 0;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 += v;
			}
		}
	}
	if(fabs(gtD0-pD0[0]) >= TOL){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu (TOL=%f)!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0, TOL);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_sum_huge){
	/**
	 * We test here a reduction of a huge 1D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i;
	size_t dims[1]  = {100000000};
	size_t prodDims = dims[0];
	const int reduxList[] = {0};
	const float TOL = 1e-2;
	double gtD0;

	float*  pS0 = calloc(1, sizeof(*pS0) * dims[0]);
	float*  pD0 = calloc(1, sizeof(*pD0));

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = pcgRand01()-0.5;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_FLOAT, 1, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_FLOAT, 0, NULL, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_SUM);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0), &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD0 = 0;
	for(i=0;i<dims[0];i++){
		double  v   = pS0[i];
		gtD0 += v;
	}
	if(fabs(gtD0-pD0[0]) >= TOL){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu (TOL=%f)!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0, TOL);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_prod_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};
	const float TOL = 1e-4;

	float*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0)    *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = (pcgRand01()-0.5)*0.1 + 1;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_PROD);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*dims[1], &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		float  gtD0 = 1;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 *= v;
			}
		}

		if(fabs(gtD0-pD0[j]) >= TOL){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu (TOL=%f)!\n",
				__func__, __LINE__, gtD0, pD0[j], j, TOL);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_prod_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};
	const float TOL    = 1e-4;

	float*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	float*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = (pcgRand01()-0.5)*0.1 + 1;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_PROD);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*rdxProdDims, &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					float gtD0 = 1;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD0 *= v;
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(fabs(gtD0-pD0[dstIdx]) >= TOL){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu (TOL=%f)!\n",
							__func__, __LINE__, gtD0, pD0[dstIdx], dstIdx, TOL);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_prod_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	const float TOL = 1e-4;
	float  gtD0;

	float*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0)                             );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = (pcgRand01()-0.5)*0.1 + 1;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_PROD);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0), &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD0 = 1;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 *= v;
			}
		}
	}
	if(fabs(gtD0-pD0[0]) >= TOL){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu (TOL=%f)!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0, TOL);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_prodnz_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};
	const float TOL = 1e-4;

	float*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0)    *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = (pcgRand01()-0.5)*0.1 + 1;
		if(pcgRand01()<0.1){
			pS0[i] = 0;
		}
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_FLOAT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_PRODNZ);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*dims[1], &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		float  gtD0 = 1;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 *= v==0 ? 1 : v;
			}
		}

		if(fabs(gtD0-pD0[j]) >= TOL){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu (TOL=%f)!\n",
				__func__, __LINE__, gtD0, pD0[j], j, TOL);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_prodnz_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};
	const float TOL    = 1e-4;

	float*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	float*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = (pcgRand01()-0.5)*0.1 + 1;
		if(pcgRand01()<0.1){
			pS0[i] = 0;
		}
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_FLOAT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_FLOAT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_PRODNZ);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*rdxProdDims, &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					float gtD0 = 1;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									float v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD0 *= v==0 ? 1 : v;
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(fabs(gtD0-pD0[dstIdx]) >= TOL){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu (TOL=%f)!\n",
							__func__, __LINE__, gtD0, pD0[dstIdx], dstIdx, TOL);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_prodnz_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	const float TOL = 1e-4;
	float  gtD0;

	float*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	float*  pD0 = calloc(1, sizeof(*pD0)                             );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		pS0[i] = (pcgRand01()-0.5)*0.1 + 1;
		if(pcgRand01()<0.1){
			pS0[i] = 0;
		}
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_FLOAT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_FLOAT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_PRODNZ);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0), &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD0 = 1;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				float v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 *= v==0 ? 1 : v;
			}
		}
	}
	if(fabs(gtD0-pD0[0]) >= TOL){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %f != %f UUT @ %zu (TOL=%f)!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0, TOL);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_and_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	uint32_t*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0)    *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-AND, so the bits should be 1 with high
		 * probability.
		 */

		pS0[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_AND);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*dims[1], &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		uint32_t  gtD0 = -1;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 &= v;
			}
		}

		if(gtD0 != pD0[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
				__func__, __LINE__, gtD0, pD0[j], j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_and_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	uint32_t*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-AND, so the bits should be 1 with high
		 * probability.
		 */

		pS0[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_AND);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*rdxProdDims, &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					uint32_t gtD0 = -1;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									uint32_t v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD0 &= v;
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD0 != pD0[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
							__func__, __LINE__, gtD0, pD0[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_and_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	uint32_t  gtD0;

	uint32_t*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0)                             );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-AND, so the bits should be 1 with high
		 * probability.
		 */

		pS0[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] |= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_AND);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0), &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD0 = -1;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 &= v;
			}
		}
	}
	if(gtD0 != pD0[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_or_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	uint32_t*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0)    *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-OR, so the bits should be 0 with high
		 * probability.
		 */

		pS0[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_OR);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*dims[1], &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		uint32_t  gtD0 = 0;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 |= v;
			}
		}

		if(gtD0 != pD0[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
				__func__, __LINE__, gtD0, pD0[j], (size_t)j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_or_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	uint32_t*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-OR, so the bits should be 0 with high
		 * probability.
		 */

		pS0[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_OR);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*rdxProdDims, &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					uint32_t gtD0 = 0;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									uint32_t v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD0 |= v;
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD0 != pD0[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
							__func__, __LINE__, gtD0, pD0[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_or_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	uint32_t  gtD0;

	uint32_t*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0)                             );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-OR, so the bits should be 0 with high
		 * probability.
		 */

		pS0[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
		pS0[i] &= (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_OR);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0), &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD0 = 0;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 |= v;
			}
		}
	}
	if(gtD0 != pD0[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_xor_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	uint32_t*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0)    *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-XOR, so the bits should be 1 with even
		 * probability.
		 */

		pS0[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_XOR);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*dims[1], &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		uint32_t  gtD0 = 0;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 ^= v;
			}
		}

		if(gtD0 != pD0[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
				__func__, __LINE__, gtD0, pD0[j], (size_t)j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_xor_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	uint32_t*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-XOR, so the bits should be 1 with even
		 * probability.
		 */

		pS0[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_XOR);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*rdxProdDims, &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					uint32_t gtD0 = 0;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									uint32_t v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD0 ^= v;
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD0 != pD0[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
							__func__, __LINE__, gtD0, pD0[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_xor_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	uint32_t  gtD0;

	uint32_t*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0)                             );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-XOR, so the bits should be 1 with even
		 * probability.
		 */

		pS0[i]  = (uint32_t)(pcgRand01() * (uint32_t)-1);
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_XOR);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0), &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD0 = 0;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 ^= v;
			}
		}
	}
	if(gtD0 != pD0[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_any_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	uint32_t*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0)    *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ANY, so the values should be 0 with high
		 * probability.
		 */

		pS0[i]  = pcgRand01() < 0.05;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ANY);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*dims[1], &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		uint32_t  gtD0 = 0;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 = gtD0 || v;
			}
		}

		if(gtD0 != pD0[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
				__func__, __LINE__, gtD0, pD0[j], (size_t)j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_any_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	uint32_t*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ANY, so the values should be 0 with high
		 * probability.
		 */

		pS0[i]  = pcgRand01() < 0.05;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ANY);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*rdxProdDims, &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					uint32_t gtD0 = 0;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									uint32_t v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD0 = gtD0 || v;
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD0 != pD0[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
							__func__, __LINE__, gtD0, pD0[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_any_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	uint32_t  gtD0;

	uint32_t*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0)                             );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ANY, so the values should be 0 with high
		 * probability.
		 */

		pS0[i]  = pcgRand01() < 0.05;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ANY);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0), &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD0 = 0;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 = gtD0 || v;
			}
		}
	}
	if(gtD0 != pD0[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_all_reduction){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,2};

	uint32_t*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0)    *         dims[1]        );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ALL, so the values should be non-0 with high
		 * probability.
		 */

		pS0[i]  = pcgRand01() > 0.05;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 1, &dims[1], GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ALL);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*dims[1], &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(j=0;j<dims[1];j++){
		uint32_t  gtD0 = 1;

		for(i=0;i<dims[0];i++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 = gtD0 && v;
			}
		}

		if(gtD0 != pD0[j]){
			errCnt++;
			if(errCnt <= MAXERRPRINT){
				fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
				__func__, __LINE__, gtD0, pD0[j], (size_t)j);
				fflush (stderr);
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_all_veryhighrank){
	/**
	 * Here we test a reduction of a random 8D tensor on four dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0, dstIdx;
	size_t i,j,k,l,m,n,o,p;
	size_t dims   [8]  = {1171,373,2,1,2,1,2,1};
	size_t prodDims    = dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5]*dims[6]*dims[7];
	size_t rdxDims[4]  = {1171,373,1,2};
	size_t rdxProdDims = rdxDims[0]*rdxDims[1]*rdxDims[2]*rdxDims[3];
	const int reduxList[] = {2,4,7,5};

	uint32_t*  pS0 = calloc(1, sizeof(*pS0) * prodDims);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0) * rdxProdDims);

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ALL, so the values should be non-0 with high
		 * probability.
		 */

		pS0[i]  = pcgRand01() > 0.05;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 8, dims,    GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 4, rdxDims, GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ALL);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0)*rdxProdDims, &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(l=0;l<dims[3];l++){
				for(o=0;o<dims[6];o++){
					uint32_t gtD0 = 1;

					for(k=0;k<dims[2];k++){
						for(m=0;m<dims[4];m++){
							for(p=0;p<dims[7];p++){
								for(n=0;n<dims[5];n++){
									uint32_t v = pS0[(((((((i)*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m)*dims[5] + n)*dims[6] + o)*dims[7] + p];
									gtD0 = gtD0 && v;
								}
							}
						}
					}

					dstIdx = (((i)*dims[1] + j)*dims[3] + l)*dims[6] + o;
					if(gtD0 != pD0[dstIdx]){
						errCnt++;
						if(errCnt <= MAXERRPRINT){
							fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
							__func__, __LINE__, gtD0, pD0[dstIdx], dstIdx);
							fflush (stderr);
						}
					}
				}
			}
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);


	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

START_TEST(test_all_alldimsreduced){
	/**
	 * We test here a reduction of some random 3D tensor on all dimensions.
	 */

	GpuArray gaS0;
	GpuArray gaD0;
	GpuReductionAttr* grAttr;
	GpuReduction*     gr;
	size_t errCnt      = 0;
	size_t i,j,k;
	size_t dims[3]  = {32,50,79};
	size_t prodDims = dims[0]*dims[1]*dims[2];
	const int reduxList[] = {0,1,2};
	uint32_t  gtD0;

	uint32_t*  pS0 = calloc(1, sizeof(*pS0)    * dims[0]*dims[1]*dims[2]);
	uint32_t*  pD0 = calloc(1, sizeof(*pD0)                             );

	ck_assert_ptr_nonnull(pS0);
	ck_assert_ptr_nonnull(pD0);


	/**
	 * Initialize source data.
	 */

	pcgSeed(1);
	for(i=0;i<prodDims;i++){
		/**
		 * We are testing logic-ALL, so the values should be non-0 with high
		 * probability.
		 */

		pS0[i]  = pcgRand01() > 0.05;
	}


	/**
	 * Run the kernel.
	 */

	ga_assert_ok(GpuArray_empty (&gaS0, ctx, GA_UINT, 3, &dims[0], GA_C_ORDER));
	ga_assert_ok(GpuArray_empty (&gaD0, ctx, GA_UINT, 0, NULL,     GA_C_ORDER));

	ga_assert_ok(GpuArray_write (&gaS0, pS0, sizeof(*pS0)*prodDims));
	ga_assert_ok(GpuArray_memset(&gaD0, -1));  /* 0xFFFFFFFF is a qNaN. */

	GpuReductionAttr_new(&grAttr, GpuArray_context(&gaS0));
	ck_assert_ptr_nonnull(grAttr);
	GpuReductionAttr_setop    (grAttr, GA_REDUCE_ALL);
	GpuReductionAttr_setdims  (grAttr, gaS0.nd, gaD0.nd);
	GpuReductionAttr_sets0type(grAttr, gaS0.typecode);
	GpuReductionAttr_setd0type(grAttr, gaD0.typecode);
	GpuReduction_new(&gr, grAttr);
	ck_assert_ptr_nonnull(gr);
	ga_assert_ok(GpuReduction_call(gr, &gaD0, NULL, &gaS0, gaS0.nd-gaD0.nd, reduxList, 0));
	GpuReduction_free(gr);
	GpuReductionAttr_free(grAttr);

	ga_assert_ok(GpuArray_read  (pD0,   sizeof(*pD0), &gaD0));


	/**
	 * Check that the destination tensors are correct.
	 */

	gtD0 = 1;

	for(i=0;i<dims[0];i++){
		for(j=0;j<dims[1];j++){
			for(k=0;k<dims[2];k++){
				uint32_t v = pS0[(i*dims[1] + j)*dims[2] + k];
				gtD0 = gtD0 && v;
			}
		}
	}
	if(gtD0 != pD0[0]){
		errCnt++;
		if(errCnt <= MAXERRPRINT){
			fprintf(stderr, "%s:%d: Mismatch GT %u != %u UUT @ %zu!\n",
			__func__, __LINE__, gtD0, pD0[0], (size_t)0);
			fflush (stderr);
		}
	}
	ck_assert_msg(errCnt == 0, "%zu mismatches!", errCnt);

	/**
	 * Deallocate.
	 */

	free(pS0);
	free(pD0);
	GpuArray_clear(&gaS0);
	GpuArray_clear(&gaD0);
}END_TEST

Suite *get_suite(void) {
	Suite *s  = suite_create("reduction");
	TCase *tc = tcase_create("basic");
	tcase_add_checked_fixture(tc, setup, teardown);
	tcase_set_timeout(tc, 120.0);

	tcase_add_test(tc, test_maxandargmax_reduction);
	tcase_add_test(tc, test_maxandargmax_idxtranspose);
	tcase_add_test(tc, test_maxandargmax_bigdestination);
	tcase_add_test(tc, test_maxandargmax_veryhighrank);
	tcase_add_test(tc, test_maxandargmax_alldimsreduced);

	tcase_add_test(tc, test_minandargmin_reduction);
	tcase_add_test(tc, test_minandargmin_veryhighrank);
	tcase_add_test(tc, test_minandargmin_alldimsreduced);

	tcase_add_test(tc, test_argmax_reduction);
	tcase_add_test(tc, test_argmax_veryhighrank);
	tcase_add_test(tc, test_argmax_alldimsreduced);

	tcase_add_test(tc, test_argmin_reduction);
	tcase_add_test(tc, test_argmin_veryhighrank);
	tcase_add_test(tc, test_argmin_alldimsreduced);

	tcase_add_test(tc, test_max_reduction);
	tcase_add_test(tc, test_max_veryhighrank);
	tcase_add_test(tc, test_max_alldimsreduced);

	tcase_add_test(tc, test_min_reduction);
	tcase_add_test(tc, test_min_veryhighrank);
	tcase_add_test(tc, test_min_alldimsreduced);

	tcase_add_test(tc, test_sum_reduction);
	tcase_add_test(tc, test_sum_veryhighrank);
	tcase_add_test(tc, test_sum_alldimsreduced);
	tcase_add_test(tc, test_sum_huge);

	tcase_add_test(tc, test_prod_reduction);
	tcase_add_test(tc, test_prod_veryhighrank);
	tcase_add_test(tc, test_prod_alldimsreduced);

	tcase_add_test(tc, test_prodnz_reduction);
	tcase_add_test(tc, test_prodnz_veryhighrank);
	tcase_add_test(tc, test_prodnz_alldimsreduced);

	tcase_add_test(tc, test_and_reduction);
	tcase_add_test(tc, test_and_veryhighrank);
	tcase_add_test(tc, test_and_alldimsreduced);

	tcase_add_test(tc, test_or_reduction);
	tcase_add_test(tc, test_or_veryhighrank);
	tcase_add_test(tc, test_or_alldimsreduced);

	tcase_add_test(tc, test_xor_reduction);
	tcase_add_test(tc, test_xor_veryhighrank);
	tcase_add_test(tc, test_xor_alldimsreduced);

	tcase_add_test(tc, test_any_reduction);
	tcase_add_test(tc, test_any_veryhighrank);
	tcase_add_test(tc, test_any_alldimsreduced);

	tcase_add_test(tc, test_all_reduction);
	tcase_add_test(tc, test_all_veryhighrank);
	tcase_add_test(tc, test_all_alldimsreduced);

	suite_add_tcase(s, tc);
	return s;
}

