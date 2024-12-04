/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_HBATCHED_H
#define MAGMA_HBATCHED_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void
magma_hdisplace_pointers(
    magmaHalf **output_array,
    magmaHalf **input_array, magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);


magma_int_t
magma_hrecommend_cublas_gemm_batched(
    magma_trans_t transa, magma_trans_t transb,
    magma_int_t m, magma_int_t n, magma_int_t k);

magma_int_t
magma_hrecommend_cublas_gemm_stream(
    magma_trans_t transa, magma_trans_t transb,
    magma_int_t m, magma_int_t n, magma_int_t k);
void magma_get_hgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb);
magma_int_t magma_get_htrsm_batched_stop_nb(magma_side_t side, magma_int_t m, magma_int_t n);
magma_int_t magma_get_hgetrf_batched_ntcol(magma_int_t m, magma_int_t n);

void magma_hset_pointer(
    magmaHalf **output_array,
    magmaHalf *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batch_offset,
    magma_int_t batchCount, 
    magma_queue_t queue);

magma_int_t 
magmablas_hgemm_batched(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k, 
    magmaHalf alpha,
    magmaHalf const * const * dAarray, magma_int_t ldda,
    magmaHalf const * const * dBarray, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf **dCarray, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_hgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaHalf alpha,
    magmaHalf const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaHalf const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void magma_check_address_alignment(magmaHalf **A, int n);

void
magma_hgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaHalf alpha,
    magmaHalf const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaHalf const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magma_hgemm_batched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaHalf alpha,
    magmaHalf const * const * dA_array, magma_int_t ldda,
    magmaHalf const * const * dB_array, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_ihamax_batched(
        magma_int_t length,
        magmaHalf **x_array, magma_int_t xi, magma_int_t xj, magma_int_t lda, magma_int_t incx,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t step, magma_int_t gbstep, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_hswap_batched(
    magma_int_t n, magmaHalf **x_array, magma_int_t xi, magma_int_t xj, magma_int_t incx,
    magma_int_t step, magma_int_t** ipiv_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_hscal_hger_batched(
    magma_int_t m, magma_int_t n,
    magmaHalf **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_hcomputecolumn_batched(
    magma_int_t m, magma_int_t paneloffset, magma_int_t step,
    magmaHalf **dA_array,  magma_int_t lda,
    magma_int_t ai, magma_int_t aj,
    magma_int_t **ipiv_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_hgetf2trsm_batched(
    magma_int_t ib, magma_int_t n,
    magmaHalf **dA_array,  magma_int_t j, magma_int_t lda,
    magma_int_t batchCount, magma_queue_t queue);


magma_int_t
magma_hgetf2_batched(
    magma_int_t m, magma_int_t n,
    magmaHalf **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t **ipiv_array,
    magma_int_t **dpivinfo_array,
    magma_int_t *info_array,
    magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_hgetrf_recpanel_batched(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,
    magmaHalf** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue);

magma_int_t
magma_hgetrf_batched(
    magma_int_t m, magma_int_t n,
    magmaHalf **dA_array,
    magma_int_t lda,
    magma_int_t **ipiv_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_hgetf2_fused_batched(
    magma_int_t m, magma_int_t n,
    magmaHalf **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **dipiv_array,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue);

magma_int_t
magma_hgetrf_batched_smallsq_noshfl(
    magma_int_t n,
    magmaHalf** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue );

void
magma_hlaswp_rowparallel_batched(
    magma_int_t n,
    magmaHalf**  input_array, magma_int_t  input_i, magma_int_t  input_j, magma_int_t ldi,
    magmaHalf** output_array, magma_int_t output_i, magma_int_t output_j, magma_int_t ldo,
    magma_int_t k1, magma_int_t k2,
    magma_int_t **pivinfo_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_htrsm_small_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaHalf alpha,
        magmaHalf **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaHalf **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_htrsm_recursive_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaHalf alpha,
        magmaHalf **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaHalf **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_htrsm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t m, magma_int_t n,
        magmaHalf alpha,
        magmaHalf **dA_array, magma_int_t ldda,
        magmaHalf **dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue );

#ifdef __cplusplus
}
#endif


#endif /* MAGMA_HBATCHED_H */
