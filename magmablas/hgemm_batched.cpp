/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Wang Yihan

*/
#include "magma_internal.h"
#include "commonblas_s.h"

void
magma_hgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaHalf alpha,
    magmaHalf const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaHalf const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t use_cublas  = magma_hrecommend_cublas_gemm_batched(transA, transB, m, n, k);
    magma_int_t zero_offset = (Ai == 0 && Aj == 0 && Bi == 0 && Bj == 0 && Ci == 0 && Cj == 0);
    if(use_cublas){
        if(zero_offset){
            cublasHgemmBatched(
                    queue->cublas_handle(), cublas_trans_const(transA), cublas_trans_const(transB),
                    int(m), int(n), int(k),
                    (magmaHalf*)&alpha, (const magmaHalf**)dA_array, int(ldda),
                            (const magmaHalf**)dB_array, int(lddb),
                    (magmaHalf*)&beta,  (magmaHalf**)dC_array, int(lddc), int(batchCount) );
        }
        else{
            magmaHalf** dAarray = (magmaHalf**)queue->get_dAarray();
            magmaHalf** dBarray = (magmaHalf**)queue->get_dBarray();
            magmaHalf** dCarray = (magmaHalf**)queue->get_dCarray();
            magma_int_t max_batchCount   = queue->get_maxBatch();
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
                magma_int_t batch = min(max_batchCount, batchCount-i);
                magma_hdisplace_pointers(dAarray, (magmaHalf**)dA_array + i, ldda, Ai, Aj, batch, queue);
                magma_hdisplace_pointers(dBarray, (magmaHalf**)dB_array + i, lddb, Bi, Bj, batch, queue);
                magma_hdisplace_pointers(dCarray, (magmaHalf**)dC_array + i, lddc, Ci, Cj, batch, queue);
                cublasHgemmBatched(
                        queue->cublas_handle(), cublas_trans_const(transA), cublas_trans_const(transB),
                        int(m), int(n), int(k),
                        (magmaHalf*)&alpha, (const magmaHalf**)dAarray, int(ldda),
                                (const magmaHalf**)dBarray, int(lddb),
                        (magmaHalf*)&beta,  (magmaHalf**)dCarray, int(lddc), int(batch) );
            }
        }
    }
    else{
        magmablas_hgemm_batched_core(
            transA, transB,
            m, n, k,
            alpha, dA_array, Ai, Aj, ldda,
                   dB_array, Bi, Bj, lddb,
            beta,  dC_array, Ci, Cj, lddc,
            batchCount, queue);
    }
}

/******************************************************************************/
extern "C" void
magma_hgemm_batched( magma_trans_t transA, magma_trans_t transB,
                     magma_int_t m, magma_int_t n, magma_int_t k,
                     magmaHalf alpha,
                     magmaHalf const * const * dA_array, magma_int_t ldda,
                     magmaHalf const * const * dB_array, magma_int_t lddb,
                     magmaHalf beta,
                     magmaHalf **dC_array, magma_int_t lddc,
                     magma_int_t batchCount, magma_queue_t queue )
{
    magma_hgemm_batched_core(
            transA, transB, m, n, k,
            alpha, dA_array, 0, 0, ldda,
                   dB_array, 0, 0, lddb,
            beta,  dC_array, 0, 0, lddc,
            batchCount, queue );
}
