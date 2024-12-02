/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Wang Yihan
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "magma_types.h"

#define PRECISION_h
#include "hgemm_template_kernel_batched.cuh"
#include "./gemm_config/hgemm_param.h"
#define version(v) NN_V_ ## v

extern "C" magma_int_t
magmablas_hgemm_batched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaHalf alpha,
    magmaHalf const * const * dAarray, magma_int_t ldda,
    magmaHalf const * const * dBarray, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf **dCarray, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

#ifdef MAGMA_HAVE_HIP
     /* for now, fall back on hipblas */
     hipblasHgemmBatched(
         queue->hipblas_handle(),
        hipblas_trans_const(transA), hipblas_trans_const(transB),
        int(m), int(n), int(k), (const hipblasHalf*)&alpha,
        (const hipblasHalf**)dAarray, int(ldda),
        (const hipblasHalf**)dBarray, int(lddb),
		(const hipblasHalf*)&beta,  (hipblasHalf**)dCarray, int(lddc), int(batchCount) );
#else

    magma_int_t arch = magma_getdevice_arch();
    if(arch < 700) {
        printf("%s: architecture %lld is not supported\n", __func__, (long long)arch);
        return -14;
    }

#if CUDA_VERSION >= 9000
    magma_int_t shape = 0;
    if      (transA == MagmaNoTrans   && transB == MagmaNoTrans)   { shape = 0; } // nn
    else if (transA == MagmaNoTrans   && transB == MagmaTrans)     { shape = 1; } // nt
    else if (transA == MagmaNoTrans   && transB == MagmaConjTrans) { shape = 2; } // nc
    else if (transA == MagmaTrans     && transB == MagmaNoTrans)   { shape = 3; } // tn
    else if (transA == MagmaTrans     && transB == MagmaTrans)     { shape = 4; } // tt
    else if (transA == MagmaTrans     && transB == MagmaConjTrans) { shape = 5; } // tc
    else if (transA == MagmaConjTrans && transB == MagmaNoTrans)   { shape = 6; } // cn
    else if (transA == MagmaConjTrans && transB == MagmaTrans)     { shape = 7; } // ct
    else if (transA == MagmaConjTrans && transB == MagmaConjTrans) { shape = 8; } // cc

    switch(shape){
        case 0:    // nn
        {
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_nn<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nn<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nn<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nn<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nn<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nn<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nn<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nn<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nn<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_nn<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nn<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nn<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nn<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nn<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nn<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nn<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nn<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
        }
        break;
        case 1:    // nt
        case 2:    // nc
        {
            // TODO: tune for nt case (now using same tuning as nn)
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_nt<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nt<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nt<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nt<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nt<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nt<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nt<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nt<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nt<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_nt<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nt<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nt<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nt<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nt<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nt<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nt<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nt<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
        }
        break;
        case 3:    // tn
        case 6:    // cn
        {
            // TODO: tune for nt case (now using same tuning as nn)
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_tn<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tn<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tn<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tn<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tn<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tn<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tn<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tn<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tn<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_tn<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tn<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tn<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tn<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tn<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tn<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tn<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tn<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
        }
        break;
        case 4:    // tt
        case 5:    // tc
        case 7:    // ct
        case 8:    // cc
        {
            // TODO: tune for nt case (now using same tuning as nn)
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_tt<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tt<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tt<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tt<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tt<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tt<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tt<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tt<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tt<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_tt<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tt<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tt<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tt<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tt<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tt<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tt<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tt<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, 0, 0, 0, 0, 0, 0, batchCount, queue );
                }
            }
        }
        break;
        default:; // propose something
    }
#endif

#endif    // MAGMA_HAVE_HIP
    return 0;
}

/***************************************************************************//**
    Purpose
    -------
    SGEMM performs one of the matrix-matrix operations
    
        C = alpha*op( A )*op( B ) + beta*C,
    
    where op( X ) is one of
    
        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,
    
    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
    
    Parameters
    ----------
    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A**T.
      -     = MagmaConjTrans:  op( A ) = A**H.
    
    @param[in]
    transB  magma_trans_t.
            On entry, transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( B ) = B.
      -     = MagmaTrans:      op( B ) = B**T.
      -     = MagmaConjTrans:  op( B ) = B**H.
    
    @param[in]
    m       INTEGER.
            On entry,  M  specifies  the number  of rows  of the  matrix
            op( A )  and of the  matrix C.  M  must  be at least  zero.
    
    @param[in]
    n       INTEGER.
            On entry,  N  specifies the number  of columns of the matrix
            op( B ) and the number of columns of the matrix C. N must be
            at least zero.
    
    @param[in]
    k       INTEGER.
            On entry,  K  specifies  the number of columns of the matrix
            op( A ) and the number of rows of the matrix op( B ). K must
            be at least  zero.
    
    @param[in]
    alpha   REAL
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA_array      Array of pointers, dimension (batchCount). 
             Each is a REAL array A of DIMENSION ( ldda, ka ), where ka is
             k  when  transA = MagmaNoTrans,  and is  m  otherwise.
             Before entry with  transA = MagmaNoTrans,  the leading  m by k
             part of the array A must contain the matrix A, otherwise
             the leading  k by m  part of the array A must contain  the
             matrix A.
    
    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of each array A as 
            declared in the calling (sub) program. When  transA = MagmaNoTrans then
            ldda must be at least  max( 1, m ), otherwise  ldda must be at
            least  max( 1, k ).
    
    @param[in]
    dB_array      Array of pointers, dimension (batchCount).
             Each is a REAL array B of DIMENSION ( lddb, kb ), where kb is
             n  when  transB = MagmaNoTrans,  and is  k  otherwise.
             Before entry with  transB = MagmaNoTrans,  the leading  k by n
             part of the array B must contain the matrix B, otherwise
             the leading  n by k  part of the array B must contain  the
             matrix B.
    
    @param[in]
    lddb    INTEGER.
            On entry, lddb specifies the first dimension of each array B as 
            declared in the calling (sub) program. When  transB = MagmaNoTrans then
            lddb must be at least  max( 1, k ), otherwise  lddb must be at
            least  max( 1, n ).
    
    @param[in]
    beta    REAL.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then C need not be set on input.
    
    @param[in,out]
    dC_array      Array of pointers, dimension (batchCount).
             REAL array of DIMENSION ( lddc, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, each array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).
    
    @param[in]
    lddc    INTEGER.
            On entry, lddc specifies the first dimension of each array C as declared
            in  the  calling  (sub)  program.   lddc  must  be  at  least
            max( 1, m ).
    
    @param[in]
    Ai   INTEGER
            Row offset for all 'A' matrices.
    
    @param[in]
    Aj   INTEGER
            Column offset for all 'A' matrices.
    
    @param[in]
    Bi   INTEGER
            Row offset for all 'B' matrices.
    
    @param[in]
    Bj   INTEGER
            Column offset for all 'B' matrices.
    
    @param[in]
    Ci   INTEGER
            Row offset for all 'C' matrices.
    
    @param[in]
    Cj   INTEGER
            Column offset for all 'C' matrices.
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemm_batched
*******************************************************************************/
void
magmablas_hgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaHalf alpha,
    magmaHalf const * const * dAarray, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaHalf const * const * dBarray, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf **dCarray, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        printf("arch < 200 not supported \n"); // TODO call cublas
        return;
    }
    
    if ( m <= 0 || n <= 0 || k <= 0 )
        return;
#if 0
    // special case for small square matrices 
    if( m == n && n == k && m <= magma_get_sgemm_batched_smallsq_limit(m) ){
        magmablas_hgemm_batched_smallsq(
                transA, transB, 
                m, n, k, 
                alpha, dAarray, Ai, Aj, ldda, 
                       dBarray, Bi, Bj, lddb, 
                beta,  dCarray, Ci, Cj, lddc, batchCount, queue );
        return;
    }
#endif
    #if CUDA_VERSION >= 9000
    magma_int_t shape = 0;
    if      (transA == MagmaNoTrans   && transB == MagmaNoTrans)   { shape = 0; } // nn
    else if (transA == MagmaNoTrans   && transB == MagmaTrans)     { shape = 1; } // nt
    else if (transA == MagmaNoTrans   && transB == MagmaConjTrans) { shape = 2; } // nc
    else if (transA == MagmaTrans     && transB == MagmaNoTrans)   { shape = 3; } // tn
    else if (transA == MagmaTrans     && transB == MagmaTrans)     { shape = 4; } // tt
    else if (transA == MagmaTrans     && transB == MagmaConjTrans) { shape = 5; } // tc
    else if (transA == MagmaConjTrans && transB == MagmaNoTrans)   { shape = 6; } // cn
    else if (transA == MagmaConjTrans && transB == MagmaTrans)     { shape = 7; } // ct
    else if (transA == MagmaConjTrans && transB == MagmaConjTrans) { shape = 8; } // cc

    switch(shape){
        case 0:    // nn
        {
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_nn<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nn<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nn<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nn<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nn<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nn<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nn<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nn<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nn<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_nn<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nn<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nn<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nn<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nn<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nn<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nn<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nn<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
            }
        }
        break;
        case 1:    // nt
        case 2:    // nc
        {
            // TODO: tune for nt case (now using same tuning as nn)
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_nt<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nt<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nt<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nt<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nt<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nt<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nt<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nt<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nt<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_nt<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_nt<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_nt<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_nt<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_nt<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_nt<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_nt<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_nt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_nt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_nt<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_nt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_nt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_nt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_nt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_nt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else {
                    hgemm_template_batched_nt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
            }
        }
        break;
        case 3:    // tn
        case 6:    // cn
        {
            // TODO: tune for nt case (now using same tuning as nn)
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_tn<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tn<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tn<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tn<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tn<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tn<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tn<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tn<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tn<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tn<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tn<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tn<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_tn<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tn<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tn<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tn<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tn<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tn<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tn<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tn<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tn<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tn<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tn<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tn<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
            }
        }
        break;
        case 4:    // tt
        case 5:    // tc
        case 7:    // ct
        case 8:    // cc
        {
            // TODO: tune for nt case (now using same tuning as nn)
            if(m == n && k <= 16){
                if(m <= 16) {
                    hgemm_template_batched_tt<magmaHalf, version(455)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tt<magmaHalf, version(3957)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tt<magmaHalf, version(4090)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tt<magmaHalf, version(2208)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tt<magmaHalf, version(5157)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tt<magmaHalf, version(4409)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tt<magmaHalf, version(1092)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tt<magmaHalf, version(5354)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tt<magmaHalf, version(1334)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tt<magmaHalf, version(2325)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tt<magmaHalf, version(2211)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tt<magmaHalf, version(4009)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
            }
            else{    // tuning here is based on square sizes
                if(m <= 16) {
                    hgemm_template_batched_tt<magmaHalf, version(4)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 32){
                    hgemm_template_batched_tt<magmaHalf, version(4019)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 48){
                    hgemm_template_batched_tt<magmaHalf, version(1109)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 64){
                    hgemm_template_batched_tt<magmaHalf, version(4143)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 80){
                    hgemm_template_batched_tt<magmaHalf, version(2014)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 96){
                    hgemm_template_batched_tt<magmaHalf, version(1110)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 112){
                    hgemm_template_batched_tt<magmaHalf, version(3318)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 128){
                    hgemm_template_batched_tt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 144){
                    hgemm_template_batched_tt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 160){
                    hgemm_template_batched_tt<magmaHalf, version(2210)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 176){
                    hgemm_template_batched_tt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 192){
                    hgemm_template_batched_tt<magmaHalf, version(1286)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 208){
                    hgemm_template_batched_tt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 224){
                    hgemm_template_batched_tt<magmaHalf, version(2339)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else if(m <= 240){
                    hgemm_template_batched_tt<magmaHalf, version(2112)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
                else {
                    hgemm_template_batched_tt<magmaHalf, version(4428)>
                    (m, n, k, dAarray, ldda, dBarray, lddb, dCarray, lddc, alpha, beta, Ai, Aj, Bi, Bj, Ci, Cj, batchCount, queue );
                }
            }
        }
        break;
        default:; // propose something
    }
#endif
}

