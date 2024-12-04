/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

*/

#ifndef MAGMABLAS_H_H
#define MAGMABLAS_H_H

#include "magma_types.h"
#include "magma_copy.h"

// Half precision routines are available for C++ compilers only
#ifdef __cplusplus
extern "C" {

// =============================================================================
// copying sub-matrices (contiguous columns)

/// Type-safe version of magma_setmatrix() for magmaHalf arrays.
/// @ingroup magma_setmatrix
#define magma_hsetmatrix(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_hsetmatrix_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getmatrix() for magmaHalf arrays.
/// @ingroup magma_getmatrix
#define magma_hgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_hgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copymatrix() for magmaHalf arrays.
/// @ingroup magma_copymatrix
#define magma_hcopymatrix(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_hcopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_setmatrix_async() for magmaHalf arrays.
/// @ingroup magma_setmatrix
#define magma_hsetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        magma_hsetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getmatrix_async() for magmaHalf arrays.
/// @ingroup magma_getmatrix
#define magma_hgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        magma_hgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copymatrix_async() for magmaHalf arrays.
/// @ingroup magma_copymatrix
#define magma_hcopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_hcopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_hsetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaHalf const    *hA_src, magma_int_t lda,
    magmaHalf_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setmatrix_internal( m, n, sizeof(magmaHalf),
                              hA_src, lda,
                              dB_dst, lddb, queue,
                              func, file, line );
}

static inline void
magma_hgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaHalf_const_ptr dA_src, magma_int_t ldda,
    magmaHalf          *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getmatrix_internal( m, n, sizeof(magmaHalf),
                              dA_src, ldda,
                              hB_dst, ldb, queue,
                              func, file, line );
}

static inline void
magma_hcopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaHalf_const_ptr dA_src, magma_int_t ldda,
    magmaHalf_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copymatrix_internal( m, n, sizeof(magmaHalf),
                               dA_src, ldda,
                               dB_dst, lddb, queue,
                               func, file, line );
}

static inline void
magma_hsetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaHalf const    *hA_src, magma_int_t lda,
    magmaHalf_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setmatrix_async_internal( m, n, sizeof(magmaHalf),
                                    hA_src, lda,
                                    dB_dst, lddb, queue,
                                    func, file, line );
}

static inline void
magma_hgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaHalf_const_ptr dA_src, magma_int_t ldda,
    magmaHalf          *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getmatrix_async_internal( m, n, sizeof(magmaHalf),
                                    dA_src, ldda,
                                    hB_dst, ldb, queue,
                                    func, file, line );
}

static inline void
magma_hcopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaHalf_const_ptr dA_src, magma_int_t ldda,
    magmaHalf_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copymatrix_async_internal( m, n, sizeof(magmaHalf),
                                     dA_src, ldda,
                                     dB_dst, lddb, queue,
                                     func, file, line );
}

// =============================================================================
// conversion routines
void
magmablas_slag2h(
    magma_int_t m, magma_int_t n,
    float const * dA, magma_int_t lda,
    magmaHalf* dHA, magma_int_t ldha,
    magma_int_t *info, magma_queue_t queue);

void
magmablas_hlag2s(
    magma_int_t m, magma_int_t n,
    magmaHalf_const_ptr dA, magma_int_t lda,
    float             *dSA, magma_int_t ldsa,
    magma_queue_t queue );

void
magmablas_slag2h_batched(
    magma_int_t m, magma_int_t n,
    float const * const * dAarray, magma_int_t lda,
    magmaHalf** dHAarray, magma_int_t ldha,
    magma_int_t *info_array, magma_int_t batchCount, 
    magma_queue_t queue);

void
magmablas_hlag2s_batched(
    magma_int_t m, magma_int_t n,
    magmaHalf const * const * dAarray, magma_int_t lda,
    float               **dSAarray, magma_int_t ldsa,
    magma_int_t batchCount, magma_queue_t queue );

// =============================================================================
// Level 3 BLAS (alphabetical order)
void
magma_hgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaHalf alpha,
    magmaHalf_const_ptr dA, magma_int_t ldda,
    magmaHalf_const_ptr dB, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_hgemmx(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaHalf_const_ptr dA, magma_int_t ldda,
    magmaHalf_const_ptr dB, magma_int_t lddb,
    float beta,
    float *dC, magma_int_t lddc,
    magma_queue_t queue );
}

#endif
#endif // MAGMABLAS_H_H
