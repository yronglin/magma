/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

   @precisions normal z -> s d c
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

//#define DBG
//#define ZGETF2_BATCHED_FUSED_UPDATE

/***************************************************************************//**
    Purpose
    -------
    ZGETF2 computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ai      INTEGER
            Row offset for A.

    @param[in]
    aj      INTEGER
            Column offset for A.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    gbstep  INTEGER
            internal use.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    this is an internal routine that might have many assumption.


    @ingroup magma_getf2_batched
*******************************************************************************/
static magma_int_t
magma_zgetf2_batched_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **ipiv_array, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount,
    magma_queue_t queue)
{
    #define dAarray(i, j)  dA_array, i, j

    magma_int_t arginfo = 0;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magma_int_t nb = BATF2_NB;

    magma_int_t min_mn = min(m, n);
    magma_int_t gbj, panelj, step, ib;

    for( panelj=0; panelj < min_mn; panelj += nb)
    {

        ib = min(nb, min_mn-panelj);

        for (step=0; step < ib; step++) {
            gbj = panelj+step;
            if ((m-panelj) > MAX_NTHREADS) {
                // find the max of the column
                arginfo = magma_izamax_batched(m-gbj, dA_array, ai+gbj, aj+gbj, ldda, 1, ipiv_array, ai+gbj, gbj, gbstep, info_array, batchCount, queue);
                if (arginfo != 0 ) return arginfo;
                // Apply the interchange to columns 1:N. swap the whole row
                arginfo = magma_zswap_batched(n, dA_array, ai, aj, ldda, gbj, ipiv_array, batchCount, queue);
                if (arginfo != 0 ) return arginfo;
                // Compute elements J+1:M of J-th column.
                if (gbj < m) {
                    arginfo = magma_zscal_zgeru_batched( m-gbj, ib-step, dA_array, ai+gbj, aj+gbj, ldda, info_array, gbj, gbstep, batchCount, queue );
                    if (arginfo != 0 ) return arginfo;
                }
            }
            else {
                //printf("running --- shared version\n");
                //printf("calling zcomputecolumn, panel ai=%2d, panelj = %d, step = %2d\n", ai, panelj, step);
                arginfo = magma_zcomputecolumn_batched(m-panelj, panelj, step, dA_array, ai, aj, ldda, ipiv_array, info_array, gbstep, batchCount, queue);
                if (arginfo != 0 ) return arginfo;
                // Apply the interchange to columns 1:N. swap the whole row

                //printf("calling zswap batched, ai = %d, step = %d\n", ai, gbj);
                arginfo = magma_zswap_batched(n, dA_array, ai, aj, ldda, gbj, ipiv_array, batchCount, queue);
                if (arginfo != 0 ) return arginfo;
            }
        }


        if ( (n-panelj-ib) > 0) {
            // continue the update of the selected ib row column panelj+ib:n(TRSM)
            magma_zgetf2trsm_batched(ib, n-panelj-ib, dA_array, ai+panelj, ldda, batchCount, queue);
            // do the blocked DGER = DGEMM for the remaining panelj+ib:n columns
            //magma_zdisplace_pointers(dW0_displ, dA_array, ldda, ib+panelj, panelj, batchCount, queue);
            //magma_zdisplace_pointers(dW1_displ, dA_array, ldda, panelj, ib+panelj, batchCount, queue);
            //magma_zdisplace_pointers(dW2_displ, dA_array, ldda, ib+panelj, ib+panelj, batchCount, queue);

            magma_zgemm_batched_core( MagmaNoTrans, MagmaNoTrans, m-(panelj+ib), n-(panelj+ib), ib,
                                 c_neg_one, dAarray(ai+ib+panelj, aj+panelj   ), ldda,
                                            dAarray(ai+panelj   , aj+ib+panelj), ldda,
                                 c_one,     dAarray(ai+ib+panelj, aj+ib+panelj), ldda,
                                 batchCount, queue );
        }
    }

    //magma_free_cpu(cpuAarray);

    return 0;

    #undef dAarray
}


static magma_int_t
magma_zgetf2_batched_v2(
    magma_int_t m, magma_int_t n, magma_int_t stop_nb,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **ipiv_array, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
#define dA_array(i,j) dA_array, i, j
#define ipiv_array(i) ipiv_array, i

    #ifdef DBG
    magmaDoubleComplex* a;
    magma_getvector(1, sizeof(magmaDoubleComplex*), dA_array, 1, &a, 1, queue);
    magma_zprint_gpu(m, n, a, ldda, queue);
    #endif

    //printf("(%d, %d) -- %d\n", m, n, stop_nb);
    magma_int_t arginfo = 0;
    if(n <= stop_nb){
        //printf("(%d, %d) -- %d\n", m, n, stop_nb);
        arginfo = magma_zgetf2_fused_batched(m, n, dA_array(ai,aj), ldda, ipiv_array, info_array, batchCount, queue);
    }
    else{
        magma_int_t n1, n2;
        n1 = stop_nb; // n1 = n/2
        n2 = n - n1;

        // panel 1
        arginfo = magma_zgetf2_batched_v2(
                    m, n1, stop_nb,
                    dA_array(ai,aj), ldda,
                    ipiv_array, dpivinfo_array, info_array,
                    batchCount, queue);

        if(arginfo != 0) return arginfo;

        // swap right
        setup_pivinfo_batched(dpivinfo_array, ipiv_array(ai), m, n1, batchCount, queue);

        #ifdef ZGETF2_BATCHED_FUSED_UPDATE
        arginfo = -1;
        for(int inb = 32; inb >= 2; inb /= 2) {
            arginfo = magma_zgetf2_update_batched(
                        m, n1, n2, min(n2,inb),
                        dA_array, ai, aj, ldda,
                        dpivinfo_array, 0,
                        batchCount, queue );
            if(arginfo == 0) break;
        }

        if(arginfo != 0) {
        #endif
            // fused update failed to launch, use classic update
            magma_zlaswp_rowparallel_batched(
                n2,
                dA_array(ai,aj+n1), ldda,
                dA_array(ai,aj+n1), ldda,
                0, n1, dpivinfo_array,
                batchCount, queue);

            // trsm
            magmablas_ztrsm_recursive_batched(
                MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                n1, n2, MAGMA_Z_ONE,
                dA_array(ai,   aj), ldda,
                dA_array(ai,aj+n1), ldda,
                batchCount, queue );

            // gemm
            magma_zgemm_batched_core(
                MagmaNoTrans, MagmaNoTrans,
                m-n1, n2, n1,
                MAGMA_Z_NEG_ONE, dA_array(ai+n1,    aj), ldda,
                                 dA_array(ai   , aj+n1), ldda,
                MAGMA_Z_ONE,     dA_array(ai+n1, aj+n1), ldda,
                batchCount, queue );

        #ifdef ZGETF2_BATCHED_FUSED_UPDATE
            arginfo = 0;
        }
        #endif

        #ifdef DBG
        magmaDoubleComplex* aa;
        magma_getvector(1, sizeof(magmaDoubleComplex*), dA_array, 1, &aa, 1, queue);
        magma_zprint_gpu(m, n, aa, ldda, queue);
        #endif


        // panel 2
        magma_zgetf2_batched_v2(
                m-n1, n2, stop_nb,
                dA_array(ai+n1,aj+n1), ldda,
                ipiv_array, dpivinfo_array, info_array,
                batchCount, queue);

        // swap left
        setup_pivinfo_batched(dpivinfo_array, ipiv_array(ai+n1), m-n1, n2, batchCount, queue);
        adjust_ipiv_batched(ipiv_array(ai+n1), n2, n1, batchCount, queue);
        magma_zlaswp_rowparallel_batched(
                n1,
                dA_array(ai+n1,aj), ldda,
                dA_array(ai+n1,aj), ldda,
                n1, n, dpivinfo_array,
                batchCount, queue);
    }
    return arginfo;
#undef dA_array
#undef ipiv_array
}

extern "C" magma_int_t
magma_zgetf2_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **ipiv_array,
    magma_int_t** dpivinfo_array,
    magma_int_t *info_array,
    magma_int_t gbstep,
    magma_int_t batchCount,
    magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    if (m < 0) {
        arginfo = -1;
    } else if (n < 0 ) {
        arginfo = -2;
    } else if (ai < 0) {
        arginfo = -4;
    } else if (aj < 0 || aj != ai) {
        arginfo = -5;
    } else if (ldda < max(1,m)) {
        arginfo = -6;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    // first, test the fused panel
    arginfo = -1;
    for(magma_int_t inb = 32; inb >= 2; inb/=2 ) {
        arginfo = magma_zgetf2_batched_v2(m, n, inb, dA_array, ai, aj, ldda, ipiv_array, dpivinfo_array, info_array, batchCount, queue);
        if(arginfo == 0) break;
    }

    // negative arginfo means that fused panel did not launch
    if( arginfo != 0 ) {
        arginfo = magma_zgetf2_batched_v1(m, n, dA_array, ai, aj, ldda, ipiv_array, info_array, gbstep, batchCount, queue);
    }

    return arginfo;
}
