/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "magma_types.h"
#include "shuffle.cuh"
#include "hgetf2_devicefunc.cuh"

#define PRECISION_s

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

/******************************************************************************/
__global__ void
ihamax_kernel_batched(
        int length, magmaHalf **x_array, int xi, int xj, int lda, int incx,
        magma_int_t** ipiv_array, int ipiv_i,
        magma_int_t *info_array, int step, int gbstep)
{
    extern __shared__ magmaHalf sdata[];
    const int batchid = blockIdx.x;

    int tx = threadIdx.x;
    const magmaHalf *x = x_array[batchid] + xj * lda + xi;
    magma_int_t *ipiv           = ipiv_array[batchid] + ipiv_i;
    magma_int_t *info = &info_array[batchid];
    int linfo = ( (gbstep+step) == 0) ? 0 : *info;

    magmaHalf *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);

    ihamax_devfunc(length, x, incx, shared_x, shared_idx);

    if (tx == 0) {
        *ipiv  = shared_idx[0] + step + 1; // Fortran Indexing & adjust ipiv
        linfo  = ( shared_x[0] == MAGMA_H_ZERO && linfo == 0) ? (shared_idx[0]+step+gbstep+1) : linfo;
        *info = (magma_int_t)linfo;
    }
}


/******************************************************************************/
__global__ void
ihamax_kernel_native(
        int length, magmaHalf_ptr x, int incx,
        magma_int_t* ipiv, magma_int_t *info,
        int step, int gbstep)
{
    extern __shared__ magmaHalf sdata[];
    const int tx = threadIdx.x;

    magmaHalf *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);
    int linfo = ( (gbstep+step) == 0) ? 0 : *info;

    ihamax_devfunc(length, x, incx, shared_x, shared_idx);
    if (tx == 0) {
        *ipiv  = shared_idx[0] + step + 1; // Fortran Indexing
        linfo  = ( shared_x[0] == MAGMA_H_ZERO && linfo == 0) ? (shared_idx[0]+step+gbstep+1) : linfo;
        *info = (magma_int_t)linfo;
    }
}


/***************************************************************************//**
    Purpose
    -------

    ISAMAX find the index of max absolute value of elements in x and store the index in ipiv

    This is an internal routine that might have many assumption.

    Arguments
    ---------

    @param[in]
    length       INTEGER
            On entry, length specifies the size of vector x. length >= 0.


    @param[in]
    x_array     Array of pointers, dimension (batchCount).
            Each is a REAL array of dimension


    @param[in]
    xi      INTEGER
            Row offset, internal use

    @param[in]
    xj      INTEGER
            Column offset, internal use

    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    step    INTEGER
            the offset of ipiv

    @param[in]
    lda    INTEGER
            The leading dimension of each array A, internal use to find the starting position of x.

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
    gbstep    INTEGER
            the offset of info, internal use

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_iamax_batched
*******************************************************************************/
extern "C" magma_int_t
magma_ihamax_batched(
        magma_int_t length,
        magmaHalf **x_array, magma_int_t xi, magma_int_t xj, magma_int_t lda, magma_int_t incx,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t step, magma_int_t gbstep, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    if (length == 0 ) return 0;

    dim3 grid(batchCount, 1, 1);
    dim3 threads(zamax, 1, 1);

    int chunk = magma_ceildiv( length, zamax );

    ihamax_kernel_batched<<< grid, threads, zamax * (sizeof(magmaHalf) + sizeof(int)), queue->cuda_stream() >>>
    (length, x_array, xi, xj, lda, incx, ipiv_array, ipiv_i, info_array, step, gbstep);

    return 0;
}


/******************************************************************************/
// For use in magma_isamax_native only
// cublasIsamax always writes 32bit pivots, so make sure it is magma_int_t
// __global__ void magma_hpivcast(magma_int_t* dipiv)
// {
//     // uses only 1 thread
//     int* address = (int*)dipiv;
//     int pivot = *address;          // read the value written by cuBLAS (int)
//     *dipiv = (magma_int_t)pivot;    // write it back in the same address as dipiv
// }

/******************************************************************************/
extern "C" magma_int_t
magma_ihamax_native(
    magma_int_t length,
    magmaHalf_ptr x, magma_int_t incx,
    magma_int_t* ipiv, magma_int_t *info,
    magma_int_t step, magma_int_t gbstep, magma_queue_t queue)
{
    if (length == 0 ) return 0;

    // TODO: decide the best isamax for all precisions
    //if( length <= 15360 ) {
        dim3 grid(1, 1, 1);
        dim3 threads(zamax, 1, 1);

        ihamax_kernel_native<<< grid, threads, zamax * (sizeof(magmaHalf) + sizeof(int)), queue->cuda_stream() >>>
        (length, x, incx, ipiv, info, step, gbstep);
    // cublas does not support Ihamax
    // }
    // else {
    // #ifdef MAGMA_HAVE_CUDA
    //     cublasPointerMode_t ptr_mode;
    //     cublasGetPointerMode(queue->cublas_handle(), &ptr_mode);
    //     cublasSetPointerMode(queue->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
    //     cublasIsamax(queue->cublas_handle(), length, x, 1, (int*)(ipiv));
    //     magma_spivcast<<< 1, 1, 0, queue->cuda_stream() >>>( ipiv );

    //     cublasSetPointerMode(queue->cublas_handle(), ptr_mode);
    // #elif defined(MAGMA_HAVE_HIP)
    //     hipblasPointerMode_t ptr_mode;
    //     hipblasGetPointerMode(queue->hipblas_handle(), &ptr_mode);
    //     hipblasSetPointerMode(queue->hipblas_handle(), CUBLAS_POINTER_MODE_DEVICE);

    //     hipblasIsamax(queue->hipblas_handle(), length, (const magmaHalf*)x, 1, (int*)(ipiv));
    //     magma_spivcast<<< 1, 1, 0, queue->cuda_stream() >>>( ipiv );

    //     hipblasSetPointerMode(queue->hipblas_handle(), ptr_mode);
    // #endif

    //     adjust_ipiv( ipiv, 1, step, queue);
    // }
    return 0;
}

/******************************************************************************/
__global__
void hswap_kernel_batched(
        magma_int_t n,
        magmaHalf **x_array, magma_int_t xi, magma_int_t xj, magma_int_t incx,
        magma_int_t step, magma_int_t** ipiv_array)
{
    const int batchid = blockIdx.x;
    magmaHalf *x = x_array[batchid] + xj * incx + xi;
    magma_int_t *ipiv = ipiv_array[batchid] + xi;

    magmaHalf* xpiv = x_array[batchid] + (xj+step) * incx + (xi+step);
    magmaHalf rx_abs = __habs( MAGMA_H_REAL(xpiv[0]) ) + __habs( MAGMA_H_IMAG(xpiv[0]) );
    if( rx_abs != MAGMA_H_ZERO) {
        hswap_device(n, x, incx, step, ipiv);
    }
}


/******************************************************************************/
__global__
void sswap_kernel_native( magma_int_t n,
                          magmaHalf_ptr x, magma_int_t incx,
                          magma_int_t step, magma_int_t* ipiv)
{
    hswap_device(n, x, incx, step, ipiv);
}


/***************************************************************************//**
    Purpose
    -------

    sswap two row in x.  index (ipiv[step]-1)-th and index step -th

    This is an internal routine that might have many assumption.

    Arguments
    ---------

    @param[in]
    n       INTEGER
            On entry, n specifies the size of vector x. n >= 0.


    @param[in]
    dA_array  Array of pointers, dimension (batchCount).
            Each is a REAL array of dimension


    @param[in]
    ai      INTEGER
            Row offset, internal use.

    @param[in]
    aj      INTEGER
            Column offset, internal use.

    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    step    INTEGER
            The starting address of matrix C in A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).


    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_swap_batched
*******************************************************************************/
extern "C" magma_int_t
magma_hswap_batched( magma_int_t n,
                     magmaHalf **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t incx,
                     magma_int_t step, magma_int_t** ipiv_array,
                     magma_int_t batchCount, magma_queue_t queue)
{
    /*
    sswap two row: (ipiv[step]-1)th and step th
    */
    if ( n  > MAX_NTHREADS)
    {
        fprintf( stderr, "%s nb=%lld > %lld, not supported\n",
                 __func__, (long long) n, (long long) MAX_NTHREADS );
        return -15;
    }
    dim3 grid(batchCount, 1, 1);
    dim3 threads(zamax, 1, 1);

    hswap_kernel_batched
        <<< grid, threads, 0, queue->cuda_stream() >>>
        (n, dA_array, ai, aj, incx, step, ipiv_array);
    return 0;
}


/******************************************************************************/
extern "C" void
magma_hswap_native( magma_int_t n, magmaHalf_ptr x, magma_int_t incx,
                    magma_int_t step, magma_int_t* ipiv,
                    magma_queue_t queue)
{
    /*
    sswap two row: (ipiv[step]-1)th and step th
    */
    if ( n  > MAX_NTHREADS){
        fprintf( stderr, "%s nb=%lld > %lld, not supported\n",
                 __func__, (long long) n, (long long) MAX_NTHREADS );
    }
    dim3 grid(1, 1, 1);
    dim3 threads(zamax, 1, 1);

    sswap_kernel_native
        <<< grid, threads, 0, queue->cuda_stream() >>>
        (n, x, incx, step, ipiv);
}

/******************************************************************************/
template<int N>
__global__
void hscal_hger_1d_kernel_native( int m,
                                magmaHalf_ptr dA, int lda,
                                magma_int_t *info, int step, int gbstep)
{
    // This dev function has a return statement inside, be sure
    // not to merge it with another dev function. Otherwise, the
    // return statement should be converted into an if-statement
    hscal_hger_device<N>(m, dA, lda, info, step, gbstep);
}


/******************************************************************************/
__global__
void hscal_hger_1d_generic_kernel_native( int m, int n,
                                magmaHalf_ptr dA, int lda,
                                magma_int_t *info, int step, int gbstep)
{
    // This dev function has a return statement inside, be sure
    // not to merge it with another dev function. Otherwise, the
    // return statement should be converted into an if-statement
    hscal_hger_generic_device(m, n, dA, lda, info, step, gbstep);
}


/******************************************************************************/
template<int N>
__global__
void hscal_hger_1d_kernel_batched(int m, magmaHalf **dA_array, int ai, int aj, int lda, magma_int_t *info_array, int step, int gbstep)
{
    const int batchid = blockIdx.z;
    magmaHalf* dA = dA_array[batchid] + aj * lda + ai;
    magma_int_t *info = &info_array[batchid];
    hscal_hger_device<N>(m, dA, lda, info, step, gbstep);
}


/******************************************************************************/
__global__
void hscal_hger_1d_generic_kernel_batched(int m, int n, magmaHalf **dA_array, int ai, int aj, int lda, magma_int_t *info_array, int step, int gbstep)
{
    const int batchid = blockIdx.z;
    magmaHalf* dA = dA_array[batchid] + aj * lda + ai;
    magma_int_t *info = &info_array[batchid];
    hscal_hger_generic_device(m, n, dA, lda, info, step, gbstep);
}


/******************************************************************************/
extern "C"
magma_int_t
magma_hscal_hger_batched(
    magma_int_t m, magma_int_t n,
    magmaHalf **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
    magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized kernel which merged sscal and sger the two kernels
    1) sscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a sger Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */
    if ( n == 0) return 0;
    if ( n > MAX_NTHREADS ) {
        fprintf( stderr, "%s nb=%lld, > %lld, not supported\n", __func__, (long long) n, (long long) MAX_NTHREADS );
        return -15;
    }

    magma_int_t max_batchCount = queue->get_maxBatch();
    const int tbx = 256;
    dim3 threads(tbx, 1, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid(magma_ceildiv(m,tbx), 1, ibatch);

        switch(n){
            case  1: hscal_hger_1d_kernel_batched< 1><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA_array+i, ai, aj, lda, info_array+i, step, gbstep);break;
            case  2: hscal_hger_1d_kernel_batched< 2><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA_array+i, ai, aj, lda, info_array+i, step, gbstep);break;
            case  3: hscal_hger_1d_kernel_batched< 3><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA_array+i, ai, aj, lda, info_array+i, step, gbstep);break;
            case  4: hscal_hger_1d_kernel_batched< 4><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA_array+i, ai, aj, lda, info_array+i, step, gbstep);break;
            case  5: hscal_hger_1d_kernel_batched< 5><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA_array+i, ai, aj, lda, info_array+i, step, gbstep);break;
            case  6: hscal_hger_1d_kernel_batched< 6><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA_array+i, ai, aj, lda, info_array+i, step, gbstep);break;
            case  7: hscal_hger_1d_kernel_batched< 7><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA_array+i, ai, aj, lda, info_array+i, step, gbstep);break;
            case  8: hscal_hger_1d_kernel_batched< 8><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA_array+i, ai, aj, lda, info_array+i, step, gbstep);break;
            default: hscal_hger_1d_generic_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>(m, n, dA_array+i, ai, aj, lda, info_array+i, step, gbstep);
        }
    }
    return 0;
}


/******************************************************************************/
extern "C"
magma_int_t
magma_hscal_hger_native(
    magma_int_t m, magma_int_t n,
    magmaHalf_ptr dA, magma_int_t lda,
    magma_int_t *info, magma_int_t step, magma_int_t gbstep,
    magma_queue_t queue)
{
    /*
    Specialized kernel which merged sscal and sger the two kernels
    1) sscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a sger Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */
    if ( n == 0) return 0;
    if ( n > MAX_NTHREADS ) {
        fprintf( stderr, "%s nb=%lld, > %lld, not supported\n", __func__, (long long) n, (long long) MAX_NTHREADS );
        return -15;
    }
    const int tbx = 256;
    dim3 grid(magma_ceildiv(m,tbx), 1, 1);
    dim3 threads(tbx, 1, 1);
    switch(n){
        case 1: hscal_hger_1d_kernel_native<1><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA, lda, info, step, gbstep);break;
        case 2: hscal_hger_1d_kernel_native<2><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA, lda, info, step, gbstep);break;
        case 3: hscal_hger_1d_kernel_native<3><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA, lda, info, step, gbstep);break;
        case 4: hscal_hger_1d_kernel_native<4><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA, lda, info, step, gbstep);break;
        case 5: hscal_hger_1d_kernel_native<5><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA, lda, info, step, gbstep);break;
        case 6: hscal_hger_1d_kernel_native<6><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA, lda, info, step, gbstep);break;
        case 7: hscal_hger_1d_kernel_native<7><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA, lda, info, step, gbstep);break;
        case 8: hscal_hger_1d_kernel_native<8><<<grid, threads, 0, queue->cuda_stream()>>>( m, dA, lda, info, step, gbstep);break;
        default: hscal_hger_1d_generic_kernel_native<<<grid, threads, 0, queue->cuda_stream()>>>( m, n, dA, lda, info, step, gbstep);
    }
    return 0;
}


/******************************************************************************/
__global__
void hgetf2trsm_kernel_batched(int ib, int n, magmaHalf **dA_array, int step, int lda)
{

    extern __shared__ magmaHalf shared_data[];

    /*
        this kernel does the safe nonblocked TRSM operation
        B = A^-1 * B
    */
    const int batchid = blockIdx.x;

    magmaHalf *A_start = dA_array[batchid];
    magmaHalf *A = &(A_start[step + step * lda]);
    magmaHalf *B = &(A_start[step + (step+ib) * lda]);
    magmaHalf *shared_a = shared_data;
    magmaHalf *shared_b = shared_data+ib*ib;

    int tid = threadIdx.x;
    int i,d;


    // Read A and B at the same time to the shared memory (shared_a shared_b)
    // note that shared_b = shared_a+ib*ib so its contiguous
    // I can make it in one loop reading
    if ( tid < ib) {
        #pragma unroll
        for (i=0; i < n+ib; i++) {
            shared_a[tid + i*ib] = A[tid + i*lda];
        }
    }
    __syncthreads();

    if (tid < n) {
        #pragma unroll
        for (d=0;  d < ib-1; d++) {
            for (i=d+1; i < ib; i++) {
                shared_b[i+tid*ib] += (MAGMA_H_NEG_ONE) * shared_a[i+d*ib] * shared_b[d+tid*ib];
            }
        }
    }
    __syncthreads();

    // write back B
    if ( tid < ib) {
        #pragma unroll
        for (i=0; i < n; i++) {
            B[tid + i*lda] = shared_b[tid + i*ib];
        }
    }
}


/***************************************************************************//**
    Purpose
    -------

    hgetf2trsm solves one of the matrix equations on gpu

     B = C^-1 * B

    where C, B are part of the matrix A in dA_array,

    This version load C, B into shared memory and solve it
    and copy back to GPU device memory.
    This is an internal routine that might have many assumption.

    Arguments
    ---------
    @param[in]
    ib       INTEGER
            The number of rows/columns of each matrix C, and rows of B.  ib >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix B.  n >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a REAL array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[in]
    step    INTEGER
            The starting address of matrix C in A.  LDDA >= max(1,M).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getf2_batched
*******************************************************************************/
extern "C" void
magma_hgetf2trsm_batched(magma_int_t ib, magma_int_t n, magmaHalf **dA_array,
                         magma_int_t step, magma_int_t ldda,
                         magma_int_t batchCount, magma_queue_t queue)
{
    if ( n == 0 || ib == 0 ) return;
    size_t shared_size = sizeof(magmaHalf)*(ib*(ib+n));

    // TODO TODO TODO
    if ( shared_size > (MAX_SHARED_ALLOWED*1024) ) // limit the shared memory to 46K leaving 2K for extra
    {
        fprintf( stderr, "%s: error out of shared memory\n", __func__ );
        return;
    }

    dim3 grid(batchCount, 1, 1);
    dim3 threads(max(n,ib), 1, 1);

    hgetf2trsm_kernel_batched
    <<< grid, threads, shared_size, queue->cuda_stream() >>>
    (ib, n, dA_array, step, ldda);
}


/******************************************************************************/
template<int NB>
__global__ void
hgetf2trsm_2d_kernel( int m, int n,
                           magmaHalf_ptr dA, int ldda,
                           magmaHalf_ptr dB, int lddb)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ magmaHalf sA[NB * NB];
    __shared__ magmaHalf sB[NB * NB];

    // init sA & sB
    sA[ ty * NB + tx ] = MAGMA_H_ZERO;
    sB[ ty * NB + tx ] = MAGMA_H_ZERO;

    const int nblocks = magma_ceildiv(n, NB);
    const int n_ = n - (nblocks-1) * NB;

    // load A
    if( ty < m && tx < m && tx > ty){
        sA[ty * NB + tx] = dA[ty * ldda + tx];
    }

    if( ty == tx ){
        // ignore diagonal elements
        sA[tx * NB + tx] = MAGMA_H_ONE;
    }
    __syncthreads();

    #pragma  unroll
    for(int s = 0; s < nblocks-1; s++){
        // load B
        if( tx < m ){
            sB[ ty * NB + tx] = dB[ ty * lddb + tx ];
        }

        // no need to sync because each thread column is less than 32
        // solve
        #pragma unroll
        for(int i = 0; i < NB; i++){
            if(tx >  i){
                 sB[ ty * NB + tx ] -= sA[ i * NB + tx ] * sB[ ty * NB + i ];
            }
        }

        // write B
        if( tx < m){
            dB[ ty * lddb + tx ] = sB[ ty * NB + tx ];
        }
        dB += NB * lddb;
    }

    // last, possible partial, block
    if( ty < n_ && tx < m){
        sB[ ty * NB + tx] = dB[ ty * lddb + tx ];
    }

    #pragma unroll
    for(int i = 0; i < NB; i++){
        if(tx >  i){
             sB[ ty * NB + tx ] -= sA[ i * NB + tx ] * sB[ ty * NB + i ];
        }
    }

    if( ty < n_ && tx < m){
        dB[ ty * lddb + tx ] = sB[ ty * NB + tx ];
    }
}

#if 0
/******************************************************************************/
extern"C" void
magma_hgetf2trsm_2d_native(
    magma_int_t m, magma_int_t n,
    magmaHalf_ptr dA, magma_int_t ldda,
    magmaHalf_ptr dB, magma_int_t lddb,
    magma_queue_t queue)
{
    if( m > 32 ){
        magma_strsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                     m, n, MAGMA_H_ONE,
                     dA, ldda,
                     dB, lddb, queue );
        return;
    }

    const int m8 = magma_roundup(m, 8);
    dim3 grid(1, 1, 1);
    dim3 threads(m8, m8, 1);

    switch(m8){
        case  8: hgetf2trsm_2d_kernel< 8><<<grid, threads, 0, queue->cuda_stream() >>>( m, n, dA, ldda, dB, lddb ); break;
        case 16: hgetf2trsm_2d_kernel<16><<<grid, threads, 0, queue->cuda_stream() >>>( m, n, dA, ldda, dB, lddb ); break;
        case 24: hgetf2trsm_2d_kernel<24><<<grid, threads, 0, queue->cuda_stream() >>>( m, n, dA, ldda, dB, lddb ); break;
        case 32: hgetf2trsm_2d_kernel<32><<<grid, threads, 0, queue->cuda_stream() >>>( m, n, dA, ldda, dB, lddb ); break;
        default:;
    }
}
#endif

/******************************************************************************/
__global__ void
zcomputecolumn_kernel_shared_batched( int m, int paneloffset, int step,
                                      magmaHalf **dA_array, int ai, int aj,
                                      int lda, magma_int_t **ipiv_array, magma_int_t *info_array, int gbstep)
{
    const int batchid = blockIdx.x;
    extern __shared__ magmaHalf shared_data[];

    int gboff = paneloffset+step;
    magma_int_t *ipiv           = ipiv_array[batchid] + ai;
    magmaHalf *A_start = dA_array[batchid] + aj * lda + ai;
    magmaHalf *A0j     = &(A_start[paneloffset + (paneloffset+step) * lda]);
    magmaHalf *A00     = &(A_start[paneloffset + paneloffset * lda]);

    magmaHalf *shared_A = shared_data;
    __shared__ magmaHalf  shared_x[zamax];
    __shared__ int     shared_idx[zamax];
    __shared__ magmaHalf alpha;
    int tid   = threadIdx.x;
    int linfo = ((gboff + gbstep) == 0 ) ? 0 : info_array[batchid];

    // checkinfo to avoid computation of the singular matrix
    //if (info_array[batchid] != 0 ) return;


    int nchunk = magma_ceildiv( m, MAX_NTHREADS );
    // read the current column from dev to shared memory
    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m ) shared_A[tid + s * MAX_NTHREADS] = A0j[tid + s * MAX_NTHREADS];
    }
    __syncthreads();

    // update this column
    if ( step > 0 ) {
        zupdate_device( m, step, A00, lda, shared_A, 1);
        __syncthreads();
    }

    // if ( tid < (m-step) ) // DO NO PUT THE IF CONDITION HERE SINCE isamax_devfunc HAS __syncthreads INSIDE.
    // So let all htreads call this routine it will handle correctly based on the size
    // note that isamax need only 128 threads, s
    ihamax_devfunc(m-step, shared_A+step, 1, shared_x, shared_idx);
    if (tid == 0) {
        ipiv[gboff]  = shared_idx[0] + gboff + 1; // Fortran Indexing
        alpha = shared_A[shared_idx[0]+step];
        //printf("@ step %d ipiv=%d where gboff=%d  shared_idx %d alpha %5.3f\n",step,ipiv[gboff],gboff,shared_idx[0],alpha);
        linfo  = ( shared_x[0] == MAGMA_H_ZERO && linfo == 0) ? (shared_idx[0]+gboff+gbstep+1) : linfo;
        info_array[batchid] = (magma_int_t)linfo;
        //if (shared_x[0] == MAGMA_H_ZERO) {
        //    info_array[batchid] = shared_idx[0] + gboff + gbstep + 1;
        //}
    }
    __syncthreads();

    //if (shared_x[0] == MAGMA_H_ZERO) return;
    //__syncthreads();

    if( shared_x[0] != MAGMA_H_ZERO ) {
        hscal5_device( m-step, shared_A+step, alpha);
        // there is sync at the end of sscal5_device
    }

    // put back the pivot that has been scaled with itself menaing =1
    if (tid == 0)  shared_A[shared_idx[0] + step] = alpha;
    __syncthreads();

    // write back from shared to dev memory
    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m )
        {
            A0j[tid + s * MAX_NTHREADS] = shared_A[tid + s * MAX_NTHREADS];
            //printf("@ step %d tid %d updating A=x*alpha after A= %5.3f\n",step,tid,shared_A[tid]);
        }
    }
    __syncthreads();
}


/******************************************************************************/
extern "C"
magma_int_t magma_hcomputecolumn_batched( magma_int_t m, magma_int_t paneloffset, magma_int_t step,
                                          magmaHalf **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
                                          magma_int_t **ipiv_array,
                                          magma_int_t *info_array, magma_int_t gbstep,
                                          magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized kernel which merged sscal and sger the two kernels
    1) sscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a sger Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */
    if ( m == 0) return 0;

    size_t all_shmem_size = zamax*(sizeof(magmaHalf)+sizeof(int)) + (m+2)*sizeof(magmaHalf);
    if ( all_shmem_size >  (MAX_SHARED_ALLOWED*1024) ) // limit the shared memory to 44K leaving 4K for extra
    {
        fprintf( stderr, "%s error out of shared memory\n", __func__ );
        return -20;
    }

    size_t shared_size = sizeof(magmaHalf)*m;
    dim3 grid(batchCount, 1, 1);
    dim3 threads(min(m, MAX_NTHREADS), 1, 1);

    zcomputecolumn_kernel_shared_batched
    <<< grid, threads, shared_size, queue->cuda_stream() >>>
    (m, paneloffset, step, dA_array, ai, aj, lda, ipiv_array, info_array, gbstep);

    return 0;
}

/******************************************************************************/
template<int N>
__global__ void
hgetf2_fused_kernel_batched( int m,
                           magmaHalf** dA_array, int ai, int aj, int ldda,
                           magma_int_t** dipiv_array, magma_int_t* info_array, int batchCount)
{
    const int tx = threadIdx.x;
    const int batchid = blockIdx.x * blockDim.y + threadIdx.y;
    if(batchid >= batchCount)return;

    int rowid, gbstep = aj;
    int linfo = (gbstep == 0) ? 0 : info_array[batchid];

    // shared memory workspace
    extern __shared__ magmaHalf zdata[];
    magmaHalf* swork = (magmaHalf*)zdata;

    // read
    magmaHalf* dA = dA_array[batchid] + aj * ldda + ai;
    magmaHalf  rA[N] = {MAGMA_H_ZERO};
    #pragma unroll
    for(int i = 0; i < N; i++){
        rA[i] = dA[ i * ldda + tx ];
    }

     hgetf2_fused_device<N>(
             m, min(m,N), rA,
             dipiv_array[batchid] + ai,
             swork, linfo, gbstep, rowid);

    // write
    if(tx == 0){
        info_array[batchid] = (magma_int_t)( linfo );
    }

    #pragma unroll
    for(int i = 0; i < N; i++){
        dA[ i * ldda + rowid ] = rA[i];
    }

}

/******************************************************************************/
template<int N>
static magma_int_t
magma_hgetf2_fused_kernel_driver_batched(
    magma_int_t m,
    magmaHalf **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **dipiv_array,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    magma_int_t ntcol = (m >= 32)? 1 : (32/m);
    int shmem = 0, shmem_max = 0;   // not magma_int_t (causes problems with 64bit builds)
    shmem += N * sizeof(magmaHalf);
    shmem += m * sizeof(magmaHalf);
    shmem += m * sizeof(int);    // not magma_int_t
    shmem += N * sizeof(int);    // not magma_int_t
    shmem *= ntcol;

    dim3 grid(magma_ceildiv(batchCount,ntcol), 1, 1);
    dim3 threads(m, ntcol, 1);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, nthreads = m * ntcol;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(hgetf2_fused_kernel_batched<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    void *kernel_args[] = {&m, &dA_array, &ai, &aj, &ldda, &dipiv_array, &info_array, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)hgetf2_fused_kernel_batched<N>, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    magma_hgetf2_reg_batched computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges. This routine is used for batch LU panel
    factorization, and has specific assumption about the value of N

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is a right-looking unblocked version of the algorithm. The routine is a batched
    version that factors batchCount M-by-N matrices in parallel.

    This version load an entire matrix (m*n) into registers and factorize it with pivoting
    and copy back to GPU device memory.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  ib >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a REAL array on the GPU, dimension (LDDA,N).
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
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
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
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getf2_batched
*******************************************************************************/
extern "C" magma_int_t
magma_hgetf2_fused_batched(
    magma_int_t m, magma_int_t n,
    magmaHalf **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **dipiv_array,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    if(m < 0) {
        info = -1;
    }
    else if(n < 0 || n > 32){
        fprintf( stderr, "%s: n = %4lld not supported, must be between 0 and %4lld\n",
                 __func__, (long long) m, (long long) 32);
        info = -2;
    }

    if(info < 0) return info;

    switch(n) {
        case  1: info = magma_hgetf2_fused_kernel_driver_batched< 1>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  2: info = magma_hgetf2_fused_kernel_driver_batched< 2>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  3: info = magma_hgetf2_fused_kernel_driver_batched< 3>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  4: info = magma_hgetf2_fused_kernel_driver_batched< 4>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  5: info = magma_hgetf2_fused_kernel_driver_batched< 5>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  6: info = magma_hgetf2_fused_kernel_driver_batched< 6>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  7: info = magma_hgetf2_fused_kernel_driver_batched< 7>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  8: info = magma_hgetf2_fused_kernel_driver_batched< 8>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case  9: info = magma_hgetf2_fused_kernel_driver_batched< 9>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 10: info = magma_hgetf2_fused_kernel_driver_batched<10>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 11: info = magma_hgetf2_fused_kernel_driver_batched<11>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 12: info = magma_hgetf2_fused_kernel_driver_batched<12>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 13: info = magma_hgetf2_fused_kernel_driver_batched<13>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 14: info = magma_hgetf2_fused_kernel_driver_batched<14>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 15: info = magma_hgetf2_fused_kernel_driver_batched<15>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 16: info = magma_hgetf2_fused_kernel_driver_batched<16>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 17: info = magma_hgetf2_fused_kernel_driver_batched<17>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 18: info = magma_hgetf2_fused_kernel_driver_batched<18>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 19: info = magma_hgetf2_fused_kernel_driver_batched<19>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 20: info = magma_hgetf2_fused_kernel_driver_batched<20>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 21: info = magma_hgetf2_fused_kernel_driver_batched<21>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 22: info = magma_hgetf2_fused_kernel_driver_batched<22>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 23: info = magma_hgetf2_fused_kernel_driver_batched<23>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 24: info = magma_hgetf2_fused_kernel_driver_batched<24>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 25: info = magma_hgetf2_fused_kernel_driver_batched<25>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 26: info = magma_hgetf2_fused_kernel_driver_batched<26>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 27: info = magma_hgetf2_fused_kernel_driver_batched<27>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 28: info = magma_hgetf2_fused_kernel_driver_batched<28>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 29: info = magma_hgetf2_fused_kernel_driver_batched<29>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 30: info = magma_hgetf2_fused_kernel_driver_batched<30>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 31: info = magma_hgetf2_fused_kernel_driver_batched<31>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        case 32: info = magma_hgetf2_fused_kernel_driver_batched<32>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue); break;
        default: info = -100;
    }

    return info;
}
