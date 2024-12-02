/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver

   @author Ahmad Abdelfattah
   @author Azzam Haidar

 */


#ifndef MAGMABLAS_SGETF2_DEVICES_Z_H
#define MAGMABLAS_SGETF2_DEVICES_Z_H

/******************************************************************************/
static __device__ __inline__ int
ihamax_devfunc(int length, const magmaHalf *x, int incx, magmaHalf *shared_x, int *shared_idx)
{
    int tx = threadIdx.x;
    magmaHalf res;
    magmaHalf  res1;
    int nchunk = magma_ceildiv( length, zamax );

    if ( tx < zamax ) {
        shared_x[tx]   = 0.0;
        shared_idx[tx] = tx; //-1; // -1 will crash the code in case matrix is singular, better is to put =tx and make check info at output
    }
    __syncthreads();

    for (int s =0; s < nchunk; s++)
    {
        if ( (tx + s * zamax < length) && (tx < zamax) )
        {
            res = x[(tx + s * zamax) * incx];
            res1 = __habs(MAGMA_H_REAL(res)) + __habs(MAGMA_H_IMAG(res));

            if ( res1  > shared_x[tx] )
            {
                shared_x[tx] = res1;
                shared_idx[tx] = tx + s * zamax;
            }
        }
    }
    __syncthreads();

    if (length >= zamax) // there are more than 128 threads working ==> all shared_x shared_idx are initialized here so I can call the fixed getidmax
        magma_getidmax<zamax>(tx, shared_x, shared_idx);
    else
        magma_getidmax_n(min(zamax,length), tx, shared_x, shared_idx);
    return shared_idx[0];
}

/******************************************************************************/
static __device__ __inline__
void hswap_device( magma_int_t n,
                   magmaHalf_ptr x, magma_int_t incx,
                   magma_int_t step, magma_int_t* ipiv)
{
    const int tx = threadIdx.x;

    __shared__ int jp;

    if (tx == 0){
        jp = ipiv[step] - 1;
    }
    __syncthreads();

    if (jp == step) return; // no pivot

    if (tx < n) {
        magmaHalf tmp = x[jp + tx * incx];
        x[jp + tx * incx] = x[step + tx * incx];
        x[step + tx * incx] = tmp;
    }
}

/******************************************************************************/
// This version swaps two rows that are specified at the input
// the logic deciding these two rows is assumed to be at the
// kernel level (unlike sswap_device)
static __device__ __inline__
void hswap_device_v2(
            magma_int_t n,
            magmaHalf_ptr x1, magma_int_t incx1,
            magmaHalf_ptr x2, magma_int_t incx2 )
{
    const int tx = threadIdx.x;

    if (tx < n) {
        magmaHalf tmp  = x1[tx * incx1];
        x1[tx * incx1]          = x2[tx * incx2];
        x2[tx * incx2]          = tmp;
    }
}

/******************************************************************************/
template<int N>
static __device__ __inline__
void hscal_hger_device( int m,
                         magmaHalf_ptr dA, int lda,
                         magma_int_t *info, int step, int gbstep)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * blockDim.x + tx;

    magmaHalf rA[N], reg;
    __shared__ magmaHalf shared_y[N];

    if (tx < N) {
        shared_y[tx] = dA[lda * tx];
    }
    __syncthreads();

    // terminate threads that are out of the range
    if (gtx == 0 || gtx >= m) return;

    magmaHalf rTmp = __habs(MAGMA_H_REAL( shared_y[0] ) ) + __habs( MAGMA_H_IMAG( shared_y[0] ) );

    reg = (rTmp == MAGMA_H_ZERO) ? MAGMA_H_ONE : MAGMA_H_DIV(MAGMA_H_ONE, shared_y[0]);

    #pragma unroll
    for(int i = 0; i < N; i++)
        rA[i] = dA[ i* lda + gtx ];

    rA[0] *= reg;

    #pragma unroll
    for(int i = 1; i < N; i++)
        rA[i] -= rA[0] * shared_y[i];

    #pragma unroll
    for(int i = 0; i < N; i++)
        dA[gtx + i * lda] = rA[i];
}

/******************************************************************************/
static __device__ __inline__
void hscal_hger_generic_device( int m, int n,
                         magmaHalf_ptr dA, int lda,
                         magma_int_t *info, int step, int gbstep)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * blockDim.x + tx;
    if (gtx == 0 || gtx >= m) return;

    magmaHalf rA, reg;
    magmaHalf rTmp;
    rA   = dA[0];
    rTmp = __habs(MAGMA_H_REAL( rA ) ) + __habs( MAGMA_H_IMAG( rA ) );

    reg = (rTmp == MAGMA_H_ZERO) ? MAGMA_H_ONE : MAGMA_H_DIV(MAGMA_H_ONE, rA);
    rA  = dA[ gtx ];
    rA *= reg;

    dA[ gtx ] = rA;
    #pragma unroll
    for(int i = 1; i < n; i++)
        dA[i * lda + gtx] -= rA * dA[i * lda + 0];
}

/******************************************************************************/
static __device__ __inline__
void
zupdate_device(int m, int step, magmaHalf* x, int ldx,  magmaHalf *A, int lda)
{
    int tid = threadIdx.x;
    int nchunk = magma_ceildiv( m, MAX_NTHREADS );
    int indx;
    //magmaHalf reg = MAGMA_H_ZERO;

    // update the current column by all the previous one
    #pragma unroll
    for (int i=0; i < step; i++) {
        for (int s=0; s < nchunk; s++)
        {
            indx = tid + s * MAX_NTHREADS;
            if ( indx > i  && indx < m ) {
                A[indx] -=  A[i] * x[indx + i*ldx];
                //printf("         @ step %d tid %d updating x[tid]*y[i]=A %5.3f %5.3f = %5.3f  at i %d\n", step, tid, x[tid + i*ldx], A[i], A[tid],i);
            }
        }
        __syncthreads();
    }

    //printf("         @ step %d tid %d adding %5.3f to A %5.3f make it %5.3f\n",step,tid,-reg,A[tid],A[tid]-reg);
}


/******************************************************************************/
static __device__ __inline__
void
hscal5_device(int m, magmaHalf* x, magmaHalf alpha)
{
    int tid = threadIdx.x;
    int nchunk = magma_ceildiv( m, MAX_NTHREADS );

    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m ) {
            #if 0
            x[tid + s * MAX_NTHREADS] *= MAGMA_H_DIV(MAGMA_H_ONE, alpha);
            #else
            x[tid + s * MAX_NTHREADS] = x[tid + s * MAX_NTHREADS]/alpha;
            #endif
        }
    }
    __syncthreads();
}

/******************************************************************************/
template<int WIDTH>
static __device__ __inline__
void
hgetf2_fused_device( int m, int minmn, magmaHalf rA[WIDTH], magma_int_t* dipiv,
                     magmaHalf* swork, int &linfo, int gbstep, int &rowid)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    magmaHalf reg       = MAGMA_H_ZERO;

    int max_id;
    magmaHalf rx_abs_max = MAGMA_H_ZERO;

    magmaHalf *sx = (magmaHalf*)(swork);
    magmaHalf* dsx = (magmaHalf*)(sx + blockDim.y * WIDTH);
    int* isx    = (int*)(dsx + blockDim.y * m);
    int* sipiv  = (int*)(isx + blockDim.y * m);
    sx    += ty * WIDTH;
    dsx   += ty * m;
    isx   += ty * m;
    sipiv += ty * WIDTH;

    rowid = tx;

    // init sipiv
    if(tx < WIDTH){
        sipiv[tx] = 0;
    }

    #pragma unroll
    for(int i = 0; i < WIDTH; i++){
        // isamax and find pivot
        dsx[ rowid ] = __habs(MAGMA_H_REAL( rA[i] )) + __habs(MAGMA_H_IMAG( rA[i] ));
        isx[ tx ] = tx;
        __syncthreads();
        magma_getidmax_n(m-i, tx, dsx+i, isx+i); // this devfunc has syncthreads at the end
        rx_abs_max = dsx[i];
        max_id = isx[i];
        linfo  = ( rx_abs_max == MAGMA_H_ZERO && linfo == 0) ? (gbstep+i+1) : linfo;
        if(tx == 0) {
            sipiv[i] = max_id;
        }
        __syncthreads();

        if( rowid == max_id ) {
            #pragma unroll
            for(int j = 0; j < WIDTH; j++){
                sx[j] = rA[j];
            }
        }
        __syncthreads();

        if( rx_abs_max != MAGMA_H_ZERO ) {
            if(rowid == max_id){
                rowid = i;
            }
            else if(rowid == i){
                rowid = max_id;
            }
        }
        __syncthreads();

        reg = (rx_abs_max == MAGMA_H_ZERO ) ? MAGMA_H_ONE : MAGMA_H_DIV(MAGMA_H_ONE, sx[i] );
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < WIDTH; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
    }

    // write
    if(tx < minmn){
        dipiv[tx] = (magma_int_t)(sipiv[tx] + 1); // fortran indexing
        //printf("--- ipiv[%d] --- = %d\n", tx, dipiv[tx]);
    }
}


#endif // MAGMABLAS_SGETF2_DEVICES_Z_H
