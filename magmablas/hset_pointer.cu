/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Ahmad Abdelfattah
       
       dedicated src for pointer arithmetic in fp16
*/

#include "magma_internal.h"
#include "magma_types.h"

#define PRECISION_h

__global__ void kernel_hset_pointer(
    magmaHalf **output_array,
    magmaHalf *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batch_offset)
{
    output_array[blockIdx.x] =  input + blockIdx.x * batch_offset + row + column * lda;
    assert((uintptr_t)output_array[blockIdx.x] % alignof(magmaHalf) == 0);
}

extern "C"
void magma_hset_pointer(
    magmaHalf **output_array,
    magmaHalf *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batch_offset,
    magma_int_t batchCount, 
    magma_queue_t queue)
{
    kernel_hset_pointer
        <<< batchCount, 1, 0, queue->cuda_stream() >>>
        (output_array, input, lda,  row, column, batch_offset);
}

/******************************************************************************/
__global__ void hdisplace_pointers_kernel(magmaHalf **output_array,
               magmaHalf **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column)
{
    magmaHalf *inpt = input_array[blockIdx.x];
    output_array[blockIdx.x] = &inpt[row + column * lda];
}

/***************************************************************************//**
    Purpose
    -------

    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row + lda * column; 
    
    Arguments
    ----------

    @param[out]
    output_array    Array of pointers, dimension (batchCount).
             Each pointer points to the new displacement of array A in input_array on the GPU
   
    @param[in]
    input_array     Array of pointers, dimension (batchCount).
             Each is a REAL array A of DIMENSION ( lda, column ) on the GPU

    @param[in]
    lda    INTEGER
            LDA specifies the leading dimension of A.

    @param[in]
    row       INTEGER
            On entry, row specifies the number of rows of the matrix A.

    @param[in]
    column       INTEGER
            On entry, column specifies the number of columns of the matrix A

    @param[in]
    batch_offset  INTEGER
                The starting pointer of each matrix A in input arrray

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
*******************************************************************************/
extern "C"
void magma_hdisplace_pointers(magmaHalf **output_array,
               magmaHalf **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount, magma_queue_t queue)
{
    hdisplace_pointers_kernel
        <<< batchCount, 1, 0, queue->cuda_stream() >>>
        (output_array, input_array, lda, row, column);
}